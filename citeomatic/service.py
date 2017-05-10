#!/usr/bin/env python3

import collections
import hashlib
import json
import logging
import os
import re
from typing import List

import boto3
import flask
import numpy as np
import requests
from citeomatic import display, elastic
from citeomatic.cache import LocalCache, S3Cache
from citeomatic.elastic import fetch_citations
from citeomatic.features import Corpus, Document, Featurizer
from citeomatic.neighbors import ANN, EmbeddingModel
from flask import Flask, Response, request, Blueprint

app = Flask(__name__, template_folder='.', static_folder='client/build/')

Prediction = collections.namedtuple(
    'Prediction',
    ['score', 'document', 'position', 'explanation', 'cited', 'pdf']
)


class APIModel(object):
    def __init__(
        self,
        models,
        featurizer: Featurizer,
        corpus: Corpus=None,
        ann: ANN=None,
        ann_embedding_model=None,
        max_neighbors=1000,
        candidate_min_in_citations=4,
    ):
        self.model = models['citeomatic']
        self.embedding_model = EmbeddingModel(featurizer, models['embedding']) if \
            ann_embedding_model is None else ann_embedding_model
        self.featurizer = featurizer
        self.explanation = None  # Explanation(self.model, featurizer)
        self._ann = ann
        self.corpus = corpus
        self.max_neighbors = max_neighbors
        self.candidate_min_in_citations = candidate_min_in_citations

    def get_ann_similar_documents(self, doc, top_n=NUM_ES_PAPERS):
        doc_embedded = self.embedding_model.embed(doc)
        return self._ann.get_nns_by_vector(doc_embedded, top_n)

    def predict(self, doc, top_n=DEFAULT_NUM_CITATIONS) -> List[Prediction]:
        bulk_ids = self.get_ann_similar_documents(doc, top_n=self.max_neighbors)
        bulk_ids = [
            bulk_id for bulk_id in bulk_ids
            if self.corpus[bulk_id].in_citation_count >=
            self.candidate_min_in_citations
        ]

        # Extend the candidate set with their citations
        citations_of_candidates = []
        for id in bulk_ids:
            if self.citation_source == 'es':
                citations_of_candidates.extend(fetch_citations(id))
            else:
                citations_of_candidates.extend(self.corpus[id].citations)
        bulk_ids = list(set(citations_of_candidates + bulk_ids))

        logging.info('Fetching %d documents ' % len(bulk_ids))

        if self.use_es_neighbors:
            candidates = elastic.fetch_all(
                bulk_ids[:TOTAL_CANDIDATES]
            )  # Add method to make multiple requests if count exceed 1k
        else:
            if self.fetch_docs_from_es:
                candidates = elastic.fetch_all(bulk_ids[:TOTAL_CANDIDATES])
            else:
                logging.info("Fetching documents from corpus")
                candidates = [self.corpus[paper_id] for paper_id in bulk_ids]

        logging.info('Featurizing... %d documents ' % len(candidates))
        features = self.featurizer.transform_query_and_results(doc, candidates)
        logging.info('Predicting...')
        scores = self.model.predict(features, batch_size=64).flatten()
        best_matches = np.argsort(scores)[::-1]

        predictions = []
        for i, match_idx in enumerate(best_matches[:top_n]):
            if candidates[match_idx].title.lower() == doc.title.lower():
                continue
            predictions.append(
                Prediction(
                    score=float(scores[match_idx]),
                    document=candidates[match_idx],
                    pdf=sha_to_url(str(candidates[match_idx].id)),
                    position=i,
                    explanation={},
                    cited=candidates[match_idx].title.lower() in doc.citations
                )
            )
        logging.info("Done! Found %s predictions." % len(predictions))
        return predictions


def document_from_dict(doc):
    defaults = {
        'title': '',
        'abstract': '',
        'authors': [],
        'citations': [],
        'year': 2016,
        'id': 0,
        'venue': '',
        'in_citation_count': 0,
        'out_citation_count': 0,
        'key_phrases': []
    }
    defaults.update(doc)

    return Document(**defaults)


def dict_from_document(doc):
    doc_dict = {}
    for field in doc._fields:
        doc_dict[field] = getattr(doc, field)
    return doc_dict


def find_citations(source_file, doc):
    predictions = app.config['API_MODEL'].predict(doc)

    response = {
        'predictions':
            [
                {
                    'document': p.document._asdict(),
                    'score': p.score,
                    'explanation': p.explanation,
                    'cited': str(p.cited) if p.cited else '',
                    'pdf': p.pdf,
                    'bibtex': display.document_to_bibtex(p.document)
                } for p in predictions
            ]
    }
    response.update(doc._asdict())
    response['source_file'] = source_file

    # logging.debug("Citeomatic response %s", predictions)
    return response


@app.route('/api/predictions/json', methods=['GET', 'POST'])
def predict_json():
    doc = document_from_dict(request.get_json())
    predictions = app.config['API_MODEL'].predict(doc)
    return flask.jsonify(
        {
            'predictions':
                [
                    {
                        'document': p.document._asdict(),
                        'score': p.score,
                        'explanation': p.explanation
                    } for p in predictions
                ]
        }
    )


@app.route('/api/pdfs', methods=['GET'])
def fetch_pdfs():
    ids = request.args['ids'].split(',')
    return flask.Response('Not yet implemented. Sorry!', mimetype='text/plain')


@app.route("/upload/json", methods=['POST'])
def upload_form():
    logging.debug(request.get_json())
    req_body = request.get_json()
    title = req_body['title']
    abstract = req_body['abstract']
    authors = req_body['authors'].split(',')

    json_body = {'title': title, 'abstract': abstract, 'authors': authors}
    return flask.jsonify(find_citations('', document_from_dict(json_body)))
