#!/usr/bin/env python3

import collections
import logging
from typing import List

import flask
import numpy as np
from flask import Flask, request

from citeomatic import display
from citeomatic.common import Document, FieldNames
from citeomatic.corpus import Corpus
from citeomatic.features import Featurizer
from citeomatic.neighbors import ANN, EmbeddingModel

NUM_ANN_CANDIDATES = 1000
DEFAULT_NUM_CITATIONS = 50
TOTAL_CANDIDATES = 1000

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

    def get_ann_similar_documents(self, doc, top_n=NUM_ANN_CANDIDATES):
        doc_embedded = self.embedding_model.embed(doc)
        return self._ann.get_nns_by_vector(doc_embedded, top_n)

    @staticmethod
    def _sha_to_url(sha):
        return "https://pdfs.semanticscholar.org/" + sha[0:4] + "/" + sha[4:] + ".pdf"

    def predict(self, doc, top_n=DEFAULT_NUM_CITATIONS) -> List[Prediction]:
        candidate_ids = self.get_ann_similar_documents(doc, top_n=self.max_neighbors)
        candidate_ids = [
            bulk_id for bulk_id in candidate_ids
            if self.corpus[bulk_id].in_citation_count >=
            self.candidate_min_in_citations
        ]

        # Extend the candidate set with their citations
        citations_of_candidates = []
        for id in candidate_ids:
            citations_of_candidates.extend(self.corpus[id].citations)
        candidate_ids = list(set(citations_of_candidates + candidate_ids))

        logging.info('Fetching %d documents ' % len(candidate_ids))
        candidates = [self.corpus[paper_id] for paper_id in candidate_ids]

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
                    pdf=APIModel._sha_to_url(str(candidates[match_idx].id)),
                    position=i,
                    explanation={},
                    cited=candidates[match_idx].title.lower() in doc.citations
                )
            )
        logging.info("Done! Found %s predictions." % len(predictions))
        return predictions


def document_from_dict(doc):
    defaults = {
        FieldNames.TITLE: '',
        FieldNames.ABSTRACT: '',
        FieldNames.AUTHORS: [],
        FieldNames.OUT_CITATIONS: [],
        FieldNames.YEAR: 2016,
        FieldNames.PAPER_ID: 0,
        FieldNames.VENUE: '',
        FieldNames.IN_CITATION_COUNT: 0,
        FieldNames.OUT_CITATION_COUNT: 0,
        FieldNames.KEY_PHRASES: []
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