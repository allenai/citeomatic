#!/usr/bin/env python3
import atexit
import collections
import logging
import os
import random

import base.config
import numpy as np
import tqdm
from citeomatic import DEFAULT_BASE_DIR, ROOT, model_from_directory
from citeomatic.elastic import fetch_citations, fetch_level2_citations
from citeomatic.features import Corpus
from citeomatic.neighbors import EmbeddingModel, ANN
from citeomatic.service import APIModel
from base import file_util
from traitlets import Int, Unicode, Bool, Enum

DETAILED_PAPERS = [
    'Piccolo: Building Fast, Distributed Programs with Partitioned Tables',
    'Holographic Embeddings of Knowledge Graphs',
    'Identifying Relations for Open Information Extraction',
    'Question Answering over Freebase with Multi-Column Convolutional Neural Networks',
    'Optimizing Cauchy Reed-Solomon Codes for Fault-Tolerant Network Storage Applications',
    'Wikification and Beyond: The Challenges of Entity and Concept Grounding',
    'Named Entity Recognition in Tweets: An Experimental Study',
    'Training Input-Output Recurrent Neural Networks through Spectral Methods',
    'End-To-End Memory Networks',
]

EVAL_KEYS = [1, 5, 10, 20, 50, 100, 1000]

CITE_CACHE = {}
DONE_IDS = []


def _load_cite_cache():
    if os.path.exists('/tmp/citation.cache.json'):
        return file_util.read_json('/tmp/citation.cache.json')
    return {}


CITE_CACHE = _load_cite_cache()


def _save_cache():
    file_util.write_json('/tmp/citation.cache.json', CITE_CACHE)


atexit.register(_save_cache)


class TestCiteomatic(base.config.App):
    """
    Test the citation prediction model and calculate Precision and Recall@K.

    Parameters
    ----------
    model_dir : string
        Required argument
        Location of the saved model weights and config files.

    test_samples : Int, default=10
        Default number of samples to evaluate on

    min_citation_count : Int, default=10
        The minimum number of citations a test document should have

    filter_method : str, default='es'
        What method to use to pre-fetch the document.
        'es' is elastic search.
        'ann' is approximate nearest neighbors.

    ann_path : str, default=ROOT + '/data/citeomatic-approx-nn.index'
        Location of the ANN index.

    corpus_path: str, default='corpus-small.json'
        Location of corpus file to use.

    """
    defaults = {
        'base_dir': os.path.join(ROOT, DEFAULT_BASE_DIR),
    }

    model_dir = Unicode(allow_none=False)
    test_samples = Int(default_value=10)
    min_citation_count = Int(default_value=10)
    max_neighbors = Int(default_value=1000)
    corpus_path = Unicode(default_value=os.path.join(defaults['base_dir'], 'corpus.msgpack'))
    filter_method = Unicode(default_value='es')
    ann_path = Unicode(default_value=None, allow_none=True)
    ann_model_dir = Unicode(default_value=None, allow_none=True)
    candidate_min_in_citations = Int(default_value=4, allow_none=True)
    limit_candidate_to_train_ids = Bool(default_value=False)
    extend_candidate_citations = Bool(default_value=False)

    def _fetch_citations(self, paper_id, level):
        key = '%s/%d' % (paper_id, level)
        if key not in CITE_CACHE:
            if level == 1:
                citations = self.corpus[paper_id].citations
                citations = [c for c in citations if c in self.corpus.train_ids]
                CITE_CACHE[key] = citations
            else:
                if self.citation_source == 'es':
                    second_level_citations = list(
                        fetch_level2_citations(self._fetch_citations(paper_id, 1))
                    )
                else:
                    second_level_citations = []
                    second_level_citations.extend(self.corpus[paper_id].citations)
                    for c in self.corpus[paper_id].citations:
                        second_level_citations.extend(self.corpus[c].citations)
                    second_level_citations = [
                        c for c in second_level_citations if c in self.corpus.train_ids
                    ]
                CITE_CACHE[key] = second_level_citations
        return CITE_CACHE[key]

    def _predict(self, paper_id):
        # Obtain out-citations of a paper. We cannot use the ones
        # in the `corpus` object because they were filtered down to contain
        # IDs that are in the corpus itself.
        gold_citations = set(self._fetch_citations(paper_id, 1))
        citations_of_citations = set(self._fetch_citations(paper_id, 2))
        gold_citations_2 = gold_citations.union(citations_of_citations)

        if len(gold_citations) < self.min_citation_count:
            return None

        logging.info("No. of gold citations of %s = %d" % (paper_id, len(gold_citations)))
        document = self.corpus[paper_id]

        def _found(lst):
            return len([id for id in lst if id in self.corpus])

        best_recall_1 = _found(gold_citations) / len(gold_citations)
        best_recall_2 = _found(gold_citations_2) / len(gold_citations_2)
        logging.info(
            'Corpus recall for paper %s = %f %s' % (paper_id, best_recall_1, best_recall_2)
        )

        predictions = self.model.predict(document, top_n=np.max(list(EVAL_KEYS)))

        paper_results = []
        for prediction in predictions:
            if prediction.document.id == paper_id: continue
            paper_results.append(
                {
                    'title': prediction.document.title,
                    'id': prediction.document.id,
                    'correct_1': prediction.document.id in gold_citations,
                    'correct_2': prediction.document.id in gold_citations_2,
                    'score': prediction.score,
                }
            )

        def _mrr(p):
            try:
                idx = p.index(True)
                return 1. / (idx + 1)
            except ValueError:
                return 0.0

        p1 = [p['correct_1'] for p in paper_results]
        mrr1 = _mrr(p1)
        p2 = [p['correct_2'] for p in paper_results]
        mrr2 = _mrr(p2)

        logging.info('Level 1 P@10 = %f ' % np.mean(p1[:10]))
        logging.info('Level 2 P@10 = %f ' % np.mean(p2[:10]))
        logging.info('Level 1 MRR = %f' % mrr1)
        logging.info('Level 2 MRR = %f' % mrr2)
        candidate_set_recall = np.sum(p1) / len(gold_citations)
        logging.info('Candidate set recall = %f ' % candidate_set_recall)

        DONE_IDS.append(paper_id)
        logging.info('======================== %d' % len(DONE_IDS))
        return {
            'title': document.title,
            'id': document.id,
            'predictions': paper_results,
            'recall_1': best_recall_1,
            'recall_2': best_recall_2,
            'num_gold_1': len(gold_citations),
            'num_gold_2': len(gold_citations_2),
            'mrr_1': mrr1,
            'mrr_2': mrr2
        }

    def _init_model(self):
        featurizer, models = model_from_directory(self.model_dir)
        corpus = Corpus.load(self.corpus_path, featurizer.training_fraction)

        if self.filter_method == "ann":
            ann = ANN.load(self.ann_path)
            if self.ann_model_dir:
                featurizer_ann, models_ann = model_from_directory(self.ann_model_dir)
            else:
                featurizer_ann, models_ann = featurizer, models

            ann_doc_embedding_model = EmbeddingModel(featurizer_ann, models_ann['embedding'])
            api_model = APIModel(
                models,
                featurizer,
                ann=ann,
                ann_embedding_model=ann_doc_embedding_model,
                corpus=corpus,
                max_neighbors=self.max_neighbors,
                candidate_min_in_citations=self.candidate_min_in_citations,
                limit_candidate_to_train_ids=self.limit_candidate_to_train_ids,
                extend_candidate_citations=self.extend_candidate_citations,
                citation_source=self.citation_source
            )
        else:
            api_model = APIModel(
                models,
                featurizer,
                max_neighbors=self.max_neighbors,
                candidate_min_in_citations=self.candidate_min_in_citations,
                limit_candidate_to_train_ids=self.limit_candidate_to_train_ids,
                extend_candidate_citations=self.extend_candidate_citations,
                citation_source=self.citation_source
            )

        self.corpus = corpus
        self.model = api_model
        return corpus, api_model

    def main(self, rest):
        corpus, api_model = self._init_model()
        logging.info(
            'Found %d ids in training and %d ids for testing' %
            (len(corpus.train_ids), len(corpus.test_ids))
        )

        query_doc_ids = []
        for doc in tqdm.tqdm(corpus):
            if doc.title in DETAILED_PAPERS:
                query_doc_ids.append(doc.id)

        for doc_id in query_doc_ids:
            logging.info('Query Doc Title ---> %s ' % corpus[doc_id].title)
            citations = self._fetch_citations(doc_id, 1)
            predictions = api_model.predict(corpus[doc_id], top_n=50)
            for prediction in predictions:
                logging.info(
                    '\t%f\t%s\t%s' % (
                        prediction.score, prediction.document.id in citations,
                        prediction.document.title
                    )
                )

        random.seed(110886)
        shuffled_test_ids = sorted(np.sort(list(corpus.test_ids)), key=lambda k: random.random())
        filtered_test_ids = []
        for test_id in tqdm.tqdm(shuffled_test_ids):
            if len(self._fetch_citations(test_id, 1)) >= self.min_citation_count:
                filtered_test_ids.append(test_id)
            if len(filtered_test_ids) == self.test_samples:
                break
        shuffled_test_ids = filtered_test_ids
        results = [self._predict(paper_id) for paper_id in shuffled_test_ids]
        results = [r for r in results if r is not None]

        precision_at_1 = collections.defaultdict(list)
        recall_at_1 = collections.defaultdict(list)

        precision_at_2 = collections.defaultdict(list)
        recall_at_2 = collections.defaultdict(list)

        for r in results:
            p1 = [p['correct_1'] for p in r['predictions']]
            p2 = [p['correct_2'] for p in r['predictions']]
            for k in EVAL_KEYS:
                patk = np.mean(p1[:k])
                ratk = np.sum(p1[:k]) / r['num_gold_1']
                precision_at_1[k].append(patk)
                recall_at_1[k].append(ratk)

                patk = np.mean(p2[:k])
                ratk = np.sum(p2[:k]) / r['num_gold_2']
                precision_at_2[k].append(patk)
                recall_at_2[k].append(ratk)

        self.write_json(
            'test_results.json', {
                'precision_1': {k: np.mean(v)
                                for (k, v) in precision_at_1.items()},
                'recall_1': {k: np.mean(v)
                             for (k, v) in recall_at_1.items()},
                'precision_2': {k: np.mean(v)
                                for (k, v) in precision_at_2.items()},
                'recall_2': {k: np.mean(v)
                             for (k, v) in recall_at_2.items()},
                'mrr_1': np.mean([r['mrr_1'] for r in results]),
                'mrr_2': np.mean([r['mrr_2'] for r in results]),
                'results': results,
            }
        )

        logging.info("\n====\nResults on %d randomly sampled papers" % len(precision_at_1[1]))
        logging.info("Precision @K")
        logging.info("K\tLevel 1\tLevel 2")
        for k in np.sort(list(precision_at_1.keys())):
            logging.info(
                "K=%d:\t%f\t%f" % (k, np.mean(precision_at_1[k]), np.mean(precision_at_2[k]))
            )
        logging.info("Recall @k")
        for k in np.sort(list(recall_at_1.keys())):
            logging.info("K=%d:\t%f\t%f" % (k, np.mean(recall_at_1[k]), np.mean(recall_at_2[k])))
        logging.info("Best possible recall = %f ", np.mean([r['recall_1'] for r in results]))
        logging.info('Level 1 MRR = %f' % np.mean([r['mrr_1'] for r in results]))
        logging.info('Level 2 MRR = %f' % np.mean([r['mrr_2'] for r in results]))


TestCiteomatic.run(__name__)
