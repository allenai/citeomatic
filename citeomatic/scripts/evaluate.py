import json

import pickle

from citeomatic.candidate_selectors import BM25CandidateSelector, ANNCandidateSelector, \
    OracleCandidateSelector
from citeomatic.common import DatasetPaths
from citeomatic.config import App
from traitlets import Int, Unicode, Enum

from citeomatic.corpus import Corpus
from citeomatic.neighbors import EmbeddingModel, ANN
from citeomatic.ranker import NoneRanker, Ranker
from citeomatic.serialization import model_from_directory
from citeomatic.training import eval_text_model, EVAL_DATASET_KEYS
import os


class Evaluate(App):
    dataset_type = Enum(('dblp', 'pubmed', 'oc'), default_value='pubmed')
    candidate_selector_type = Enum(('bm25', 'ann', 'oracle'), default_value='bm25')
    metric = Enum(('precision', 'recall', 'f1'), default_value='recall')
    split = Enum(('train', 'test', 'valid'), default_value='valid')

    # ann options
    paper_embedder_dir = Unicode(default_value=None, allow_none=True)

    # Candidate selector options
    num_candidates = Int(default_value=None, allow_none=True)

    ranker_type = Enum(('none', 'neural'), default_value='none')
    n_eval = Int(default_value=None, allow_none=True)

    # ranker options
    citation_ranker_dir = Unicode(default_value=None, allow_none=True)

    _embedder = None
    _ann = None

    def embedder(self, featurizer, embedding_model) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel(featurizer, embedding_model)
        return self._embedder

    def ann(self, embedder, corpus) -> ANN:
        if corpus.corpus_type == 'oc' and os.path.exists(DatasetPaths.OC_ANN_FILE + ".pickle"):
            self._ann = ANN.load(DatasetPaths.OC_ANN_FILE)
            return self._ann
        if self._ann is None:
            self._ann = ANN.build(embedder, corpus, ann_trees=100)
            if self.dataset_type == 'oc':
                self._ann.save(DatasetPaths.OC_ANN_FILE)
        return self._ann

    def _make_ann_candidate_selector(self, corpus, featurizer, embedding_model, num_candidates):
        e = self.embedder(featurizer, embedding_model)
        return ANNCandidateSelector(
            corpus=corpus,
            ann=self.ann(e, corpus),
            paper_embedding_model=e,
            top_k=num_candidates,
            extend_candidate_citations=True
        )

    def main(self, args):
        dp = DatasetPaths()
        if self.dataset_type == 'oc':
            corpus = Corpus.load_pkl(dp.get_pkl_path(self.dataset_type))
        else:
            corpus = Corpus.load(dp.get_db_path(self.dataset_type))

        if self.ranker_type == 'none':
            citation_ranker = NoneRanker()
        elif self.ranker_type == 'neural':
            assert self.citation_ranker_dir is not None
            ranker_featurizer, ranker_models = model_from_directory(self.citation_ranker_dir,
                                                                    on_cpu=True)
            citation_ranker = Ranker(
                corpus=corpus,
                featurizer=ranker_featurizer,
                citation_ranker=ranker_models['citeomatic'],
                num_candidates_to_rank=100
            )
        else:
            assert False

        candidate_results_map = {}
        if self.num_candidates is None:
            if self.dataset_type == 'oc':
                num_candidates_list = [100]
            else:
                num_candidates_list = [1, 5, 10, 15, 25, 50, 75, 100]
        else:
            num_candidates_list = [self.num_candidates]

        for num_candidates in num_candidates_list:

            if self.candidate_selector_type == 'bm25':
                index_path = dp.get_bm25_index_path(self.dataset_type)
                candidate_selector = BM25CandidateSelector(
                    corpus,
                    index_path,
                    num_candidates,
                    False
                )
            elif self.candidate_selector_type == 'ann':
                assert self.paper_embedder_dir is not None
                featurizer, models = model_from_directory(self.paper_embedder_dir, on_cpu=True)
                candidate_selector = self._make_ann_candidate_selector(corpus=corpus,
                                                                       featurizer=featurizer,
                                                                       embedding_model=models['embedding'],
                                                                       num_candidates=num_candidates)
            elif self.candidate_selector_type == 'oracle':
                candidate_selector = OracleCandidateSelector(corpus)
            else:
                assert False

            results = eval_text_model(corpus, candidate_selector, citation_ranker,
                                      papers_source=self.split, n_eval=self.n_eval)
            candidate_results_map[num_candidates] = results

        best_k = -1
        best_metric = 0.0
        metric_key = self.metric + "_1"
        for k, v in candidate_results_map.items():
            if best_metric < v[metric_key][EVAL_DATASET_KEYS[self.dataset_type]]:
                best_k = k
                best_metric = v[metric_key][EVAL_DATASET_KEYS[self.dataset_type]]

        print(json.dumps(candidate_results_map, indent=4, sort_keys=True))
        print(best_k)
        print(best_metric)

Evaluate.run(__name__)
