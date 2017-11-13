import numpy as np

from citeomatic.corpus import Corpus
from citeomatic.features import Featurizer


class Ranker:
    def __init__(self, corpus: Corpus, featurizer: Featurizer, citation_ranker,
                 num_candidates_to_rank):
        self.corpus = corpus
        self.featurizer = featurizer
        self.citation_ranker = citation_ranker
        self.num_candidates_to_rank = num_candidates_to_rank

    def rank(self, query_id, candidate_ids, similarities):
        query = self.corpus[query_id]
        candidates = [self.corpus[id] for id in candidate_ids]

        features = self.featurizer.transform_query_and_results(query, candidates, similarities)
        scores = self.citation_ranker.predict(features, batch_size=1024).flatten()
        best_matches = np.argsort(scores)[::-1]

        predictions = []
        pred_scores = []

        for i, match_idx in enumerate(best_matches[:self.num_candidates_to_rank]):
            predictions.append(candidates[match_idx].id)
            pred_scores.append(float(scores[match_idx]))

        return predictions, pred_scores


class NoneRanker(object):

    def rank(self, query_id, candidate_ids, similarities):
        return candidate_ids, similarities
