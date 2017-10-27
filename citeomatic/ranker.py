import numpy as np


class Ranker:
    def __init__(self, corpus, featurizer, citation_ranker):
        self.corpus = corpus
        self.featurizer = featurizer
        self.citation_ranker = citation_ranker

    def rank(self, query_id, candidate_ids):
        query = self.corpus[query_id]
        candidates = [self.corpus[id] for id in candidate_ids]
        features = self.featurizer.transform_query_and_results(query, candidates)
        scores = self.citation_ranker.predict(features, batch_size=1024).flatten()
        best_matches = np.argsort(scores)[::-1]

        predictions = []
        query_doc_citations = set(query.out_citations)
        for i, match_idx in enumerate(best_matches[:model_options.num_candidates_to_rank]):
            predictions.append(
                {
                    'score':float(scores[match_idx]),
                    'document': candidates[match_idx],
                    'position': i,
                    'is_cited': candidates[match_idx].id in query_doc_citations

                }
            )

        return predictions