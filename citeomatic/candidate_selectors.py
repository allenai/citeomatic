from abc import ABC

from citeomatic.neighbors import ANN


class CandidateSelector(ABC):
    def __init__(self, top_k=100):
        self.top_k = top_k

    def fetch_candidates(self, doc_id) -> list:
        """
        For each query paper, return a list of candidates and associated scores
        :param doc_id: Document ID to get candidates for
        :param top_k: How many top candidates to fetch
        :return:
        """
        pass


class ANNCandidateSelector(CandidateSelector):
    def __init__(self, ann: ANN, top_k):
        super().__init__(top_k)
        self.ann = ann

    def fetch_candidates(self, doc_id):
        # TODO: Need scores form ANN index.
        return self.ann.get_nns_by_id(doc_id, self.top_k)


class BM25CandidateSelector(CandidateSelector):
    def __init__(self, top_k):
        super().__init__(top_k)
        self.bm25 = None

    def fetch_candidates(self, doc_id):
        # Implement BM25 index builder and return
        pass
