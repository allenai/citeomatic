from abc import ABC

from citeomatic.neighbors import EmbeddingModel, ANN
from citeomatic.corpus import Corpus


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
    def __init__(
            self,
            corpus: Corpus,
            ann: ANN,
            paper_embedding_model: EmbeddingModel,
            top_k: int,
            extend_candidate_citations: bool
    ):
        super().__init__(top_k)
        self.corpus = corpus
        self.ann = ann
        self.paper_embedding_model = paper_embedding_model
        self.extend_candidate_citations = extend_candidate_citations

    def fetch_candidates(self, doc_id, candidate_ids_pool):
        doc = self.corpus[doc_id]
        doc_embedding = self.paper_embedding_model.embed(doc)
        # 1. Fetch candidates from ANN index
        nn_candidates = self.ann.get_nns_by_vector(doc_embedding, self.top_k + 1)
        # 2. Remove the current document from candidate list
        if doc_id in nn_candidates:
            nn_candidates.remove(doc_id)
        candidate_ids = nn_candidates[:self.top_k]

        # 3. Check if we need to include citations of candidates found so far.
        if self.extend_candidate_citations:
            extended_candidate_ids = []
            for candidate_id in candidate_ids:
                extended_candidate_ids.extend(self.corpus[candidate_id].out_citations)
            candidate_ids = candidate_ids + extended_candidate_ids
        logging.debug("Number of candidates found: {}".format(len(candidate_ids)))
        candidate_ids = set(candidate_ids).intersection(candidate_ids_pool)
        return candidate_ids


class BM25CandidateSelector(CandidateSelector):
    def __init__(self, top_k):
        super().__init__(top_k)
        self.bm25 = None

    def fetch_candidates(self, doc_id):
        # Implement BM25 index builder and return
        pass
