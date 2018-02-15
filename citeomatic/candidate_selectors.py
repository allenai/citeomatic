import logging
from abc import ABC

from whoosh import scoring, qparser
from whoosh.filedb.filestore import FileStorage, copy_to_ram
from whoosh.index import FileIndex
from whoosh.qparser import MultifieldParser

from citeomatic.common import schema, FieldNames
from citeomatic.corpus import Corpus
from citeomatic.neighbors import ANN
from citeomatic.neighbors import EmbeddingModel
import numpy as np


class CandidateSelector(ABC):
    def __init__(self, top_k=100):
        self.top_k = top_k

    def fetch_candidates(self, doc_id, candidates_id_pool) -> tuple:
        """
        For each query paper, return a list of candidates and associated scores
        :param doc_id: Document ID to get candidates for
        :param top_k: How many top candidates to fetch
        :param candidates_id_pool: Set of candidate IDs to limit candidates to
        :return:
        """
        pass

    def confidence(self, doc_id, candidate_ids):
        """

        :param doc_id:
        :param candidate_ids:
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

    def fetch_candidates(self, doc_id, candidate_ids_pool: set):
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
        if doc_id in candidate_ids:
            candidate_ids.remove(doc_id)
        candidate_ids_list = list(candidate_ids)

        candidate_ids_list = [candidate_doc_id for candidate_doc_id in candidate_ids_list if
                              self.corpus[candidate_doc_id].year <= self.corpus[doc_id].year]

        confidence_scores = self.confidence(doc_id, candidate_ids_list)
        sorted_pairs = sorted(zip(candidate_ids_list, confidence_scores), key=lambda x: x[1],
                              reverse=True)

        sorted_candidate_ids = []
        sorted_scores = []
        for pair in sorted_pairs:
            sorted_candidate_ids.append(pair[0])
            sorted_scores.append(pair[1])

        return sorted_candidate_ids, sorted_scores

    def confidence(self, doc_id, candidate_ids):
        doc = self.corpus[doc_id]
        doc_embedding = self.paper_embedding_model.embed(doc)
        return self.ann.get_similarities(doc_embedding, candidate_ids)


class BM25CandidateSelector(CandidateSelector):
    def __init__(
            self,
            corpus: Corpus,
            index_path: str,
            top_k,
            extend_candidate_citations: bool
    ):
        super().__init__(top_k)
        self.index_path = index_path

        storage = FileStorage(self.index_path, readonly=True)
        self._bm25_index = FileIndex(copy_to_ram(storage), schema=schema)
        self.searcher = self._bm25_index.searcher(weighting=scoring.BM25F)
        self.query_parser = MultifieldParser([FieldNames.TITLE, FieldNames.ABSTRACT],
                                             self._bm25_index.schema, group=qparser.OrGroup)
        self.corpus = corpus
        self.extend_candidate_citations = extend_candidate_citations

    def fetch_candidates(self, doc_id, candidate_ids_pool):

        title_key_terms = ' '.join([
            t for t,_ in self.searcher.key_terms_from_text('title', self.corpus[doc_id].title,
                                                           numterms=3)]
        )
        abstract_key_terms = ' '.join([
            t for t,_ in self.searcher.key_terms_from_text('abstract', self.corpus[doc_id].abstract)]
        )
        # Implement BM25 index builder and return
        query = self.query_parser.parse(title_key_terms + " " + abstract_key_terms)
        results = self.searcher.search(query, limit=self.top_k + 1, optimize=True, scored=True)

        candidate_ids_pool = set(candidate_ids_pool)
        candidate_ids = []
        candidate_scores = []
        for result in results:
            if result['id'] in candidate_ids_pool and result['id'] != doc_id:
                candidate_ids.append(result['id'])
                candidate_scores.append(result.score)

        return candidate_ids, candidate_scores


class OracleCandidateSelector(CandidateSelector):
    def __init__(self, corpus: Corpus):
        super().__init__()
        self.corpus = corpus

    def fetch_candidates(self, doc_id, candidate_ids_pool):
        candidates = set(self.corpus.get_citations(doc_id))
        candidates.intersection_update(candidate_ids_pool)

        return list(candidates), np.ones(len(candidates))
