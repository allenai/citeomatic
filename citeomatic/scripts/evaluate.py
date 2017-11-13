from citeomatic.candidate_selectors import BM25CandidateSelector, ANNCandidateSelector
from citeomatic.common import DatasetPaths
from citeomatic.config import App
from traitlets import Int, Unicode, Enum

from citeomatic.corpus import Corpus
from citeomatic.neighbors import EmbeddingModel, ANN
from citeomatic.ranker import NoneRanker
from citeomatic.serialization import model_from_directory
from citeomatic.training import eval_text_model


class Evaluate(App):
    dataset_type = Enum(('dblp', 'pubmed', 'oc'), default_value='pubmed')
    candidate_selector_type = Enum(('bm25', 'ann'), default_value='bm25')

    # ann options
    paper_embedder_dir = Unicode(default_value=None, allow_none=True)

    # Candidate selector options
    num_candidates = Int(default_value=100)

    ranker_type = Enum(('none', 'neural'), default_value='none')
    n_eval = Int(default_value=None, allow_none=True)

    @staticmethod
    def _make_ann_candidate_selector(corpus, featurizer, embedding_model, k):
        embedder = EmbeddingModel(featurizer, embedding_model)
        ann = ANN.build(embedder, corpus, ann_trees=100)
        return ANNCandidateSelector(
            corpus=corpus,
            ann=ann,
            paper_embedding_model=embedder,
            top_k=k,
            extend_candidate_citations=True
        )

    def main(self, args):
        dp = DatasetPaths()
        if self.dataset_type == 'oc':
            corpus = Corpus.load_pkl(dp.get_pkl_path(self.dataset_type))
        else:
            corpus = Corpus.load(dp.get_db_path(self.dataset_type))
        index_path = dp.get_bm25_index_path(self.dataset_type)
        if self.candidate_selector_type == 'bm25':
            candidate_selector = BM25CandidateSelector(
                corpus,
                index_path,
                self.num_candidates,
                False
            )
        elif self.candidate_selector_type == 'ann':
            assert self.paper_embedder_dir is not None
            featurizer, models = model_from_directory(self.paper_embedder_dir, on_cpu=True)
            candidate_selector = Evaluate._make_ann_candidate_selector(corpus=corpus,
                                                                       featurizer=featurizer,
                                                                       embedding_model=models[
                                                                           'embedding'],
                                                                       k=self.num_candidates)

        if self.ranker_type == 'none':
            citation_ranker = NoneRanker()
        else:
            assert False

        results = eval_text_model(corpus, candidate_selector, citation_ranker, papers_source='test', n_eval=self.n_eval)
        print(results)


Evaluate.run(__name__)