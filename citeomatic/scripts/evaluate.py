from citeomatic.candidate_selectors import BM25CandidateSelector
from citeomatic.common import DatasetPaths
from citeomatic.config import App
from traitlets import Int, Unicode, Enum

from citeomatic.corpus import Corpus
from citeomatic.ranker import NoneRanker
from citeomatic.training import eval_text_model


class Evaluate(App):
    dataset_type = Enum(('dblp', 'pubmed', 'oc'), default_value='pubmed')
    candidate_selector_type = Enum(('bm25', 'ann'), default_value='bm25')
    ranker_type = Enum(('none', 'neural'), default_value='none')

    def main(self, args):
        dp = DatasetPaths()
        corpus = Corpus.load(dp.get_db_path(self.dataset_type))
        index_path = dp.get_bm25_index_path(self.dataset_type)
        if self.candidate_selector_type == 'bm25':
            candidate_selector = BM25CandidateSelector(
                corpus,
                index_path,
                100,
                False
            )
        else:
            assert False

        if self.ranker_type == 'none':
            citation_ranker = NoneRanker()
        else:
            assert False

        results = eval_text_model(corpus, candidate_selector, citation_ranker,
                                  papers_source='test', n_eval=None)
        print(results)


Evaluate.run(__name__)