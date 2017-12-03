from collections import Counter

from citeomatic.common import DatasetPaths
from citeomatic.config import App
from citeomatic.corpus import Corpus
from citeomatic.traits import Enum
import numpy as np


class CorpusStat(App):

    dataset_type = Enum(('dblp', 'pubmed', 'oc'), default_value='pubmed')

    def main(self, args):
        dp = DatasetPaths()
        if self.dataset_type == 'oc':
            corpus = Corpus.load_pkl(dp.get_pkl_path(self.dataset_type))
        else:
            corpus = Corpus.load(dp.get_db_path(self.dataset_type))

        authors = Counter()
        key_phrases = Counter()
        years = Counter()
        venues = Counter()
        num_docs_with_kp = 0

        in_citations_counts = []
        out_citations_counts = []
        for doc in corpus:
            authors.update(doc.authors)
            key_phrases.update(doc.key_phrases)
            if len(doc.key_phrases) > 0:
                num_docs_with_kp += 1
            in_citations_counts.append(doc.in_citation_count)
            out_citations_counts.append(doc.out_citation_count)
            years.update([doc.year])
            venues.update([doc.venue])

        training_years = [corpus[doc_id].year for doc_id in corpus.train_ids]
        validation_years = [corpus[doc_id].year for doc_id in corpus.valid_ids]
        testing_years = [corpus[doc_id].year for doc_id in corpus.test_ids]

        print("No. of documents = {}".format(len(corpus)))
        print("Unique number of authors = {}".format(len(authors)))
        print("Unique number of key phrases = {}".format(len(key_phrases)))
        print("Unique number of venues = {}".format(len(venues)))
        print("No. of docs with key phrases = {}".format(num_docs_with_kp))
        print("Average in citations = {} (+/- {})".format(np.mean(in_citations_counts),
                                                          np.std(in_citations_counts)))
        print("Average out citations = {} (+/- {})".format(np.mean(out_citations_counts),
                                                           np.std(out_citations_counts)))
        print("No. of training examples = {} ({} to {})".format(len(corpus.train_ids),
                                                                np.min(training_years),
                                                                np.max(training_years)))
        print("No. of validation examples = {} ({} to {})".format(len(corpus.valid_ids),
                                                                  np.min(validation_years),
                                                                  np.max(validation_years)))
        print("No. of testing examples = {} ({} to {})".format(len(corpus.test_ids),
                                                               np.min(testing_years),
                                                               np.max(testing_years)))
        print(authors.most_common(10))




CorpusStat.run(__name__)