import collections
import logging
import math
import re
import resource

import numpy as np
import pandas as pd
import six
import tqdm
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

from citeomatic.utils import flatten

CLEAN_TEXT_RE = re.compile('[^ a-z]')

# filters for authors and docs
MAX_AUTHORS_PER_DOCUMENT = 8
MIN_TRUE_CITATIONS = 2
MAX_TRUE_CITATIONS = 100

# Adjustments to how we boost heavily cited documents.
CITATION_SLOPE = 0.01
MAX_CITATION_BOOST = 0.02

# Parameters for soft-margin data generation.
TRUE_CITATION_OFFSET = 0.3
HARD_NEGATIVE_OFFSET = 0.2
NN_NEGATIVE_OFFSET = 0.1
EASY_NEGATIVE_OFFSET = 0.0

# ANN jaccard percentile cutoff
ANN_JACCARD_PERCENTILE = 0.05

def label_for_doc(d, offset):
    sigmoid = 1 / (1 + np.exp(-d.in_citation_count * CITATION_SLOPE))
    return offset + (sigmoid * MAX_CITATION_BOOST)

def order_preserving_unique(seq):
    """
    A function that returns the unique values from a sequence while preserving their order.
    """
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def agresti_coull_interval(p, n, z=1.96):
    """
    Binomial proportion confidence interval.
    from: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Agresti-Coull_Interval
    """
    n = float(n)
    n_tilde = n + z**2
    successes = np.round(p * n)
    p_tilde = (successes + 0.5 * z**2) / n_tilde
    bound_val = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    lower_bound = np.maximum(0, p_tilde - bound_val)
    upper_bound = np.minimum(1, p_tilde + bound_val)
    return lower_bound, upper_bound


def jaccard(featurizer, x, y):
    x_title, x_abstract = featurizer._cleaned_document_words(x)
    y_title, y_abstract = featurizer._cleaned_document_words(y)
    a = set(x_title + x_abstract)
    b = set(y_title + y_abstract)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def build_index_dict(sequences, offset=1):
    """
    Builds a dictionary mapping words to indices.
    """
    unique_symbols = set()
    for seq in sequences:
        for c in seq:
            unique_symbols.add(c)
    return {c: i + offset for (i, c) in enumerate(unique_symbols)}


def _clean(text):
    return CLEAN_TEXT_RE.sub(' ', text.lower())


class Featurizer(object):
    '''
    This class uses the corpus to turn text into features expected by the neural network.

    Parameters
    ----------
    max_title_len : int, default=32
        Maximum number of tokens allowed in the paper title.
    max_abstract_len : int, default=256
        Maximum number of tokens allowed in the abstract title.
    '''
    STOPWORDS = {
        'abstract', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
        'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the',
        'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with',
        'the', 'we', 'our', 'which'
    }

    MIN_AUTHOR_PAPERS = 5  # minimum number of papers for an author to get an embedding.

    def __init__(
        self,
        max_features=200000,
        max_title_len=32,
        max_abstract_len=256,
        allow_duplicates=True
    ):
        self.max_features = max_features
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len
        self.allow_duplicates = allow_duplicates

        self.word_count = {}
        self.author_to_index = {}
        self.word_indexer = None

    @property
    def n_authors(self):
        if not hasattr(self, 'author_to_index'):
            self.author_to_index = {}
        return len(self.author_to_index) + 1

    def fit(self, corpus, max_df_frac=0.90, min_df_frac=0.000025):

        logging.info(
            'Usage at beginning of featurizer fit: %s',
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        )

        # Fitting authors
        logging.info('Fitting authors')
        author_counts = collections.Counter()
        for doc in tqdm.tqdm(corpus):
            if doc.id in corpus.train_ids:
                author_counts.update(doc.authors)

        for author, count in author_counts.items():
            if count >= Featurizer.MIN_AUTHOR_PAPERS:
                self.author_to_index[author] = 1 + len(self.author_to_index)

        # Step 1: filter out some words and make a word_count dictionary
        logging.info('Cleaning text.')
        all_docs_text = [
            ' '.join((_clean(doc.title), _clean(doc.abstract)))
            for doc in tqdm.tqdm(corpus)
            if doc.id in corpus.train_ids
        ]

        logging.info('Fitting vectorizer...')
        if self.max_features is not None:
            count_vectorizer = CountVectorizer(
                max_df=max_df_frac,
                max_features=self.max_features,
                stop_words=self.STOPWORDS
            )
        else:
            count_vectorizer = CountVectorizer(
                max_df=max_df_frac,
                min_df=min_df_frac,
                stop_words=self.STOPWORDS
            )
        count_vectorizer.fit(tqdm.tqdm(all_docs_text))
        self.word_count = count_vectorizer.vocabulary_

        logging.info(
            'Usage after word count: %s',
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        )
        logging.info('Number of words = %d' % len(self.word_count))

        # Step 4: Initialize mapper from word to index
        self.n_features = (1 + len(self.word_count))

        self.word_indexer = FeatureIndexer(
            word_count=self.word_count, allow_duplicates=self.allow_duplicates
        )

        logging.info(
            'Usage after word_indexer: %s',
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        )
        logging.info(
            'Usage at end of fit: %s',
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        )
        logging.info('Total words %d ' % len(self.word_indexer.word_to_index))

    def __setstate__(self, state):
        for k, v in state.items():
            try:
                setattr(self, k, v)
            except AttributeError as e:
                logging.warning('Ignoring renamed attribute: %s', k)
                continue

    def _citation_features(self, documents):
        return np.log([doc.in_citation_count + 1 for doc in documents])

    def _intersection_features(self, source_features, candidate_features):
        feats_intersection_lst = [
            np.intersect1d(source, result)
            for (source, result) in zip(source_features, candidate_features)
        ]
        feats_intersection = np.zeros_like(source_features)
        for i, intersection in enumerate(feats_intersection_lst):
            feats_intersection[i, :len(intersection)] = intersection
        return feats_intersection

    def _words_from_text(self, text):
        return [word for word in text if word in self.word_count]

    def _text_features(self, text, max_len):
        return np.asarray(
            pad_sequences(
                self.word_indexer.transform([self._words_from_text(text)]),
                max_len
            )[0],
            dtype=np.int32
        )

    def _cleaned_document_words(self, document):
        title = _clean(document.title).split(' ')
        abstract = _clean(document.abstract).split(' ')
        return title, abstract

    def transform_query_candidate(self, query_docs, candidate_docs):
        """
        Parameters
        ----------
        query_docs - a list of query documents
        candidate_docs - a list of candidate documents corresponding to each query doc in query_docs
        Returns
        -------
        [feats1, feats2] - feats1 and feats2 are transformed versions of the text in the dicts
        of documents.
        """
        source_features = self.transform_list(query_docs)
        candidate_features = self.transform_list(candidate_docs)
        candidate_citation_features = self._citation_features(candidate_docs)

        query_candidate_title_intersection = self._intersection_features(
            source_features=source_features['title'],
            candidate_features=candidate_features['title']
        )
        query_candidate_abstract_intersection = self._intersection_features(
            source_features=source_features['abstract'],
            candidate_features=candidate_features['abstract']
        )

        features = {
            'query-authors':
                source_features['authors'],
            'query-title-txt':
                source_features['title'],
            'query-abstract-txt':
                source_features['abstract'],
            'candidate-title-txt':
                candidate_features['title'],
            'candidate-abstract-txt':
                candidate_features['abstract'],
            'candidate-authors':
                candidate_features['authors'],
            'query-candidate-title-intersection':
                query_candidate_title_intersection,
            'query-candidate-abstract-intersection':
                query_candidate_abstract_intersection,
            'candidate-citation-count':
                candidate_citation_features
        }
        return features

    def transform_query_and_results(self, query, list_of_documents):
        """
        Parameters
        ----------
        query - a single query document
        list_of_documents - a list of possible result documents
        Returns
        -------
        [feats1, feats2] - feats1 and feats2 are transformed versions of the text in the
            in the tuples of documents.
        """

        query_docs = []
        for i in range(len(list_of_documents)):
            query_docs.append(query)
        return self.transform_query_candidate(query_docs, list_of_documents)

    def transform_doc(self, document):
        """
        Converts a document into its title and abstract word sequences
        :param document: Input document of type Document
        :return: a tuple containing the title and abstract's transformed word sequences. Word
        sequences are np arrays
        """
        title, abstract = self._cleaned_document_words(document)
        features = {
            'title':
                self._text_features(title, self.max_title_len),
            'abstract':
                self._text_features(abstract, self.max_abstract_len),
            'authors':
                [
                    self.author_to_index[author] for author in document.authors
                    if author in self.author_to_index
                ]
        }

        return features

    def transform_list(self, list_of_documents):
        docs = []
        for document in list_of_documents:
            docs.append(self.transform_doc(document))

        # pull out individual columns and convert to arrays
        return {
            'title':
                np.asarray([doc['title'] for doc in docs]),
            'abstract':
                np.asarray([doc['abstract'] for doc in docs]),
            'authors':
                np.asarray(
                    pad_sequences(
                        [doc['authors']
                         for doc in docs], MAX_AUTHORS_PER_DOCUMENT
                    )
                ),
        }


class CachingFeaturizer(Featurizer):
    def __init__(self, featurizer):
        for k, v in featurizer.__dict__.items():
            setattr(self, k, v)

        self._cache = {}

    def transform_doc(self, document):
        if not document.id in self._cache:
            features = Featurizer.transform_doc(self, document)
            self._cache[document.id] = features

        return self._cache[document.id]


class FeatureIndexer(object):
    """
    A class to transform raw tokens into formatted indices.

    Parameters
    ----------
    word_count : dict/set/list
        The set of words to index.

    offset : int, default=1
        Index offset. Default is 1 because Keras reserves index 0 for the mask.
    """

    def __init__(self, word_count, allow_duplicates=True, offset=1):
        self.word_to_index = {}
        self.offset = offset
        for i, word in enumerate(word_count):
            self.word_to_index[word] = i + offset
        self.allow_duplicates = allow_duplicates

    def transform(self, raw_X):
        """
        Transforms raw strings into hashed indices.

        Input should be e.g. raw_X = [['the', 'first', 'string'],['the', 'second']],
        """

        if not self.allow_duplicates:
            for i, raw_x in enumerate(raw_X):
                raw_X[i] = order_preserving_unique(raw_x)

        indexed_X = []
        for raw_x in raw_X:
            indexed_x = [
                self.word_to_index[word] for word in raw_x
                if word in self.word_to_index
            ]
            indexed_X.append(indexed_x)

        return indexed_X


class DataGenerator(object):
    """
    Class to yield batches of data to train Keras models.

    Parameters
    ----------
    corpus : Corpus
        The corpus with all of the documents.

    featurizer : Featurizer
        Featurizer to turn documents into indices.
    """
    KEYS = ['hard_negatives', 'nn', 'easy']

    def __init__(self, corpus, featurizer, ann=None):
        self.corpus = corpus
        self.featurizer = CachingFeaturizer(featurizer)
        self.ann = ann

    def _listwise_examples(
            self,
            paper_ids,
            candidate_ids=None,
            neg_to_pos_ratio=6,
            margin_multiplier=1
    ):
        # the id pool should only have IDs that are in the corpus
        paper_ids_list = np.array(list(self.corpus.filter(paper_ids)))

        # candidate_ids is decides where candidates come from
        if candidate_ids is None:
            candidate_ids = self.corpus.train_ids
        else:
            candidate_ids = self.corpus.filter(candidate_ids)

        # these are reused
        candidate_ids_set = set(candidate_ids)
        candidate_ids_list = np.array(list(candidate_ids_set))

        while True:
            candidate_ids_list = np.random.permutation(candidate_ids_list)
            paper_ids_list = np.random.permutation(paper_ids_list)
            for doc_id in paper_ids_list:
                examples = []
                labels = []
                query = self.corpus[doc_id]
                true_citations = candidate_ids_set.intersection(query.citations)
                if len(true_citations) < MIN_TRUE_CITATIONS:
                    continue

                if len(true_citations) > MAX_TRUE_CITATIONS:
                    true_citations = np.random.choice(
                        list(true_citations), MAX_TRUE_CITATIONS, replace=False
                    )

                n_positive = len(true_citations)
                n_per_type = {
                    'hard_negatives':  np.ceil(n_positive * neg_to_pos_ratio / 3.0),
                    'easy': np.ceil(n_positive * neg_to_pos_ratio / 3.0),
                    'nn':  np.ceil(n_positive * neg_to_pos_ratio / 3.0)
                }

                if self.ann is not None:
                    pos_jaccard_sims = [
                        jaccard(self.featurizer, query, self.corpus[i])
                        for i in true_citations
                    ]
                    ann_jaccard_cutoff = np.percentile(pos_jaccard_sims, ANN_JACCARD_PERCENTILE)
                else:
                    ann_jaccard_cutoff = None

                hard_negatives, nn_negatives, easy_negatives = self.get_negatives(
                    candidate_ids_set, candidate_ids_list, n_per_type, query, ann_jaccard_cutoff
                )

                for c in true_citations:
                    doc = self.corpus[c]
                    labels.append(label_for_doc(doc, TRUE_CITATION_OFFSET * margin_multiplier))
                    examples.append(doc)

                for doc in hard_negatives:
                    labels.append(label_for_doc(doc, HARD_NEGATIVE_OFFSET * margin_multiplier))
                    examples.append(doc)

                for doc in nn_negatives:
                    labels.append(label_for_doc(doc, NN_NEGATIVE_OFFSET * margin_multiplier))
                    examples.append(doc)

                for doc in easy_negatives:
                    labels.append(label_for_doc(doc, EASY_NEGATIVE_OFFSET * margin_multiplier))
                    examples.append(doc)

                labels = np.asarray(labels)
                sorted_idx = np.argsort(labels)[::-1]
                labels = labels[sorted_idx]
                examples = [examples[i] for i in sorted_idx]

                yield query, examples, labels

    def triplet_generator(
            self,
            paper_ids,
            candidate_ids=None,
            batch_size=1024,
            neg_to_pos_ratio=6,
            margin_multiplier=1
    ):
        queries = []
        batch_ex = []
        batch_labels = []

        # Sample examples from our sorted list.  The margin between each example is the difference in their label:
        # easy negatives (e.g. very bad results) should be further away from a true positive than hard negatives
        # (less embarrassing).
        for q, ex, labels in self._listwise_examples(paper_ids, candidate_ids, neg_to_pos_ratio):
            num_true = len([l for l in labels if l >= TRUE_CITATION_OFFSET])
            # ignore cases where we didn't find enough negatives...
            if len(labels) < num_true * 2:
                continue
            # Sample pairs of (positive, negative)
            pos = np.random.randint(0, num_true, len(labels) * 2)
            neg = np.random.randint(num_true, len(labels), len(labels) * 2)

            for ai, bi in zip(pos, neg):
                queries.extend([q, q])
                batch_ex.extend([ex[ai], ex[bi]])
                batch_labels.extend([labels[ai], labels[bi]])
                if len(queries) > batch_size:
                    yield self.featurizer.transform_query_candidate(
                        queries, batch_ex
                    ), np.asarray(batch_labels)
                    del queries[:]
                    del batch_ex[:]
                    del batch_labels[:]

    def get_negatives(
        self, candidate_ids_set, candidate_ids_list, n_per_type, document, ann_jaccard_cutoff=1
    ):
        '''
        :param n_per_type: dictionary with keys: 'easy', 'hard_negatives', 'nn'
        :param document: query document
        :return: documents
        '''

        def sample(document_ids, n):
            document_ids = document_ids.intersection(candidate_ids_set)
            if len(document_ids) > n:
                document_ids = np.random.choice(
                    list(document_ids), size=int(n), replace=False
                )
            return document_ids

        # initialize some variables
        doc_citations = set(document.citations)
        doc_citations.add(document.id)
        result_ids_dict = {}
        for key in self.KEYS:
            result_ids_dict[key] = set()

        # step 0: make sure we heed the limitations about NN negatives
        if self.ann is None:
            n_per_type['easy'] += np.ceil(n_per_type['nn'] / 2.0)
            n_per_type['hard_negatives'] += np.ceil(n_per_type['nn'] / 2.0)
            n_per_type['nn'] = 0

        # step 1: find hard citation negatives, and remove true positives from it
        if n_per_type['hard_negatives'] > 0:
            result_ids_dict['hard_negatives'] = set(
                flatten(
                    [
                        list(self.corpus[id].citations) for id in document.citations
                        if id in self.corpus
                    ]
                )
            )
            result_ids_dict['hard_negatives'].difference_update(doc_citations)
            # adding hard_negatives to doc_citations so we can remove them later
            doc_citations.update(result_ids_dict['hard_negatives'])

        # step 2: get nearest neighbors from embeddings, and remove the true positives, hard citations and es negatives
        if n_per_type['nn'] > 0:
            # getting more than we need because of the jaccard cutoff
            candidate_nn_ids = self.ann.get_nns_by_id(
                document.id, 10 * n_per_type['nn']
            )
            if ann_jaccard_cutoff < 1:
                candidate_nn_ids = [
                    i for i in candidate_nn_ids
                    if jaccard(self.featurizer, document, self.corpus[i]) < ann_jaccard_cutoff
                ]
            result_ids_dict['nn'] = set(candidate_nn_ids)
            result_ids_dict['nn'].difference_update(doc_citations)
            # adding ann_negatives to doc_citations so we can remove them later
            doc_citations.update(result_ids_dict['nn'])

        # step 3: get easy negatives
        if n_per_type['easy'] > 0:
            random_index = np.random.randint(len(candidate_ids_list))
            random_index_range = np.arange(random_index, random_index + int(n_per_type['easy']))
            result_ids_dict['easy'] = set(np.take(candidate_ids_list, random_index_range, mode='wrap'))
            result_ids_dict['easy'].difference_update(doc_citations)

        # step 4: trim down the requested number of ids per type and get the actual documents
        result_docs = []
        for key in self.KEYS:
            docs = [
                self.corpus[doc_id]
                for doc_id in sample(result_ids_dict[key], n_per_type[key])
            ]
            result_docs.append(docs)

        return result_docs
