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

from base import flatten

MAX_AUTHORS_PER_DOCUMENT = 8
CLEAN_TEXT_RE = re.compile('[^ a-z]')

# Adjustments to how we boost heavily cited documents.
CITATION_SLOPE = 0.01
MAX_CITATION_BOOST = 0.02

MIN_TRUE_CITATIONS = 3
MAX_TRUE_CITATIONS = 100

# Parameters for soft-margin data generation.
TRUE_CITATION_OFFSET = 0.3
HARD_NEGATIVE_OFFSET = 0.25
ES_NEGATIVE_OFFSET = 0.2
NN_NEGATIVE_OFFSET = 0.1
EASY_NEGATIVE_OFFSET = 0.0


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

    use_unigrams_from_corpus : bool, default=True
        Whether to use the unigrams found in the corpus.
    use_bigrams_from_corpus : bool, default=False
        Whether to use bigrams found in the corpus.
    keyphrases_path : string, default=None
        Path of the keyphrase dictionary/set. If not None, will load the keyphrase dictionary and use the
        sufficiently common keyphrases therein as tokens.
    index_type : {'exact', 'hashed'}, default='exact'
        Whether to use exact indexing or the hashing trick. If bigrams are turned on, 'hashed' is recommended.
    training_fraction : float, default=0.95
        What fraction of the corpus to use for training all of the word filtering and indexing.
    '''
    STOPWORDS = {
        'abstract', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
        'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the',
        'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with',
        'the', 'we', 'our', 'which'
    }

    N_KP_MAX_LENGTH = 4
    N_KP_MIN_COUNT = 10
    MIN_AUTHOR_PAPERS = 5  # minimum number of papers for an author to get an embedding.

    def __init__(
        self,
        max_title_len=32,
        max_abstract_len=256,
        use_unigrams_from_corpus=True,
        use_bigrams_from_corpus=False,
        allow_duplicates=True,
        training_fraction=0.95
    ):
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len
        self.use_unigrams_from_corpus = use_unigrams_from_corpus
        self.use_bigrams_from_corpus = use_bigrams_from_corpus
        self.allow_duplicates = allow_duplicates
        self.training_fraction = training_fraction
        self.word_count = {}
        self.author_to_index = {}
        self.word_indexer = None

    @property
    def n_authors(self):
        if not hasattr(self, 'author_to_index'):
            self.author_to_index = {}
        return len(self.author_to_index) + 1

    def fit(
        self, corpus, max_df_frac=0.90, min_df_frac=0.000025, max_features=None
    ):

        logging.info(
            'Usage at beginning of featurizer fit: %s',
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        )

        # Fitting authors:
        logging.info('Fitting authors')
        author_counts = collections.Counter()
        for doc in tqdm.tqdm(corpus):
            author_counts.update(doc.authors)

        for author, count in author_counts.items():
            if count >= Featurizer.MIN_AUTHOR_PAPERS:
                self.author_to_index[author] = 1 + len(self.author_to_index)

        # Step 1: filter out some words and make a word_count dictionary
        logging.info('Cleaning text.')
        all_docs_text = [
            ' '.join((_clean(doc.title), _clean(doc.abstract)))
            for doc in tqdm.tqdm(corpus)
        ]

        logging.info('Fitting vectorizer...')
        if max_features is not None:
            count_vectorizer = CountVectorizer(
                max_df=max_df_frac,
                max_features=max_features,
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
        self.n_features = (
            1 + len(self.word_count) * self.use_unigrams_from_corpus
        )

        if self.use_unigrams_from_corpus:
            wc = self.word_count
        else:
            wc = {}
        self.word_indexer = FeatureIndexer(
            word_count=wc, allow_duplicates=self.allow_duplicates
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
        return np.asarray(
            [math.log(doc.in_citation_count + 1) for doc in documents]
        )

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

    es_negatives : dict
        A dictionary of elasticsearch negative training examples for each training document in the Corpus.
    """
    KEYS = ['hard_negatives', 'es', 'nn', 'easy']

    def __init__(
        self,
        corpus,
        featurizer,
        ann=None,
    ):
        self.ann = ann
        self.corpus = corpus
        self.featurizer = CachingFeaturizer(featurizer)
        self.es_negatives = None

    def _listwise_examples(self, id_pool, id_filter=None, neg_to_pos_ratio=5):
        id_pool = self.corpus.filter(id_pool)
        if id_filter is None:
            id_filter = self.corpus.train_ids
        else:
            id_filter = self.corpus.filter(id_filter)

        id_filter = set(id_filter)
        id_list = np.array(list(id_filter))

        def _label_for_doc(d, offset):
            sigmoid = 1 / (1 + np.exp(-d.in_citation_count * CITATION_SLOPE))
            return offset + (sigmoid * MAX_CITATION_BOOST)

        while True:
            for doc_id in np.random.permutation(id_list):
                examples = []
                labels = []
                query = self.corpus[doc_id]
                true_citations = id_filter.intersection(query.citations)
                if len(true_citations) < MIN_TRUE_CITATIONS:
                    continue

                if len(true_citations) > MAX_TRUE_CITATIONS:
                    true_citations = np.random.choice(
                        list(true_citations), MAX_TRUE_CITATIONS, replace=False
                    )

                n_positive = len(true_citations)
                n_per_type = {}
                n_per_type['hard_negatives'] = n_positive
                n_per_type['es'] = n_positive
                n_per_type['easy'] = int(n_positive * neg_to_pos_ratio / 2.0)
                n_per_type['nn'] = int(n_positive * neg_to_pos_ratio / 2.0)

                pos_jaccard_sims = [
                    jaccard(self.featurizer, query, self.corpus[i])
                    for i in true_citations
                ]
                ann_jaccard_cutoff = np.percentile(pos_jaccard_sims, 0.05)
                hard_negatives, es_negatives, nn_negatives, easy_negatives = self.get_negatives(
                    id_filter, id_list, n_per_type, query, ann_jaccard_cutoff
                )

                for c in true_citations:
                    doc = self.corpus[c]
                    labels.append(_label_for_doc(doc, TRUE_CITATION_OFFSET))
                    examples.append(doc)

                for doc in es_negatives:
                    labels.append(_label_for_doc(doc, ES_NEGATIVE_OFFSET))
                    examples.append(doc)

                for doc in hard_negatives:
                    labels.append(_label_for_doc(doc, HARD_NEGATIVE_OFFSET))
                    examples.append(doc)

                for doc in easy_negatives:
                    labels.append(_label_for_doc(doc, EASY_NEGATIVE_OFFSET))
                    examples.append(doc)

                for doc in nn_negatives:
                    labels.append(_label_for_doc(doc, NN_NEGATIVE_OFFSET))
                    examples.append(doc)

                labels = np.asarray(labels)
                sorted_idx = np.argsort(labels)[::-1]
                labels = labels[sorted_idx]
                examples = [examples[i] for i in sorted_idx]

                yield query, examples, labels

    def listwise_generator(self, id_pool, id_filter=None):
        for query, examples, labels in self._listwise_examples(
            id_pool, id_filter
        ):
            yield (
                self.featurizer.transform_query_and_results(query, examples),
                labels,
            )

    def triplet_generator(
        self, id_pool, id_filter=None, batch_size=1024, neg_to_pos_ratio=5
    ):
        queries = []
        batch_ex = []
        batch_labels = []

        # Sample examples from our sorted list.  The margin between each example is the difference in their label:
        # easy negatives (e.g. very bad results) should be further away from a true positive than hard negatives
        # (less embarrassing).
        for q, ex, labels in self._listwise_examples(
            id_pool,
            id_filter,
            neg_to_pos_ratio=neg_to_pos_ratio,
        ):
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
        self, id_filter, id_list, n_per_type, document, ann_jaccard_cutoff=1
    ):
        '''
        :param n_per_type: dictionary with keys: 'easy', 'hard_negatives', 'es', 'nn'
        :param document: query document
        :return: documents
        '''

        def sample(document_ids, n):
            document_ids = document_ids.intersection(id_filter)
            if len(document_ids) > n:
                document_ids = np.random.choice(
                    list(document_ids), size=n, replace=False
                )
            return document_ids

        # initialize some variables
        all_docs = set(document.citations)
        all_docs.add(document.id)
        result_ids_dict = {}
        for key in self.KEYS:
            result_ids_dict[key] = set()

        # step 0: make sure we heed the limitations about ES and NN negatives
        if self.es_negatives is None:
            n_per_type['hard_negatives'] += n_per_type['es']
            n_per_type['es'] = 0

        if self.ann is None:
            n_per_type['easy'] += n_per_type['nn']
            n_per_type['nn'] = 0

        # step 1: find ALL hard citation negatives, and remove true positives from it
        if n_per_type['hard_negatives'] > 0:
            result_ids_dict['hard_negatives'] = set(
                flatten(
                    [
                        list(self.corpus[id].citations) for id in document.citations
                        if id in self.corpus
                    ]
                )
            )
            result_ids_dict['hard_negatives'].difference_update(all_docs)
            all_docs.update(result_ids_dict['hard_negatives'])

        # step 2: get ALL es_negatives, and remove the true positives and hard citations from it
        if n_per_type['es'] > 0:
            if document.id in self.es_negatives:
                result_ids_dict['es'] = set(self.es_negatives[document.id])
                result_ids_dict['es'].difference_update(all_docs)
                all_docs.update(result_ids_dict['es'])

        # step 3: get nearest neighbors from embeddings, and remove the true positives, hard citations and es negatives
        if n_per_type['nn'] > 0:
            candidate_nn_ids = self.ann.get_nns_by_id(
                document.id, 10 * n_per_type['nn']
            )
            if ann_jaccard_cutoff < 1:
                candidate_nn_ids = [
                    i for i in candidate_nn_ids
                    if jaccard(self.featurizer, document, self.corpus[i]) < ann_jaccard_cutoff
                ]
            result_ids_dict['nn'] = set(candidate_nn_ids)
            result_ids_dict['nn'].difference_update(all_docs)
            all_docs.update(result_ids_dict['nn'])

        # step 4: get easy negatives
        if n_per_type['easy'] > 0:
            result_ids_dict['easy'] = set(
                np.random.choice(id_list, size=n_per_type['easy'], replace=False)
            )
            result_ids_dict['easy'].difference_update(all_docs)

        # trim down the requested number of ids per type and get the actual documents
        result_docs = []
        for key in self.KEYS:
            docs = [
                self.corpus[doc_id]
                for doc_id in sample(result_ids_dict[key], n_per_type[key])
            ]
            result_docs.append(docs)

        return result_docs
