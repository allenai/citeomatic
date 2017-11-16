import unittest

import numpy as np

from citeomatic.corpus import Corpus
from citeomatic.features import Featurizer, DataGenerator
from citeomatic.models.layers import triplet_loss
from citeomatic.models.options import ModelOptions
from citeomatic.utils import import_from
from tests.test_corpus import build_test_corpus
import keras.backend as K

create_model = import_from("citeomatic.models.citation_ranker", "create_model")
embedder_create_model = import_from("citeomatic.models.paper_embedder", "create_model")

class TestModelBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')
        corpus = Corpus.load('/tmp/foo.sqlite')

        options = ModelOptions(**{})

        featurizer = Featurizer(max_title_len=options.max_title_len, max_abstract_len=options.max_abstract_len)
        featurizer.fit(corpus, max_df_frac=1.0)

        options.n_features = featurizer.n_features
        options.n_authors = featurizer.n_authors
        options.n_venues = featurizer.n_venues
        options.n_keyphrases = featurizer.n_keyphrases

        cls.corpus = corpus
        cls.featurizer = featurizer
        cls.options = options

    def test_build_paper_embedder_sum(self):
        try:
            models = embedder_create_model(self.options)
            assert 'embedding' in models
            assert 'citeomatic' in models
            self._test_train(models)
            assert True
        except Exception:
            assert False

    def test_build_magdir(self):
        try:
            models = embedder_create_model(self.options)
            self.options.use_magdir = False
            assert 'embedding' in models
            assert 'citeomatic' in models
            self._test_train(models)
            assert True
        except Exception:
            assert False

    def test_build_paper_embedder_cnn(self):
        try:
            self.options.embedding_type = 'cnn'
            models = embedder_create_model(self.options)
            assert 'embedding' in models
            assert 'citeomatic' in models
            self._test_train(models)
            assert True
        except Exception:
            assert False

    def test_build_paper_embedder_cnn2(self):
        try:
            self.options.embedding_type = 'cnn2'
            models = embedder_create_model(self.options)
            assert 'embedding' in models
            assert 'citeomatic' in models
            self._test_train(models)
            assert True
        except Exception:
            assert False

    def test_cnn(self):
        self.options.embedding_type = 'cnn'
        try:
            models = create_model(self.options)
            self._test_train(models)
        except Exception:
            assert False

    def test_lstm(self):
        self.options.embedding_type = 'lstm'
        try:
            models = create_model(self.options)
            self._test_train(models)
        except Exception:
            assert False

    def test_build_paper_embedder_lstm(self):
        try:
            self.options.embedding_type = 'lstm'
            models = embedder_create_model(self.options)
            assert 'embedding' in models
            assert 'citeomatic' in models
            self._test_train(models)
            assert True
        except Exception:
            assert False

    def test_build_train_ranker(self):
        try:
            models = create_model(self.options)
            assert models['embedding'] is None
            assert 'citeomatic' in models
            self._test_train(models)
        except Exception:
            assert False

    def test_use_author(self):
        self.options.use_authors = True
        try:
            models = create_model(self.options)
            self._test_train(models)
        except Exception:
            assert False

    def test_use_venue(self):
        self.options.use_venue = True
        try:
            models = create_model(self.options)
            self._test_train(models)
        except Exception:
            assert False

    def test_use_keyphrases(self):
        self.options.use_keyphrases = True
        try:
            models = create_model(self.options)
            self._test_train(models)
        except Exception:
            assert False

    def test_use_citations(self):
        self.options.use_citations = True
        try:
            models = create_model(self.options)
            self._test_train(models)
        except Exception:
            assert False

        self.options.use_citations = False
        try:
            models = create_model(self.options)
            self._test_train(models)
        except Exception:
            assert False

    def test_use_sparse(self):
        self.options.use_sparse = True
        try:
            models = create_model(self.options)
            self._test_train(models)
        except Exception:
            assert False

    def test_siamese(self):
        self.options.use_src_tgt_embeddings = True
        try:
            models = create_model(self.options)
            self._test_train(models)
        except Exception:
            assert False

    def _test_train(self, models: dict):
        model = models['citeomatic']
        model.compile(optimizer='nadam', loss=triplet_loss)
        dg = DataGenerator(self.corpus, self.featurizer, candidate_selector=TestCandidateSelector())

        training_generator = dg.triplet_generator(paper_ids=self.corpus.train_ids, batch_size=2)

        model.fit_generator(training_generator, steps_per_epoch=1, epochs=10)
        K.clear_session()


class TestCandidateSelector(object):
    def confidence(self, doc_id, candidate_ids):
        return np.ones(len(candidate_ids))
