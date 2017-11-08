import unittest

from citeomatic.corpus import Corpus
from citeomatic.features import Featurizer, DataGenerator
from citeomatic.models.layers import triplet_loss
from citeomatic.models.options import ModelOptions
from citeomatic.utils import import_from
from tests.test_corpus import build_test_corpus

create_model = import_from("citeomatic.models.citation_ranker", "create_model")


class TestModelBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')
        corpus = Corpus.load('/tmp/foo.sqlite')

        featurizer = Featurizer()
        featurizer.fit(corpus, max_df_frac=1.0)

        options = ModelOptions(**{})
        options.n_features = featurizer.n_features
        options.n_authors = featurizer.n_authors
        options.n_venues = featurizer.n_venues

        cls.corpus = corpus
        cls.featurizer = featurizer
        cls.options = options

    def test_build_paper_embedder(self):
        embedder_create_model = import_from("citeomatic.models.paper_embedder", "create_model")
        try:
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

    def _test_train(self, models: dict):
        model = models['citeomatic']
        model.compile(optimizer='nadam', loss=triplet_loss)
        dg = DataGenerator(self.corpus, self.featurizer)

        training_generator = dg.triplet_generator(paper_ids=self.corpus.train_ids, batch_size=2)

        model.fit_generator(training_generator, steps_per_epoch=1, epochs=10)



