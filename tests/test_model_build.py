import unittest

from citeomatic.models.options import ModelOptions
from citeomatic.utils import import_from


class TestModelBuild(unittest.TestCase):
    def test_build_paper_embedder(self):
        options = ModelOptions(**{})
        create_model = import_from("citeomatic.models.paper_embedder", "create_model")
        try:
            models = create_model(options)
            assert 'embedding' in models
            assert 'citeomatic' in models
            assert True
        except Exception:
            assert False

    def test_build_citation_ranker(self):
        options = ModelOptions(**{})
        create_model = import_from("citeomatic.models.citation_ranker", "create_model")
        try:
            models = create_model(options)
            assert models['embedding'] is None
            assert 'citeomatic' in models
        except Exception:
            assert False

        options.use_authors = True
        options.n_authors = 10
        try:
            models = create_model(options)
        except Exception:
            assert False

        options.use_venue = True
        options.n_venues = 10
        try:
            models = create_model(options)
        except Exception:
            assert False

        options.use_citations = True
        try:
            models = create_model(options)
        except Exception:
            assert False
        options.use_citations = False
        try:
            models = create_model(options)
        except Exception:
            assert False

        options.use_sparse = True
        try:
            models = create_model(options)
        except Exception:
            assert False

        options.use_src_tgt_embeddings = True
        try:
            models = create_model(options)
        except Exception:
            assert False


