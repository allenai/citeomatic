#!/usr/bin/env python
"""
Helpers for pickle compatibility across module renames.
"""
from typing import Tuple, Any

from base import file_util
import os
import importlib
import pickle

from citeomatic.features import Featurizer


class ModelLoader(pickle.Unpickler):
    def find_class(self, mod_name, klass_name):
        if mod_name[:4] == 'ai2.':
            mod_name = mod_name[4:]

        mod = importlib.import_module(mod_name)
        return getattr(mod, klass_name)


def load_pickle(filename):
    with file_util.open(filename) as f:
        return ModelLoader(f).load()


def model_from_directory(dirname: str) -> Tuple[Featurizer, Any]:
    config = load_pickle(os.path.join(dirname, 'config.pickle'))  # type: dict
    featurizer = load_pickle(os.path.join(dirname, 'featurizer.pickle')
                            )  # type: Featurizer

    try:
        options = load_pickle(os.path.join(dirname, 'options.pickle')
                             )  # type: ModelOptions
        create_model = import_from(
            'citeomatic.models.%s' % config['model_name'], 'create_model'
        )
        models = create_model(options)
    except FileNotFoundError:
        create_model = import_from(
            'citeomatic.models.%s' % config['model_name'], 'create_model'
        )
        opts = dict(
            n_features=featurizer.n_features,
            n_authors=featurizer.n_authors,
            dense_dim=config['dense_dim'],
            enable_citation_feature=config['enable_citation_feature'],
        )
        models = create_model(**opts)

    print("Loading model from %s " % dirname)
    print(models['citeomatic'].summary())
    if dirname.startswith('s3://'):
        models['citeomatic'].load_weights(
            file_util.cache_file(os.path.join(dirname, 'weights.h5'))
        )
        models['embedding'].load_weights(
            file_util.cache_file(os.path.join(dirname, 'embedding_weights.h5'))
        )
    else:
        models['citeomatic'].load_weights(os.path.join(dirname, 'weights.h5'))
        models['embedding'
              ].load_weights(os.path.join(dirname, 'embedding_weights.h5'))
    return featurizer, models


def import_from(module, name):
    module = importlib.import_module(module)
    module = importlib.reload(module)
    return getattr(module, name)
