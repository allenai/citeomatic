#!/usr/bin/env python
"""
Helpers for pickle compatibility across module renames.
"""
from typing import Tuple, Any

import os
import json
import importlib
import pickle

from citeomatic import file_util
from citeomatic.utils import import_from
from citeomatic.features import Featurizer
from citeomatic.model.options import ModelOptions


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
    featurizer = file_util.read_pickle(
        os.path.join(dirname, 'featurizer.pickle')
    )  # type: Featurizer

    options_json = file_util.read_json(
        os.path.join(dirname, 'options.json'),
    )
    options = ModelOptions(**json.loads(options_json))
    options.n_authors = featurizer.n_authors
    options.n_features = featurizer.n_features
    create_model = import_from(
        'citeomatic.models.%s' % options.model_name, 'create_model'
    )
    models = create_model(options)
    citeomatic_model, embedding_model = models['citeomatic'], models['embedding']

    print("Loading model from %s " % dirname)
    print(citeomatic_model.summary())
    if dirname.startswith('s3://'):
        citeomatic_model.load_weights(
            file_util.cache_file(os.path.join(dirname, 'weights.h5'))
        )
        embedding_model.load_weights(
            file_util.cache_file(os.path.join(dirname, 'embedding.h5'))
        )
    else:
        citeomatic_model.load_weights(os.path.join(dirname, 'weights.h5'))
        embedding_model.load_weights(os.path.join(dirname, 'embedding.h5'))
    return featurizer, models
