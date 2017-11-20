#!/usr/bin/env python
"""
Helpers for pickle compatibility across module renames.
"""
import json
import os
from typing import Tuple, Any

import tensorflow as tf

from citeomatic import file_util
from citeomatic.common import DatasetPaths
from citeomatic.features import Featurizer
from citeomatic.models.options import ModelOptions
from citeomatic.utils import import_from


def model_from_directory(dirname: str, on_cpu=False) -> Tuple[Featurizer, Any]:
    dp = DatasetPaths()

    options_json = file_util.read_json(
        os.path.join(dirname, dp.OPTIONS_FILENAME),
    )
    options = ModelOptions(**json.loads(options_json))

    featurizer_file_prefix = 'pretrained_' if options.use_pretrained else 'corpus_fit_'

    featurizer = file_util.read_pickle(
        os.path.join(dirname, featurizer_file_prefix + dp.FEATURIZER_FILENAME)
    )  # type: Featurizer

    options.n_authors = featurizer.n_authors
    options.n_features = featurizer.n_features
    options.n_venues = featurizer.n_venues
    options.n_keyphrases = featurizer.n_keyphrases
    create_model = import_from(
        'citeomatic.models.%s' % options.model_name, 'create_model'
    )
    if on_cpu:
        with tf.device('/cpu:0'):
            models = create_model(options)
    else:
        models = create_model(options)

    print("Loading model from %s " % dirname)
    print(models['citeomatic'].summary())
    if dirname.startswith('s3://'):
        models['citeomatic'].load_weights(
            file_util.cache_file(os.path.join(dirname, dp.CITEOMATIC_WEIGHTS_FILENAME))
        )
        models['embedding'].load_weights(
            file_util.cache_file(os.path.join(dirname, dp.EMBEDDING_WEIGHTS_FILENAME))
        )
    else:
        models['citeomatic'].load_weights(os.path.join(dirname, dp.CITEOMATIC_WEIGHTS_FILENAME))
        if models['embedding'] is not None:
            models['embedding'].load_weights(os.path.join(dirname, dp.EMBEDDING_WEIGHTS_FILENAME))
    return featurizer, models
