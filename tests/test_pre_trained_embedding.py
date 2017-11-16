import random
import unittest
import os

import h5py
from sklearn.preprocessing import normalize

from citeomatic.models.options import ModelOptions
from citeomatic.models.text_embeddings import TextEmbeddingSum
import numpy as np

FIXTURES = os.path.join('tests', 'fixtures')
EMBEDDINGS_FILE = os.path.join(FIXTURES, 'weights.h5')


def almost_equal(x, y, threshold=0.0001):
    return abs(x-y) < threshold


class TestPreTrainedEmbedding(unittest.TestCase):

    def test_pre_trained_layer(self):

        with h5py.File(EMBEDDINGS_FILE, 'r') as f:
            pretrained_embeddings = f['embedding'][...]

        options = ModelOptions()
        options.use_pretrained = True
        options.dense_dim = 300
        options.n_features = 200
        t_embedding_sum = TextEmbeddingSum(options=options,
                                           pretrained_embeddings=pretrained_embeddings,
                                           magnitudes_initializer='ones'
                                           )

        embedding_model, outputs = t_embedding_sum.create_text_embedding_model(
            prefix='test', final_l2_norm=False)

        idx = random.randint(0, 200)

        pred = embedding_model.predict(np.asarray([idx + 1]))[0]
        input_embedding = normalize(pretrained_embeddings[idx].reshape(1, -1))[0]
        assert all(map(almost_equal, pred, input_embedding))
