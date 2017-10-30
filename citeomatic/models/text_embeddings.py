from abc import ABC
import numpy as np

from citeomatic.models.layers import L2Normalize, ScalarMul, Sum, EmbeddingZero
from citeomatic.models.options import ModelOptions
from keras.layers import Bidirectional, Embedding, Input, LSTM, Concatenate, SpatialDropout1D
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l1, l2

import keras.backend as K


def _prefix(tuple):
    return '-'.join(tuple)

'''
TODO: we reuse a lot of code across the three TextEmbedding classes.
There should be a parent class that does all the embedding initialization logic.
Like this: class TextEmbedding(abc)
'''

class TextEmbeddingSum(object):
    """
    Text embedding models class.
    """

    def __init__(self, options: ModelOptions, pretrained_embeddings=None):
        self.n_features = options.n_features
        self.dense_dim = options.dense_dim
        self.l1_lambda = options.l1_lambda
        self.l2_lambda = options.l2_lambda * (pretrained_embeddings is None)
        self.dropout_p = options.dropout_p

        # shared layers
        self.embed_direction = EmbeddingZero(
            output_dim=self.dense_dim,
            input_dim=self.n_features,
            activity_regularizer=l2(self.l2_lambda),
            mask_zero=False,
            trainable=pretrained_embeddings is None
        )
        if pretrained_embeddings is not None:
            self.embed_direction.build((None,))
            set_embedding_layer_weights(self.embed_direction, pretrained_embeddings)

        self.embed_magnitude = EmbeddingZero(
            output_dim=1,
            input_dim=self.n_features,
            activity_regularizer=l1(self.l1_lambda),
            # will induce sparsity if large enough
            mask_zero=False
        )

    def create_text_embedding_model(self, prefix="", final_l2_norm=True):
        """
        :param prefix: Preferred prefix to add to each layer in the model
        :return: A model that takes a sequence of words as inputs and outputs the normed sum of
        word embeddings that occur in the document.
        """
        _input = Input(shape=(None,), dtype='int32', name='%s-txt' % prefix)
        dir_embedding = self.embed_direction(_input)
        direction = L2Normalize.invoke(dir_embedding, name='%s-dir-norm' % prefix)
        magnitude = self.embed_magnitude(_input)
        _embedding = ScalarMul.invoke([direction, magnitude], name='%s-embed' % prefix)
        _embedding = SpatialDropout1D(self.dropout_p)(_embedding)
        summed = Sum.invoke(_embedding, name='%s-sum-title' % prefix)
        if final_l2_norm:
            normed_sum = L2Normalize.invoke(
                summed, name='%s-l2_normed_sum' % prefix
            )
            outputs_list = [normed_sum]
        else:
            outputs_list = [summed]
        return Model(
            inputs=_input, outputs=outputs_list, name="%s-embedding-model" % prefix
        ), outputs_list


class TextEmbeddingConv(object):
    """
    Text embedding models class.
    """

    def __init__(self, options: ModelOptions, pretrained_embeddings=None):
        self.n_features = options.n_features
        self.dense_dim = options.dense_dim
        self.nb_filter = options.n_filter
        self.max_filter_length = options.max_filter_length
        self.l1_lambda = options.l1_lambda
        self.l2_lambda = options.l2_lambda * (pretrained_embeddings is None)
        self.dropout_p = options.dropout_p

        # shared embedding layers
        self.embed_direction = EmbeddingZero(
            output_dim=self.dense_dim,
            input_dim=self.n_features,
            activity_regularizer=l2(self.l2_lambda) * (pretrained_embeddings is None),
            mask_zero=False,
            trainable=pretrained_embeddings is None
        )
        if pretrained_embeddings is not None:
            self.embed_direction.build((None,))
            set_embedding_layer_weights(self.embed_direction, pretrained_embeddings)

        self.embed_magnitude = EmbeddingZero(
            output_dim=1,
            input_dim=n_features,
            activity_regularizer=l1(self.l1_lambda),
            mask_zero=False
        )

        # shared convolution layers
        self.conv_layers = []
        for i in range(1, max_filter_length + 1):
            self.conv_layers.append(
                Convolution1D(
                    n_filter, i, activation='linear', border_mode='same'
                )
            )

        # convolution gate layers
        # see: https://arxiv.org/abs/1612.08083
        self.conv_gate_layers = []
        for i in range(1, max_filter_length + 1):
            self.conv_gate_layers.append(
                Convolution1D(
                    n_filter, i, activation='sigmoid', border_mode='same'
                )
            )

    def create_text_embedding_model(self, prefix="", final_l2_norm=True):
        """
        :param prefix: Preferred prefix to add to each layer in the model
        :return: A model that takes a sequence of words as inputs and outputs the normed sum of
        word embeddings that occur in the document.
        """
        _input = Input(shape=(None,), dtype='int32', name='%s-txt' % prefix)
        dir_embedding = self.embed_direction(_input)
        direction = L2Normalize.invoke(dir_embedding, name='%s-dir-norm' % prefix)
        magnitude = self.embed_magnitude(_input)
        _embedding = ScalarMul.invoke([direction, magnitude], name='%s-embed' % prefix)
        _embedding = SpatialDropout1D(self.dropout_p)(_embedding)
        # perform convolutions of various lengths and concat them all
        # we multiply the convolutions by their "gates"
        list_of_gated_convs = [
            ScalarMul.invoke([conv(_embedding), conv_gate(_embedding)])
            for conv, conv_gate in zip(self.conv_layers, self.conv_gate_layers)
        ]
        if len(list_of_gated_convs) > 1:
            # last axis should have size nb_filter * max_filter_length
            concatted_embeddings = Concatenate(
                axis=-1, name='%s-concatted-embeddings' % prefix
            )(list_of_gated_convs)
        else:
            concatted_embeddings = list_of_gated_convs[0]
        # global max pool
        # pool = GlobalMaxPooling1D(name='%s-max-pool' % prefix)(concatted_embeddings)
        pool = GlobalAveragePooling1D(name='%s-ave-pool' % prefix
                                     )(concatted_embeddings)
        if final_l2_norm:
            normed_pool = L2Normalize.invoke(
                pool, name='%s-l2_normed_max_pool' % prefix
            )
            outputs_list = [normed_pool]
        else:
            outputs_list = [pool]

        return Model(
            inputs=_input, outputs=outputs_list, name='%s-embedding-model'
        ), outputs_list


class TextEmbeddingLSTM(object):
    """
    Text embedding models class.
    """

    def __init__(self, options: ModelOptions, pretrained_embeddings=None):
        self.n_features = options.n_features
        self.dense_dim = options.dense_dim
        self.lstm_dim = options.lstm_dim
        self.l2_lambda = options.l2_lambda * (pretrained_embeddings is None)
        self.dropout_p = options.dropout_p

        # shared embedding layers
        self.embedding = EmbeddingZero(
            output_dim=self.dense_dim,
            input_dim=self.n_features,
            activity_regularizer=l2(self.l2_lambda) * (pretrained_embeddings is None),
            mask_zero=True,
            trainable=pretrained_embeddings is None
        )
        if pretrained_embeddings is not None:
            self.embedding.build((None,))
            set_embedding_layer_weights(self.embedding, pretrained_embeddings)

        self.bilstm = Bidirectional(LSTM(lstm_dim))

    def create_text_embedding_model(self, prefix="", final_l2_norm=True):
        """
        :param prefix: Preferred prefix to add to each layer in the model
        :return: A model that takes a sequence of words as inputs and outputs the normed sum of
        word embeddings that occur in the document.
        """
        _input = Input(shape=(None,), dtype='int32', name='%s-txt' % prefix)
        _embedding = self.embedding(_input)
        _embedding = SpatialDropout1D(self.dropout_p)(_embedding)
        lstm_embedding = self.bilstm(_embedding)
        if final_l2_norm:
            normed_lstm_embedding = L2Normalize.invoke(
                lstm_embedding, name='%s-l2_normed_bilstm_embedding' % prefix
            )
            outputs_list = [normed_lstm_embedding]
        else:
            outputs_list = [lstm_embedding]
        return Model(
            inputs=_input, outputs=outputs_list, name="%s-embedding-model"
        ), outputs_list

def set_embedding_layer_weights(embedding_layer, pretrained_embeddings):
    dense_dim = pretrained_embeddings.shape[1]
    weights = np.vstack((np.zeros(dense_dim), pretrained_embeddings))
    embedding_layer.set_weights([weights])