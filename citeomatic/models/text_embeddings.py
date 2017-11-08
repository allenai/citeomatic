from abc import ABC
import numpy as np

from citeomatic.models.layers import L2Normalize, ScalarMul, Sum, EmbeddingZero
from citeomatic.models.options import ModelOptions
from keras.layers import Bidirectional, Input, LSTM, Concatenate, SpatialDropout1D
from keras.layers import Conv1D, Lambda
from keras.models import Model
from keras.regularizers import l1, l2

import keras.backend as K


def _prefix(tuple):
    return '-'.join(tuple)

def set_embedding_layer_weights(embedding_layer, pretrained_embeddings):
    dense_dim = pretrained_embeddings.shape[1]
    weights = np.vstack((np.zeros(dense_dim), pretrained_embeddings))
    embedding_layer.set_weights([weights])

def valid_conv_kernel_size(input_kernel_size, h, r):
    return int(np.floor((input_kernel_size - h)/r + 1))

'''
TODO: we reuse a lot of code across the three TextEmbedding classes.
There should be a parent class that does all the embedding initialization logic.
Like this: class TextEmbedding(abc)
'''

class TextEmbeddingSum(object):
    """
    Text embedding models class.
    """

    def __init__(self, options: ModelOptions, pretrained_embeddings=None, field_type='text',
                 magnitudes_initializer='uniform'):
        """

        :param options:
        :param pretrained_embeddings:
        :param embedding_type: Takes one of three values: text / authors / venues depending on
        which field is being embedded
        """
        self.field_type = field_type
        if self.field_type == 'text':
            self.n_features = options.n_features
            self.dense_dim = options.dense_dim
        elif self.field_type == 'authors':
            self.n_features = options.n_authors
            self.dense_dim = options.author_dim
        elif self.field_type == 'venue':
            self.n_features = options.n_venues
            self.dense_dim = options.venue_dim
        else:
            assert False

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
            mask_zero=False,
            embeddings_initializer=magnitudes_initializer
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

    def __init__(self, options: ModelOptions, pretrained_embeddings=None, field_type='text', max_sequence_len=None,
                 magnitudes_initializer='uniform'):
        self.field_type = field_type
        if self.field_type == 'text':
            self.n_features = options.n_features
            self.dense_dim = options.dense_dim
        elif self.field_type == 'authors':
            self.n_features = options.n_authors
            self.dense_dim = options.author_dim
        elif self.field_type == 'venue':
            self.n_features = options.n_venues
            self.dense_dim = options.venue_dim
        else:
            assert False

        self.max_sequence_len = max_sequence_len
        self.l1_lambda = options.l1_lambda
        self.l2_lambda = options.l2_lambda * (pretrained_embeddings is None)
        self.dropout_p = options.dropout_p
        self.kernel_width = options.kernel_width
        self.stride = options.stride

        # shared embedding layers
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
            mask_zero=False
        )

        # shared convolution layers
        conv1_output_length = valid_conv_kernel_size(max_sequence_len, self.kernel_width, self.stride)
        conv2_output_length = valid_conv_kernel_size(conv1_output_length, self.kernel_width, self.stride)

        self.conv1 = Conv1D(filters=self.dense_dim,
                            kernel_size=self.kernel_width,
                            strides=self.stride,
                            padding='valid',
                            activation='elu')

        self.conv2 = Conv1D(filters=self.dense_dim,
                            kernel_size=self.kernel_width,
                            strides=self.stride,
                            padding='valid',
                            activation='elu')

        self.conv3 = Conv1D(filters=self.dense_dim,
                            kernel_size=conv2_output_length,
                            strides=self.stride,
                            padding='valid',
                            activation='elu')

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
        conved1 = self.conv1(_embedding)
        conved2 = self.conv2(conved1)
        conved3 = self.conv3(conved2)
        conved3 = Lambda(lambda x: K.squeeze(x, axis=1))(conved3)
        if final_l2_norm:
            normed_conved3 = L2Normalize.invoke(
                conved3, name='%s-l2_normed_conv_encoding' % prefix
            )
            outputs_list = [normed_conved3]
        else:
            outputs_list = [conved3]

        return Model(
            inputs=_input, outputs=outputs_list, name='%s-embedding-model'
        ), outputs_list


class TextEmbeddingLSTM(object):
    """
    Text embedding models class.
    """

    def __init__(self, options: ModelOptions, pretrained_embeddings=None, field_type='text',
                 magnitudes_initializer='uniform'):
        self.field_type = field_type
        if self.field_type == 'text':
            self.n_features = options.n_features
            self.dense_dim = options.dense_dim
        elif self.field_type == 'authors':
            self.n_features = options.n_authors
            self.dense_dim = options.author_dim
        elif self.field_type == 'venue':
            self.n_features = options.n_venues
            self.dense_dim = options.venue_dim
        else:
            assert False

        self.l2_lambda = options.l2_lambda * (pretrained_embeddings is None)
        self.l1_lambda = options.l1_lambda
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
            mask_zero=False,
            embeddings_initializer=magnitudes_initializer
        )

        self.bilstm = Bidirectional(LSTM(self.dense_dim))

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
