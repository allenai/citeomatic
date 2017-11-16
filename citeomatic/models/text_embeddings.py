from abc import ABC
import numpy as np

from citeomatic.models.layers import L2Normalize, ScalarMul, Sum, EmbeddingZero
from citeomatic.models.options import ModelOptions
from keras.layers import Bidirectional, Input, LSTM, Concatenate, SpatialDropout1D
from keras.layers import Conv1D, Lambda, Dense, GlobalMaxPooling1D, Embedding
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

def make_embedder(options, pretrained_embeddings):
    if options.embedding_type == 'sum':
        embedder_title = TextEmbeddingSum(options=options, pretrained_embeddings=pretrained_embeddings)
        embedder_abstract = embedder_title
    elif options.embedding_type == 'cnn':
        embedder_title = TextEmbeddingConv(options=options, pretrained_embeddings=pretrained_embeddings, max_sequence_len=options.max_title_len)
        embedder_abstract = TextEmbeddingConv(options=options, pretrained_embeddings=pretrained_embeddings, max_sequence_len=options.max_abstract_len)
        # no reason not to share the embedding itself
        embedder_abstract.embed_direction = embedder_title.embed_direction
        embedder_abstract.embed_magnitude = embedder_title.embed_magnitude
    elif options.embedding_type == 'cnn2':
        embedder_title = TextEmbeddingConv2(options=options, pretrained_embeddings=pretrained_embeddings)
        embedder_abstract = TextEmbeddingConv2(options=options, pretrained_embeddings=pretrained_embeddings)
        # no reason not to share the embedding itself
        embedder_abstract.embed_direction = embedder_title.embed_direction
        embedder_abstract.embed_magnitude = embedder_title.embed_magnitude
    elif options.embedding_type == 'lstm':
        embedder_title = TextEmbeddingLSTM(options=options, pretrained_embeddings=pretrained_embeddings)
        embedder_abstract = embedder_title
    else:
        assert False, 'Unknown embedding type %s' % options.embedding_type
    return embedder_title, embedder_abstract


class TextEmbedding(object):

    def __init__(self,
                 options: ModelOptions,
                 pretrained_embeddings=None,
                 field_type='text',
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
            self.dense_dim = options.metadata_dim
        elif self.field_type == 'venue':
            self.n_features = options.n_venues
            self.dense_dim = options.metadata_dim
        elif self.field_type == 'keyphrases':
            self.n_features = options.n_keyphrases
            self.dense_dim = options.metadata_dim
        else:
            assert False

        self.l1_lambda = options.l1_lambda
        self.l2_lambda = options.l2_lambda * (pretrained_embeddings is None)
        self.dropout_p = options.dropout_p
        self.use_magdir = options.use_magdir
        self.magnitudes_initializer = magnitudes_initializer
        self.enable_fine_tune = options.enable_fine_tune
        self.pretrained_embeddings = pretrained_embeddings
        self.mask = None

    def define_embedding_layers(self):
        # shared layers
        self.embed_direction = EmbeddingZero(
            output_dim=self.dense_dim,
            input_dim=self.n_features,
            activity_regularizer=l2(self.l2_lambda),
            mask_zero=self.mask,
            trainable=self.pretrained_embeddings is None or self.enable_fine_tune
        )
        if self.pretrained_embeddings is not None:
            self.embed_direction.build((None,))
            set_embedding_layer_weights(self.embed_direction,
                                        self.pretrained_embeddings)

        self.embed_magnitude = EmbeddingZero(
            output_dim=1,
            input_dim=self.n_features,
            activity_regularizer=l1(self.l1_lambda),
            # will induce sparsity if large enough
            mask_zero=self.mask,
            embeddings_initializer=self.magnitudes_initializer
        )

        self.dropout = SpatialDropout1D(self.dropout_p)

    def embedding_constructor(self, prefix):
        _input = Input(shape=(None,), dtype='int32', name='%s-txt' % prefix)
        if self.use_magdir:
            dir_embedding = self.embed_direction(_input)
            direction = L2Normalize.invoke(dir_embedding,
                                           name='%s-dir-norm' % prefix)
            magnitude = self.embed_magnitude(_input)
            _embedding = ScalarMul.invoke([direction, magnitude],
                                          name='%s-embed' % prefix)
        else:
            _embedding = self.embed_direction(_input)
        _embedding = self.dropout(_embedding)
        return _input, _embedding


class TextEmbeddingSum(TextEmbedding):
    """
    Text embedding models class.
    """
    def __init__(self, **kwargs):
        super(TextEmbeddingSum, self).__init__(**kwargs)
        self.mask = True
        self.define_embedding_layers()

    def create_text_embedding_model(self, prefix="", final_l2_norm=True):
        """
        :param prefix: Preferred prefix to add to each layer in the model
        :return: A model that takes a sequence of words as inputs and outputs the normed sum of
        word embeddings that occur in the document.
        """
        _input, _embedding = self.embedding_constructor(prefix)
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


class TextEmbeddingConv(TextEmbedding):
    """
    Text embedding models class.
    """

    def __init__(self, max_sequence_len=None, **kwargs):
        super(TextEmbeddingConv, self).__init__(**kwargs)

        self.max_sequence_len = max_sequence_len
        self.kernel_width = kwargs['options'].kernel_width
        self.stride = kwargs['options'].stride
        self.mask = False

        self.define_embedding_layers()

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
        :param final_l2_norm: Whether to l2 norm final output
        :return: A model that takes a sequence of words as inputs and outputs the normed sum of
        word embeddings that occur in the document.
        """
        _input, _embedding = self.embedding_constructor(prefix)
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


class TextEmbeddingConv2(TextEmbedding):
    """
    Text embedding models class.

    More or less:
    https://arxiv.org/pdf/1408.5882v2.pdf
    """

    def __init__(self, **kwargs):
        super(TextEmbeddingConv2, self).__init__(**kwargs)

        self.filters = kwargs['options'].filters
        self.max_kernel_size = kwargs['options'].max_kernel_size
        self.mask = False

        self.define_embedding_layers()

        # shared conv layers
        self.conv_layers = []
        for kernel_size in range(2, self.max_kernel_size + 1):
            self.conv_layers.append(Conv1D(filters=self.filters,
                                           kernel_size=kernel_size,
                                           padding='same'))

        self.dense = Dense(self.dense_dim, activation='elu')

    def create_text_embedding_model(self, prefix="", final_l2_norm=True):
        """
        :param prefix: Preferred prefix to add to each layer in the model
        :return: A model that takes a sequence of words as inputs and outputs the normed sum of
        word embeddings that occur in the document.
        """
        _input, _embedding = self.embedding_constructor(prefix)
        list_of_convs = [GlobalMaxPooling1D()(conv(_embedding))
                         for conv in self.conv_layers]
        z = Concatenate()(list_of_convs) if len(list_of_convs) > 1 else list_of_convs[0]
        encoded = self.dense(z)

        if final_l2_norm:
            normed_encoded = L2Normalize.invoke(
                encoded, name='%s-l2_normed_conv_encoding' % prefix
            )
            outputs_list = [normed_encoded]
        else:
            outputs_list = [encoded]

        return Model(
            inputs=_input, outputs=outputs_list, name='%s-embedding-model'
        ), outputs_list


class TextEmbeddingLSTM(TextEmbedding):
    """
    Text embedding models class.
    """

    def __init__(self, **kwargs):
        super(TextEmbeddingLSTM, self).__init__(**kwargs)
        self.mask = True
        self.define_embedding_layers()
        self.bilstm = Bidirectional(LSTM(self.dense_dim))
        self.dense = Dense(self.dense_dim, activation='elu')

    def create_text_embedding_model(self, prefix="", final_l2_norm=True):
        """
        :param prefix: Preferred prefix to add to each layer in the model
        :return: A model that takes a sequence of words as inputs and outputs the normed sum of
        word embeddings that occur in the document.
        """
        _input, _embedding = self.embedding_constructor(prefix)
        lstm_embedding = self.dense(self.bilstm(_embedding))
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
