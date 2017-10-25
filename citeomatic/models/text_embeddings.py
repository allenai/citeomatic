from citeomatic.models.layers import L2Normalize, ScalarMul, Sum, ZeroMaskedEntries
from keras.layers import Bidirectional, Embedding, Input, LSTM, Concatenate
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l1, l2


def _prefix(tuple):
    return '-'.join(tuple)


class TextEmbedding(object):
    """
    Text embedding models class.
    """

    def __init__(self, n_features, dense_dim, l1_lambda=0, l2_lambda=0):
        self.n_features = n_features
        self.dense_dim = dense_dim
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # shared layers
        self.embed_direction = Embedding(
            output_dim=self.dense_dim,
            input_dim=self.n_features,
            activity_regularizer=l2(self.l2_lambda),
            mask_zero=True
        )
        self.mask_direction = ZeroMaskedEntries()
        self.embed_magnitude = Embedding(
            output_dim=1,
            input_dim=self.n_features,
            activity_regularizer=l1(self.l1_lambda),
            # will induce sparsity if large enough
            mask_zero=True
        )
        self.mask_magnitude = ZeroMaskedEntries()

    def create_text_embedding_model(self, prefix="", final_l2_norm=True):
        """
        :param prefix: Preferred prefix to add to each layer in the model
        :return: A model that takes a sequence of words as inputs and outputs the normed sum of
        word embeddings that occur in the document.
        """
        _input = Input(shape=(None,), dtype='int32', name='%s-txt' % prefix)
        dir_embedding = self.mask_direction(self.embed_direction(_input))

        direction = L2Normalize.invoke(
            dir_embedding, name='%s-dir-norm' % prefix
        )
        magnitude = self.mask_magnitude(self.embed_magnitude(_input))
        _embedding = ScalarMul.invoke(
            [direction, magnitude], name='%s-embed' % prefix
        )
        sum = Sum.invoke(_embedding, name='%s-sum-title' % prefix)
        if final_l2_norm:
            normed_sum = L2Normalize.invoke(
                sum, name='%s-l2_normed_sum' % prefix
            )
            outputs_list = [normed_sum]
        else:
            outputs_list = [sum]
        return Model(
            inputs=_input, outputs=outputs_list, name="%s-embedding-model" % prefix
        ), outputs_list


class TextEmbeddingConv(object):
    """
    Text embedding models class.
    """

    def __init__(
        self, n_features, dense_dim, n_filter, max_filter_length, l1_lambda=0, l2_lambda=0
    ):
        self.n_features = n_features
        self.dense_dim = dense_dim
        self.nb_filter = n_filter
        self.max_filter_length = max_filter_length
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # shared embedding layers
        self.embed_direction = Embedding(
            output_dim=self.dense_dim,
            input_dim=self.n_features,
            activity_regularizer=l2(self.l2_lambda),
            mask_zero=True
        )
        self.mask_direction = ZeroMaskedEntries()
        self.embed_magnitude = Embedding(
            output_dim=1,
            input_dim=n_features,
            activity_regularizer=l1(self.l1_lambda),
            mask_zero=True
        )
        self.mask_magnitude = ZeroMaskedEntries()

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
        dir_embedding = self.mask_direction(self.embed_direction(_input))

        direction = L2Normalize.invoke(
            dir_embedding, name='%s-dir-norm' % prefix
        )
        magnitude = self.mask_magnitude(self.embed_magnitude(_input))
        _embedding = ScalarMul.invoke(
            [direction, magnitude], name='%s-embed' % prefix
        )
        # perform convolutions of various lengths and concat them all
        # we are multiply the convolutions by their "gates"
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

    def __init__(self, n_features, dense_dim, lstm_dim, l2_lambda=0, **kw):
        self.n_features = n_features
        self.dense_dim = dense_dim
        self.lstm_dim = lstm_dim
        self.l2_lambda = l2_lambda

        # shared embedding layers
        self.embedding = Embedding(
            output_dim=self.dense_dim,
            input_dim=self.n_features,
            activity_regularizer=l2(self.l2_lambda),
            mask_zero=True
        )
        self.bilstm = Bidirectional(LSTM(lstm_dim))

    def create_text_embedding_model(self, prefix="", final_l2_norm=True):
        """
        :param prefix: Preferred prefix to add to each layer in the model
        :return: A model that takes a sequence of words as inputs and outputs the normed sum of
        word embeddings that occur in the document.
        """
        _input = Input(shape=(None,), dtype='int32', name='%s-txt' % prefix)
        _embedding = self.embedding(_input)
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
