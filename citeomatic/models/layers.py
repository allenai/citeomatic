import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda
from keras.optimizers import Nadam
from keras.regularizers import l2

REGULARIZER = l2
LEARNING_RATE = 0.002
REG_LAMBDA = 1e-5
OPTIMIZER = Nadam()
PAIRWISE_LOSS = 'binary_crossentropy'


class ZeroMaskedEntries(Layer):
    """
    This layer is called after Embedding.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


class NamedLambda(Lambda):
    def __init__(self, name=None):
        Lambda.__init__(self, self.fn, name=name)

    @classmethod
    def invoke(cls, args, **kw):
        return cls(**kw)(args)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.name)


class L2Normalize(NamedLambda):
    def fn(self, x):
        return K.l2_normalize(x, axis=-1)


class ScalarMul(NamedLambda):
    def fn(self, x):
        return x[0] * x[1]


class Sum(NamedLambda):
    def fn(self, x):
        return K.sum(x, axis=1)


class Sigmoid(NamedLambda):
    def fn(self, x):
        return K.sigmoid(x)


class ScalarMultiply(Layer):
    def __init__(self, **kwargs):
        super(ScalarMultiply, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(
            shape=(1, 1), initializer='one', trainable=True
        )
        super(ScalarMultiply, self).build(input_shape)

    def call(self, x, mask=None):
        return self.w * x

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1]


def elu(x):
    alpha = 1.0
    pos = K.relu(x)
    neg = (x - abs(x)) * 0.5
    return pos + alpha * (K.exp(neg) - 1.)


def expnorm(x):
    exp_x = K.exp(x - K.max(x))
    return exp_x / K.sum(exp_x)


def logsumexp(x):
    max_x = K.max(x)
    return max_x + K.log(K.sum(K.exp(x - max_x)))


def min_max_kernel_loss(y_true, y_pred):
    p_true = expnorm(y_true)
    p_pred = expnorm(y_pred)
    min_max = K.sum(K.minimum(p_true, p_pred)
                   ) / K.sum(K.maximum(p_true, p_pred))
    return 1 - min_max


def listmle_loss(y_true, y_pred):
    """
    Find the negative log likelihood of the ground truth as per the current model. This loss
    function assumes that the batch contains one single document and the examples are sorted in
    descending order of their true score
    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred = K.l2_normalize(y_pred, axis=0)
    log_cum_sum_exp = K.log(tf.cumsum(K.exp(y_pred[::-1])))[::-1]
    return K.sum(-(tf.mul(y_pred, y_true)) + log_cum_sum_exp)


def listnet_loss(y_true, y_pred):
    """
    Inputs are treated as unnormalized log-probabilities.
    For inspiration code, see:
    https://github.com/koreyou/listnet_chainer/blob/master/listnet/training.py
    """
    log_p_true = y_true - logsumexp(y_true)  # log-sum-exp trick
    log_p_pred = y_pred - logsumexp(y_pred)
    return K.sum(
        K.abs(K.exp(log_p_true) * log_p_true - K.exp(log_p_true) * log_p_pred)
    )


def triplet_loss(y_true, y_pred):
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    pos = y_pred[::2]
    neg = y_pred[1::2]

    # margin is given by the difference in label for the correct order
    margin = y_true[::2] - y_true[1::2]
    delta = K.maximum(margin + neg - pos, 0)
    return K.mean(delta, axis=-1)
