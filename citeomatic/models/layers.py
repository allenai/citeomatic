import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda
from keras.layers import Concatenate, Dot, Reshape, Flatten

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


class ScalarMultiply(Layer):
    def __init__(self, **kwargs):
        super(ScalarMultiply, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(
            shape=(1, 1), initializer='one', trainable=True, name='w'
        )
        super(ScalarMultiply, self).build(input_shape)

    def call(self, x, mask=None):
        return self.w * x

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1]


def custom_dot(a, b, d, normalize=True):
    # keras is terrible...
    reshaped_a = Reshape((1, d))(a)
    reshaped_b = Reshape((1, d))(b)
    reshaped_in = [reshaped_a, reshaped_b]
    dotted = Dot(axes=(2, 2), normalize=normalize)(reshaped_in)
    return Flatten()(dotted)


def triplet_loss(y_true, y_pred):
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    pos = y_pred[::2]
    neg = y_pred[1::2]
    # margin is given by the difference in labels
    margin = y_true[::2] - y_true[1::2]
    delta = K.maximum(margin + neg - pos, 0)
    return K.mean(delta, axis=-1)
