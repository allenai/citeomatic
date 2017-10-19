import json
from citeomatic import file_util
from traitlets import Bool, HasTraits, Int, Unicode, Enum, Float


class ModelOptions(HasTraits):
    model_name = Enum(values=['model_ann', 'model_full'], default_value='model_full')
    n_features = Int()
    n_authors = Int()

    dense_dim = Int(default_value=75)
    embedding_type = Enum(values=['sum', 'cnn', 'lstm'], default_value='sum')

    use_nn_negatives = Bool(default_value=False)
    # todo: have a field here with the embedding_model directory?
    use_dense = Bool(default_value=True)
    use_citations = Bool(default_value=True)
    use_authors = Bool(default_value=False)
    author_dim = Int(default_value=10)
    sparse_option = Enum(values=['none', 'linear', 'attention'], default_value='linear')
    use_attention = Bool(default_value=False)
    use_holographic = Bool(default_value=False)
    use_src_tgt_embeddings = Bool(default_value=False)
    optimizer = Unicode(default_value='tfopt')

    lr = Float(default_value=0.0001)
    margin_multiplier = Float(default_value=1)

    # training and feature params
    train_frac = Float(default_value=0.8) # the rest will be divided 50/50 val/test
    max_features = Int(default_value=200000)
    neg_to_pos_ratio = Int(default_value=6) # ideally divisible by 2 and 3
    batch_size = Int(default_value=1024)
    samples_per_epoch = Int(default_value=1000000)
    total_samples = Int(default_value=5000000)
    reduce_lr_flag = Bool(default_value=False)

    # regularization params for embedding layer: l1 for mag/sparse, l2 for dir
    l2_lambda = Float(default_value=0.00001)
    l1_lambda = Float(default_value=0.0000001)

    # convolutions
    n_filter = Int()
    max_filter_len = Int()

    # dense layers
    dense_type = Unicode(default_value='dense')
    dense_config = Unicode(default_value='20,20')

    def __repr__(self):
        return json.dumps(self._trait_values, indent=2, sort_keys=True)

    def to_json(self):
        return self.__repr__()

    @staticmethod
    def load(filename):
        kw = file_util.read_json(filename)
        return ModelOptions(**kw)
