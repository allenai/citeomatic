import json
from citeomatic import file_util
from citeomatic.common import PAPER_EMBEDDING_MODEL, CITATION_RANKER_MODEL
from traitlets import Bool, HasTraits, Int, Unicode, Enum, Float


class ModelOptions(HasTraits):
    model_name = Enum(values=[PAPER_EMBEDDING_MODEL, CITATION_RANKER_MODEL], default_value=PAPER_EMBEDDING_MODEL)
    n_features = Int()
    n_authors = Int()

    dense_dim = Int(default_value=75)
    embedding_type = Enum(values=['sum', 'cnn', 'lstm'], default_value='sum')

    use_nn_negatives = Bool(default_value=False)
    use_dense = Bool(default_value=True)
    use_citations = Bool(default_value=True)
    use_authors = Bool(default_value=False)
    author_dim = Int(default_value=10)
    use_sparse = Bool(default_value=True)
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
    dropout_p = Float(default_value=0)

    # convolutions
    n_filter = Int()
    max_filter_len = Int()

    # dense layers
    dense_config = Unicode(default_value='20,20')

    num_ann_nbrs_to_fetch = Int(default_value=100)
    num_candidates_to_rank = Int(default_value=100) # No. of candidates to fetch from ANN at eval time
    extend_candidate_citations = Bool(default_value=True) # Whether to include citations of ANN
    # similar docs as possible candidates or not

    use_pretrained = Bool(default_value=False)
    num_oov_buckets = 100 # for hashing out of vocab terms
    dense_dim_pretrained = 300 # just a fact - don't change
    oov_term_prefix = '#OOV_'
    subset_vocab_to_training = False

    # minimum number of papers for an author to get an embedding.
    min_author_papers = 5

    def __repr__(self):
        return json.dumps(self._trait_values, indent=2, sort_keys=True)

    def to_json(self):
        return self.__repr__()

    @staticmethod
    def load(filename):
        kw = file_util.read_json(filename)
        return ModelOptions(**kw)
