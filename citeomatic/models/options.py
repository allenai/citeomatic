import json
from citeomatic import file_util
from citeomatic.common import PAPER_EMBEDDING_MODEL, CITATION_RANKER_MODEL
from traitlets import Bool, HasTraits, Int, Unicode, Enum, Float


class ModelOptions(HasTraits):

    # The type of candidate selector to use. Okapi BM25 (https://en.wikipedia.org/wiki/Okapi_BM25)
    # ranking model or an Approximate Nearest Neighbor index built on embeddings of documents
    # obtained from the paper_embedding model
    candidate_selector_type = Enum(('ann', 'bm25'), default_value='ann')

    # Whether to use the citation_ranker model to re-rank selected candidates or not
    citation_ranker_type = Enum(('neural', 'none'), default_value='neural')

    # Model name to train: citation_ranker or paper_embedder
    model_name = Enum(values=[PAPER_EMBEDDING_MODEL, CITATION_RANKER_MODEL], default_value=PAPER_EMBEDDING_MODEL)

    # No. of features (words) to retain from the corpus for training
    n_features = Int()

    # No. of authors to retain from the corpus for training
    n_authors = Int()

    # No. of venues to retain from the corpus for training
    n_venues = Int()

    # No. of key phrases to retain from the corpus for training
    n_keyphrases = Int()

    # Dimension of word embedding
    dense_dim = Int(default_value=75)

    # Dimension of embeddings for author, venue or keyphrase
    metadata_dim = Int(default_value=10)

    # Embedding type to use for text fields
    embedding_type = Enum(values=['sum', 'cnn', 'cnn2', 'lstm'], default_value='sum')

    # Architecture changing options
    use_dense = Bool(default_value=True)
    use_citations = Bool(default_value=True)
    use_sparse = Bool(default_value=True)
    use_src_tgt_embeddings = Bool(default_value=False)
    use_metadata = Bool(default_value=True)
    use_authors = Bool(default_value=False)
    use_venue = Bool(default_value=False)
    use_keyphrases = Bool(default_value=False)

    # training and feature params
    optimizer = Unicode(default_value='tfopt')
    lr = Float(default_value=0.0001)
    use_nn_negatives = Bool(default_value=True)
    margin_multiplier = Float(default_value=1)
    use_variable_margin = Bool(default_value=True)
    train_frac = Float(default_value=0.8) # the rest will be divided 50/50 val/test
    max_features = Int(default_value=200000)
    max_title_len = Int(default_value=50)
    max_abstract_len = Int(default_value=500)
    neg_to_pos_ratio = Int(default_value=6) # ideally divisible by 2 and 3
    batch_size = Int(default_value=512)
    samples_per_epoch = Int(default_value=1000000)
    total_samples = Int(default_value=5000000)
    reduce_lr_flag = Bool(default_value=False)

    # regularization params for embedding layer: l1 for mag/sparse, l2 for dir
    l2_lambda = Float(default_value=0.00001)
    l1_lambda = Float(default_value=0.0000001)
    dropout_p = Float(default_value=0)
    use_magdir = Bool(default_value=True)

    # params for TextEmbeddingConv
    kernel_width = Int(default_value=5)
    stride = Int(default_value=2)

    # params for TextEmbeddingConv2
    filters = Int(default_value=100) # default in the paper
    max_kernel_size = Int(default_value=5) # we use 2, 3, 4, 5. paper uses 3, 4, 5

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

    # minimum number of papers for authors/venues/keyphrases to get an embedding.
    min_author_papers = Int(default_value=1)
    min_venue_papers = Int(default_value=1)
    min_keyphrase_papers = Int(default_value=5)

    use_selector_confidence = Bool(default_value=True)

    # Tensoboard logging directory
    tb_dir = Unicode(default_value=None, allow_none=True)

    # Option to fine-tune pre-trained embeddings
    enable_fine_tune = Bool(default_value=True)

    # Use both training and validation data for final training of model
    train_for_test_set = Bool(default_value=False)

    def __repr__(self):
        return json.dumps(self._trait_values, indent=2, sort_keys=True)

    def to_json(self):
        model_kw = {name: getattr(self, name) for name in ModelOptions.class_traits().keys()}
        return json.dumps(model_kw)

    @staticmethod
    def load(filename):
        kw = file_util.read_json(filename)
        return ModelOptions(**kw)
