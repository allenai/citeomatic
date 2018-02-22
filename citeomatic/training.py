import collections
import logging
import os
import resource

import h5py
import keras
import numpy as np
import tensorflow as tf
import tqdm
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.optimizers import TFOptimizer

from citeomatic import file_util
from citeomatic.candidate_selectors import CandidateSelector, ANNCandidateSelector
from citeomatic.common import DatasetPaths
from citeomatic.corpus import Corpus
from citeomatic.features import DataGenerator
from citeomatic.features import Featurizer
from citeomatic.models import layers
from citeomatic.models.options import ModelOptions
from citeomatic.neighbors import EmbeddingModel, ANN
from citeomatic.ranker import Ranker
from citeomatic.serialization import model_from_directory
from citeomatic.utils import import_from
from citeomatic.eval_metrics import precision_recall_f1_at_ks, average_results, f1

EVAL_KEYS = [1, 5, 10, 20, 50, 100, 1000]
EVAL_DATASET_KEYS = {'dblp': 10,
                     'pubmed': 20,
                     'oc': 20}

EVAL_DOC_MIN_CITATION = {
    'dblp': 10,
    'pubmed': 10,
    'oc': 1
}


class ValidationCallback(keras.callbacks.Callback):
    def __init__(self, corpus, candidate_selector, ranker, n_valid):
        super().__init__()
        self.candidate_selector = candidate_selector
        self.corpus = corpus
        self.ranker = ranker
        self.losses = []
        self.n_valid = n_valid

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        p_r_f1_mrr = eval_text_model(
            self.corpus,
            self.candidate_selector,
            self.ranker,
            papers_source='valid',
            n_eval=self.n_valid
        )
        for k, v in p_r_f1_mrr.items():
            logs[k] = v


class MemoryUsageCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        logging.info(
            '\nCurrent memory usage: %s',
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        )


class UpdateANN(keras.callbacks.Callback):
    def __init__(self,
                 corpus,
                 featurizer,
                 embedding_model,
                 training_data_generator: DataGenerator,
                 validation_data_generator: DataGenerator,
                 embed_at_epoch_end,
                 embed_at_train_begin):
        super().__init__()
        self.corpus = corpus
        self.featurizer = featurizer
        self.embedding_model = embedding_model
        self.training_data_generator = training_data_generator
        self.validation_data_generator = validation_data_generator
        self.embed_at_epch_end = embed_at_epoch_end
        self.embed_at_train_begin = embed_at_train_begin

    def _re_embed(self, load_pkl=False):
        embedder = EmbeddingModel(self.featurizer, self.embedding_model)
        if load_pkl and self.corpus.corpus_type == 'oc' and os.path.exists(
                        DatasetPaths.OC_ANN_FILE + ".pickle"):
            ann = ANN.load(DatasetPaths.OC_ANN_FILE)
        else:
            ann = ANN.build(embedder, self.corpus, ann_trees=10)
        candidate_selector = ANNCandidateSelector(
            corpus=self.corpus,
            ann=ann,
            paper_embedding_model=embedder,
            top_k=100,
            extend_candidate_citations=False
        )
        self.training_data_generator.ann = ann
        self.training_data_generator.candidate_selector = candidate_selector

        self.validation_data_generator.ann = self.training_data_generator.ann
        self.validation_data_generator.candidate_selector = self.training_data_generator.candidate_selector

    def on_train_begin(self, logs=None):
        if self.embed_at_train_begin:
            logging.info(
                'Beginning training. Embedding corpus.',
            )
            self._re_embed(load_pkl=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.embed_at_epch_end:
            logging.info(
                'Epoch %d begin. Retraining approximate nearest neighbors model.',
                epoch + 1
            )
            self._re_embed()


def train_text_model(
        corpus: Corpus,
        featurizer: Featurizer,
        model_options: ModelOptions,
        models_ann_dir=None,
        debug=False,
        tensorboard_dir=None,
):
    """
    Utility function for training citeomatic models.
    """

    # load pretrained embeddings
    if model_options.use_pretrained:
        dp = DatasetPaths()
        pretrained_embeddings_file = dp.embeddings_weights_for_corpus('shared')
        with h5py.File(pretrained_embeddings_file, 'r') as f:
            pretrained_embeddings = f['embedding'][...]
    else:
        pretrained_embeddings = None

    create_model = import_from(
        'citeomatic.models.%s' % model_options.model_name,
        'create_model'
    )
    models = create_model(model_options, pretrained_embeddings)
    model, embedding_model = models['citeomatic'], models['embedding']

    logging.info(model.summary())

    if model_options.train_for_test_set:
        paper_ids_for_training = corpus.train_ids + corpus.valid_ids
        candidates_for_training = corpus.train_ids + corpus.valid_ids + corpus.test_ids
    else:
        paper_ids_for_training = corpus.train_ids
        candidates_for_training = corpus.train_ids + corpus.valid_ids

    training_dg = DataGenerator(corpus=corpus,
                                featurizer=featurizer,
                                margin_multiplier=model_options.margin_multiplier,
                                use_variable_margin=model_options.use_variable_margin)
    training_generator = training_dg.triplet_generator(
        paper_ids=paper_ids_for_training,
        candidate_ids=candidates_for_training,
        batch_size=model_options.batch_size,
        neg_to_pos_ratio=model_options.neg_to_pos_ratio
    )

    validation_dg = DataGenerator(corpus=corpus,
                                  featurizer=featurizer,
                                  margin_multiplier=model_options.margin_multiplier,
                                  use_variable_margin=model_options.use_variable_margin)
    validation_generator = validation_dg.triplet_generator(
        paper_ids=corpus.valid_ids,
        candidate_ids=corpus.train_ids + corpus.valid_ids,
        batch_size=1024,
        neg_to_pos_ratio=model_options.neg_to_pos_ratio
    )

    if model_options.optimizer == 'tfopt':
        optimizer = TFOptimizer(
            tf.contrib.opt.LazyAdamOptimizer(learning_rate=model_options.lr)
        )
    else:
        optimizer = import_from(
            'keras.optimizers', model_options.optimizer
        )(lr=model_options.lr)

    model.compile(optimizer=optimizer, loss=layers.triplet_loss)

    # training calculation
    model_options.samples_per_epoch = int(np.minimum(
        model_options.samples_per_epoch, model_options.total_samples
    ))
    epochs = int(np.ceil(
        model_options.total_samples / model_options.samples_per_epoch
    ))
    steps_per_epoch = int(
        model_options.samples_per_epoch / model_options.batch_size
    )

    # callbacks
    callbacks_list = []
    if debug:
        callbacks_list.append(MemoryUsageCallback())
    if model_options.tb_dir is not None:
        callbacks_list.append(
            TensorBoard(
                log_dir=model_options.tb_dir, histogram_freq=1, write_graph=True
            )
        )
    if model_options.reduce_lr_flag:
        if model_options.optimizer != 'tfopt':
            callbacks_list.append(
                ReduceLROnPlateau(
                    verbose=1, patience=2, epsilon=0.01, min_lr=1e-6, factor=0.5
                )
            )

    if models_ann_dir is None:
        ann_featurizer = featurizer
        paper_embedding_model = embedding_model
        embed_at_epoch_end = True
        embed_at_train_begin = False
    else:
        ann_featurizer, ann_models = model_from_directory(models_ann_dir, on_cpu=True)
        paper_embedding_model = ann_models['embedding']
        paper_embedding_model._make_predict_function()
        embed_at_epoch_end = False
        embed_at_train_begin = True
    callbacks_list.append(
        UpdateANN(corpus, ann_featurizer, paper_embedding_model, training_dg, validation_dg,
                  embed_at_epoch_end, embed_at_train_begin)
    )

    if model_options.tb_dir is None:
        validation_data = validation_generator
    else:
        validation_data = next(validation_generator)

    # logic
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=10
    )

    return model, embedding_model


def end_to_end_training(model_options: ModelOptions, dataset_type, models_dir, models_ann_dir=None):
    # step 1: make the directory
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # step 2: load the corpus DB
    print("Loading corpus db...")
    dp = DatasetPaths()
    db_file = dp.get_db_path(dataset_type)
    json_file = dp.get_json_path(dataset_type)
    if not os.path.isfile(db_file):
        print("Have to build the database! This may take a while, but should only happen once.")
        Corpus.build(db_file, json_file)

    if dataset_type == 'oc':
        corpus = Corpus.load_pkl(dp.get_pkl_path(dataset_type))
    else:
        corpus = Corpus.load(db_file, model_options.train_frac)

    # step 3: load/make the featurizer (once per hyperopt run)
    print("Making feautrizer")
    featurizer_file_prefix = 'pretrained_' if model_options.use_pretrained else 'corpus_fit_'

    featurizer_file = os.path.join(models_dir, featurizer_file_prefix + dp.FEATURIZER_FILENAME)

    if os.path.isfile(featurizer_file):
        featurizer = file_util.read_pickle(featurizer_file)
    else:
        featurizer = Featurizer(
            max_features=model_options.max_features,
            max_title_len=model_options.max_title_len,
            max_abstract_len=model_options.max_abstract_len,
            use_pretrained=model_options.use_pretrained,
            min_author_papers=model_options.min_author_papers,
            min_venue_papers=model_options.min_venue_papers,
            min_keyphrase_papers=model_options.min_keyphrase_papers
        )
        featurizer.fit(corpus, is_featurizer_for_test=model_options.train_for_test_set)
        file_util.write_pickle(featurizer_file, featurizer)

    # update model options after featurization
    model_options.n_authors = featurizer.n_authors
    model_options.n_venues = featurizer.n_venues
    model_options.n_keyphrases = featurizer.n_keyphrases
    model_options.n_features = featurizer.n_features
    if model_options.use_pretrained:
        model_options.dense_dim = model_options.dense_dim_pretrained

    # step 4: train the model
    citeomatic_model, embedding_model = train_text_model(
        corpus,
        featurizer,
        model_options,
        models_ann_dir=models_ann_dir,
        debug=True,
        tensorboard_dir=None
    )

    # step 5: save the model
    citeomatic_model.save_weights(
        os.path.join(models_dir, dp.CITEOMATIC_WEIGHTS_FILENAME), overwrite=True
    )

    if embedding_model is not None:
        embedding_model.save_weights(
            os.path.join(models_dir, dp.EMBEDDING_WEIGHTS_FILENAME), overwrite=True
        )

    file_util.write_json(
        os.path.join(models_dir, dp.OPTIONS_FILENAME),
        model_options.to_json(),
    )

    return corpus, featurizer, model_options, citeomatic_model, embedding_model


def _gold_citations(doc_id: str, corpus: Corpus, min_citations: int, candidate_ids_pool: set):
    gold_citations_1 = set(corpus.get_citations(doc_id))

    if doc_id in gold_citations_1:
        gold_citations_1.remove(doc_id)

    citations_of_citations = []
    for c in gold_citations_1:
        citations_of_citations.extend(corpus.get_citations(c))

    gold_citations_2 = set(citations_of_citations).union(gold_citations_1)

    if doc_id in gold_citations_2:
        gold_citations_2.remove(doc_id)

    gold_citations_1.intersection_update(candidate_ids_pool)
    gold_citations_2.intersection_update(candidate_ids_pool)

    if len(gold_citations_1) < min_citations:
        return [], []

    return gold_citations_1, gold_citations_2


def eval_text_model(
        corpus: Corpus,
        candidate_selector: CandidateSelector,
        ranker: Ranker,
        papers_source='valid',
        min_citations=1,
        n_eval=None
):
    if papers_source == 'valid':
        paper_ids_for_eval = corpus.valid_ids
        candidate_ids_pool = corpus.train_ids + corpus.valid_ids
    elif papers_source == 'train':
        paper_ids_for_eval = corpus.train_ids
        candidate_ids_pool = corpus.train_ids
    else:
        logging.info("Using Test IDs")
        paper_ids_for_eval = corpus.test_ids
        candidate_ids_pool = corpus.train_ids + corpus.valid_ids + corpus.test_ids

    if corpus.corpus_type == 'dblp' or corpus.corpus_type == 'pubmed':
        # Hack to compare with previous work. BEWARE: do not do real experiments this way !!!
        candidate_ids_pool = corpus.train_ids

    candidate_ids_pool = set(candidate_ids_pool)

    logging.info(
        "Restricting Candidates pool and gold citations to {} docs".format(len(candidate_ids_pool)))

    if n_eval is not None:
        if n_eval < len(paper_ids_for_eval):
            logging.info("Selecting a random sample of {} papers for evaluation.".format(n_eval))
            np.random.seed(110886)
            paper_ids_for_eval = np.random.choice(paper_ids_for_eval, n_eval, replace=False)
        else:
            logging.info("Using all {} papers for evaluation.".format(len(paper_ids_for_eval)))

    # eval_doc_predictions = []
    results_1 = []
    results_2 = []
    for doc_id in tqdm.tqdm(paper_ids_for_eval):

        gold_citations_1, gold_citations_2 = _gold_citations(doc_id, corpus, min_citations,
                                                             candidate_ids_pool)
        if len(gold_citations_1) < EVAL_DOC_MIN_CITATION[corpus.corpus_type]:
            logging.debug("Skipping doc id : {}".format(doc_id))
            continue

        candidate_ids, confidence_scores = candidate_selector.fetch_candidates(doc_id,
                                                                               candidate_ids_pool)
        if len(candidate_ids) == 0:
            continue
        predictions, scores = ranker.rank(doc_id, candidate_ids, confidence_scores)

        logging.debug("Done! Found %s predictions." % len(predictions))
        # eval_doc_predictions.append(predictions)

        r_1 = precision_recall_f1_at_ks(
            gold_y=gold_citations_1,
            predictions=predictions,
            scores=None,
            k_list=EVAL_KEYS
        )

        r_2 = precision_recall_f1_at_ks(
            gold_y=gold_citations_2,
            predictions=predictions,
            scores=None,
            k_list=EVAL_KEYS
        )

        results_1.append(r_1)
        results_2.append(r_2)

    logging.info("Found {} papers in the test set after filtering docs with fewer than {} "
                 "citations".format(len(results_1), EVAL_DOC_MIN_CITATION[corpus.corpus_type]))

    averaged_results_1 = average_results(results_1)
    averaged_results_2 = average_results(results_2)

    return {
        'precision_1': {k: v for k, v in zip(EVAL_KEYS, averaged_results_1['precision'])},
        'recall_1': {k: v for k, v in zip(EVAL_KEYS, averaged_results_1['recall'])},
        'f1_1_per_paper': {k: v for k, v in zip(EVAL_KEYS, averaged_results_1['f1'])},
        'mrr_1': averaged_results_1['mrr'],
        'f1_1': {k: f1(p, r) for k, p, r in zip(EVAL_KEYS, averaged_results_1['precision'],
                                                averaged_results_1['recall'])}
    }
