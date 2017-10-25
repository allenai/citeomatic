import logging
import os
import resource
from collections import defaultdict

import keras
import numpy as np
import tensorflow as tf
import tqdm
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.engine import Model
from keras.optimizers import TFOptimizer

from citeomatic import file_util
from citeomatic.common import DatasetPaths, Document
from citeomatic.corpus import Corpus
from citeomatic.features import DataGenerator
from citeomatic.features import Featurizer
from citeomatic.models import layers
from citeomatic.models.options import ModelOptions
from citeomatic.neighbors import EmbeddingModel, ANN
from citeomatic.serialization import model_from_directory
from citeomatic.utils import import_from
import collections

EVAL_KEYS = [1, 5, 10, 20, 50, 100, 1000]
EVAL_DATASET_KEYS = {'dblp': 5,
                     'pubmed': 10,
                     'oc': 20}


def rank_metrics(y, preds, max_num_true_multiplier=9):
    """
    Compute various ranking metrics for citation prediction problem.
    """
    y = np.array(y)
    preds = np.array(preds)
    argsort_y = np.argsort(y)[::-1]
    y = y[argsort_y]
    preds = preds[argsort_y]
    sorted_inds = np.argsort(preds)[::-1]  # high to lower
    num_true = int(np.sum(y))
    K = int(np.minimum(len(y) / num_true, max_num_true_multiplier))
    precision_at_num_positive = []
    recall_at_num_positive = []
    # precision at i*num_true
    for i in range(1, K + 1):
        correct = np.sum(y[sorted_inds[:num_true * i]])
        precision_at_num_positive.append(correct / float(num_true * i))
        recall_at_num_positive.append(correct / float(num_true))
    # mean rank of the true indices
    rank_of_true = np.argsort(sorted_inds)[y == 1]
    mean_rank = np.mean(rank_of_true) + 1
    return mean_rank, precision_at_num_positive, recall_at_num_positive


def test_model(
        model, corpus, test_generator, n=None, print_results=False, debug=False
):
    """
    Utility function to test citation prediction for one query document at a time.
    """
    metrics = defaultdict(list)  # this function supports multiple outputs
    if n is None:
        n = len(corpus.test_ids)
    for i in range(n):
        data, labels = next(test_generator)
        predictions = model.predict(data)
        if len(predictions) == len(labels):
            predictions = [predictions]
        for i in range(len(predictions)):
            preds = predictions[i].flatten()
            metrics_loop = rank_metrics(labels, preds)
            metrics[i].append(metrics_loop)
            if debug:
                print(metrics_loop)
                print()

    rank = {}
    precision = {}
    recall = {}
    for i in range(len(metrics)):
        r, pr, rec = zip(*metrics[i])
        min_len = np.min([len(i) for i in pr])
        rank[i] = r
        precision[i] = [i[:min_len] for i in pr]
        recall[i] = [i[:min_len] for i in rec]
        if print_results:
            print("Mean rank:", np.round(np.mean(rank[i], 0), 2))
            print(
                "Average Precisions at multiples of num_true:",
                np.round(np.mean(precision[i], 0), 2)
            )
            print(
                "Average Recalls at multiples of num_true:",
                np.round(np.mean(recall[i], 0), 2)
            )

    return rank, precision, recall


class ValidationCallback(keras.callbacks.Callback):
    def __init__(self, model, corpus, validation_generator):
        super().__init__()
        self.model = model
        self.corpus = corpus
        self.validation_generator = validation_generator
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        test_model(
            self.model,
            self.corpus,
            test_generator=self.validation_generator,
            n=1000,
            print_results=True
        )
        logging.info()


class MemoryUsageCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        logging.info(
            '\nCurrent memory usage: %s',
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        )


class UpdateANN(keras.callbacks.Callback):
    def __init__(self, corpus, featurizer, embedding_model, data_generator, embed_every_epoch=True):
        self.corpus = corpus
        self.featurizer = featurizer
        self.embedding_model = embedding_model
        self.data_generator = data_generator
        self.embed_every_epoch = embed_every_epoch
        if not self.embed_every_epoch:
            self.on_epoch_end(epoch=-2)

    def on_epoch_end(self, epoch, logs=None):
        if self.embed_every_epoch:
            logging.info(
                'Epoch %d ended. Retraining approximate nearest neighbors model.',
                epoch + 1
            )
            embedder = EmbeddingModel(self.featurizer, self.embedding_model)
            ann = ANN.build(embedder, self.corpus, ann_trees=10)
            self.data_generator.ann = ann


def train_text_model(
        corpus,
        featurizer,
        model_options,
        models_ann_dir=None,
        debug=False,
        tensorboard_dir=None
):
    """
    Utility function for training citeomatic models.
    """

    create_model = import_from(
        'citeomatic.models.%s' % model_options.model_name,
        'create_model'
    )
    models = create_model(model_options)
    model, embedding_model = models['citeomatic'], models['embedding']

    logging.info(model.summary())

    training_dg = DataGenerator(corpus, featurizer)
    training_generator = training_dg.triplet_generator(
        paper_ids=corpus.train_ids,
        candidate_ids=corpus.train_ids,
        batch_size=model_options.batch_size,
        neg_to_pos_ratio=model_options.neg_to_pos_ratio,
        margin_multiplier=model_options.margin_multiplier
    )

    validation_dg = DataGenerator(corpus, featurizer)
    validation_generator = validation_dg.triplet_generator(
        paper_ids=corpus.valid_ids,
        candidate_ids=corpus.train_ids + corpus.valid_ids,
        batch_size=1024,
        neg_to_pos_ratio=model_options.neg_to_pos_ratio,
        margin_multiplier=model_options.margin_multiplier
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
    if tensorboard_dir is not None:
        callbacks_list.append(
            TensorBoard(
                log_dir=tensorboard_dir, histogram_freq=1, write_graph=True
            )
        )
    if model_options.reduce_lr_flag:
        if model_options.optimizer != 'tfopt':
            callbacks_list.append(
                ReduceLROnPlateau(
                    verbose=1, patience=2, epsilon=0.01, min_lr=1e-6, factor=0.5
                )
            )

    if model_options.use_nn_negatives:
        if models_ann_dir is None:
            ann_featurizer = featurizer
            ann_embedding_model = embedding_model
            embed_every_epoch = True
        else:
            ann_featurizer, ann_models = model_from_directory(models_ann_dir)
            ann_embedding_model = ann_models['embedding']
            embed_every_epoch = False
        callbacks_list.append(
            UpdateANN(corpus, ann_featurizer, ann_embedding_model, training_dg, embed_every_epoch)
        )

    # logic
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=10,
    )

    return model, embedding_model


def end_to_end_training(model_options, dataset_type, models_dir, models_ann_dir=None):
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
    corpus = Corpus.load(db_file, model_options.train_frac)

    # step 3: load/make the featurizer (once per hyperopt run)
    print("Making feautrizer")
    featurizer_file = os.path.join(models_dir, dp.FEATURIZER_FILENAME)
    if not os.path.isfile(featurizer_file):
        featurizer = Featurizer(
            max_features=model_options.max_features,
            allow_duplicates=False
        )
        featurizer.fit(corpus)
        file_util.write_pickle(featurizer_file, featurizer)
    else:
        featurizer = file_util.read_pickle(featurizer_file)

    # model_options = ModelOptions(**hyperopt.pyll.stochastic.sample(space))
    model_options.n_authors = featurizer.n_authors
    model_options.n_features = featurizer.n_features

    # step 4: train the model
    citeomatic_model, embedding_model = train_text_model(
        corpus,
        featurizer,
        model_options,
        models_ann_dir=models_ann_dir,
        debug=False,
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


def fetch_candidates(
        corpus: Corpus,
        candidate_ids_pool: set,
        doc: Document,
        ann_embedding_model: EmbeddingModel,
        ann: ANN,
        top_n: int,
        extend_candidate_citations: bool
):
    doc_embedding = ann_embedding_model.embed(doc)
    # 1. Fetch candidates from ANN index
    nn_candidates = ann.get_nns_by_vector(doc_embedding, top_n + 1)
    # 2. Remove the current document from candidate list
    if doc.id in nn_candidates:
        nn_candidates.remove(doc.id)
    candidate_ids = nn_candidates[:top_n]

    # 3. Check if we need to include citations of candidates found so far.
    if extend_candidate_citations:
        extended_candidate_ids = []
        for candidate_id in candidate_ids:
            extended_candidate_ids.extend(corpus[candidate_id].out_citations)
        candidate_ids = candidate_ids + extended_candidate_ids
    logging.debug("Number of candidates found: {}".format(len(candidate_ids)))
    candidate_ids = set(candidate_ids).intersection(candidate_ids_pool)
    candidates = [corpus[candidate_id] for candidate_id in candidate_ids]

    return candidates


def eval_metrics(predictions: list, corpus: Corpus, doc_id: str, min_citations):

    def _mrr(p):
        try:
            idx = p.index(True)
            return 1. / (idx + 1)
        except ValueError:
            return 0.0

    gold_citations = set(corpus[doc_id].out_citations)
    if doc_id in gold_citations:
        gold_citations.remove(doc_id)
    citations_of_citations = []
    for c in gold_citations:
        citations_of_citations.extend(corpus[c].out_citations)
    gold_citations_2 = set(citations_of_citations).union(gold_citations)
    if doc_id in gold_citations_2:
        gold_citations_2.remove(doc_id)

    if len(gold_citations) < min_citations:
        return None

    paper_results = []

    for prediction in predictions:
        paper_results.append(
            {
                'correct_1': prediction['document'].id in gold_citations,
                'correct_2': prediction['document'].id in gold_citations_2,
                'score': prediction['score'],
                'num_gold_1': len(gold_citations),
                'num_gold_2': len(gold_citations_2)
            }
        )

    p1 = [p['correct_1'] for p in paper_results]
    mrr1 = _mrr(p1)
    p2 = [p['correct_2'] for p in paper_results]
    mrr2 = _mrr(p2)

    logging.debug('Level 1 P@10 = %f ' % np.mean(p1[:10]))
    logging.debug('Level 2 P@10 = %f ' % np.mean(p2[:10]))
    logging.debug('Level 1 MRR = %f' % mrr1)
    logging.debug('Level 2 MRR = %f' % mrr2)
    candidate_set_recall = np.sum(p1) / len(gold_citations)
    logging.debug('Candidate set recall = {} '.format(candidate_set_recall))

    return {
        'id': doc_id,
        'predictions': paper_results,
        'num_gold_1': len(gold_citations),
        'num_gold_2': len(gold_citations_2),
        'mrr_1': mrr1,
        'mrr_2': mrr2,
        'p1': p1,
        'p2': p2
    }


def eval_text_model(
        corpus: Corpus,
        featurizer: Featurizer,
        model_options: ModelOptions,
        citeomatic_model: Model,
        embedding_model_for_ann: Model = None,
        featurizer_for_ann: Featurizer = None,
        papers_source='valid',
        ann: ANN = None,
        min_citations=1,
        n_eval = None
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

    if n_eval is not None:
        if n_eval < len(paper_ids_for_eval):
            logging.info("Selecting a random sample of {} papers for evaluation.".format(n_eval))
            paper_ids_for_eval = np.random.choice(paper_ids_for_eval, n_eval, replace=False)
        else:
            logging.info("Using all {} papers for evaluation.".format(len(paper_ids_for_eval)))

    candidate_ids_pool = set(candidate_ids_pool)

    if featurizer_for_ann is None:
        featurizer_for_ann = featurizer
    ann_embedding_model = EmbeddingModel(featurizer_for_ann, embedding_model_for_ann)

    if ann is None:
        ann = ANN.build(ann_embedding_model, corpus)

    eval_doc_predictions = []
    results = []
    for doc_id in tqdm.tqdm(paper_ids_for_eval):
        query = corpus[doc_id]
        candidates = fetch_candidates(
            corpus=corpus,
            candidate_ids_pool=candidate_ids_pool,
            doc=query,
            ann_embedding_model=ann_embedding_model,
            ann=ann,
            top_n=model_options.num_ann_nbrs_to_fetch,
            extend_candidate_citations=model_options.extend_candidate_citations)

        logging.debug('Featurizing... %d documents ' % len(candidates))
        features = featurizer.transform_query_and_results(query, candidates)
        logging.debug('Predicting...')
        scores = citeomatic_model.predict(features, batch_size=1024).flatten()
        best_matches = np.argsort(scores)[::-1]

        predictions = []
        query_doc_citations = set(query.out_citations)
        for i, match_idx in enumerate(best_matches[:model_options.num_candidates_to_rank]):
            predictions.append(
                {
                    'score':float(scores[match_idx]),
                    'document': candidates[match_idx],
                    'position': i,
                    'is_cited': candidates[match_idx].id in query_doc_citations

                }
            )
        logging.debug("Done! Found %s predictions." % len(predictions))
        eval_doc_predictions.append(predictions)
        r = eval_metrics(predictions, corpus, doc_id, min_citations)
        if r is not None:
            results.append(r)

    precision_at_1 = collections.defaultdict(list)
    recall_at_1 = collections.defaultdict(list)

    precision_at_2 = collections.defaultdict(list)
    recall_at_2 = collections.defaultdict(list)

    for r in results:
        p1 = r['p1']
        p2 = r['p2']
        for k in EVAL_KEYS:
            patk = np.mean(p1[:k])
            ratk = np.sum(p1[:k]) / r['num_gold_1']
            precision_at_1[k].append(patk)
            recall_at_1[k].append(ratk)

            patk = np.mean(p2[:k])
            ratk = np.sum(p2[:k]) / r['num_gold_2']
            precision_at_2[k].append(patk)
            recall_at_2[k].append(ratk)

    return {
        'precision_1': {k: np.mean(v)
                        for (k, v) in precision_at_1.items()},
        'recall_1': {k: np.mean(v)
                     for (k, v) in recall_at_1.items()},
        'precision_2': {k: np.mean(v)
                        for (k, v) in precision_at_2.items()},
        'recall_2': {k: np.mean(v)
                     for (k, v) in recall_at_2.items()},
        'mrr_1': np.mean([r['mrr_1'] for r in results]),
        'mrr_2': np.mean([r['mrr_2'] for r in results]),
    }
