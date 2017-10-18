import logging
import resource
from collections import defaultdict

import tensorflow as tf
import keras
import numpy as np
from citeomatic.neighbors import EmbeddingModel, make_ann
from citeomatic.features import DataGenerator
from citeomatic.utils import import_from
from citeomatic.models import layers
from keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback
from keras.optimizers import TFOptimizer


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
    def __init__(self, corpus, featurizer, embedding_model, data_generator):
        self.corpus = corpus
        self.featurizer = featurizer
        self.embedding_model = embedding_model
        self.data_generator = data_generator

    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            'Epoch %d ended. Retraining approximate nearest neighbors model.',
            epoch + 1
        )
        embedding_model_wrapped = EmbeddingModel(
            self.featurizer, self.embedding_model
        )
        ann = make_ann(
            embedding_model_wrapped,
            self.corpus,
            ann_trees=10,
            build_ann_index=True
        )
        self.data_generator.ann = ann


def train_text_model(
    corpus,
    featurizer,
    model_options,
    embedding_model_for_ann=None,
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
        batch_size=10000,
        neg_to_pos_ratio=model_options.neg_to_pos_ratio,
        margin_multiplier=model_options.margin_multiplier
    )

    optimizer = TFOptimizer(
        tf.contrib.opt.LazyAdamOptimizer(learning_rate=model_options.lr)
    )
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
        assert embedding_model_for_ann is not None
        callbacks_list.append(
            UpdateANN(corpus, featurizer, embedding_model_for_ann, training_generator)
        )

    # logic
    model.fit_generator(
        training_generator,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks_list,
        epochs=epochs,
        max_q_size=2,
        pickle_safe=False,
        validation_data=validation_generator,
        validation_steps=1
    )

    return model, embedding_model

