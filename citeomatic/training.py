import logging
import resource
import os
from collections import defaultdict

import tensorflow as tf
import keras
import numpy as np
from citeomatic import file_util
from citeomatic.neighbors import EmbeddingModel, make_ann
from citeomatic.serialization import model_from_directory
from citeomatic.features import DataGenerator
from citeomatic.utils import import_from
from citeomatic.models import layers
from citeomatic.corpus import Corpus
from citeomatic.common import DatasetPaths
from citeomatic.features import Featurizer
from citeomatic.models.options import ModelOptions
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
        training_generator,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks_list,
        epochs=epochs,
        max_q_size=2,
        pickle_safe=False,
        validation_data=validation_generator,
        validation_steps=10
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

    embedding_model.save_weights(
        os.path.join(models_dir, dp.EMBEDDING_WEIGHTS_FILENAME), overwrite=True
    )

    file_util.write_json(
        os.path.join(models_dir, dp.OPTIONS_FILENAME),
        model_options.to_json(),
    )

    return corpus, featurizer, model_options, citeomatic_model, embedding_model

def eval_text_model(
    corpus,
    featurizer,
    model_options,
    embedding_model_for_ann=None,
    debug=False
):
    pass

