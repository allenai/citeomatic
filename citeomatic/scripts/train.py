import datetime
import logging
import os
from pprint import pprint

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
from traitlets import Int, Unicode, Enum

from citeomatic import file_util
from citeomatic.common import PAPER_EMBEDDING_MODEL, CITATION_RANKER_MODEL, DatasetPaths
from citeomatic.config import App
from citeomatic.candidate_selectors import ANNCandidateSelector, BM25CandidateSelector
from citeomatic.models.options import ModelOptions
from citeomatic.serialization import model_from_directory
from citeomatic.neighbors import EmbeddingModel, ANN
from citeomatic.ranker import Ranker, NoneRanker
from citeomatic.training import end_to_end_training
from citeomatic.training import eval_text_model, EVAL_DATASET_KEYS
import pickle
import tensorflow as tf
import keras.backend as K


class TrainCiteomatic(App, ModelOptions):
    dataset_type = Enum(('dblp', 'pubmed', 'oc'), default_value='dblp')

    # Whether to train based on given options or to run a hyperopt
    mode = Enum(('train', 'hyperopt'))

    # Training parameters
    hyperopts_results_pkl = Unicode(default_value=None, allow_none=True)
    options_json = Unicode(default_value=None, allow_none=True)

    # hyperopt parameters
    max_evals_initial = Int(default_value=25)
    max_evals_secondary = Int(default_value=5)
    total_samples_initial = Int(default_value=5000000)
    total_samples_secondary = Int(default_value=50000000)
    n_eval = Int(default_value=500, allow_none=True)
    models_ann_dir = Unicode(default_value=None, allow_none=True)
    models_dir_base = Unicode(
        default_value='data/models/'
    )
    run_identifier = Unicode(default_value=None, allow_none=True)
    version = Unicode(default_value='v0')

    # to be filled in later
    models_dir = Unicode(default_value=None, allow_none=True)
    ann = None

    def main(self, args):

        if self.mode == 'hyperopt':
            self.run_hyperopt()
        elif self.mode == 'train':
            self.run_train()
        else:
            assert False

    def run_train(self):
        eval_params = {}
        if self.hyperopts_results_pkl is not None:
            params = pickle.load(open(self.hyperopts_results_pkl, "rb"))
            for k, v in params[1][0]['result']['params'].items():
                eval_params[k] = v
        if self.options_json is not None:
            obj = file_util.read_json(self.options_json)
            eval_params.update(obj)

        if self.model_name == PAPER_EMBEDDING_MODEL:
            self.models_ann_dir = None
            self.models_dir = os.path.join(self.models_dir_base, PAPER_EMBEDDING_MODEL)

        self.train_and_evaluate(eval_params)

    @staticmethod
    def _hyperopt_space(model_name, total_samples):

        use_pretrained = hp.choice('use_pretrained', [True, False])

        common_param_space = {
            'total_samples':
                total_samples,
            'lr':
                hp.choice('lr', [0.1, 0.01, 0.001, 0.0001, 0.00001]),
            'l1_lambda':
                hp.choice('l1_lambda', np.append(np.logspace(-7, -2, 6), 0)),
            'dropout_p':
                hp.quniform('dropout_p', 0.0, 0.75, 0.05),
            'margin_multiplier':
                hp.choice('margin_multiplier', [0.5, 0.75, 1.0, 1.25, 1.5])
        }

        pre_trained_params = {
            True: {
                'use_pretrained': True,
                'l2_lambda': 0,
                'dense_dim': 300,
                'enable_fine_tune': hp.choice('enable_fine_tune', [True, False])
            },
            False: {
                'use_pretrained': False,
                'l2_lambda': hp.choice('l2_lambda', np.append(np.logspace(-7, -2, 6), 0)),
                'dense_dim': scope.int(hp.quniform('dense_dim', 25, 325, 25)),
                'enable_fine_tune': True # doesn't matter what goes here
            }
        }

        # the search space
        # note that the scope.int code is a hack to get integers out of the sampler
        if model_name == CITATION_RANKER_MODEL:
            ranker_model_params = {
                'embedding_type':
                    hp.choice('embedding_type', ['sum']),
                'metadata_dim':
                    scope.int(hp.quniform('metadata_dim', 5, 55, 5)),
            }

            space = scope.switch(
                scope.int(use_pretrained),
                {**pre_trained_params[False],
                 **common_param_space,
                 **ranker_model_params,
                 },
                {**pre_trained_params[True],
                 **common_param_space,
                 **ranker_model_params
                 }
            )
        elif model_name == PAPER_EMBEDDING_MODEL:
            space = scope.switch(
                scope.int(use_pretrained),
                {**pre_trained_params[False],
                 **common_param_space
                 },
                {**pre_trained_params[True],
                 **common_param_space
                 }
            )
        else:
            # Should not come here. Adding this to make pycharm happy.
            assert False

        return space

    def run_hyperopt(self):
        # run identifier
        if self.run_identifier is None:
            self.run_identifier = '_'.join(
                [
                    'citeomatic_hyperopt',
                    self.model_name,
                    self.dataset_type,
                    datetime.datetime.now().strftime("%Y-%m-%d"),
                    self.version
                ]
            )

        self.models_dir = os.path.join(self.models_dir_base, self.run_identifier)
        if self.model_name == PAPER_EMBEDDING_MODEL:
            self.models_ann_dir = None

        space = self._hyperopt_space(self.model_name, self.total_samples_initial)

        # stage 1: run hyperopt for max_evals_initial
        # using a max of total_samples_initial samples
        trials = Trials()
        _ = fmin(
            fn=self.train_and_evaluate,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals_initial,
            trials=trials
        )

        sorted_results_stage_1 = sorted(
            trials.trials, key=lambda x: x['result']['loss']
        )

        # stage 2: run the top max_evals_seconadry from stage 1
        # using a max of total_samples_secondary samples
        results_stage_2 = []
        for result in sorted_results_stage_1[:self.max_evals_secondary]:
            params = result['result']['params']
            params['total_samples'] = self.total_samples_secondary
            out = self.train_and_evaluate(params)
            results_stage_2.append({'params': params, 'result': out})

        sorted_results_stage_2 = sorted(
            results_stage_2, key=lambda x: x['result']['loss']
        )

        # save and display results
        results_save_file = 'hyperopt_results.pickle'
        file_util.write_pickle(
            os.path.join(self.models_dir, results_save_file),
            (sorted_results_stage_1, sorted_results_stage_2)
        )
        pprint(sorted_results_stage_2[0])

    def train_and_evaluate(self, eval_params):
        # Needed especially for hyperopt runs
        K.clear_session()

        model_kw = {name: getattr(self, name) for name in ModelOptions.class_traits().keys()}
        model_kw.update(eval_params)
        model_options = ModelOptions(**model_kw)

        if model_options.use_metadata:
            model_options.use_keyphrases = True
            model_options.use_authors = True
            model_options.use_venue = True

        print("====== OPTIONS =====")
        print(model_options)
        print("======")

        if model_options.train_for_test_set:
            logging.info("\n\n============== TRAINING FOR TEST SET =============\n\n")

        training_outputs = end_to_end_training(
            model_options,
            self.dataset_type,
            self.models_dir,
            self.models_ann_dir
        )
        corpus, featurizer, model_options, citeomatic_model, embedding_model = training_outputs

        if self.candidate_selector_type == 'ann':
            # if no ann_dir is provided, then we use the model that was just trained
            # and have to rebuild the ANN
            if self.models_ann_dir is None:
                print(
                    'Using embedding model that was just trained for eval. Building...')
                paper_embedding_model = EmbeddingModel(
                    featurizer,
                    embedding_model
                )
                self.ann = ANN.build(paper_embedding_model, corpus)
            # if a dir is provided, then go ahead and load it
            else:
                featurizer_for_ann, ann_models = model_from_directory(
                    self.models_ann_dir, on_cpu=True
                )
                paper_embedding_model = EmbeddingModel(
                    featurizer_for_ann,
                    ann_models['embedding']
                )
                # the ANN itself needs to be only built once
                if self.ann is None:
                    if corpus.corpus_type == 'oc' and os.path.exists(DatasetPaths.OC_ANN_FILE + ".pickle"):
                        self.ann = ANN.load(DatasetPaths.OC_ANN_FILE)
                    else:
                        self.ann = ANN.build(paper_embedding_model, corpus)

            candidate_selector = ANNCandidateSelector(
                corpus=corpus,
                ann=self.ann,
                paper_embedding_model=paper_embedding_model,
                top_k=model_options.num_ann_nbrs_to_fetch,
                extend_candidate_citations=model_options.extend_candidate_citations
            )
        elif self.candidate_selector_type == 'bm25':
            dp = DatasetPaths()
            candidate_selector = BM25CandidateSelector(
                corpus=corpus,
                index_path=dp.get_bm25_index_path(self.dataset_type),
                top_k=model_options.num_ann_nbrs_to_fetch,
                extend_candidate_citations=model_options.extend_candidate_citations
            )
        else:
            # Should not come here. Adding this to make pycharm happy.
            assert False

        if self.citation_ranker_type == 'neural':
            ranker = Ranker(
                corpus=corpus,
                featurizer=featurizer,
                citation_ranker=citeomatic_model,
                num_candidates_to_rank=model_options.num_candidates_to_rank
            )
        elif self.citation_ranker_type == 'none':
            ranker = NoneRanker()
        else:
            # Should not come here. Adding this to make pycharm happy.
            assert False

        if self.mode != 'hyperopt' or model_options.total_samples == self.total_samples_secondary:
            results_training = eval_text_model(
                corpus,
                candidate_selector,
                ranker,
                papers_source='train',
                n_eval=self.n_eval
            )
        else:
            results_training = {}

        results_validation = eval_text_model(
            corpus,
            candidate_selector,
            ranker,
            papers_source='valid',
            n_eval=self.n_eval
        )

        logging.info("===== Validation Results ===== ")
        logging.info("Validation Precision\n\n{}".format(results_validation['precision_1']))
        logging.info("Validation Recall\n\n{}".format(results_validation['recall_1']))

        p = results_validation['precision_1'][EVAL_DATASET_KEYS[self.dataset_type]]
        r = results_validation['recall_1'][EVAL_DATASET_KEYS[self.dataset_type]]
        f1 = results_validation['f1_1'][EVAL_DATASET_KEYS[self.dataset_type]]

        if self.model_name == PAPER_EMBEDDING_MODEL:
            # optimizing for recall
            l = -r
        else:
            # optimizing for F1
            l = -f1

        out = {
            'loss': l,  # have to negate since we're minimizing
            'losses_training': results_training,
            'losses_validation': results_validation,
            'status': STATUS_FAIL if np.isnan(f1) else STATUS_OK,
            'params': eval_params
        }

        return out


TrainCiteomatic.run(__name__)
