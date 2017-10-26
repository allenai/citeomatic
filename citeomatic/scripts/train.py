import datetime
import logging
import os
from pprint import pprint

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
from traitlets import Int, Unicode, Enum

from citeomatic import file_util
from citeomatic.common import PAPER_EMBEDDING_MODEL, CITATION_RANKER_MODEL
from citeomatic.config import App
from citeomatic.models.options import ModelOptions
from citeomatic.serialization import model_from_directory
from citeomatic.training import end_to_end_training
from citeomatic.training import eval_text_model, EVAL_DATASET_KEYS
import pickle


class TrainCiteomatic(App, ModelOptions):

    dataset_type = Enum(('dblp', 'pubmed', 'oc'), default_value='dblp')

    # Whether to train based on given options or to run a hyperopt
    mode = Enum(('train', 'hyperopt'))

    # Training parameters
    hyperopts_results_pkl = Unicode(default_value=None, allow_none=True)

    # hyperopt parameters
    max_evals_initial = Int(default_value=75)
    max_evals_secondary = Int(default_value=10)
    total_samples_initial = Int(default_value=5000000)
    total_samples_secondary = Int(default_value=20000000)
    n_eval = Int(default_value=1000, allow_none=True)
    models_ann_dir = Unicode(default_value=None, allow_none=True)
    models_dir_base = Unicode(
        default_value='/net/nfs.corp/s2-research/citeomatic/'
    )
    run_identifier = Unicode(default_value=None, allow_none=True)
    version = Unicode(default_value='v0')

    # to be filled in later
    models_dir = None
    embedding_model_for_ann = None
    featurizer_for_ann = None

    def main(self, args):

        if self.mode == 'hyperopt':
            self.run_hyperopt()
        elif self.mode == 'train':
            self.run_train()
        else:
            assert False

    def run_train(self):
        model_kw = {name: getattr(self, name) for name in ModelOptions.class_traits().keys()}
        if self.hyperopts_results_pkl is not None:
            params = pickle.load(open(self.hyperopts_results_pkl, "rb"))
            for k, v in params[1][0]['result']['params'].items():
                model_kw[k] = v
        if self.model_name == PAPER_EMBEDDING_MODEL:
            self.models_ann_dir = None
            self.models_dir = os.path.join(self.models_dir_base, PAPER_EMBEDDING_MODEL)

        self.eval_fn(model_kw)

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
        else:
            self.models_ann_dir = os.path.join(self.models_dir_base, self.run_identifier)

        # the search space
        # note that the scope.int code is a hack to get integers out of the sampler
        if self.model_name == CITATION_RANKER_MODEL:
            space = {
                'total_samples':
                    self.total_samples_initial,
                'dense_dim':
                    scope.int(hp.quniform('dense_dim', 25, 325, 25)),
                'use_authors':
                    hp.choice('use_authors', [True, False]),
                'use_citations':
                    hp.choice('use_citations', [True, False]),
                'use_sparse':
                    hp.choice('sparse_option', [True, False]),
                'use_src_tgt_embeddings':
                    hp.choice('use_src_tgt_embeddings', [True, False]),
                'lr':
                    hp.choice('lr', [0.1, 0.01, 0.001, 0.0001, 0.00001]),
                'l2_lambda':
                    hp.choice('l2_lambda', np.append(np.logspace(-7, -2, 6), 0)),
                'l1_lambda':
                    hp.choice('l1_lambda', np.append(np.logspace(-7, -2, 6), 0)),
                'margin_multiplier':
                    hp.choice('margin_multiplier', [0.5, 0.75, 1.0, 1.25, 1.5])
            }
        elif self.model_name == PAPER_EMBEDDING_MODEL:
            space = {
                'total_samples':
                    self.total_samples_initial,
                'lr':
                    hp.choice('lr', [0.1, 0.01, 0.001, 0.0001, 0.00001]),
                'l2_lambda':
                    hp.choice('l2_lambda', np.append(np.logspace(-7, -2, 6), 0)),
                'l1_lambda':
                    hp.choice('l1_lambda', np.append(np.logspace(-7, -2, 6), 0)),
                'margin_multiplier':
                    hp.choice('margin_multiplier', [0.5, 0.75, 1.0, 1.25, 1.5])
            }

        # stage 1: run hyperopt for max_evals_initial
        # using a max of total_samples_initial samples
        trials = Trials()
        _ = fmin(
            fn=self.eval_fn,
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
            out = self.eval_fn(params)
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

    def eval_fn(self, params):
        model_options = ModelOptions(**params)
        print("====== OPTIONS =====")
        print(model_options)
        print("======")

        # TODO: if self.models_ann_dir is not none, we still make the ANN every time.
        # should be able to just do it once. Maybe need to restructure?
        # This will be a bigger deal once we are using open corpus
        training_outputs = end_to_end_training(
            model_options,
            self.dataset_type,
            self.models_dir,
            self.models_ann_dir
        )
        corpus, featurizer, model_options, model, embedding_model = training_outputs
        if self.models_ann_dir is None:
            featurizer_for_ann = featurizer
            embedding_model_for_ann = embedding_model
        elif self.embedding_model_for_ann is None:
            featurizer_for_ann, ann_models = model_from_directory(self.models_ann_dir)
            embedding_model_for_ann = ann_models['embedding']
            self.featurizer_for_ann = featurizer_for_ann
            self.embedding_model_for_ann = embedding_model_for_ann
        else:
            embedding_model_for_ann = self.embedding_model_for_ann
            featurizer_for_ann = self.featurizer_for_ann

        results_training = eval_text_model(
            corpus=corpus,
            featurizer=featurizer,
            model_options=model_options,
            citeomatic_model=model,
            embedding_model_for_ann=embedding_model_for_ann,
            featurizer_for_ann=featurizer_for_ann,
            papers_source='train',
            n_eval=self.n_eval
        )
        results_validation = eval_text_model(
            corpus=corpus,
            featurizer=featurizer,
            model_options=model_options,
            citeomatic_model=model,
            embedding_model_for_ann=embedding_model_for_ann,
            featurizer_for_ann=featurizer_for_ann,
            papers_source='valid',
            n_eval=self.n_eval
        )

        logging.info("===== Validation Results ===== ")
        logging.info(results_validation['precision_1'])
        logging.info(results_validation['recall_1'])

        p = results_validation['precision_1'][EVAL_DATASET_KEYS[self.dataset_type]]
        r = results_validation['recall_1'][EVAL_DATASET_KEYS[self.dataset_type]]
        f1 = 2 * p * r / (p + r)

        if self.model_name == PAPER_EMBEDDING_MODEL:
            l = -1 * results_validation['recall_1'][100]
        else:
            l = -f1

        out = {
            'loss': l, # have to negate since we're minimizing
            'losses_training': results_training,
            'losses_validation': results_validation,
            'status': STATUS_FAIL if np.isnan(f1) else STATUS_OK,
            'params': params
        }

        return out


TrainCiteomatic.run(__name__)
