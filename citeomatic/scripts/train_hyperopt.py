import os
import numpy as np
import hyperopt
from pprint import pprint
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
from citeomatic import file_util
from citeomatic.config import App
from citeomatic.corpus import Corpus
from citeomatic.features import Featurizer
from citeomatic.common import FilePaths
from citeomatic.training import train_text_model
from citeomatic.models.options import ModelOptions
from traitlets import Bool, Int, Unicode, Enum, Float
import datetime

class CiteomaticHyperopt(App):

    # hyperopt parameters
    max_evals_initial = Int(default_value=75)
    max_evals_secondary = Int(default_value=10)
    total_samples_initial = Int(default_value=5000000)
    total_samples_secondary = Int(default_value=20000000)
    train_frac = Float(default_value=0.80)
    models_dir_base = Unicode(
        default_value='/net/nfs.corp/s2-research/citeomatic/'
    )
    version = Unicode(default_value='v0')

    # fixed model parameters
    dataset_type = Enum(('dblp', 'pubmed', 'oc'), default_value='dblp')
    embedding_type = Enum(('rnn', 'cnn', 'basic'), default_value='basic')
    samples_per_epoch = Int(default_value=1000000)
    batch_size = Int(default_value=1024)
    use_nn_negatives = Bool(default_value=False)
    reduce_lr_flag = Bool(default_value=True)
    max_features = Int(default_value=200000)

    # to be filled in later
    models_dir = None

    def main(self, args):

        # run identifier
        run_identifier = '_'.join(
            [
                'citeomatic_hyperopt',
                self.dataset_type,
                datetime.datetime.now().strftime("%Y-%m-%d"),
                self.version
            ]
        )
        self.models_dir = os.path.join(self.models_dir_base, run_identifier)

        # the search space
        # note that the scope.int code is a hack to get integers out of the sampler
        space = {
            'dense_dim':
                scope.int(hp.quniform('dense_dim', 25, 325, 25)),
            'embedding_type':
                self.embedding_type,
            'use_nn_negatives':
                self.use_nn_negatives,
            'use_dense':
                True,
            'use_citations':
                hp.choice('use_citations', [True, False]),
            'use_authors':
                hp.choice('use_authors', [True, False]),
            'author_dim':
                10,
            'sparse_option':
                hp.choice('sparse_option', ['none', 'linear', 'attention']),
            'use_holographic':
                hp.choice('use_holographic', [True, False]),
            'use_src_tgt_embeddings':
                hp.choice('use_src_tgt_embeddings', [True, False]),
            'lr':
                hp.choice('lr', [0.1, 0.01, 0.001, 0.0001, 0.00001]),
            'batch_size':
                self.batch_size,
            'samples_per_epoch':
                self.samples_per_epoch,
            'total_samples':
                self.total_samples_initial,
            'samples_per_epoch':
                self.samples_per_epoch,
            'reduce_lr_flag':
                self.reduce_lr_flag,
            'l2_lambda':
                hp.choice('l2_lambda', np.append(np.logspace(-7, 0, 8), 0)),
            'l1_lambda':
                hp.choice('l1_lambda', np.append(np.logspace(-7, 0, 8), 0)),
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
        results_save_file = 'hyperopt_results_' + run_identifier + '.pickle'
        file_util.write_pickle(
            os.path.join(self.models_dir, results_save_file),
            (sorted_results_stage_1, sorted_results_stage_2)
        )
        pprint(sorted_results_stage_2[0])

    def eval_fn(self, params):
        # step 1: make the directory
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # step 2: load the corpus DB
        print("Loading corpus db...")
        fp = FilePaths()
        db_file = fp.get_db_path(self.dataset_type)
        json_file = fp.get_json_path(self.dataset_type)
        if not os.path.isfile(db_file):
            print("Have to build the database! This may take a while, but should only happen once.")
            Corpus.build(db_file, json_file)
        corpus = Corpus.load(db_file, self.train_frac)
        #model_options = ModelOptions(**hyperopt.pyll.stochastic.sample(params))

        # step 3: load/make the featurizer (once per hyperopt run)
        featurizer_file = os.path.join(self.models_dir, 'featurizer.pickle')
        if not os.path.isfile(featurizer_file):
            featurizer = Featurizer(
                max_features=self.max_features,
                allow_duplicates=False
            )
            featurizer.fit(corpus)
            file_util.write_pickle(featurizer_file, featurizer)
        else:
            featurizer = file_util.read_pickle(featurizer_file)

        # step 4: train the model
        model_options = ModelOptions(**params)
        model_options.n_authors = featurizer.n_authors
        model_options.n_features = featurizer.n_features

        model, embedding_model = train_text_model(
            corpus,
            featurizer,
            model_options,
            embedding_model_for_ann=None,
            debug=False,
            tensorboard_dir=None
        )

        # step 5: evaluation
        '''
        try:
            eval_file = os.path.join(
                self.models_dir, 'evaluation_results.pickle'
            )
            results = file_util.read_pickle(eval_file)
            result = results[self.metric]
            # clean up
            eval_clean_cmd = 'rm ' + os.path.join(
                self.models_dir, 'evaluation_results.pickle'
            )
            os.system(eval_clean_cmd)
        except:
            print('Could finish evaluation!')
            result = -1
            results = {}

        out = {
            'loss': result * -1,
            'other_losses': results,
            'status': STATUS_FAIL if result == -1 else STATUS_OK,
            'params': params
        }

        return out
        '''

CiteomaticHyperopt.run(__name__)
