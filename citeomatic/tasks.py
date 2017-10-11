#!/usr/bin/env python
"""
Luigi pipeline for Citeomatic.

This includes tasks for fetching the dataset, building a vocabulary and
training features and training/evaluating the model.
"""
import logging
import os
import zipfile
from os import path

import luigi
from citeomatic import file_util, features, training, corpus
from citeomatic.features import Featurizer
from citeomatic.models import layers
from citeomatic.models.options import ModelOptions
from citeomatic.serialization import import_from
from luigi.util import inherits

logger = logging.getLogger('citeomatic.tasks')

import faulthandler
faulthandler.enable()


class SharedParameters(luigi.Task):
    base_dir = luigi.Parameter(default=path.expanduser('~/citeomatic-data/'))

    @property
    def data_dir(self):
        return self.base_dir + '/data'

    @property
    def model_dir(self):
        return self.base_dir + '/model'

    def log(self, msg, *args):
        logger.info(msg, *args)


class DownloadCorpus(SharedParameters):
    corpus_url = luigi.Parameter(
        default=
        'https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/2017-02-21/papers-2017-02-21.zip'
    )

    def output(self):
        json_name = self.corpus_url.split('/')[-1]
        json_name = json_name.replace('.zip', '.json.gz')
        return luigi.LocalTarget(path.join(self.data_dir, json_name))

    def run(self):
        self.output().makedirs()

        output_dir = path.dirname(self.output().path)
        output_filename = self.output().path

        assert os.system(
            'curl "%s" > "%s/papers.zip.tmp"' % (self.corpus_url, output_dir)
        ) == 0

        with zipfile.ZipFile('%s/papers.zip.tmp' % output_dir) as zf:
            for name in zf.namelist():
                if name.endswith('.json.gz'):
                    zf.extract(name, output_dir)
                    break

        #assert os.unlink('%s/papers.zip.tmp' % output_dir) == 0


class BuildCorpus(SharedParameters):
    def requires(self):
        return {'corpus': DownloadCorpus()}

    def output(self):
        corpus_suffix = self.requires()['corpus'].corpus_url.split('/')[-1]
        corpus_name = corpus_suffix.replace('.zip', '.sqlite')
        return luigi.LocalTarget(path.join(self.data_dir, corpus_name))

    def run(self):
        try:
            corpus.build_corpus(self.output().path + '.tmp', self.input()['corpus'].path)
            os.rename(self.output().path + '.tmp', self.output().path)
        except:
            os.system("rm -rf '%s'" % self.output().path + '.tmp')
            raise


class CreateFeaturizer(SharedParameters):
    training_fraction = luigi.FloatParameter(default=0.9)
    use_bigrams = luigi.BoolParameter(default=False)
    use_unigrams = luigi.BoolParameter(default=True)
    max_features = luigi.IntParameter(default=100000000)
    name = luigi.Parameter('default')

    def requires(self):
        return {'corpus': BuildCorpus()}

    def output(self):
        return luigi.LocalTarget(
            path.join(self.model_dir, 'featurizer-%s.pickle' % self.name)
        )

    def run(self):
        logger.info(
            "Loading corpus from file %s " % self.input()['corpus'].path
        )
        c = corpus.Corpus.load(self.input()['corpus'].path, self.training_fraction)

        logger.info("Fitting featurizer and making cache...")
        featurizer = Featurizer(
            use_unigrams_from_corpus=self.use_unigrams,
            use_bigrams_from_corpus=self.use_bigrams,
            allow_duplicates=False,
            training_fraction=self.training_fraction
        )
        featurizer.fit(c, max_features=self.max_features)
        self.output().makedirs()
        file_util.write_pickle(self.output().path, featurizer)


class TrainModel(SharedParameters):
    model_config = luigi.Parameter()
    experiment_name = luigi.Parameter(default='v0')
    batch_size = luigi.IntParameter(default=1024)
    use_nn_negatives = luigi.BoolParameter(default=False)

    def requires(self):
        return {'featurizer': CreateFeaturizer(), 'corpus': BuildCorpus()}

    def output(self):
        return luigi.LocalTarget(
            path.join(self.model_dir, self.experiment_name, 'weights.h5')
        )

    def run(self):
        featurizer = file_util.read_pickle(self.input()['featurizer'].path)
        c = corpus.Corpus.load(self.input()['corpus'].path)

        model_options = ModelOptions.load(self.model_config)
        model_options.n_authors = featurizer.n_authors
        model_options.n_features = featurizer.n_features
        model_options.n_documents = len(c)
        model_name = model_options.model_name

        create_model = import_from(
            'citeomatic.models.%s' % model_options.model_name, 'create_model'
        )
        models = create_model(model_options)

        citeomatic_model, embedding_model = (
            models['citeomatic'], models['embedding']
        )

        logging.info(citeomatic_model.summary())

        training_dg = features.DataGenerator(
            c,
            featurizer,
        )
        validation_dg = features.DataGenerator(
            c,
            featurizer,
        )

        metrics = []

        training_generator = training_dg.triplet_generator(
            id_pool=c.train_ids,
            id_filter=c.train_ids,
            batch_size=self.batch_size,
            neg_to_pos_ratio=5
        )
        validation_generator = validation_dg.triplet_generator(
            id_pool=c.test_ids, id_filter=c.train_ids, batch_size=1024
        )
        loss = layers.triplet_loss

        compilation_options = {
            'optimizer': layers.OPTIMIZER,
            'loss': loss,
            'metrics': metrics
        }
        citeomatic_model.compile(**compilation_options)

        training.train_text_model(
            corpus=c,
            model=citeomatic_model,
            embedding_model=embedding_model,
            featurizer=featurizer,
            training_generator=training_generator,
            validation_generator=validation_generator,
            data_generator=training_dg,
            use_nn_negatives=self.use_nn_negatives,
            debug=False
        )

        self.output().makedirs()
        citeomatic_model.save_weights(
            path.join(self.output().path, 'weights.h5'), overwrite=True
        )

        embedding_model.save_weights(
            path.join(self.output().path, 'embedding.h5'), overwrite=True
        )

        file_util.write_json(
            model_options.to_json(),
            path.join(self.output().path, 'options.json')
        )


class TestModel(SharedParameters):
    def requires(self):
        return {
            'featurizer': CreateFeaturizer(),
            'corpus': DownloadCorpus(),
            'model': TrainModel(),
        }

    def run(self):
        from citeomatic.scripts.evaluate_citeomatic_model import \
            TestCiteomatic
        test_app = TestCiteomatic(
            model_dir=self.output_dir(),
            test_samples=self.test_samples,
            min_citation_count=10,
            corpus_path=self._corpus_path('corpus.msgpack'),
            filter_method='es',
        )
        test_app.main([])

if __name__ == '__main__':
    from luigi.cmdline import luigi_run
    luigi_run()
