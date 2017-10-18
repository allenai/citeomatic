import logging
import os

import keras.backend as K
import tensorflow as tf
from citeomatic import service
from citeomatic.serialization import model_from_directory
from citeomatic.features import Corpus
from citeomatic.grobid_parser import GrobidParser
from citeomatic.neighbors import ANN, EmbeddingModel
from citeomatic.config import setup_default_logging


def get_session():
    num_threads = os.environ.get('NUM_THREADS', None)
    gpu_fraction = float(os.environ.get('GPU_FRACTION', '1.0'))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(
            config=tf.ConfigProto(
                gpu_options=gpu_options,
                intra_op_parallelism_threads=int(num_threads)
            )
        )
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


K.set_session(get_session())

setup_default_logging(logging.INFO)

featurizer, models = model_from_directory(os.environ['MODEL_PATH'])
if 'ANN_MODEL_PATH' in os.environ:
    featurizer, ann_models = model_from_directory(os.environ['ANN_MODEL_PATH'])
    ann_model = EmbeddingModel(featurizer, ann_models['embedding'])
    ann = ANN.load(os.environ['ANN_MODEL_PATH'] + '/citeomatic_ann')
    corpus = Corpus.load(os.environ['CORPUS_PATH'])
else:
    ann = None
    ann_model = None
    corpus = None

app = service.app
app.config['DEBUG'] = True
app.config['API_MODEL'] = service.APIModel(
    corpus=corpus,
    featurizer=featurizer,
    models=models,
    ann_embedding_model=ann_model,
    ann=ann,
)

# assert os.environ.get('AWS_ACCESS_KEY_ID')
# assert os.environ.get('AWS_SECRET_ACCESS_KEY')
app.config['NODE_PROXY'] = os.environ.get('NODE_PROXY', 'http://localhost:5100')
app.config['S3_CACHE_ON'] = int(os.environ.get('S3_CACHE_ON', 0))
app.config['SECRET_KEY'] = 'top secret!'
app.config['GROBID'] = GrobidParser(
    os.environ.get('GROBID_HOST', 'http://localhost:8080')
)

logging.info("S3_CACHE_ON=%s", app.config['S3_CACHE_ON'])
