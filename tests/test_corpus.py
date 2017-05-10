#!/usr/bin/env python
import json
import tempfile
import time
import os
import random
from base import file_util

from citeomatic import features
from citeomatic.corpus import Corpus
from citeomatic.schema_pb2 import Document
from google.protobuf.json_format import MessageToDict


def _time(op):
    st = time.time()
    r = op()
    ed = time.time()
    print(op, ed - st)
    return r


WORDS = '''
Ashikaga
Boone's
Charybdis's
Decker
Eurasia
Gounod
Idaho's
Keven
Lubavitcher
Merck's
Nisan
Platonist
Rowling's
Soave's
Tomas
Wilkes
accretion
agreeably
anguishing
armor
avenues
bassoon
bier's
bobs
brightest
bystander's
carpetbags
charbroiling
civilian
collaboration
condition's
convincingly
crankcases
curtsying
deeper
designate
disbursements
divorce
duckbill's
elliptical
enviously
exiling
fateful
fixture
forces
fulcra
geologic
graffiti
gyration's
hearten
homeyness's
hyphenated
inbreed
injections
inundate
jubilantly
lamebrain
liberalism
loss
manna
memorials
miscasting
mortifies
naturalistic
noses
opened
overpopulation's
parqueted
perform
pillow
politest
preferable
pronoun
pyjamas's
rattles
referees
representation's
rhino's
rumples
scarcity's
seldom
shipments
sizes
sneeringly
speakers
stake
stratums
summoning
synthetic's
tenderness's
tingle
transiting
turncoat
uneasily
urchin's
violets
wayfaring's
wintertime
zaniest
'''.split('\n')

WORDS = WORDS * 1000
print(len(WORDS))


def build_test_corpus(source_file, target_file):
    try:
        os.unlink(target_file)
    except:
        pass

    with open(source_file, 'w') as tf:
        for i in range(1000):
            json.dump(
                dict(
                    title=' '.join(random.sample(WORDS, 10)),
                    paperAbstract=' '.join(random.sample(WORDS, 1000)),
                    authors=[],
                    outCitations=[
                        str(x) for x in random.sample(range(1000), 10)
                    ],
                    year=2011,
                    id=str(i),
                    venue='',
                ), tf
            )
            tf.write('\n')

    Corpus.build(target_file, source_file)


def test_corpus_conversion():
    build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')


def test_data_gen():
    #build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')
    corpus = Corpus.load('/home/power/citeomatic-data/data/papers-2017-02-21-sample.sqlite')
    featurizer = file_util.read_pickle('/home/power/citeomatic-data/model/featurizer-default.pickle')
#    featurizer = features.Featurizer(
#        use_unigrams_from_corpus=True,
#        use_bigrams_from_corpus=True,
#        allow_duplicates=False,
#        training_fraction=0.9
#    )
#    featurizer.fit(corpus, max_features=10000, max_df_frac=1.0)
    dg = features.DataGenerator(corpus, featurizer, es_negatives=None)
    gen = dg.triplet_generator(
        id_pool=corpus.train_ids,
        id_filter=corpus.train_ids,
        batch_size=128,
        neg_to_pos_ratio=5
    )

    for i in range(100):
        print(i)
        next(gen)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-s'])
