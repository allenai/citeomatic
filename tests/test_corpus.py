#!/usr/bin/env python
import json
import logging
import os
import random
import time

from citeomatic import features
from citeomatic.common import FieldNames
from citeomatic.corpus import Corpus


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

WORDS = WORDS * 100
print(len(WORDS))


def build_test_corpus(source_file, target_file):
    try:
        os.unlink(target_file)
    except:
        pass

    with open(source_file, 'w') as tf:
        for i in range(100):
            json.dump({
                FieldNames.TITLE: ' '.join(random.sample(WORDS, 10)),
                FieldNames.ABSTRACT: ' '.join(random.sample(WORDS, 1000)),
                FieldNames.AUTHORS: [],
                FieldNames.OUT_CITATIONS: [
                    str(x) for x in random.sample(range(100), 10)
                ],
                FieldNames.IN_CITATION_COUNT: len([
                    str(x) for x in random.sample(range(100), 10)
                ]),
                FieldNames.KEY_PHRASES: [' '.join(random.sample(WORDS, 3))],
                FieldNames.YEAR: 2011,
                FieldNames.PAPER_ID: str(i),
                FieldNames.VENUE: ''
            }, tf
            )
            tf.write('\n')

    Corpus.build(target_file, source_file)


def test_corpus_conversion():
    build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')


def test_data_gen():
    build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')
    corpus = Corpus.load('/tmp/foo.sqlite')
    featurizer = features.Featurizer(
        allow_duplicates=False
    )
    featurizer.fit(corpus, max_df_frac=1.0)
    dg = features.DataGenerator(corpus, featurizer)
    gen = dg.triplet_generator(
        paper_ids=corpus.train_ids,
        candidate_ids=corpus.train_ids,
        batch_size=128,
        neg_to_pos_ratio=5
    )

    for i in range(100):
        print(i)
        next(gen)


def test_data_isolation():
    build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')
    corpus = Corpus.load('/tmp/foo.sqlite')

    assert len(set(corpus.train_ids).intersection(set(corpus.valid_ids))) == 0
    assert len(set(corpus.train_ids).intersection(set(corpus.test_ids))) == 0
    assert len(set(corpus.valid_ids).intersection(set(corpus.test_ids))) == 0

    featurizer = features.Featurizer(
        allow_duplicates=False
    )
    featurizer.fit(corpus, max_df_frac=1.0)
    dg = features.DataGenerator(corpus, featurizer)

    query, examples, labels = next(dg._listwise_examples(corpus.train_ids))
    examples_ids = [doc.id for doc in examples]

    assert len(set(examples_ids).intersection(set(corpus.valid_ids))) == 0
    assert len(set(examples_ids).intersection(set(corpus.test_ids))) == 0

    dg = features.DataGenerator(corpus, featurizer)
    query, examples, labels = next(dg._listwise_examples(paper_ids=corpus.valid_ids,
                                                         candidate_ids=corpus.valid_ids + corpus.train_ids))
    examples_ids = [doc.id for doc in examples]

    assert len(set(examples_ids).intersection(set(corpus.train_ids))) > 0
    assert len(set(examples_ids).intersection(set(corpus.test_ids))) == 0

    dg = features.DataGenerator(corpus, featurizer)
    query, examples, labels = next(dg._listwise_examples(paper_ids=corpus.test_ids,
                                                         candidate_ids=corpus.valid_ids + corpus.train_ids))
    examples_ids = [doc.id for doc in examples]
    assert len(set(examples_ids).intersection(set(corpus.test_ids))) == 0

    dg = features.DataGenerator(corpus, featurizer)
    query, examples, labels = next(
        dg._listwise_examples(paper_ids=corpus.test_ids,
                              candidate_ids=corpus.valid_ids + corpus.train_ids + corpus.test_ids))
    examples_ids = [doc.id for doc in examples]
    assert len(set(examples_ids).intersection(set(corpus.test_ids))) != 0

if __name__ == '__main__':
    import pytest

    pytest.main([__file__, '-s'])
