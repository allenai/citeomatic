#!/usr/bin/env python
import json
import logging
import os
import random
import time

import numpy as np

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
accretion
agreeably
anguishing
armor
avenues
bassoon
bier
bobs
brightest
bystander
carpetbags
charbroiling
civilian
collaboration
condition
convincingly
crankcases
curtsying
deeper
designate
disbursements
divorce
duckbill
elliptical
enviously
exiling
fateful
fixture
forces
fulcra
geologic
graffiti
gyration
hearten
homeyness
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
overpopulation
parqueted
perform
pillow
politest
preferable
pronoun
pyjamas
rattles
referees
representation
rhino
rumples
scarcity
seldom
shipments
sizes
sneeringly
speakers
stake
stratums
summoning
synthetic
tenderness
tingle
transiting
turncoat
uneasily
urchin
violets
wayfaring
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
                    str(x) for x in random.sample(range(100), 2)
                ],
                FieldNames.IN_CITATION_COUNT: len([
                    str(x) for x in random.sample(range(100), 2)
                ]),
                FieldNames.KEY_PHRASES: random.sample(WORDS, 3),
                FieldNames.YEAR: 2011,
                FieldNames.PAPER_ID: str(i),
                FieldNames.VENUE: 'v-{}'.format(random.randint(1, 5))
            }, tf
            )
            tf.write('\n')

    Corpus.build(target_file, source_file)


def test_corpus_conversion():
    build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')


def test_corpus_iterator():
    corpus = Corpus.load('/tmp/foo.sqlite')
    iter_ids = []
    for doc in corpus:
        iter_ids.append(doc.id)
    overlap_n = len(set(iter_ids).intersection(set(corpus.all_ids)))
    assert overlap_n == corpus.n_docs


def test_featurizer_and_data_gen():
    build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')
    corpus = Corpus.load('/tmp/foo.sqlite')
    featurizer = features.Featurizer()
    featurizer.fit(corpus, max_df_frac=1.0)

    dg = features.DataGenerator(corpus, featurizer)
    gen = dg.triplet_generator(
        paper_ids=corpus.all_ids,
        candidate_ids=corpus.all_ids,
        batch_size=128,
        neg_to_pos_ratio=5
    )

    # make sure we can get features
    for i in range(10):
        print(i)
        X, y = next(gen)

    # correct batch size
    assert len(y) >= 128
    # positives, hard negatives, easy negatives
    assert len(np.unique(y)) == 3
    # correct padding
    assert X['query-abstract-txt'].shape[1] == featurizer.max_abstract_len
    assert X['query-title-txt'].shape[1] == featurizer.max_title_len
    # no new words
    assert set(featurizer.word_indexer.word_to_index.keys()).difference(WORDS) == set()

    q, ex, labels = next(dg._listwise_examples(
        corpus.all_ids,
        corpus.all_ids
    ))

    # query id should not be in candidates
    assert q.id not in [i.id for i in ex]

    # pos ids should be out_citations
    pos_docs = [i.id for i, j in zip(ex, labels) if j == np.max(labels)]
    assert set(pos_docs) == set(q.out_citations)

    # neg ids should be NOT out_citations
    neg_docs = [i.id for i, j in zip(ex, labels) if j < np.max(labels)]
    assert np.all([i not in neg_docs for i in q.out_citations])

    # test variable margin off
    dg = features.DataGenerator(corpus, featurizer, use_variable_margin=False)
    gen = dg.triplet_generator(
        paper_ids=corpus.all_ids,
        candidate_ids=corpus.all_ids,
        batch_size=128,
        neg_to_pos_ratio=5
    )

    X, y = next(gen)
    print(dg.margins_offset_dict)
    assert len(np.unique(y)) == 2



def test_data_isolation():
    build_test_corpus('/tmp/foo.json', '/tmp/foo.sqlite')
    corpus = Corpus.load('/tmp/foo.sqlite')

    assert len(set(corpus.train_ids).intersection(set(corpus.valid_ids))) == 0
    assert len(set(corpus.train_ids).intersection(set(corpus.test_ids))) == 0
    assert len(set(corpus.valid_ids).intersection(set(corpus.test_ids))) == 0

    featurizer = features.Featurizer()
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
    #assert len(set(examples_ids).intersection(set(corpus.test_ids))) != 0


if __name__ == '__main__':
    import pytest

    pytest.main([__file__, '-s'])
