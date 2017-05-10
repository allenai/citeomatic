import logging

import tensorflow as tf
from citeomatic.models.layers import Sum, ZeroMaskedEntries
from citeomatic.models.options import ModelOptions
from citeomatic.models.text_embeddings import (
    TextEmbedding, TextEmbeddingConv, TextEmbeddingLSTM, summed_embedding
)
from keras.engine import Model
from keras.layers import Dense, Embedding, Highway, Input, Merge, Reshape
from keras.layers.core import Flatten

FIELDS = ['title', 'abstract']
SOURCE_NAMES = ['query', 'candidate']


def holographic_merge(inputs):
    a, b = inputs
    a_fft = tf.fft(tf.complex(a, 0.0))
    b_fft = tf.fft(tf.complex(b, 0.0))
    ifft = tf.ifft(tf.conj(a_fft) * b_fft)
    return tf.cast(tf.real(ifft), 'float32')


def holographic_output_shape(shapes):
    return shapes[0]


def create_model(options: ModelOptions):
    logging.info('Building model: %s' % options)

    def _prefix(tuple):
        return '-'.join(tuple)

    def reshaped(model):
        return Reshape((1, options.dense_dim))(model)

    flatten = Flatten()
    embedders = {}

    def _make_embedder():
        if options.embedding_type == 'basic':
            return TextEmbedding(
                n_features=options.n_features,
                dense_dim=options.dense_dim,
                l1_lambda=options.l1_lambda
            )
        elif options.embedding_type == 'cnn':
            return TextEmbeddingConv(
                n_features=options.n_features,
                dense_dim=options.dense_dim,
                n_filter=options.n_filter,
                max_filter_length=options.max_filter_len,
                l1_lambda=options.l1_lambda
            )
        elif options.embedding_type == 'lstm':
            return TextEmbeddingLSTM(
                n_features=options.n_features, dense_dim=options.dense_dim
            )
        else:
            assert False, 'Unknown embedding type %s' % options.embedding_type

    if options.use_src_tgt_embeddings:
        embedders['query'] = _make_embedder()
        embedders['candidate'] = _make_embedder()
    else:
        embedders['query'] = embedders['candidate'] = _make_embedder()

    embedding_models = {}
    normed_sums = {}
    intermediate_outputs = []

    def _similarity(a, b):
        if options.use_holographic:
            return Merge(
                mode=holographic_merge, output_shape=holographic_output_shape
            )([reshaped(a), reshaped(b)])
        else:
            return Merge(
                mode='dot', dot_axes=(2, 2)
            )([reshaped(a), reshaped(b)])

    def _sum(v):
        return Sum()(v)

    citeomatic_inputs = []
    if options.use_dense:
        for source in SOURCE_NAMES:
            for field in FIELDS:
                prefix = _prefix((source, field))
                embedding_model, outputs = embedders[
                    source
                ].create_text_embedding_model(prefix=prefix)
                embedding_models[prefix] = embedding_model
                normed_sums[(source, field)] = outputs[0]
                citeomatic_inputs.append(embedding_models[prefix].input)

        for field in FIELDS:
            query = normed_sums[('query', field)]
            candidate = normed_sums[('candidate', field)]
            intermediate_outputs.append(flatten(_similarity(query, candidate)))

    # lookup weights for the intersection of individual terms (computed by the feature generator.)
    assert not (
        options.use_sparse and options.use_attention
    ), 'Incompatible sparse options.'
    if options.use_sparse:
        for field in FIELDS:
            sparse_input = Input(
                name='query-candidate-%s-intersection' % field, shape=(None,)
            )
            sparse_embedding = Embedding(
                input_dim=options.n_features,
                output_dim=1,
                mask_zero=True,
                name="%s-sparse-embedding" % field
            )
            elementwise_sparse = ZeroMaskedEntries(
            )(sparse_embedding(sparse_input))
            intermediate_outputs.append(_sum(elementwise_sparse))
            citeomatic_inputs.append(sparse_input)
    elif options.use_attention:
        for field in FIELDS:
            sparse_input = Input(
                name='query-candidate-%s-intersection' % field, shape=(None,)
            )
            sparse_embedding = Embedding(
                input_dim=options.n_features,
                output_dim=options.dense_dim,
                mask_zero=True,
                name="%s-sparse-embedding" % field
            )
            elementwise_sparse = ZeroMaskedEntries(
            )(sparse_embedding(sparse_input))
            intermediate_outputs.append(
                flatten(
                    _similarity(
                        normed_sums[('query', field)], _sum(elementwise_sparse)
                    )
                )
            )
            citeomatic_inputs.append(sparse_input)

    if options.use_authors:
        assert options.n_authors > 0
        assert options.author_dim > 0

        # compute candidate-author interactions
        author_input = Input(
            name='candidate-authors', shape=(None,), dtype='int32'
        )
        citeomatic_inputs.append(author_input)
        author_embeddings = summed_embedding(
            name='authors',
            input=author_input,
            n_features=options.n_authors,
            dense_dim=options.author_dim
        )

        if options.author_dim != options.dense_dim:
            author_embeddings = Dense(output_dim=options.dense_dim
                                     )(author_embeddings)

        if options.use_holographic:
            logging.info('Holographic authors.')
            author_abstract_interaction = Merge(
                mode=holographic_merge,
                output_shape=holographic_output_shape,
                name='author-abstract-interaction'
            )(
                [
                    reshaped(author_embeddings),
                    reshaped(normed_sums[('query', 'abstract')])
                ]
            )
        else:
            author_abstract_interaction = Merge(
                mode='dot', dot_axes=(2, 2), name='author-abstract-interaction'
            )(
                [
                    reshaped(author_embeddings),
                    reshaped(normed_sums[('query', 'abstract')])
                ]
            )

        intermediate_outputs.append(flatten(author_abstract_interaction))

    if options.use_citations:
        citation_count_input = Input(
            shape=(1,), dtype='float32', name='candidate-citation-count'
        )
        citeomatic_inputs.append(citation_count_input)
        intermediate_outputs.append(citation_count_input)

    if len(intermediate_outputs) > 1:
        cosine_dists_merged = Merge(mode='concat')(intermediate_outputs)
        last = cosine_dists_merged
    else:
        last = intermediate_outputs

    for i, layer_size in enumerate(options.dense_config.split(',')):
        layer_size = int(layer_size)
        if options.dense_type == 'highway':
            last = Highway(name='dense-%d' % i, activation='elu')(last)
        else:
            last = Dense(
                layer_size, name='dense-%d' % i, activation='elu'
            )(last)

    text_output = Dense(
        1, init='one', name='final-output', activation='sigmoid'
    )(last)

    citeomatic_model = Model(input=citeomatic_inputs, output=text_output)
    embedding_model, _ = embedders['query'].create_text_embedding_model(
        prefix="doc"
    )

    models = {
        'embedding': embedding_model,
        'citeomatic': citeomatic_model,
    }

    return models
