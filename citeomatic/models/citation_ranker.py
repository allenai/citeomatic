import logging

import tensorflow as tf
from citeomatic.models.layers import Sum, custom_dot, EmbeddingZero
from citeomatic.models.options import ModelOptions
from citeomatic.models.text_embeddings import (
    TextEmbeddingSum, TextEmbeddingConv, TextEmbeddingLSTM, _prefix
)
from keras.engine import Model
from keras.regularizers import l1, l2
from keras.layers import Dense, Embedding, Input, Reshape, Concatenate

FIELDS = ['title', 'abstract']
SOURCE_NAMES = ['query', 'candidate']


def create_model(options: ModelOptions, pretrained_embeddings=None):
    logging.info('Building model: %s' % options)

    embedders = {}

    def _make_embedder():
        if options.embedding_type == 'sum':
            return TextEmbeddingSum(options, pretrained_embeddings)
        elif options.embedding_type == 'cnn':
            return TextEmbeddingConv(options, pretrained_embeddings)
        elif options.embedding_type == 'lstm':
            return TextEmbeddingLSTM(options, pretrained_embeddings)
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
            cos_dist = custom_dot(
                query,
                candidate,
                options.dense_dim,
                normalize=False
            )
            intermediate_outputs.append(cos_dist)

    # lookup weights for the intersection of individual terms
    # (computed by the feature generator.)
    if options.use_sparse:
        for field in FIELDS:
            sparse_input = Input(
                name='query-candidate-%s-intersection' % field, shape=(None,)
            )
            elementwise_sparse = EmbeddingZero(
                input_dim=options.n_features,
                output_dim=1,
                mask_zero=True,
                name="%s-sparse-embedding" % field,
                activity_regularizer=l1(options.l1_lambda)
            )(sparse_input)
            intermediate_outputs.append(Sum()(elementwise_sparse))
            citeomatic_inputs.append(sparse_input)

    if options.use_authors:
        assert options.n_authors > 0
        assert options.author_dim > 0

        # compute candidate-author interactions
        author_embedder, outputs = TextEmbeddingSum(
            options=options, field_type='authors'
        ).create_text_embedding_model(
            prefix='candidate-authors',
            final_l2_norm=True
        )
        citeomatic_inputs.append(author_embedder.input)
        author_embeddings = outputs[0]
        if options.author_dim != options.dense_dim:
            author_embeddings = Dense(options.dense_dim)(author_embeddings)

        author_abstract_interaction = custom_dot(
            author_embeddings,
            normed_sums[('query', 'abstract')],
            options.dense_dim,
            normalize=True
        )
        intermediate_outputs.append(author_abstract_interaction)

    if options.n_venues:
        assert options.n_venues > 0

        # compute candidate-venue interactions
        venue_embedder, outputs = TextEmbeddingSum(
            options=options, field_type='venue'
        ).create_text_embedding_model(
            prefix='candidate-venue',
            final_l2_norm=True
        )
        citeomatic_inputs.append(venue_embedder.input)
        venue_embeddings = outputs[0]
        if options.venue_dim != options.dense_dim:
            venue_embeddings = Dense(options.dense_dim)(venue_embeddings)

        venue_abstract_interaction = custom_dot(
            venue_embeddings,
            normed_sums[('query', 'abstract')],
            options.dense_dim,
            normalize=True
        )
        intermediate_outputs.append(venue_abstract_interaction)

    if options.use_citations:
        citation_count_input = Input(
            shape=(1,), dtype='float32', name='candidate-citation-count'
        )
        citeomatic_inputs.append(citation_count_input)
        intermediate_outputs.append(citation_count_input)

    if len(intermediate_outputs) > 1:
        last = Concatenate()(intermediate_outputs)
    else:
        last = intermediate_outputs

    for i, layer_size in enumerate(options.dense_config.split(',')):
        layer_size = int(layer_size)
        last = Dense(
            layer_size, name='dense-%d' % i, activation='elu'
        )(last)

    text_output = Dense(
        1, kernel_initializer='one', name='final-output', activation='sigmoid'
    )(last)

    citeomatic_model = Model(inputs=citeomatic_inputs, outputs=text_output)

    # Setting embedding model to None to avoid its inadvertent usage for ANN embeddings
    models = {
        'embedding': None,
        'citeomatic': citeomatic_model,
    }

    return models
