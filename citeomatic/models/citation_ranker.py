import logging

import tensorflow as tf
from citeomatic.models.layers import Sum, custom_dot, EmbeddingZero
from citeomatic.models.options import ModelOptions
from citeomatic.models.text_embeddings import TextEmbeddingSum, _prefix, make_embedder
from keras.engine import Model
from keras.regularizers import l1, l2
from keras.layers import Dense, Embedding, Input, Reshape, Concatenate, multiply, Lambda, Flatten, \
    Dot
import keras.backend as K

FIELDS = ['title', 'abstract']
SOURCE_NAMES = ['query', 'candidate']


def create_model(options: ModelOptions, pretrained_embeddings=None):
    logging.info('Building model: %s' % options)

    embedders = {}
    if options.use_src_tgt_embeddings:
        # separate emebedders for query and for candidate
        embedder_title, embedder_abstract = make_embedder(options, pretrained_embeddings)
        embedders[_prefix(('query', 'title'))] = embedder_title
        embedders[_prefix(('query', 'abstract'))] = embedder_abstract

        embedder_title, embedder_abstract = make_embedder(options, pretrained_embeddings)
        embedders[_prefix(('candidate', 'title'))] = embedder_title
        embedders[_prefix(('candidate', 'abstract'))] = embedder_abstract
    else:
        # same embedders for query and for candidate
        embedder_title, embedder_abstract = make_embedder(options, pretrained_embeddings)
        embedders[_prefix(('query', 'title'))] = embedder_title
        embedders[_prefix(('query', 'abstract'))] = embedder_abstract
        embedders[_prefix(('candidate', 'title'))] = embedder_title
        embedders[_prefix(('candidate', 'abstract'))] = embedder_abstract

    normed_sums = {}
    intermediate_outputs = []
    citeomatic_inputs = []
    if options.use_dense:
        for source in SOURCE_NAMES:
            for field in FIELDS:
                prefix = _prefix((source, field))
                embedding_model, outputs = embedders[
                    prefix
                ].create_text_embedding_model(prefix=prefix)
                normed_sums[(source, field)] = outputs[0]
                citeomatic_inputs.append(embedding_model.input)

        for field in FIELDS:
            query = normed_sums[('query', field)]
            candidate = normed_sums[('candidate', field)]
            cos_dist = custom_dot(
                query,
                candidate,
                options.dense_dim,
                normalize=True
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

        embedder = TextEmbeddingSum(options=options, field_type='authors')

        # candidate author
        candidate_author_embedder, candidate_author_embeddings = embedder.create_text_embedding_model(
            prefix='candidate-authors',
            final_l2_norm=True
        )
        citeomatic_inputs.append(candidate_author_embedder.input)

        # query author
        query_author_embedder, query_author_embeddings = embedder.create_text_embedding_model(
            prefix='query-authors',
            final_l2_norm=True
        )
        citeomatic_inputs.append(query_author_embedder.input)

        # cos-sim
        author_similarity = custom_dot(
            candidate_author_embeddings[0],
            query_author_embeddings[0],
            options.metadata_dim,
            normalize=True
        )
        intermediate_outputs.append(author_similarity)

    if options.use_venue:
        assert options.n_venues > 0

        embedder = TextEmbeddingSum(options=options, field_type='venue')

        # candidate venue
        candidate_venue_embedder, candidate_venue_embeddings = embedder.create_text_embedding_model(
            prefix='candidate-venue',
            final_l2_norm=True
        )
        citeomatic_inputs.append(candidate_venue_embedder.input)

        # query venue
        query_venue_embedder, query_venue_embeddings = embedder.create_text_embedding_model(
            prefix='query-venue',
            final_l2_norm=True
        )
        citeomatic_inputs.append(query_venue_embedder.input)

        # cos-sim
        venue_similarity = custom_dot(
            candidate_venue_embeddings[0],
            query_venue_embeddings[0],
            options.metadata_dim,
            normalize=True
        )
        intermediate_outputs.append(venue_similarity)

    if options.use_keyphrases:
        assert options.n_keyphrases > 0

        if options.n_keyphrases > 1:
            # only happens if there WERE any keyphrases
            # this prevents opencorpus from having this extra layer
            embedding = TextEmbeddingSum(options=options, field_type='keyphrases')

            # candidate keyphrases
            candidate_keyphrases_embedder, candidate_keyphrases_embeddings = embedding.create_text_embedding_model(
                prefix='candidate-keyphrases',
                final_l2_norm=True
            )
            citeomatic_inputs.append(candidate_keyphrases_embedder.input)

            # query keyphrases
            query_keyphrases_embedder, query_keyphrases_embeddings = embedding.create_text_embedding_model(
                prefix='query-keyphrases',
                final_l2_norm=True
            )
            citeomatic_inputs.append(query_keyphrases_embedder.input)

            # cos-sim
            keyphrases_similarity = custom_dot(
                candidate_keyphrases_embeddings[0],
                query_keyphrases_embeddings[0],
                options.metadata_dim,
                normalize=True
            )
            intermediate_outputs.append(keyphrases_similarity)

    if options.use_citations:
        citation_count_input = Input(
            shape=(1,), dtype='float32', name='candidate-citation-count'
        )
        citeomatic_inputs.append(citation_count_input)
        intermediate_outputs.append(citation_count_input)

    if options.use_selector_confidence:
        candidate_confidence_input = Input(
            shape=(1,), dtype='float32', name='candidate-confidence'
        )
        citeomatic_inputs.append(candidate_confidence_input)
        intermediate_outputs.append(candidate_confidence_input)

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