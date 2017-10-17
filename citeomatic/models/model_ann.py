from citeomatic.models.options import ModelOptions
from citeomatic.models.layers import L2Normalize, ScalarMultiply
from citeomatic.models.text_embeddings import TextEmbedding
from keras.engine import Model
from keras.layers import Dense, Merge, Reshape

FIELDS = ['title', 'abstract']
SOURCE_NAMES = ['query', 'candidate']


def create_model(options: ModelOptions):
    def reshaped(model):
        return Reshape((1, options.dense_dim))(model)

    def _prefix(tuple):
        return '-'.join(tuple)

    text_embeddings = TextEmbedding(
        n_features=options.n_features,
        dense_dim=options.dense_dim,
        l1_lambda=options.l1_lambda
    )
    scalar_sum_models = {}
    for field in FIELDS:
        scalar_sum_models[field] = ScalarMultiply(name='scalar-mult-' + field)

    # apply text embedding model and add up title, abstract, etc
    embedding_models = {}
    normed_weighted_sum_of_normed_sums = {}
    for source in SOURCE_NAMES:
        weighted_normed_sums = []
        for field in FIELDS:
            prefix = _prefix((source, field))
            embedding_model, _ = text_embeddings.create_text_embedding_model(
                prefix=prefix, final_l2_norm=True
            )
            embedding_models[prefix] = embedding_model
            normed_sum = embedding_models[prefix].outputs[0]
            weighted_normed_sums.append(scalar_sum_models[field](normed_sum))

        weighted_sum = Merge(mode='sum')(weighted_normed_sums)
        normed_weighted_sum_of_normed_sums[source] = L2Normalize.invoke(
            weighted_sum, name='%s-l2_normed_sum' % source
        )

    # cos distance
    query = reshaped(normed_weighted_sum_of_normed_sums['query'])
    candidate = reshaped(normed_weighted_sum_of_normed_sums['candidate'])
    cos_dist = Merge(mode='dot', dot_axes=(2, 2))([query, candidate])
    text_output = Dense(
        1, init='one', bias=False, activation='sigmoid'
    )(Reshape((1,))(cos_dist))

    citeomatic_inputs = []
    for source in SOURCE_NAMES:
        for field in FIELDS:
            key = _prefix((source, field))
            citeomatic_inputs.append(embedding_models[key].input)

    citeomatic_model = Model(input=citeomatic_inputs, output=text_output)

    embedding_model = Model(
        input=citeomatic_inputs[0:len(SOURCE_NAMES)],
        output=normed_weighted_sum_of_normed_sums['query']
    )

    models = {'embedding': embedding_model, 'citeomatic': citeomatic_model}

    return models
