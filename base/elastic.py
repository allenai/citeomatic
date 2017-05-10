import functools

import elasticsearch

DEFAULT_FIELD_MAPPING = {
    'paperAbstract': 'abstract',
}

ES_URL = 'es.development.s2.dev.ai2'


@functools.lru_cache()
def default_es_client():
    return elasticsearch.Elasticsearch(ES_URL)


def response_to_pandas(resp, field_mapping=DEFAULT_FIELD_MAPPING):
    """
    Convert an ES response to a Pandas dataframe.

    Returns a row for each hit containing the id and score along with any defined fields.
    Field names can be remapped using the `field_mapping` argument.
    :param resp:
    :param field_mapping:
    :return:  pandas.DataFrame
    """
    import pandas
    records = []
    for hit in resp['hits']['hits']:
        record = {
            'id': hit['_id'],
            'score': hit.get('_score', 0.0),
        }

        fields = hit.get(
            'fields', hit['_source']
        )  # handle either place ES can put them based on query type.
        for k, v in fields.items():
            if k in field_mapping:
                k = field_mapping[k]
            if len(v) == 1:
                v = v[0]
            record[k] = v
        records.append(record)

    return pandas.DataFrame.from_records(records)


def author_lookup(author_id, es_client=default_es_client()):
    return es_client.get(
        id=author_id, index='author', doc_type='author'
    )['_source']


def paper_lookup(doc_id, es_client=default_es_client()):
    return es_client.get(id=doc_id, index='paper', doc_type='paper')['_source']
