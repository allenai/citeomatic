import logging

import elasticsearch
import six
from citeomatic.common import Document

ES_ADDRESS = "es.production.s2.prod.ai2:9200"
ES = elasticsearch.Elasticsearch(ES_ADDRESS)


def _bool(filters=[], must=[], should=[], minimum_should_match=0):
    return {
        "bool":
            {
                "must": must,
                "filter": filters,
                "should": should,
                "minimum_should_match": minimum_should_match
            }
    }


def _range(field, start, stop):
    return {"range": {field: {"gte": start, "lte": stop}}}


def _match(field, query):
    return {
        "match": {
            field: {
                "query": query,
            }
        }
    }


def _phrase(field, phrase, slop=0):
    return {"match_phrase": {field: {"query": phrase, "slop": slop}}}


def _boost_by_field(
    field_name, modifier='log1p', minimum_value=0, max_boost=10
):
    return {
        "function_score":
            {
                "query": {
                    "range": {
                        field_name: {
                            "gt": minimum_value,
                        }
                    },
                },
                "functions":
                    [
                        {
                            "field_value_factor":
                                {
                                    "field": field_name,
                                    "modifier": modifier,
                                }
                        },
                    ],
                "max_boost":
                    max_boost,
            }
    }


# Use the builtin 'more-like-this' function from ES.
def _more_like_this(
    text,
    min_term_freq=2,
    minimum_should_match='30%',
    max_query_terms=25,
    min_doc_freq=5,
    max_doc_freq=1000000
):
    return {
        "more_like_this":
            {
                "like": text,
                "min_term_freq": min_term_freq,
                "minimum_should_match": minimum_should_match,
                "max_query_terms": max_query_terms,
                "min_doc_freq": min_doc_freq,
                "max_doc_freq": max_doc_freq
            },
    }


def _hits(queries, fields=('title',), size=50):
    hits = []
    for query in queries:
        result = ES.search(
            index='paper',
            doc_type='paper',
            body={'query': query},
            _source=fields,
            size=size
        )
        for h in result['hits']['hits']:
            if '_source' in h:
                for field, value in six.iteritems(h['_source']):
                    if isinstance(value, list) and len(value) == 1:
                        h[field] = value[0]
                    else:
                        h[field] = value
                del h['_source']
            del h['_index']
            del h['_type']
            h['id'] = h['_id']
            h['score'] = h['_score']
            del h['_id']
            del h['_score']
            hits.append(h)

    return hits


def similar_documents(doc, N=None, exclude_citations=False, min_citations=5):
    if N is None:
        N = len(doc.citations) * 10

    mlt_query = _more_like_this(
        doc.title + '\n' + doc.abstract,
        min_term_freq=1,
        minimum_should_match='10%',
        max_query_terms=25,
        min_doc_freq=10,
        max_doc_freq=100000
    )
    mlt_query = _bool(
        should=[
            mlt_query,
            _boost_by_field('numCitedBy', modifier='sqrt'),
        ],
        filters=[
            _range('year', start=1970, stop=doc.year),
            _range('numCitedBy', start=min_citations, stop=None)
        ]
    )
    author_query = _bool(
        should=[
            _phrase('authors.name', author, slop=1) for author in doc.authors
        ],
        minimum_should_match=1,
        filters=[
            _range('year', start=1970, stop=doc.year),
            _range('numCitedBy', start=min_citations, stop=None)
        ]
    )
    title_query = _bool(
        should=[[_phrase('paperAbstract', w) for w in doc.title.split() if w]],
        minimum_should_match=1,
        filters=[
            _range('year', start=1970, stop=doc.year),
            _range('numCitedBy', start=min_citations, stop=None)
        ]
    )

    results = _hits([mlt_query, author_query, title_query], size=N // 3)
    results_ids = set(i['id'] for i in results)

    if exclude_citations:
        results_ids = results_ids.difference(doc.citations)

    return results_ids


def fetch_citations(id):
    logging.debug('Fetching citations for: %s', id)
    resp = ES.search(
        index='citation',
        doc_type='citation',
        q='citingPaper.id:%s AND _exists_:citedPaper.id' % id,
        _source=['citedPaper.id'],
        size=1000
    )
    citations = [
        hit['_source']['citedPaper']['id'] for hit in resp['hits']['hits']
    ]
    logging.debug('Found %s citations', len(citations))
    return citations


def fetch_level2_citations(ids):
    logging.debug('Fetching citations for %d papers', len(ids))
    citations = []
    for id in ids:
        citations.extend(fetch_citations(id))

    return set(citations)


def fetch_all(
    ids,
    with_citations=False,
    fields=(
        'title', 'paperAbstract', 'authors.name', 'year', 'venue', 'numCitedBy',
        'numCiting', 'keyPhrases'
    )
):
    fields = list(fields)
    response = ES.mget(
        body={'ids': ids}, index='paper', doc_type='paper', _source=fields
    )
    es_docs = response['docs']

    citations = {}
    if with_citations:
        for doc in es_docs:
            citations[doc['_id']] = fetch_citations(doc['_id'])

    return [
        Document(
            title=doc['_source']['title'],
            abstract=doc['_source'].get('paperAbstract', None),
            authors=[a['name'] for a in doc['_source']['authors']],
            year=doc['_source'].get('year', 2017),
            id=doc['_id'],
            citations=citations.get(doc['_id'], []),
            venue=doc['_source']['venue'],
            in_citation_count=doc['_source']['numCitedBy'],
            out_citation_count=doc['_source']['numCiting'],
            key_phrases=doc['_source'].get('keyPhrases', [])
        ) for doc in es_docs if '_source' in doc
    ]


def document_from_title(title) -> Document:
    resp = ES.search(
        index='paper', doc_type='paper', q='title:"%s"' % title, _source=['id']
    )
    if resp['hits']['total'] == 0:
        return None
    if resp['hits']['total'] != 1:
        logging.warning('Document %s was not unique: %s', title, resp['hits'])
    id = resp['hits']['hits'][0]['_id']
    return fetch_all([id], with_citations=True)[0]
