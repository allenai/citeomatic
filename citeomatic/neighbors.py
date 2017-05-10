from typing import Iterator

import arrow
import numpy as np
import tqdm

from annoy import AnnoyIndex
from base import batch_apply, file_util, flatten
from citeomatic.schema_pb2 import Document
from citeomatic.serialization import load_pickle


def hit_to_document(hit):
    """
    Convert hit from Elasticsearch into a Citeomatic document.
    :param hit: dictionary from ES
    :return: Document
    """
    date_str = hit['_source'].get('earliestAcquisitionDate')
    if date_str:
        date = arrow.get(date_str)
    else:
        date = None

    return Document(
        title=hit['_source']['title'],
        id=hit['_id'],
        abstract=hit['_source']['paperAbstract'],
        authors=flatten(
            [author['name'] for author in hit['_source']['authors']]
        ),
        year=hit['_source'].get('year', 2016),
        venue=hit['_source'].get('venue', ''),
        date=date,
        citations=[],
        in_citation_count=0,
        out_citation_count=0,
        key_phrases=[]
    )


class ANN(object):
    """
    Wraps an Annoy index and a docid mapping.

    AnnoyIndex do not pickle correctly; they need to be save/loaded as well.
    """

    def __init__(self, embeddings, annoy_index, docid_to_idx):
        self.docid_to_idx = docid_to_idx
        self.idx_to_docid = {v: k for (k, v) in docid_to_idx.items()}
        self.embeddings = embeddings
        if annoy_index is not None:
            self.annoy_dims = annoy_index.f
            self.annoy = annoy_index
        else:
            self.annoy = None

    def save(self, target):
        if self.annoy is not None:
            self.annoy.save('%s.annoy' % target)

        file_util.write_pickle('%s.pickle' % target, self)

    @staticmethod
    def load(source):
        import annoy
        ann = load_pickle('%s.pickle' % source)
        if ann.annoy is not None:
            ann.annoy = annoy.AnnoyIndex(ann.annoy_dims)
            ann.annoy.load('%s.annoy' % source)
        return ann

    def get_nns_by_vector(self, vector, top_n, **kw):
        similarity = np.dot(self.embeddings, -vector)
        idx = np.argpartition(similarity, top_n)[:top_n]
        idx = idx[np.argsort(similarity[idx])]
        return [self.idx_to_docid[i] for i in idx]

    def get_nns_by_id(self, doc_id, top_n, **kw):
        idx = self.annoy.get_nns_by_item(
            self.docid_to_idx[doc_id], top_n, search_k=-1
        )
        return [self.idx_to_docid[i] for i in idx]


class EmbeddingModel(object):
    """
    Wrap a Siamese citeomatic model and expose an interface
    to extract the embeddings for individual documents.
    """

    def __init__(self, featurizer, model):
        import keras.backend as K
        self._model = model
        self._featurizer = featurizer
        self.output_shape = K.int_shape(self._model.outputs[0])[-1]

    def embed_documents(self,
                        generator: Iterator[Document]) -> Iterator[np.ndarray]:
        """
        Compute embeddings of the provided documents.
        """

        def _run_embedding(batch) -> np.array:
            features = self._featurizer.transform_list(batch)
            doc_embedding = self._model.predict(
                {
                    'query-title-txt': features['title'],
                    'query-abstract-txt': features['abstract'],
                    'doc-txt': features['abstract'],
                }
            )
            return doc_embedding

        return batch_apply(generator, _run_embedding)

    def embed_hits(self, generator: Iterator[dict]) -> Iterator[np.ndarray]:
        return self.embed_documents(map(hit_to_document, generator))

    def embed(self, doc):
        return np.asarray(list(self.embed_documents([doc])))[0]


def make_ann(embedding_model, corpus, ann_trees=100, build_ann_index=True):
    docid_to_idx = {}
    embedding_gen = embedding_model.embed_documents(corpus)
    doc_embeddings = np.zeros((len(corpus), embedding_model.output_shape))
    for i, (doc, embedding) in enumerate(zip(tqdm.tqdm(corpus), embedding_gen)):
        docid_to_idx[doc.id] = i
        doc_embeddings[i] = embedding

    if build_ann_index:
        annoy_index = AnnoyIndex(embedding_model.output_shape)
        for i, embedding in enumerate(tqdm.tqdm(doc_embeddings)):
            annoy_index.add_item(i, embedding)

        annoy_index.build(ann_trees)
    else:
        annoy_index = None

    ann = ANN(doc_embeddings, annoy_index, docid_to_idx)

    return ann
