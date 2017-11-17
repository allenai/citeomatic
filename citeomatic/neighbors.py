from typing import Iterator

import arrow
import numpy as np
import tqdm

from annoy import AnnoyIndex
from citeomatic import file_util
from citeomatic.utils import batch_apply, flatten
from citeomatic.schema_pb2 import Document
from citeomatic.common import load_pickle
import keras.backend as K


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

    @classmethod
    def build(cls, embedding_model, corpus, ann_trees=100):
        docid_to_idx = {}

        if corpus.corpus_type == 'pubmed' or corpus.corpus_type == 'dblp':
            docs = [corpus[doc_id] for doc_id in corpus.train_ids + corpus.valid_ids]
        else:
            docs = corpus

        doc_embeddings = np.zeros((len(docs), embedding_model.output_shape))
        embedding_gen = embedding_model.embed_documents(docs, batch_size=1024)

        for i, (doc, embedding) in enumerate(
                zip(tqdm.tqdm(docs), embedding_gen)):
            docid_to_idx[doc.id] = i
            doc_embeddings[i] = embedding

        annoy_index = AnnoyIndex(embedding_model.output_shape)
        for i, embedding in enumerate(tqdm.tqdm(doc_embeddings)):
            annoy_index.add_item(i, embedding)

        annoy_index.build(ann_trees)
        ann = cls(doc_embeddings, annoy_index, docid_to_idx)

        return ann

    def save(self, target):
        if self.annoy is not None:
            self.annoy.save('%s.annoy' % target)

        file_util.write_pickle('%s.pickle' % target, self)

    def __getstate__(self):
        return self.docid_to_idx, self.idx_to_docid, self.embeddings, self.annoy_dims, None

    def __setstate__(self, state):
        self.docid_to_idx, self.idx_to_docid, self.embeddings, self.annoy_dims, self.annoy = state

    @staticmethod
    def load(source):
        import annoy
        ann = file_util.read_pickle('%s.pickle' % source)
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

    def get_similarities(self, vector, doc_ids):
        indexes = [self.docid_to_idx[doc_id] for doc_id in doc_ids]
        return np.dot(self.embeddings[indexes], vector)


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
                        generator: Iterator[Document], batch_size=256) -> Iterator[np.ndarray]:
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

        return batch_apply(generator, _run_embedding, batch_size)

    def embed(self, doc):
        return np.asarray(list(self.embed_documents([doc])))[0]
