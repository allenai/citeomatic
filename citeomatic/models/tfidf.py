import collections

import numpy as np


class TFIDFModel(object):
    '''
    A model to predict citation similarity using TFIDF vectors only.

    See http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
    for a reference on the chosen tf-idf form.
    '''

    def __init__(self):
        pass

    def fit(self, corpus, featurizer):
        self.n_d = len(corpus.train_ids)
        self.doc_freq_dict = {}
        for id in corpus.train_ids:
            title_inds, abstract_inds = featurizer.text_features(corpus[id])
            inds = set(self._inds_combine(title_inds, abstract_inds))
            for ind in inds:
                if ind not in self.doc_freq_dict:
                    self.doc_freq_dict[ind] = 0
                self.doc_freq_dict[ind] += 1

    def predict(self, X):
        # X has 4 numpy arrays
        preds = []
        for title_inds_a, abstract_inds_a, title_inds_b, abstract_inds_b in zip(
            *X
        ):
            tfidf_dict_a = self.tfidf_per_document(
                title_inds_a, abstract_inds_a
            )
            tfidf_dict_b = self.tfidf_per_document(
                title_inds_b, abstract_inds_b
            )
            preds.append(
                self.inner_product_between_dicts(tfidf_dict_a, tfidf_dict_b)
            )
        return np.array(preds)

    def tfidf_per_document(self, title_inds, abstract_inds):
        inds = self._inds_combine(title_inds, abstract_inds)
        tf_dict = collections.Counter(inds)
        tfidf_dict = {}
        for ind, count in tf_dict.items():
            tfidf_dict[ind] = count
            # this is what sklearn does by default
            tfidf_dict[ind] *= np.log(
                (1 + self.n_d) / (1 + self.doc_freq_dict.get(ind, 0))
            ) + 1
        # normalize by the norm
        norm = np.linalg.norm(list(tfidf_dict.values()))
        for ind, count in tfidf_dict.items():
            tfidf_dict[ind] = count / norm
        return tfidf_dict

    def inner_product_between_dicts(self, tfidf_dict_a, tfidf_dict_b):
        keys_overlap = set(tfidf_dict_a.keys()
                          ).intersection(set(tfidf_dict_b.keys()))
        sim = 0
        for key in keys_overlap:
            sim += tfidf_dict_a[key] * tfidf_dict_b[key]
        return sim

    def _inds_combine(self, title_inds, abstract_inds):
        return list(title_inds[title_inds > 0]
                   ) + list(abstract_inds[abstract_inds > 0])
