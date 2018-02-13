import importlib
import os

import pickle

from citeomatic import file_util
from citeomatic.schema_pb2 import Document as ProtoDoc
import spacy
from whoosh.fields import *

PAPER_EMBEDDING_MODEL = 'paper_embedder'
CITATION_RANKER_MODEL = 'citation_ranker'

nlp = spacy.load("en")
RESTRICTED_POS_TAGS = {'PUNCT', 'SYM', 'DET', 'NUM', 'SPACE', 'PART'}

schema = Schema(title=TEXT,
                abstract=TEXT,
                id=ID(stored=True))


def global_tokenizer(text, restrict_by_pos=False, lowercase=True, filter_empty_token=True):
    if restrict_by_pos:
        token_list = [
            w.text for w in nlp(text) if w.pos_ not in RESTRICTED_POS_TAGS
        ]
    else:
        token_list = [w.text for w in nlp(text)]

    if lowercase:
        token_list = [w.lower() for w in token_list]

    if filter_empty_token:
        token_list = [w for w in token_list if len(w) > 0]

    return token_list


class FieldNames(object):
    PAPER_ID = "id"
    TITLE = "title"
    ABSTRACT = "abstract"
    AUTHORS = "authors"

    VENUE = "venue"
    YEAR = "year"

    IN_CITATIONS = "in_citations"
    OUT_CITATIONS = "out_citations"
    KEY_PHRASES = "key_phrases"

    URLS = "pdf_urls"
    S2_URL = "s2_url"

    OUT_CITATION_COUNT = 'out_citation_count'
    IN_CITATION_COUNT = 'in_citation_count'

    DATE = 'date'

    TITLE_RAW = "title_raw"
    ABSTRACT_RAW = "abstract_raw"


class DatasetPaths(object):
    BASE_DIR = os.path.abspath("./data")

    DBLP_GOLD_DIR = os.path.join(BASE_DIR, 'comparison/dblp/gold')
    DBLP_CORPUS_JSON = os.path.join(BASE_DIR, 'comparison/dblp/corpus.json')
    DBLP_DB_FILE = os.path.join(BASE_DIR, 'db/dblp.sqlite.db')
    DBLP_BM25_INDEX = os.path.join(BASE_DIR, 'bm25_index/dblp/')

    PUBMED_GOLD_DIR = os.path.join(BASE_DIR, 'comparison/pubmed/gold')
    PUBMED_CORPUS_JSON = os.path.join(BASE_DIR, 'comparison/pubmed/corpus.json')
    PUBMED_DB_FILE = os.path.join(BASE_DIR, 'db/pubmed.sqlite.db')
    PUBMED_BM25_INDEX = os.path.join(BASE_DIR, 'bm25_index/pubmed/')

    OC_FILE = os.path.join(BASE_DIR, 'open_corpus/papers-2017-02-21.json.gz')
    OC_CORPUS_JSON = os.path.join(BASE_DIR, 'open_corpus/corpus.json')
    OC_DB_FILE = os.path.join(BASE_DIR, 'db/oc.sqlite.db')
    OC_BM25_INDEX = os.path.join(BASE_DIR, 'bm25_index/oc/')
    OC_PKL_FILE = os.path.join(BASE_DIR, 'open_corpus/corpus.pkl')
    OC_ANN_FILE = os.path.join(BASE_DIR, 'open_corpus/ann.pkl')

    PRETRAINED_DIR = os.path.join(BASE_DIR, 'pretrained')
    EMBEDDING_WEIGHTS_FILENAME = 'embedding.h5'
    PRETRAINED_VOCAB_FILENAME = 'vocab.txt'
    FEATURIZER_FILENAME = 'featurizer.pickle'
    OPTIONS_FILENAME = 'options.json'
    CITEOMATIC_WEIGHTS_FILENAME = 'weights.h5'

    def embeddings_weights_for_corpus(self, corpus_name):
        return os.path.join(
            self.PRETRAINED_DIR,
            corpus_name + '_' + self.EMBEDDING_WEIGHTS_FILENAME
        )

    def vocab_for_corpus(self, corpus_name):
        return os.path.join(
            self.PRETRAINED_DIR,
            corpus_name + '_' + self.PRETRAINED_VOCAB_FILENAME
        )

    def get_json_path(self, corpus_name):
        if corpus_name.lower() == 'dblp':
            return self.DBLP_CORPUS_JSON
        elif corpus_name.lower() == 'pubmed':
            return self.PUBMED_CORPUS_JSON
        elif (corpus_name.lower() == 'oc'
              or corpus_name.lower() == 'open_corpus'
              or corpus_name.lower() == 'opencorpus'):
            return self.OC_CORPUS_JSON
        else:
            return None

    def get_bm25_index_path(self, corpus_name):
        if corpus_name.lower() == 'dblp':
            return self.DBLP_BM25_INDEX
        elif corpus_name.lower() == 'pubmed':
            return self.PUBMED_BM25_INDEX
        elif (corpus_name.lower() == 'oc'
              or corpus_name.lower() == 'open_corpus'
              or corpus_name.lower() == 'opencorpus'):
            return self.OC_BM25_INDEX
        else:
            return None

    def get_db_path(self, corpus_name):
        if corpus_name.lower() == 'dblp':
            return self.DBLP_DB_FILE
        elif corpus_name.lower() == 'pubmed':
            return self.PUBMED_DB_FILE
        elif (corpus_name.lower() == 'oc'
              or corpus_name.lower() == 'open_corpus'
              or corpus_name.lower() == 'opencorpus'):
            return self.OC_DB_FILE
        else:
            return None

    def get_pkl_path(self, corpus_name):
        if (corpus_name.lower() == 'oc'
            or corpus_name.lower() == 'open_corpus'
            or corpus_name.lower() == 'opencorpus'):
            return self.OC_PKL_FILE
        else:
            assert False


class Document(object):
    _fields = [
        FieldNames.TITLE,
        FieldNames.ABSTRACT,
        FieldNames.AUTHORS,
        FieldNames.OUT_CITATIONS,
        FieldNames.YEAR,
        FieldNames.PAPER_ID,
        FieldNames.VENUE,
        FieldNames.IN_CITATION_COUNT,
        FieldNames.OUT_CITATION_COUNT,
        FieldNames.KEY_PHRASES,
        FieldNames.DATE,
        FieldNames.TITLE_RAW,
        FieldNames.ABSTRACT_RAW,
    ]

    def __init__(
            self,
            title,
            abstract,
            authors,
            out_citations,
            year,
            id: str,
            venue,
            in_citation_count,
            out_citation_count,
            key_phrases,
            title_raw,
            abstract_raw,
            date=None,
            candidate_selector_confidence=None

    ):
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.out_citations = out_citations
        self.year = year
        self.id = id
        self.venue = venue
        self.in_citation_count = in_citation_count
        self.out_citation_count = out_citation_count
        self.key_phrases = key_phrases
        self.date = date

        self.title_raw = title_raw
        self.abstract_raw = abstract_raw
        self.candidate_selector_confidence = candidate_selector_confidence

    def __iter__(self):
        for k in self._fields:
            yield getattr(self, k)

    def _asdict(self):
        return dict(**self.__dict__)

    @staticmethod
    def from_proto_doc(doc: ProtoDoc):
        out_citations = [c for c in doc.out_citations]
        return Document(
            title=doc.title,
            abstract=doc.abstract,
            authors=[a for a in doc.authors],
            out_citations=out_citations,
            in_citation_count=doc.in_citation_count,
            year=doc.year,
            id=doc.id,
            venue=doc.venue,
            out_citation_count=len(out_citations),
            key_phrases=[p for p in doc.key_phrases],
            title_raw=doc.title_raw,
            abstract_raw=doc.abstract_raw,
        )


class ModelLoader(pickle.Unpickler):
    def find_class(self, mod_name, klass_name):
        if mod_name[:4] == 'ai2.':
            mod_name = mod_name[4:]

        mod = importlib.import_module(mod_name)
        return getattr(mod, klass_name)


def load_pickle(filename):
    with file_util.open(filename) as f:
        return ModelLoader(f).load()