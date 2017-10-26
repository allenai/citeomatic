import os
from citeomatic.schema_pb2 import Document as ProtoDoc
import spacy

PAPER_EMBEDDING_MODEL = 'paper_embedder'
CITATION_RANKER_MODEL = 'citation_ranker'

nlp = spacy.load("en")
RESTRICTED_POS_TAGS = {'PUNCT', 'SYM', 'DET', 'NUM', 'SPACE', 'PART'}


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
    BASE_DIR = '/net/nfs.corp/s2-research/citeomatic/naacl2017/'

    DBLP_GOLD_DIR = os.path.join(BASE_DIR, 'comparison/dblp/gold')
    DBLP_CORPUS_JSON = os.path.join(BASE_DIR, 'comparison/dblp/corpus.json')
    DBLP_DB_FILE = os.path.join(BASE_DIR, 'db/dblp.sqlite.db')

    PUBMED_GOLD_DIR = os.path.join(BASE_DIR, 'comparison/pubmed/gold')
    PUBMED_CORPUS_JSON = os.path.join(BASE_DIR, 'comparison/pubmed/corpus.json')
    PUBMED_DB_FILE = os.path.join(BASE_DIR, 'db/pubmed.sqlite.db')

    OC_FILE = os.path.join(BASE_DIR, 'open_corpus/papers-2017-02-21.json.gz')
    OC_CORPUS_JSON = os.path.join(BASE_DIR, 'open_corpus/corpus.json')
    OC_DB_FILE = os.path.join(BASE_DIR, 'db/oc.sqlite.db')

    FEATURIZER_FILENAME = 'featurizer.pickle'
    OPTIONS_FILENAME = 'options.json'
    CITEOMATIC_WEIGHTS_FILENAME = 'weights.h5'
    EMBEDDING_WEIGHTS_FILENAME = 'embedding.h5'

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
            date=None

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