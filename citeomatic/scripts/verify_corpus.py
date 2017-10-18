import logging

from citeomatic.common import DatasetPaths
from citeomatic.config import App
from citeomatic.corpus import Corpus


class VerifyCorpus(App):
    def main(self, args):

        def _verify(db_filename, corpus_json):
            try:
                Corpus.build(db_filename=db_filename, source_json=corpus_json)
            except Exception as e:
                logging.critical("Failed to build corpus {} for file {}".format(db_filename, corpus_json))
                print(e)

        _verify(DatasetPaths.DBLP_DB_FILE, DatasetPaths.DBLP_CORPUS_JSON)

        _verify(DatasetPaths.PUBMED_DB_FILE, DatasetPaths.PUBMED_CORPUS_JSON)

        _verify(DatasetPaths.OC_DB_FILE, DatasetPaths.OC_CORPUS_JSON)

VerifyCorpus.run(__name__)