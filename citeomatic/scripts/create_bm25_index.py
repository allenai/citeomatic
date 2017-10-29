import os

import tqdm
from whoosh.index import create_in

from citeomatic import file_util
from citeomatic.common import DatasetPaths
from citeomatic.common import schema
from citeomatic.config import App
from citeomatic.traits import Enum


class CreateBM25Index(App):
    #
    # Caveat: It is unclear how to really separate the train, validation and test sets from the
    # index. We currently index all documents (including test docs) which somewhat pollutes the
    # tf df scores.
    #
    #
    dataset_name = Enum(options=['dblp', 'pubmed', 'oc'])

    def main(self, args):
        dp = DatasetPaths()

        corpus_json = dp.get_json_path(self.dataset_name)
        index_location = dp.get_bm25_index_path(self.dataset_name)

        if os.path.exists(index_location):
            assert False
        else:
            os.mkdir(index_location)

        bm25_index = create_in(index_location, schema)
        writer = bm25_index.writer()

        for doc in tqdm.tqdm(file_util.read_json_lines(corpus_json)):
            writer.add_document(
                id=doc['id'],
                title=doc['title'],
                abstract=doc['abstract']
            )

        writer.commit()


CreateBM25Index.run(__name__)