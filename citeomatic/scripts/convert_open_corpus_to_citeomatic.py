import logging

import tqdm

from citeomatic.common import DatasetPaths, FieldNames, global_tokenizer
from citeomatic.config import App
from citeomatic.corpus import Corpus
from citeomatic.traits import Unicode
import os
import json
from citeomatic import file_util
import pickle


class ConvertOpenCorpusToCiteomatic(App):
    input_path = Unicode(default_value=DatasetPaths.OC_FILE)
    output_path = Unicode(default_value=DatasetPaths.OC_CORPUS_JSON)

    def main(self, args):
        logging.info("Reading Open Corpus file from: {}".format(self.input_path))
        logging.info("Writing json file to: {}".format(self.output_path))

        dp = DatasetPaths()

        assert os.path.exists(self.input_path)
        assert not os.path.exists(self.output_path)
        assert not os.path.exists(dp.get_pkl_path('oc'))

        with open(self.output_path, 'w') as f:
            for obj in tqdm.tqdm(file_util.read_json_lines(self.input_path)):
                if 'year' not in obj:
                    continue
                translated_obj = {
                    FieldNames.PAPER_ID: obj['id'],
                    FieldNames.TITLE_RAW: obj['title'],
                    FieldNames.ABSTRACT_RAW: obj['paperAbstract'],
                    FieldNames.AUTHORS: [a['name'] for a in obj['authors']],
                    FieldNames.IN_CITATION_COUNT: len(obj['inCitations']),
                    FieldNames.KEY_PHRASES: obj['keyPhrases'],
                    FieldNames.OUT_CITATIONS: obj['outCitations'],
                    FieldNames.URLS: obj['pdfUrls'],
                    FieldNames.S2_URL: obj['s2Url'],
                    FieldNames.VENUE: obj['venue'],
                    FieldNames.YEAR: obj['year'],
                    FieldNames.TITLE: ' '.join(global_tokenizer(obj['title'])),
                    FieldNames.ABSTRACT: ' '.join(global_tokenizer(obj['paperAbstract']))
                }
                f.write(json.dumps(translated_obj))
                f.write("\n")
        f.close()
        oc_corpus = Corpus.build(dp.get_db_path('oc'), dp.get_json_path('oc'))
        pickle.dump(oc_corpus, open(dp.get_pkl_path('oc')))

ConvertOpenCorpusToCiteomatic.run(__name__)