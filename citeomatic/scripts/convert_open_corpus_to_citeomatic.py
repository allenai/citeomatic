import logging

import tqdm

from citeomatic.common import FilePaths, FieldNames
from citeomatic.config import App
from citeomatic.traits import Unicode
import os
import json
from citeomatic import file_util


class ConvertOpenCorpusToCiteomatic(App):
    input_path = Unicode(default_value=FilePaths.OC_FILE)
    output_path = Unicode(default_value=FilePaths.OC_CORPUS_JSON)

    def main(self, args):
        logging.info("Reading Open Corpus file from: {}".format(self.input_path))
        logging.info("Writing json file to: {}".format(self.output_path))

        assert os.path.exists(self.input_path)
        assert not os.path.exists(self.output_path)

        with open(self.output_path, 'w') as f:
            for obj in tqdm.tqdm(file_util.read_json_lines(self.input_path)):
                if 'year' not in obj:
                    continue
                translated_obj = {
                    FieldNames.PAPER_ID: obj['id'],
                    FieldNames.TITLE: obj['title'],
                    FieldNames.ABSTRACT: obj['paperAbstract'],
                    FieldNames.AUTHORS: [a['name'] for a in obj['authors']],
                    FieldNames.IN_CITATIONS: obj['inCitations'],
                    FieldNames.KEY_PHRASES: obj['keyPhrases'],
                    FieldNames.OUT_CITATIONS: obj['outCitations'],
                    FieldNames.URLS: obj['pdfUrls'],
                    FieldNames.S2_URL: obj['s2Url'],
                    FieldNames.VENUE: obj['venue'],
                    FieldNames.YEAR: obj['year']
                }
                f.write(json.dumps(translated_obj))
                f.write("\n")
        f.close()

ConvertOpenCorpusToCiteomatic.run(__name__)