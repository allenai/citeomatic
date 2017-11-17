import json

import os

from citeomatic.config import App
from citeomatic.models.options import ModelOptions
from citeomatic.traits import Unicode
import copy


class GenerateOcConfigs(App):
    input_config_file = Unicode(default_value=None, allow_none=True)
    out_dir = Unicode(default_value="config/")

    def write_change_to_file(self, filename, base_options, change):
        filename = os.path.join(self.out_dir, filename)
        base_options_2 = copy.deepcopy(base_options)
        base_options_2.update(change)
        json.dump(base_options_2, open(filename, "w"), sort_keys=True, indent=2)

    def main(self, args):
        if self.input_config_file is None:
            base_config = ModelOptions().to_json()
        else:
            base_config = json.load(open(self.input_config_file))

        changes_file_list = [
            ({'use_citations': False}, "oc.citation_ranker.canonical-citation.options.json"),
            ({'use_magdir': False}, "oc.citation_ranker.canonical-magdir.options.json"),
            ({'use_variable_margin': False}, "oc.citation_ranker.canonical-var_margin.options.json"),
            ({
                 'use_metadata': False,
                 'use_authors': False,
                 'use_keyphrases': False,
                 'use_venue': False,
             }, "oc.citation_ranker.canonical-metadata.options.json"),
            ({'use_src_tgt_embeddings': False}, "oc.citation_ranker.canonical-non_siamese.options.json"),
            ({'use_pretrained': True, 'enable_fine_tune': False},
             "oc.citation_ranker.canonical-pretrained_no_finetune.options.json"),
            ({'use_pretrained': True, 'enable_fine_tune': True},
             "oc.citation_ranker.canonical-pretrained_with_finetime.options.json"),
            ({'use_sparse': False},
             "oc.citation_ranker.canonical-sparse.options.json"),
        ]

        for change, filename in changes_file_list:
            self.write_change_to_file(filename=filename,
                                      base_options=base_config,
                                      change=change
                                      )

GenerateOcConfigs.run(__name__)
