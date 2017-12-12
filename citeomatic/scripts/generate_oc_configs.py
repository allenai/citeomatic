import json

import os

from citeomatic.config import App
from citeomatic.models.options import ModelOptions
from citeomatic.traits import Unicode, Enum
import copy


class GenerateOcConfigs(App):

    dataset_type = Enum(('dblp', 'pubmed', 'oc'), default_value='pubmed')
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
            ({'use_citations': False, 'use_selector_confidence': False},
             "{}.citation_ranker.canonical-extra_features.options.json".format(self.dataset_type)),
            ({'use_magdir': False}, "{}.citation_ranker.canonical-magdir.options.json".format(
                self.dataset_type)),
            ({'use_variable_margin': False},
             "{}.citation_ranker.canonical-var_margin.options.json".format(self.dataset_type)),
            ({
                 'use_metadata': False,
                 'use_authors': False,
                 'use_keyphrases': False,
                 'use_venue': False,
             }, "{}.citation_ranker.canonical-metadata.options.json".format(self.dataset_type)),
            ({'use_src_tgt_embeddings': True},
             "{}.citation_ranker.canonical-siamese.options.json".format(self.dataset_type)),
            ({'use_src_tgt_embeddings': False},
             "{}.citation_ranker.canonical-non_siamese.options.json".format(self.dataset_type)),
            ({'use_pretrained': True, 'enable_fine_tune': False},
             "{}.citation_ranker.canonical-pretrained_no_finetune.options.json".format(self.dataset_type)),
            ({'use_pretrained': True, 'enable_fine_tune': True},
             "{}.citation_ranker.canonical-pretrained_with_finetune.options.json".format(
                 self.dataset_type
             )),
            ({'use_sparse': False},
             "{}.citation_ranker.canonical-sparse.options.json".format(self.dataset_type)),
            ({'batch_size': 512},
             "{}.citation_ranker.canonical-large_batch.options.json".format(self.dataset_type)),
            ({'use_nn_negatives': False},
             "{}.citation_ranker.canonical-nn_negatives.options.json".format(self.dataset_type)),
            ({'embedding_type': 'cnn2'},
             "{}.citation_ranker.canonical+cnn.options.json".format(self.dataset_type))
        ]

        for change, filename in changes_file_list:
            self.write_change_to_file(filename=filename,
                                      base_options=base_config,
                                      change=change
                                      )

GenerateOcConfigs.run(__name__)
