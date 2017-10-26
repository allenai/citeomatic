#!/usr/bin/env bash

protoc --python_out=./ citeomatic/schema.proto
python citeomatic/scripts/convert_kdd_to_citeomatic.py --dataset_name dblp
python citeomatic/scripts/convert_kdd_to_citeomatic.py --dataset_name pubmed
python citeomatic/scripts/convert_open_corpus_to_citeomatic.py