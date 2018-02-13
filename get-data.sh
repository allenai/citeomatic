#!/usr/bin/env bash
LOC=$1

aws s3 sync s3://ai2-s2-research-public/citeomatic-2018-02-12/ ${LOC}

# The `data` directory is hard coded as the base directory in common.py
# If a different location on disk is provided, we create a symlink from ./data/ -> $LOC

if [[ ${LOC} != 'data' ]]; then
  ln -s $LOC 'data'
fi