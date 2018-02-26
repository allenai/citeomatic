#!/usr/bin/env bash
set -x
set -e

LOC=$1

wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/citeomatic-2018-02-12.tar.gz -P ${LOC}
tar -xzvf ${LOC}/citeomatic-2018-02-12.tar.gz -C ${LOC}

# The `data` directory is hard coded as the base directory in common.py
# If a different location on disk is provided, we create a symlink from ./data/ -> $LOC

if [[ ${LOC} != 'data' ]]; then
  ln -s $LOC/citeomatic-2018-02-12/ 'data'
fi