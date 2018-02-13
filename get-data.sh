#!/usr/bin/env bash
LOC=$1

aws s3 sync s3://ai2-s2-research-public/citeomatic-2018-02-12/ ${LOC}