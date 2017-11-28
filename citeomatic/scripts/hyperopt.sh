#!/usr/bin/env bash

DATASET=$1
BASE_DIR=/net/nfs.corp/s2-research/citeomatic/naacl2017/
SHA=`git log --pretty=format:'%H' -n 1`
echo ${SHA}
echo "python citeomatic/scripts/train.py \
  --mode hyperopt \
  --dataset_type ${DATASET} \
  --n_eval 500 \
  --model_name paper_embedder \
  --models_dir_base  ${BASE_DIR}/hyperopts/${DATASET} \
  --version ${SHA} &> /tmp/${DATASET}.${SHA}.hyperopt.log"

echo "\n\n"

echo "python citeomatic/scripts/train.py \
  --mode hyperopt \
  --dataset_type ${DATASET} \
  --n_eval 500 \
  --model_name citation_ranker \
  --models_dir_base  ${BASE_DIR}/hyperopts/${DATASET} \
  --version ${SHA} \
  --models_ann_dir $BASE_DIR/comparison/$DATASET/models/paper_embedder/ \
  &> $BASE_DIR/comparison/$DATASET/logs/${DATASET}.citation_ranker.${SHA}.hyperopt.log"