#!/usr/bin/env bash

DATASET=$1
BASE_DIR=/net/nfs.corp/s2-research/citeomatic/naacl2017/
SHA=`git log --pretty=format:'%H' -n 1`
echo ${SHA}
echo "python citeomatic/scripts/train.py \
  --mode hyperopt \
  --dataset_type ${DATASET} \
  --use_nn_negatives True \
  --max_evals_initial 75 \
  --max_evals_secondary 10 \
  --total_samples_initial 5000000 \
  --total_samples_secondary 50000000 \
  --samples_per_epoch 1000000 \
  --n_eval 500 \
  --model_name paper_embedder_${SHA} \
  --models_dir_base  ${BASE_DIR}/hyperopts/${DATASET} \
  --version ${SHA} &> /tmp/${DATASET}.hyperopt.log"