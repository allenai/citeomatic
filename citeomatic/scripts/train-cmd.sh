#!/usr/bin/env bash

DATASET=$1
BASE_DIR=/net/nfs.corp/s2-research/citeomatic/naacl2017/
HYPEROPTS_DIR=${BASE_DIR}/hyperopts/${DATASET}/
VERSION=1
OPTIONS_JSON=test.json
SHA=`git log --format=%H -n 1`

# Step 1: Hyperopt ANN model
python citeomatic/scripts/train.py \
  --mode hyperopt \
  --dataset_type ${DATASET} \
  --use_pretrained False \
  --use_nn_negatives True \
  --max_evals_initial 75 \
  --max_evals_secondary 10 \
  --total_samples_initial 5000000 \
  --total_samples_secondary 50000000 \
  --samples_per_epoch 1000000 \
  --n_eval 500 \
  --model_name paper_embedder \
  --models_dir_base  ${HYPEROPTS_DIR} \
  --version "no_pretrained" &> ${HYPEROPTS_DIR}/${DATASET}.hyperopt.log

# Step 2: Train the best ANN model, save weights and ANN index
# TODO: Train the best ANN model found by hyperopt and save somewhere
python citeomatic/scripts/train.py \
  --dataset_type ${DATASET} \
  --mode train \
  --model_name paper_embedder \
  --models_dir_base  ${BASE_DIR} \
  --options_json ${OPTIONS_JSON} &> ${BASE_DIR}/${DATASET}/logs/paper_embedder.log


# Step 3: Use trained ANN model and hyperopt citeomatic model
echo "python citeomatic/scripts/train.py \
  --mode hyperopt \
  --dataset_type ${DATASET}  \
  --use_nn_negatives True \
  --max_evals_initial 75 \
  --max_evals_secondary 10 \
  --total_samples_initial 5000000 \
  --total_samples_secondary 50000000 \
  --samples_per_epoch 1000000  \
  --n_eval 500 \
  --model_name citation_ranker \
  --models_dir_base  ${HYPEROPTS_DIR} \
  --version \"ranker_hyperopt_${SHA}\" \
  --models_ann_dir ${BASE_DIR}/comparison/${DATASET}/models/paper_embedder/  &> ${BASE_DIR}/comparison/${DATASET}/logs/${DATASET}.${SHA}.ranker.hyperopt.log"

# Step 4: Train best citeomatic model with ANN from Step 2, save in final location