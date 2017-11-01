#!/usr/bin/env bash

DATASET=dblp
BASE_DIR=/net/nfs.corp/s2-research/citeomatic/naacl2017/
HYPEROPTS_DIR=${BASE_DIR}/hyperopts/${DATASET}/
VERSION=1
OPTIONS_JSON=test.json

# Step 1: Hyperopt ANN model
python citeomatic/scripts/train.py \
  --mode hyperopt \
  --dataset_type pubmed \
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
#python citeomatic/scripts/train_hyperopt.py \
#  --dataset_type ${DATASET} \
#  --models_dir ${MODEL_DIR} \
#  --models_ann_dir ${MODEL_DIR_ANN} \
#  --mode hyperopt \
#  --model_name citation_ranker

# Step 4: Train best citeomatic model with ANN from Step 2, save in final location