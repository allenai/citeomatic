#!/usr/bin/env bash

DATASET=dblp
BASE_DIR=/net/nfs.corp/s2-research/citeomatic/naacl2017/
HYPEROPTS_DIR=${BASE_DIR}/hyperopts/${DATASET}/
VERSION=1

# Step 1: Hyperopt ANN model
python citeomatic/scripts/train_hyperopt.py \
  --dataset_type ${DATASET} \
  --models_dir_base ${HYPEROPTS_DIR} \
  --mode hyperopt \
  --model_name model_ann \
  --max_evals_initial 5 \
  --max_evals_secondary 1 \
  --run_identifier hyperopt_model_ann_${VERSION}

# Step 2: Train the best ANN model, save weights and ANN index
# TODO: Train the best ANN model found by hyperopt and save somewhere
python citeomatic/scripts/train_hyperopt.py \
  --dataset_type ${DATASET} \
  --models_dir_base ${BASE_DIR}/${DATASET} \
  --mode train \
  --hyperopts_pkl ${HYPEROPTS_DIR}/hyperopt_model_ann_${VERSION}/hyperopt_results.pickle \
  --model_name model_ann


# Step 3: Use trained ANN model and hyperopt citeomatic model
#python citeomatic/scripts/train_hyperopt.py \
#  --dataset_type ${DATASET} \
#  --models_dir ${MODEL_DIR} \
#  --models_ann_dir ${MODEL_DIR_ANN} \
#  --mode hyperopt \
#  --model_name model_full

# Step 4: Train best citeomatic model with ANN from Step 2, save in final location