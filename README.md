# Citeomatic

This is the source distribution for the [Citeomatic](citeomatic.semanticscholar.org) service.

## Clone the repo
```
git clone git@github.com:allenai/citeomatic.git
```

## Setup direnv (Optional)
Citeomatic uses direnv to activate the `ai2-citeomatic` conda environment whenever you `cd` into the root directory. Alternatively, you can activate the conda environment yourself. 
1. To install `direnv` on:
##### Ubuntu: 
```
sudo apt-get install direnv
```
##### Mac OSX:
```
brew install direnv
```

Then:

`cd citeomatic/`

`direnv allow .`


## Setup

Use the provided `env.sh` script to setup the package and install dependencies:

```
./env.sh
```

Verify that you now have the `ai2-citeomatic` conda environment installed and activated

## Download data

Execute the data download script
```
./get-data.sh <location>

```
The script will internally add a symlink from a local `data` directory to the provided `<location>`. Alternatively, you can provide `data/` as the location to save all required data. 

## Running the model

#### Pubmed
```
python citeomatic/scripts/evaluate.py --dataset_type pubmed --candidate_selector_type ann --split valid --paper_embedder_dir data/comparison/pubmed/models/paper_embedder/ --num_candidates 10 --ranker_type neural --citation_ranker_dir data/comparison/pubmed/models/citation_ranker/ --n_eval 1000

```

#### DBLP
```
python citeomatic/scripts/evaluate.py --dataset_type dblp --candidate_selector_type ann --split test --paper_embedder_dir data/comparison/dblp/models/paper_embedder/ --num_candidates 10 --ranker_type neural --citation_ranker_dir data/comparison/dblp/models/citation_ranker/
```

#### Open Corpus
```
python citeomatic/scripts/evaluate.py --dataset_type oc --candidate_selector_type ann --split test --paper_embedder_dir /net/nfs.corp/s2-research/citeomatic/naacl2017/open_corpus/models/paper_embedder/ --num_candidates 5 --ranker_type neural --n_eval 20000 --citation_ranker_dir /net/nfs.corp/s2-research/citeomatic/naacl2017/open_corpus/models/stddev/citation_ranker_canonical_9f6f4b2aefae04f4f58e4de0f9522f6083fd981e_1/
```

## Training the model


## Evaluation



# Starting with scratch

The steps described above download pre-built BM25 indexes, neural models etc. and allow you to execute the evaluation method. If you intend to modify any of these components, please follow the instructions below:

1. Build BM25 indexes for a given corpus

```
python citeomatic/scripts/create_bm25_index.py --dataset_name <dataset name>
```

This script will create an index at this location: `data/bm25_index/<dataset name>/`

2. Re-Create SQLite DB for dataset
```
python citeomatic/scripts/convert_kdd_to_citeomatic.py --dataset_name <dataset name>
```
This script will create an index at this location: `data/db/<dataset name>.sqlite.db`

For the open corpus dataset:
```
python citeomatic/scripts/convert_open_corpus_to_citeomatic.py
```

3. We use the hyperopt package to tune hyperparameters. Use the following command to run the hyperopt for citeomatic

```
python citeomatic/scripts/train.py --mode hyperopt --dataset_type ${DATASET} --max_evals_initial 75 --max_evals_secondary 10 --total_samples_initial 5000000 --total_samples_secondary 50000000 --samples_per_epoch 1000000 --n_eval 500 --model_name paper_embedder --models_dir_base  ${HYPEROPTS_DIR} &> logs/hyperopt.paper_embedder.{DATASET}.log
```

