# Citeomatic

This is the source distribution for the [Citeomatic](citeomatic.semanticscholar.org) service and 
for the paper **Content-based Citation Recommendation**.

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

If you have access to the AI2 Corp network, you can set `location` to  `/net/nfs.corp/s2-research/citeomatic/public/`  

# Citeomatic Evaluation

This section details how to run the end-to-end system using pre-trained models for each dataset 
and evaluate performance of Citeomatic from the NAACL paper. The `--num_candidates` option below   
  
#### Pubmed
```
python citeomatic/scripts/evaluate.py --dataset_type pubmed --candidate_selector_type ann --split test --paper_embedder_dir data/comparison/pubmed/models/paper_embedder/ --num_candidates 10 --ranker_type neural --citation_ranker_dir data/comparison/pubmed/models/citation_ranker/

```

#### DBLP
```
python citeomatic/scripts/evaluate.py --dataset_type dblp --candidate_selector_type ann --split test --paper_embedder_dir data/comparison/dblp/models/paper_embedder/ --num_candidates 10 --ranker_type neural --citation_ranker_dir data/comparison/dblp/models/citation_ranker/
```

#### Open Corpus
```
python citeomatic/scripts/evaluate.py --dataset_type oc --candidate_selector_type ann --split test --paper_embedder_dir data/open_corpus/models/paper_embedder/ --num_candidates 5 --ranker_type neural --citation_ranker_dir data/open_corpus/models/citation_ranker/ --n_eval 20000
```


# BM25 Baseline

## Create Index
The steps described above download pre-built BM25 indexes, neural models etc. and allow you to execute the evaluation method.

Build BM25 indexes for a given corpus

```
python citeomatic/scripts/evaluate.py --dataset_type <dataset name> --candidate_selector_type bm25 --split test --ranker_type none --num_candidates 10 
```

Modify `CreateBM25Index` to change the way the BM25 index is built. We use the [whoosh](https://pypi.python.org/pypi/Whoosh/) package to build the BM25 index.
To change the way the index is queried, change the `fetch_candidates` implementation in `BM25CandidateSelector`

This script will create an index at this location: `data/bm25_index/<dataset name>/`

## Evaluate

```
python citeomatic/scripts/evaluate.py --dataset_type <dataset>   --candidate_selector_type bm25 --split test --ranker_type none --num_candidates 10
```

2. Re-Create SQLite DB for dataset

This following scripts will create an index at this location: `data/db/<dataset name>.sqlite.db`

  * For the DBLP and Pubmed datasets:
	```
	python citeomatic/scripts/convert_kdd_to_citeomatic.py --dataset_name <dataset name>
	```

  * For the open corpus dataset:
	```
	python citeomatic/scripts/convert_open_corpus_to_citeomatic.py
	```

The SQLite DB is used to speed-up retrieving documents for a particular document id. 

3. The main script to train and tune hyperparameters for various models is `train.py`. Usage:

	```
	python train.py [options]
	```

  * General Parameters:
	  * `--mode` (Required): The mode to run the `train.py` script in. Possible values: `train` or 
	  `hyperopt`. The `train` mode will train a single model and save to a given location. The 
	  `hyperopt` mode will run hyperparamter-optimization and return the best found model.
	  * `--dataset_type`: Dataset to use. Possible values: `dblp` (default), `pubmed` or `oc`
	  * `--model_name`: Possible values: `paper_embedder` (default) or `citation_ranker`
	  
  * Parameters specific to Hyperparameter Optimization
	  * `max_evals_initial`: No. of models to train in the first phase. Our hyperparameter 
	  optimization method runs in two steps. In the first step, a large number of models are run 
	  for a few epochs and the best performing 10 models are run for more number of epochs in the
	   second phase.
	  * `max_evals_secondary`: No. of models to train in the second phase. Best 
	  `max_evals_secondary` models from Phase 1 are trained for a longer time
	  * `total_samples_initial`: No. of samples to train first phase models on
	  * `total_samples_secondary`: No. of samples to train second phase models on
	  * `models_dir_base`: Base directory to store hyperopt results in 
	  * `--n_eval`: No. of validation examples to evaluate a trained model
	  * `run_identifier`: A string to identify the experiment
	  * `version`: Version string to be appended to the directory used to store model in
  
  * Parameters specific to Training a single model
      * `hyperopts_results_pkl`: Path to the `.pkl` file generated by the hyperopt mode
      * `options_json`: Optional json file containing all options required to train a model
      
    Refer to the `ModelOptions` class for more options.

3. We use the [hyperopt](https://github.com/hyperopt/hyperopt) package to tune hyperparameters. 
Use the following command to run the hyperopt on a particular dataset

  * Paper Embedder Model 

```
python citeomatic/scripts/train.py   --mode hyperopt   --dataset_type <dataset> --n_eval 500 --model_name paper_embedder   --models_dir_base  data/hyperopts/<dataset>/ --version <version> &> data/hyperopts/dblp/dblp
.paper_embedder.hyperopt.log
```

  * Citation Ranker Model
  


4. 