Experiments
===

Baseline
---
* BM25

`python citeomatic/scripts/evaluate.py --dataset_type <dataset> --candidate_selector_type bm25 --num_candidates 100`

where `<dataset>` is one of `pubmed`, `dblp`, `oc`.

For `oc` add the `--n_eval 5000` argument to test on 5000 test documents.

* Paper Embedder 


`python citeomatic/scripts/evaluate.py --dataset_type <dataset> --candidate_selector_type ann --paper_embedder_dir <model_dir>`

`model_dir` for 


| `dblp` |  `/net/nfs.corp/s2-research/citeomatic/naacl2017/comparison/dblp/models/paper_embedder/`

| `pubmed` |  `/net/nfs.corp/s2-research/citeomatic/naacl2017/comparison/pubmed/models/paper_embedder/`

| `oc` |  `/net/nfs.corp/s2-research/citeomatic/naacl2017/open_corpus/models/paper_embedder`

