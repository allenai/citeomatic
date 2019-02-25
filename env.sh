#!/bin/bash

set -x

CONDAENV=ai2-citeomatic

if ! (which conda); then
	echo "No conda installation found.  Install Conda or Miniconda for your OS."
	exit 1;
fi

source deactivate ${CONDAENV}

conda remove -y --name ${CONDAENV} --all

conda create -n ${CONDAENV} -y python==3.5.2 numpy scikit-learn notebook scikit-learn spacy pandas cython pytest

echo "Activating Conda Environment ----->"
source activate ${CONDAENV}

pip install -r requirements.in

TF_VERSION=1.12.0
if (which nvidia-smi); then
    HAS_GPU=true
fi

if [ "$HAS_GPU" = true ]; then
    pip install tensorflow-gpu==${TF_VERSION}
else
	pip install tensorflow==${TF_VERSION}
fi

if [[ $(uname) == "Linux" ]]; then
  sudo apt-get install -y protobuf-compiler
fi

python -m spacy download en
python setup.py develop