#!/bin/bash

CONDAENV=ai2-citeomatic

if ! (which conda); then
	echo "No `conda` installation found.  Installing..."
	if [[ $(uname) == "Darwin" ]]; then
	  wget --continue http://repo.continuum.io/archive/Anaconda3-4.3.1-MacOSX-x86_64.sh
	  bash Anaconda3-4.3.1-MacOSX-x86_64.sh -b
	else
	  wget --continue http://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
	  bash Anaconda3-4.3.1-Linux-x86_64.sh -b
	fi
fi

export PATH=$HOME/anaconda3/bin:$PATH

source ~/anaconda3/bin/deactivate ${CONDAENV}

conda remove -y --name ${CONDAENV} --all

conda create -n ${CONDAENV} -y python==3.5.2 numpy scikit-learn notebook scikit-learn spacy pandas cython pytest || true

echo "Activating Conda Environment ----->"
source ~/anaconda3/bin/activate ${CONDAENV}

pip install -r requirements.in

TF_VERSION=1.2.0

if [ "$TEAM_CITY_AGENT" ]; then
    pip install tensorflow==${TF_VERSION}
else
	pip install tensorflow-gpu==${TF_VERSION}
fi

sudo apt-get install protobuf-compiler
python -m spacy download en
python setup.py develop