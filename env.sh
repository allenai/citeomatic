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

conda create -n ${CONDAENV} -y python==3.5.2 numpy scikit-learn notebook scikit-learn spacy pandas cython pytest || true

echo "Activating Conda Environment ----->"
source ~/anaconda3/bin/activate ${CONDAENV}

pip install -r requirements.in

python setup.py install