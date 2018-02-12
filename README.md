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


## Running the model

For convenience, we provide a pre-built model and service you can quickly run to
get started on your own.

```
docker build -t citeomatic-server .
docker run -it -p5000 citeomatic-server
```

## Training the model

Training the default model takes a few hours on a modern GPU, and about a day on
a CPU machine.  To download the open corpus data (6M documents), train, and
evaluate the default model, run:

```
python -m citeomatic.tasks TrainDefaultModel
```

## Evaluation

