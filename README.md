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
./get-data.sh

```

## Running the model



## Training the model


## Evaluation

