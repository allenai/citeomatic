# Citeomatic

This is the source distribution for the [Citeomatic](citeomatic.semanticscholar.org) service.

## Setup

Use `pip` to setup the package and install dependencies:

```
pip install -e .
```

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

