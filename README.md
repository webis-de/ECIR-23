# ECIR 23: Bootstrapped nDCG Estimation in the Presence of Unjudged Documents

This repository contains the code and a TrecTools Plugin that allows to use bootstrapping to generate a distribution of nDCG scores by sampling judgments for the unjudged documents using run-based and/or pool-based priors.
This is helpful to estimate the true nDCG score if you have unjudged documents in your ranking and our experiments on three different corpora shows that bootstrapping works better than assuming unjudged documents are non-relevant or using condensed lists.

## Setup Environment and run unit tests

Run `make` to get an overview of all possible make targets.
With `make clean .venv`, you can set up the environment and afterwards you can run `make tests` to run all unit tests to check your environment and the code.

Running the tests with `make tests` should yield an output like this:

```
......................................................................................................................................................................
----------------------------------------------------------------------
Ran 166 tests in 13.630s

OK
```

## Add Our new Method as Plugin to TrecTools

Right now, the bootstrapping plugin is not yet integrated into the official main branch of TrecTools (but we will go for this upon acceptance).
However, you can already use our Bootstrapping approach with the following code (assumed you have this repository located in <THIS-REPOSITORY>):

```
from trectools import TrecRun, TrecQrel
import sys
sys.path.append('<THIS-REPOSITORY>/src/main/python')
from bootstrap_utils import evaluate_ndcg

run = TrecRun('path-to-run-file')
qrels = TrecQrel('path-to-qrel-file')

evaluate_ndcg(run, qrels, depth=10)
```

The notebook [src/main/ipynb/beir-evaluation.ipynb](src/main/ipynb/beir-evaluation.ipynb) provides use cases.

## Cached Data for Faster Experimentation

All judgment pools with different simulated incomplete judgments and normalized run files are created/downloaded using a single script (see [the section on "Download and Prepare all Resources"](#download-and-prepare-all-resources)).
For a fresh installation, this script runs for roughly 10 hours on a single CPU (so that we use adequate pauses while downloading run files etc.).
Hence, all experiments can run from cached metadata contained in this repository to allow fast experimentation and reproduction of the results in the paper.
To add new evaluation experiments, you must first run [this script](#download-and-prepare-all-resources) to get all the raw data.

## Download and Prepare all Resources

The run files used for the experiments are not publicly available but can be downloaded and processed (i.e., normalized by breaking score ties, potentially fixing ranks, etc.) by executing the following command (please replace `<PASSWORD>` and `<USER>` with your credentials to access the password protected area of past TREC results):

```
./src/main/resources/download-and-prepare-all-resources.py --user <USER> --password <PASSWORD> --directory src/main/resources/
```

## Experiments reported in the Paper

You can find all the notebooks for the experiments in the paper in [src/main/ipynb/](src/main/ipynb/).


