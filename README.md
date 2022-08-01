# ECIR 23: Incomplete Judgments

## Setup Environment and run unit tests

Run `make` to get an overview of all possible make targets.
With `make clean .venv`, you can set up the environment and afterwards you can run `make tests` to run all unit tests to check your environment and the code.

## Add Our new Method as Plugin to trectools

TBD.

## Experiments reported in the Paper

TBD.


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

