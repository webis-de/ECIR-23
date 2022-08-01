# ECIR 23: Incomplete Judgments

## Setup Environment and run unit tests

Run `make` to get an overview of all possible make targets.
With `make clean .venv`, you can set up the environment and afterwards you can run `make tests` to run all unit tests to check your environment and the code.

## Download and Prepare all Resources

The run files used for the experiments are not publicly available but can be downloaded and processed by executing the following command (please replace `<PASSWORD>` and `<USER>` with your credentials to access the password protected area of past TREC results):

```
./src/main/resources/download-and-prepare-all-resources.py --user <USER> --password <PASSWORD> --directory src/main/resources/
```

