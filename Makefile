VENV_NAME?=.venv

.DEFAULT: help
help:
	@echo "make .venv"
	@echo "       setup virtual environment"
	@echo "make tests"
	@echo "       run all tests"
	@echo "make clean"
	@echo "       clean up everything"


clean:
	@rm -Rf ${VENV_NAME}

# Requirements are in setup.py, so whenever setup.py is changed, re-run installation of dependencies.
.venv:
	@test -d $(VENV_NAME) || python3.6 -m venv $(VENV_NAME)
	@sh -c ". $(VENV_NAME)/bin/activate && \
		python3.6 -m pip install --upgrade pip && \
		python3.6 -m pip install wheel && \
		python3.6 -m pip install -r requirements.txt"

tests: .venv
	@PYTHONPATH=src/main/python .venv/bin/nosetests src/test/python

run-tasks-in-slurm:
	sbatch --array=0-100 src/main/sh/evaluate-all-runs-with-slurm.sh

