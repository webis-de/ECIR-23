VENV_NAME?=.venv

.DEFAULT: help
help:
	@echo "make .venv"
	@echo "       setup virtual environment"
	@echo "make tests"
	@echo "       run all tests"
	@echo "make clean"
	@echo "       clean up everything"
	@echo "make run-evaluation-tasks"
	@echo "       run all evaluations"
	@echo "make run-evaluation-tasks-in-slurm"
	@echo "       run all evaluations in slurm (idempotent)"
	@echo "make jupyterlab"
	@echo "       start a jupyterlab server"


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

run-evaluation-tasks:
	@for i in {0..$(shell cat src/main/resources/all-tasks.jsonl|wc -l)}; do ./src/main/python/run_evaluation_task.py --taskDefinititionFile src/main/resources/all-tasks.jsonl --taskNumber  $$i; done

run-evaluation-tasks-in-slurm:
	sbatch --array=0-$(shell cat src/main/resources/all-tasks.jsonl|wc -l) src/main/sh/run-evaluation-tasks-in-slurm.sh

jupyterlab:
	.venv/bin/jupyter-lab --ip 0.0.0.0

