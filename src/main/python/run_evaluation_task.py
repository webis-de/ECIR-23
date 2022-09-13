#!.venv/bin/python3
import argparse
import subprocess
from os.path import exists
import os
import sys
import json
from pathlib import Path
if 'src/main/python/' not in sys.path:
    sys.path.append('src/main/python/')
from run_file_processing import IncompletePools
from evaluation_util import evaluate_on_pools, evaluate_on_original_pool_only
from cross_validation_util import cross_validation_experiment, DEFAULT_SEARCH_SPACE
from parametrized_bootstrapping_model import ParametrizedBootstrappingModel, ReturnAlways1Model, ReturnAlways0Model,\
    LowerBoundFixedBudgetBootstrappingModel, UpperBoundFixedBudgetBootstrappingModel

SHARED_TASKS = {
    'trec-system-runs/trec18/web.adhoc': {
        'seed_url': 'https://trec.nist.gov/results/trec18/web.adhoc.input.html',
        'qrels': 'src/main/resources/unprocessed/topics-and-qrels/qrels.web.1-50.txt'
    },

    'trec-system-runs/trec19/web.adhoc': {
        'seed_url': 'https://trec.nist.gov/results/trec19/web.adhoc.input.html',
        'qrels': 'src/main/resources/unprocessed/topics-and-qrels/qrels.web.51-100.txt'
    },

    'trec-system-runs/trec20/web.adhoc': {
        'seed_url': 'https://trec.nist.gov/results/trec20/web.adhoc.input.html',
        'qrels': 'src/main/resources/unprocessed/topics-and-qrels/qrels.web.101-150.txt'
    },

    'trec-system-runs/trec21/web.adhoc': {
        'seed_url': 'https://trec.nist.gov/results/trec21/web.adhoc.input.html',
        'qrels': 'src/main/resources/unprocessed/topics-and-qrels/qrels.web.151-200.txt'
    },

    'trec-system-runs/trec22/web.adhoc': {
        'seed_url': 'https://trec.nist.gov/results/trec22/web.adhoc.input.html',
        'qrels': 'src/main/resources/unprocessed/topics-and-qrels/qrels.web.201-250.txt'
    },

    'trec-system-runs/trec23/web.adhoc': {
        'seed_url': 'https://trec.nist.gov/results/trec23/web.adhoc.input.html',
        'qrels': 'src/main/resources/unprocessed/topics-and-qrels/qrels.web.251-300.txt'
    }
}


def normalize_identifier(identifier):
    return identifier.replace('/', '-')


def __parse_args():
    parser = argparse.ArgumentParser(description='Evaluate runs.')

    parser.add_argument('--taskNumber', type=int, required=True,
                        help='In case the task is read from a file (and is not specified via arguments),' +
                             'only the task with the number will be specified.')

    parser.add_argument('--taskDefinititionFile', type=str, required=True,
                        help='A file in jsonl format that specified the tasks to execute')

    parser.add_argument('--methodToExecute', type=str, required=False, default=None,
                        choices=['run_task', 'run_task_on_qrels', 'run_cross_validation'])

    return parser.parse_args()


def task_already_executed(task):
    if type(task) == list:
        return all(task_already_executed(i) for i in task)
    
    out_file = task['working_directory'] + '/' + task['out_file_name']
    return exists(out_file)


def run_task(task):
    out_file = task['working_directory'] + '/' + task['out_file_name']
    
    subprocess.check_output(['mkdir', '-p', os.path.abspath(str(Path(task['working_directory'] + '/' + task['out_file_name'])/ '..'))])
    
    qrel_file = SHARED_TASKS[task['trec_identifier']]['qrels']
    pooling = IncompletePools(pool_per_run_file=task['working_directory'] + '/processed/pool-documents-per-run-' + normalize_identifier(task['trec_identifier']) + '.json.gz')
    pools = {k: v for k, v in pooling.create_incomplete_pools_for_run(task['run'])}

    if task['measure'].lower().startswith('bs-'):
        # Skip unused computations
        pools = {k: v for k, v in pools.items() if k.lower().startswith('depth-10-pool-incomplete-for')}
    
    eval_result = evaluate_on_pools(task['run'], qrel_file, pools, task['measure'])
    eval_result['task'] = task
    
    with open(out_file, 'w+') as f:
        f.write(json.dumps(eval_result) + '\n')


def run_task_on_qrels(task):
    out_file = task['working_directory'] + '/' + task['out_file_name']

    subprocess.check_output(
        ['mkdir', '-p', os.path.abspath(str(Path(task['working_directory'] + '/' + task['out_file_name']) / '..'))])

    qrel_file = SHARED_TASKS[task['trec_identifier']]['qrels']
    eval_result = evaluate_on_original_pool_only(task['run'], qrel_file, task['measure'])
    eval_result['task'] = task

    with open(out_file, 'w+') as f:
        f.write(json.dumps(eval_result) + '\n')


def run_cross_validation(task):
    out_dir = task['working_directory'] + '/' + task['out_file_name']

    cross_validation_experiment(
        trec=task['trec'],
        input_measure=['bs-p-1000-ndcg@10-ndcg@10', 'bs-run-and-pool-dependent-1000-ndcg@10-ndcg@10',
                       'bs-pool-dependent-1000-ndcg@10-ndcg@10', 'bs-run-dependent-1000-ndcg@10-ndcg@10'],
        models=[ParametrizedBootstrappingModel('rmse', DEFAULT_SEARCH_SPACE),
                LowerBoundFixedBudgetBootstrappingModel(0.01, DEFAULT_SEARCH_SPACE),
                LowerBoundFixedBudgetBootstrappingModel(0.05, DEFAULT_SEARCH_SPACE),
                UpperBoundFixedBudgetBootstrappingModel(0.01, DEFAULT_SEARCH_SPACE),
                UpperBoundFixedBudgetBootstrappingModel(0.05, DEFAULT_SEARCH_SPACE),
                ],
        out_dir=out_dir,
        clean=True
    )

    cross_validation_experiment(
        trec=task['trec'],
        input_measure=['ndcg@10'],
        models=[ReturnAlways1Model(), ReturnAlways0Model()],
        out_dir=out_dir,
        clean=False
    )


if __name__ == '__main__':
    args = __parse_args()
    tasks = [json.loads(i) for i in open(args.taskDefinititionFile)]
    
    task = tasks[args.taskNumber]

    if not task_already_executed(task):
        if type(task) == list:
            for t in task:
                method_to_execute = locals()[t['task_to_execute']]
                try:
                    method_to_execute(t)
                except Exception as e:
                    print(f'Got error/exception during task {t}')
                    raise e
        else:
            method_to_execute = locals()[task['task_to_execute']]
            try:
                method_to_execute(task)
            except Exception as e:
                print(f'Got error/exception during task {task}')
                raise e
