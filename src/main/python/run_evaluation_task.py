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
                        choices=['run_task', 'run_task_on_qrels'])

    return parser.parse_args()


def task_already_executed(task):
    if type(task) == list:
        return all(task_already_executed(i) for i in task)
    
    out_file = task['working_directory'] + '/' + task['out_file_name']
    return exists(out_file)


def run_task(task):
    out_file = task['working_directory'] + '/' + task['out_file_name']
    
    subprocess.check_output(['mkdir', '-p', os.path.abspath(str(Path(task['working_directory'] + '/' + task['out_file_name'])/ '..'))])
    
    qrel_file = 'src/main/resources/unprocessed/topics-and-qrels/qrels.robust04.txt'
    pooling = IncompletePools(pool_per_run_file=task['working_directory'] + '/processed/pool-documents-per-run-' + normalize_identifier(task['trec_identifier']) + '.json.gz')
    pools = {k: v for k, v in pooling.create_incomplete_pools_for_run(task['run'])}
    
    eval_result = evaluate_on_pools(task['run'], qrel_file, pools, task['measure'])
    eval_result['task'] = task
    
    with open(out_file, 'w+') as f:
        f.write(json.dumps(eval_result) + '\n')


def run_task_on_qrels(task):
    out_file = task['working_directory'] + '/' + task['out_file_name']

    subprocess.check_output(
        ['mkdir', '-p', os.path.abspath(str(Path(task['working_directory'] + '/' + task['out_file_name']) / '..'))])

    qrel_file = 'src/main/resources/unprocessed/topics-and-qrels/qrels.core17.txt'

    eval_result = evaluate_on_original_pool_only(task['run'], qrel_file, task['measure'])
    eval_result['task'] = task

    with open(out_file, 'w+') as f:
        f.write(json.dumps(eval_result) + '\n')


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
                    print(f'Got exception during task {t}')
                    raise e
        else:
            method_to_execute = locals()[task['task_to_execute']]
            try:
                method_to_execute(task)
            except Exception as e:
                print(f'Got exception during task {task}')
                raise e
