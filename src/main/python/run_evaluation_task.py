#!.venv/bin/python3
import argparse
import subprocess
from os.path import exists
import os
import sys
import json
from pathlib import Path
import gzip

if 'src/main/python/' not in sys.path:
    sys.path.append('src/main/python/')

from run_file_processing import IncompletePools, load_all_runs, normalize_run
from evaluation_util import evaluate_on_pools


def normalize_identifier(identifier):
    return identifier.replace('/', '-')


def __parse_args():
    parser = argparse.ArgumentParser(description='Evaluate runs.')

    parser.add_argument('--taskNumber', type=int, required=True,
                        help='In case the task is read from a file (and is not specified via arguments), only the task with the number will be specified.')

    parser.add_argument('--taskDefinititionFile', type=str, required=True,
                        help='A file in jsonl format that specified the tasks to execute')

    return parser.parse_args()


def run_task(task):
    out_file = task['working_directory'] + '/' + task['out_file_name']
    if exists(out_file):
        return
    
    subprocess.check_output(['mkdir', '-p', os.path.abspath(str(Path(task['working_directory'] + '/' + task['out_file_name'])/ '..'))])
    
    qrel_file = 'src/main/resources/unprocessed/topics-and-qrels/qrels.robust04.txt'
    pooling = IncompletePools(pool_per_run_file=task['working_directory'] + '/processed/pool-documents-per-run-' + normalize_identifier(task['trec_identifier']) + '.json.gz')
    pools = {k:set(v) for k,v in pooling.create_incomplete_pools_for_run(task['run'])}
    
    eval_result = evaluate_on_pools(task['run'], qrel_file, pools, task['measure'])
    eval_result['task'] = task
    
    with open(out_file, 'w') as f:
        f.write(json.dumps(eval_result))


if __name__ == '__main__':
    args = __parse_args()
    tasks = [json.loads(i) for i in open(args.taskDefinititionFile)]
    run_task(tasks[args.taskNumber])

