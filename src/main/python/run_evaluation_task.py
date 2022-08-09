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

def __parse_args():
    parser = argparse.ArgumentParser(description='Evaluate runs.')

    parser.add_argument('--taskNumber', type=int, required=True,
                        help='In case the task is read from a file (and is not specified via arguments), only the task with the number will be specified.')

    parser.add_argument('--taskDefinititionFile', type=str, required=True,
                        help='A file in jsonl format that specified the tasks to execute')

    return parser.parse_args()


def run_task(task):
    print(task)


if __name__ == '__main__':
    args = __parse_args()
    tasks = [json.loads(i) for i in open(args.taskDefinititionFile)]
    run_task(tasks[args.taskNumber])

