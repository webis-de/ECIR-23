#!.venv/bin/python3
import argparse
import subprocess
from os.path import exists
import sys
import json
from pathlib import Path
import gzip
from numpy import array_split

if 'src/main/python/' not in sys.path:
    sys.path.append('src/main/python/')

from deduplication_util import ClueWebRunDeduplication
from run_file_processing import IncompletePools, load_all_runs, normalize_run

SEED_URLS = {
    'trec-system-runs/trec13/robust': 'https://trec.nist.gov/results/trec13/robust.input.html',

    # Got the query reformulation runs thankfully from Timo Breuer
    'uqv-runs-ecir21-paper': 'https://th-koeln.sciebo.de/s/PJm1l9JnoeHvkB7',

    'trec-system-runs/trec18/web.adhoc': 'https://trec.nist.gov/results/trec18/web.adhoc.input.html',
    'trec-system-runs/trec19/web.adhoc': 'https://trec.nist.gov/results/trec19/web.adhoc.input.html',
    'trec-system-runs/trec20/web.adhoc': 'https://trec.nist.gov/results/trec20/web.adhoc.input.html',
    'trec-system-runs/trec21/web.adhoc': 'https://trec.nist.gov/results/trec21/web.adhoc.input.html',
    'trec-system-runs/trec22/web.adhoc': 'https://trec.nist.gov/results/trec22/web.adhoc.input.html',
    'trec-system-runs/trec23/web.adhoc': 'https://trec.nist.gov/results/trec23/web.adhoc.input.html',
}

BACKUP_DIR = '/mnt/ceph/storage/data-in-progress/data-research/web-search/web-search-trec/'


def normalize_identifier(identifier):
    return identifier.replace('/', '-')


def __parse_args():
    parser = argparse.ArgumentParser(description='Download and prepare all resources.')

    parser.add_argument('--password', type=str, required=True,
                        help='The password to access the protected area')

    parser.add_argument('--user', type=str, required=True,
                        help='The user to access the protected area')

    parser.add_argument('--directory', type=str, required=True,
                        help='The directory to download and process all resources.')

    return parser.parse_args()


def __rsync_from_backup_if_possible(trec_identifier, working_directory):
    source_dir = BACKUP_DIR + trec_identifier
    if not exists(source_dir):
        return

    subprocess.check_output(['rsync', '-ar', source_dir, str(Path(working_directory + '/' + trec_identifier) / '..')])


def __download_runs(target_directory, seed_url_for_runs, trec_user, trec_password):
    print(f'Download using {target_directory}, {seed_url_for_runs}, {trec_user}, {trec_password}.')
    # subprocess.check_output(['bash', '-c', 'src/main/resources/trec-results-downloader.py']))
    pass


def download(trec_identifier, trec_url, trec_user, trec_password, working_directory):
    target_directory = working_directory + '/' + trec_identifier
    subprocess.check_output(['mkdir', '-p', target_directory])

    __rsync_from_backup_if_possible(trec_identifier, working_directory)
    __download_runs(target_directory, trec_url, trec_user, trec_password)


def __normalize_runs(trec_identifier, working_directory):
    target_dir = working_directory + '/processed/normalized-runs/' + trec_identifier
    
    if exists(target_dir):
        return

    subprocess.check_output(['mkdir', '-p', target_dir])
    for run_name, run in load_all_runs(working_directory + '/unprocessed/' + trec_identifier).items():

        if '/web.adhoc' in trec_identifier:
            run = ClueWebRunDeduplication().deduplicated(run)

        run = normalize_run(run)
        run_name = target_dir + '/' + run_name.split('/')[-1]
        run.run_data.to_csv(run_name, sep=' ', header=False, index=False)


def __create_pools_per_run(trec_identifier, working_directory):
    output_file = working_directory + 'processed/pool-documents-per-run-' + normalize_identifier(trec_identifier) + '.json.gz'
    if exists(output_file):
        return IncompletePools(pool_per_run_file=output_file)

    pooling = IncompletePools(working_directory + '/processed/normalized-runs/' + trec_identifier, working_directory + '/processed/trec-system-runs-groups.json', trec_identifier).pool_per_runs()
    
    with gzip.open(output_file, 'w') as f:
        f.write(json.dumps(pooling, sort_keys=True, indent=4).encode('UTF-8'))

    return IncompletePools(pool_per_run_file=output_file)


def __create_evaluation_tasks(trec_identifier, working_directory):
    output_file = working_directory + 'all-tasks.jsonl'
    ret = []

    for run_name in load_all_runs(working_directory + '/processed/normalized-runs/' + trec_identifier).keys():
        for measure in [
            'unjudged@10', 'unjudged@20',
            'ndcg@10', 'condensed-ndcg@10', 'residual-ndcg@10',
            'ndcg@20',  'condensed-ndcg@20',  'residual-ndcg@20',
            'rbp@10', 'condensed-rbp@10', 'residual-rbp@10',
            'rbp@20', 'condensed-rbp@20', 'residual-rbp@20',
            'bs-1000-ndcg@10', 'bs-p-1000-ndcg@10', 'bs-pool-dependent-1000-ndcg@10',
            'bs-run-dependent-1000-ndcg@10', 'bs-run-and-pool-dependent-1000-ndcg@10',
        ]:
            ret += [{'run': run_name, 'measure': measure, 'trec_identifier': trec_identifier,
                     'working_directory': working_directory,
                     'out_file_name': 'eval/' + trec_identifier + '-' + str(len(ret)) + '.jsonl',
                     'task_to_execute': 'run_task'
                     }]
        
    with open(output_file, 'w') as f:
        for split in array_split(ret, 900):
            f.write(json.dumps(list(split)) + '\n')


def prepare(trec_identifier, working_directory):
    __normalize_runs(trec_identifier, working_directory)
    __create_pools_per_run(trec_identifier, working_directory)
    __create_evaluation_tasks(trec_identifier, working_directory)


if __name__ == '__main__':
    args = __parse_args()
    
    for directory, seed_url in SEED_URLS.items():
        print('Download and prepare: ' + directory)
        download(directory, seed_url, args.user, args.password, args.directory + '/unprocessed/')
        print('   [\033[92mo\033[0m] Download done.')
        prepare(directory, args.directory)
        print('   [\033[92mo\033[0m] Preparation done.')
