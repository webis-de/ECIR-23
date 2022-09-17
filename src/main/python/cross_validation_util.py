from tqdm import tqdm
from glob import glob
import pandas as pd
from result_analysis_utils import load_ground_truth_data, load_evaluations, run_cross_validation
import subprocess

DEFAULT_SEARCH_SPACE = (0, 1, 2) + tuple(range(5, 96, 5)) + (98, 99, 100)


def cross_validation_experiment(trec, input_measure, models, out_dir, clean, working_dir, max_runs=1000, failsave=True):
    eval_df = load_evaluations(tqdm(glob(working_dir + '/eval/trec-system-runs/' + trec + '/*.jsonl')))

    if clean:
        subprocess.check_output(['rm', '-rf', out_dir])

    for bootstrap_type in input_measure:
        run_cross_validation_on_runs(min(max_runs, len(eval_df)), eval_df, bootstrap_type, trec, models,
                                     failsave, out_dir)


def run_cross_validation_on_runs(runs, eval_df, input_measure, trec, models, failsave, out_dir):
    ret = []
    for i in tqdm(list(range(runs)), f'{input_measure} on {trec}'):
        try:
            ground_truth_data = load_ground_truth_data(
                df=eval_df[eval_df['run'] == eval_df.iloc[i]['run']],
                ground_truth_measure='ndcg',
                depth=10,
                input_measure=input_measure,
                random_state=3
            )
        except Exception as e:
            if not failsave:
                raise e

            continue

        if ground_truth_data is None:
            continue

        out_file = f'{out_dir}/{input_measure}-results.jsonl'
        if type(input_measure) is tuple or type(input_measure) is list:
            out_file = f'{out_dir}/{"-".join(input_measure)}-results.jsonl'

        subprocess.check_output(['mkdir', '-p', out_dir])
        subprocess.check_output(['touch', out_file])

        for model in models:
            tmp = run_cross_validation(ground_truth_data, model)

            ret += [tmp]

            tmp = tmp.to_json(lines=True, orient='records')
            with open(out_file, 'a+') as f:
                f.write(tmp)

    return pd.concat(ret)
