from tqdm import tqdm
from glob import glob
import pandas as pd
from parametrized_bootstrapping_model import ParametrizedBootstrappingModel, ReturnAlways1Model, ReturnAlways0Model,\
    LowerBoundFixedBudgetBootstrappingModel, UpperBoundFixedBudgetBootstrappingModel
from result_analysis_utils import load_ground_truth_data, load_evaluations, run_cross_validation
import subprocess

DEFAULT_SEARCH_SPACE = (0, 1, 2) + tuple(range(5, 96, 5)) + (98, 99, 100)


def cross_validation_experiment(trec, bootstrap_types, max_runs=100000, search_space=DEFAULT_SEARCH_SPACE):
    eval_df = load_evaluations(tqdm(glob('../resources/eval/trec-system-runs/' + trec + '/*.jsonl')))

    out_dir = f'cross-validation-results/{trec}'
    subprocess.check_output(['rm', '-rf', out_dir])

    for bootstrap_type in bootstrap_types:
        run_cross_validation_on_runs(min(max_runs, len(eval_df)), eval_df, bootstrap_type,
                                     trec, search_space, bootstrap_type == bootstrap_types[0])


def run_cross_validation_on_runs(runs, eval_df, bootstrap_type, trec, search_space, trivial_ones):
    ret = []
    for i in tqdm(list(range(runs)), bootstrap_type + ' on ' + trec):
        ground_truth_data = load_ground_truth_data(
            df=eval_df[eval_df['run'] == eval_df.iloc[i]['run']],
            ground_truth_measure='ndcg',
            depth=10,
            input_measure=bootstrap_type,
            random_state=3
        )

        out_dir = f'cross-validation-results/{trec}'
        out_file = f'{out_dir}/{bootstrap_type}-results.jsonl'

        models = [ParametrizedBootstrappingModel(i, search_space) for i in
                  ['rmse[0.8,1]', 'rmse', 'rmse[0.1,5]']]

        models += [LowerBoundFixedBudgetBootstrappingModel(0.01, search_space),
                   LowerBoundFixedBudgetBootstrappingModel(0.05, search_space),
                   UpperBoundFixedBudgetBootstrappingModel(0.01, search_space),
                   UpperBoundFixedBudgetBootstrappingModel(0.05, search_space),
                   ]

        subprocess.check_output(['mkdir', '-p', out_dir])
        subprocess.check_output(['touch', out_file])

        if trivial_ones:
            models += [ReturnAlways1Model(), ReturnAlways0Model()]

        for model in models:
            tmp = run_cross_validation(ground_truth_data, model)

            ret += [tmp]

            tmp = tmp.to_json(lines=True, orient='records')
            with open(out_file, 'a+') as f:
                f.write(tmp)

    return pd.concat(ret)
