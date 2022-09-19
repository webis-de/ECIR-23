import pandas as pd
from tqdm import tqdm
from result_analysis_utils import load_evaluations, load_cross_validation_results
import json
from os.path import exists
from glob import glob
from io import StringIO


class DataConstruction:
    def __init__(self, lower, actual, upper):
        self.lower = lower
        self.actual = actual
        self.upper = upper

    def construct_data_for_reconstruction_evaluation(self, data_per_row):
        ret = {}
        for topic in data_per_row.keys():
            assert topic not in ret

            ret[topic] = [self.transform(i) for i in data_per_row[topic]]

        return ret

    def transform(self, i):
        for k in [self.upper, self.actual, self.lower]:
            if k not in i:
                raise ValueError(f'I cand find the expected key {k}. Available are: {i.keys()}.')

        return {
            'system': i['system'],
            'ground_truth': i['ground_truth'],
            'prediction': {'lower': i[self.lower], 'actual': i[self.actual], 'upper': i[self.upper]}
        }


class InterpolationDataConstruction:
    def __init__(self, lower, actual, upper, interpolation):
        self.lower = lower
        self.actual = actual
        self.upper = upper
        self.interpolation = interpolation - 1.0

    def construct_data_for_reconstruction_evaluation(self, data_per_row):
        ret = {}
        for topic in data_per_row.keys():
            assert topic not in ret

            ret[topic] = [self.transform(i) for i in data_per_row[topic]]

        return ret

    def transform(self, i):
        interpolation_start = min(i[self.actual], i[self.upper])
        interpolation_upper = max(i[self.actual], i[self.upper])

        if self.interpolation >= 0:
            upper = interpolation_start + ((interpolation_upper - interpolation_start) * self.interpolation)
        else:
            interpolation_start = max(i[self.actual], i[self.lower])
            interpolation_lower = min(i[self.actual], i[self.lower])
            upper = interpolation_start + ((interpolation_start - interpolation_lower) * self.interpolation)

        return {
            'system': i['system'],
            'ground_truth': i['ground_truth'],
            'prediction': {'lower': i[self.lower], 'actual': upper, 'upper': upper}
        }


def to_ground_truth(i):
    return i['ground_truth']


def lower_bound(i):
    return min(i['prediction'].values())


def upper_bound(i):
    return max(i['prediction'].values())


class ReconstructionEvaluation:
    def __init__(self, threshold=0):
        self.__threshold = threshold

    def construct_reconstruction_scenarious(self, systems):
        ret = []

        for post_hoc_system in systems:
            reconstruction_scenario = {
                'post_hoc': {'system': post_hoc_system['system'], 'prediction': post_hoc_system['prediction']},
                'existing': [{'system': i['system'], 'ground_truth': i['ground_truth']}
                             for i in systems if i['system'] != post_hoc_system['system']]
            }

            reconstruction_scenario['expected_pairs'] = self.ground_truth_ranking(
                post_hoc_system, reconstruction_scenario['existing']
            )
            reconstruction_scenario['predicted_pairs'] = self.predicted_ranking(
                reconstruction_scenario['post_hoc'], reconstruction_scenario['existing']
            )
            ret += [reconstruction_scenario]

        return ret

    def precision(self, systems):
        reconstruction_scenarious = self.construct_reconstruction_scenarious(systems)
        predicted = 0
        correct_predicted = 0

        for scenario in reconstruction_scenarious:
            predicted += len(scenario['predicted_pairs'])
            correct_predicted += len(set(scenario['predicted_pairs']).intersection(scenario['expected_pairs']))

        if predicted == 0:
            return 0

        return correct_predicted/predicted

    def recall(self, systems):
        reconstruction_scenarious = self.construct_reconstruction_scenarious(systems)
        correct_predicted = 0
        expected = 0

        for scenario in reconstruction_scenarious:
            expected += len(scenario['expected_pairs'])
            correct_predicted += len(set(scenario['predicted_pairs']).intersection(scenario['expected_pairs']))

        if expected == 0:
            return 0

        return correct_predicted/expected

    def predicted_ranking(self, a, systems):
        ret = []

        for b in systems:
            diff_a_wins = lower_bound(a) - to_ground_truth(b)

            diff_b_wins = to_ground_truth(b) - upper_bound(a)

            if diff_a_wins >= self.__threshold:
                ret += [(a['system'], b['system'])]
            elif diff_b_wins >= self.__threshold:
                ret += [(b['system'], a['system'])]

        return set(ret)

    def ground_truth_ranking(self, a, systems):
        ret = []

        for b in systems:
            diff = to_ground_truth(a) - to_ground_truth(b)

            if a['system'] == b['system']:
                continue
            if diff >= self.__threshold:
                ret += [(a['system'], b['system'])]
            elif diff <= -1 * self.__threshold:
                ret += [(b['system'], a['system'])]

        return set(ret)


class AllApproachesDidNotParticipateInPoolingReconstructionEvaluation:
    def __init__(self, threshold=0):
        self.__threshold = threshold

    def precision(self, systems):
        ground_truth = set(self.ground_truth_pairs(systems))
        predicted = set(self.predicted_pairs(systems))

        if len(predicted) == 0:
            return 0

        return len(ground_truth.intersection(predicted))/len(predicted)

    def recall(self, systems):
        ground_truth = set(self.ground_truth_pairs(systems))
        predicted = set(self.predicted_pairs(systems))

        if len(ground_truth) == 0:
            return 1

        return len(ground_truth.intersection(predicted))/len(ground_truth)

    def ground_truth_pairs(self, systems):
        return self.__all_pairs(systems, to_ground_truth, to_ground_truth)

    def predicted_pairs(self, systems):
        return self.__all_pairs(systems, lower_bound, upper_bound)

    def __all_pairs(self, systems, lower, upper):
        ret = set()

        for s1 in systems:
            for s2 in systems:
                diff = lower(s1) - upper(s2)

                if s1['system'] != s2['system'] and diff > self.__threshold:
                    ret.add((s1['system'], s2['system']))

        return ret


def __persist_dfs(dfs, file_name):
    if exists(file_name):
        raise ValueError('')

    ret = {k: v.to_json(lines=True, orient='records') for k, v in dfs.items()}

    json.dump(ret, open(file_name, 'w'))


def __load_dfs_or_none(file_name):
    try:
        if exists(file_name):
            ret = json.load(open(file_name, 'r'))
            return {k:pd.read_json(StringIO(v), lines=True, orient='records') for k, v in ret.items()}
    except:
        pass


def load_preprocessed_reconstruction_or_from_cache(cache_file='processed-evaluation-results.json'):
    dfs = __load_dfs_or_none(cache_file)
    if dfs is not None:
        return dfs


    dfs = {}
    corpora = {
        'Robust04': ['trec13'],
        'CW09': ['trec18', 'trec19', 'trec20', 'trec21'],
        'CW12': ['trec22', 'trec23']
    }

    # We follow the preprocessing steps of Zobel et al. and include only the top 75% of the runs to mitigate the effects of low-performing runs
    runs_to_include = {'Robust04': 82, 'CW09': 24, 'CW12': 22}

    for corpus, trecs in corpora.items():
        df_for_corpus = []
        for trec in trecs:
            df_for_corpus += [
                load_df_reconstruction(trec, runs_to_include[corpus], True, min_unjudged=0, max_unjudged=0.7)]
        dfs[corpus] = pd.concat(df_for_corpus)

    __persist_dfs(dfs, cache_file)
    return __load_dfs_or_none(cache_file)


def load_df(trec):
    eval_predictions = glob(f'../resources/eval/trec-system-runs/{trec}/*.jsonl')

    # eval_predictions += list(load_cross_validation_results(open(f'../resources/processed/cross-validation-results/{trec}/bs-p-1000-ndcg@10-ndcg@10-results.jsonl'), depth=10, return_buffers=True))
    eval_predictions += list(load_cross_validation_results(open(
        f'../resources/processed/cross-validation-results/{trec}/bs-pool-dependent-1000-ndcg@10-ndcg@10-results.jsonl'),
                                                           depth=10, return_buffers=True))
    eval_predictions += list(load_cross_validation_results(open(
        f'../resources/processed/cross-validation-results/{trec}/bs-run-and-pool-dependent-1000-ndcg@10-ndcg@10-results.jsonl'),
                                                           depth=10, return_buffers=True))
    # eval_predictions += list(load_cross_validation_results(open(f'../resources/processed/cross-validation-results/{trec}/bs-run-and-pool-dependent2-1000-ndcg@10-ndcg@10-results.jsonl'), depth=10, return_buffers=True))
    eval_predictions += list(load_cross_validation_results(open(
        f'../resources/processed/cross-validation-results/{trec}/bs-run-dependent-1000-ndcg@10-ndcg@10-results.jsonl'),
                                                           depth=10, return_buffers=True))

    # eval_predictions += list(load_cross_validation_results(open(f'../resources/processed/cross-validation-results/{trec}/bs-p-1000-ndcg@10-ndcg@10-condensed-ndcg@10-results.jsonl'), depth=10, return_buffers=True))
    eval_predictions += list(load_cross_validation_results(open(
        f'../resources/processed/cross-validation-results/{trec}/bs-pool-dependent-1000-ndcg@10-ndcg@10-condensed-ndcg@10-results.jsonl'),
                                                           depth=10, return_buffers=True))
    eval_predictions += list(load_cross_validation_results(open(
        f'../resources/processed/cross-validation-results/{trec}/bs-run-and-pool-dependent-1000-ndcg@10-ndcg@10-condensed-ndcg@10-results.jsonl'),
                                                           depth=10, return_buffers=True))
    # eval_predictions += list(load_cross_validation_results(open(f'../resources/processed/cross-validation-results/{trec}/bs-run-and-pool-dependent2-1000-ndcg@10-ndcg@10-condensed-ndcg@10-results.jsonl'), depth=10, return_buffers=True))
    eval_predictions += list(load_cross_validation_results(open(
        f'../resources/processed/cross-validation-results/{trec}/bs-run-dependent-1000-ndcg@10-ndcg@10-condensed-ndcg@10-results.jsonl'),
                                                           depth=10, return_buffers=True))

    eval_predictions += list(load_cross_validation_results(
        open(f'../resources/processed/cross-validation-results/{trec}/condensed-ndcg@10-results.jsonl'), depth=10,
        return_buffers=True))
    eval_predictions += list(load_cross_validation_results(
        open(f'../resources/processed/cross-validation-results/{trec}/ndcg@10-results.jsonl'), depth=10,
        return_buffers=True))

    return load_evaluations(tqdm(eval_predictions))


def report_for_row(df_row, measure, depth):
    tmp = {'run': df_row['run'].split('/')[-1].replace('input.', '').replace('.gz', '')}
    measures = [
        ('unjudged', (f'depth-{depth}-incomplete', f'unjudged@{depth}')),
        (f'ground-truth-{measure}@{depth}', (f'depth-{depth}-complete', f'ndcg@{depth}')),
        (f'min-residual-{measure}@{depth}', (f'depth-{depth}-incomplete', f'residual-{measure}@{depth}-min')),
        (f'condensed-{measure}@{depth}', (f'depth-{depth}-incomplete', f'condensed-{measure}@{depth}')),
        (f'max-residual-{measure}@{depth}', (f'depth-{depth}-incomplete', f'residual-{measure}@{depth}-max')),
        (f'always-1', (f'depth-{depth}-incomplete', 'always-1')),
        (f'always-0', (f'depth-{depth}-incomplete', 'always-0')),
    ]

    for k, v in [('PBS-P', f'bs-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-RP', f'bs-run-and-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-R', f'bs-run-dependent-1000-{measure}@{depth}-{measure}@{depth}')]:
        for m in ['']:
            measures += [(f'{k}-RMSE{m}-{measure}@{depth}', (f'depth-{depth}-incomplete', f'pbs-rmse{m}-{v}'))]

    for k, v in [('PBS-P', f'bs-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-RP', f'bs-run-and-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-R', f'bs-run-dependent-1000-{measure}@{depth}-{measure}@{depth}')]:
        for m in ['0.8', '0.9', '0.95', '0.99']:
            internal_name = f'bs-ci-{m}-{v}-{v}-condensed-{measure}@{depth}'
            measures += [
                (f'{k}-CL-{m}-{measure}@{depth}', (f'depth-{depth}-incomplete', f'{internal_name}-{internal_name}'))]

    for k, v in [('PBS-P', f'bs-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-RP', f'bs-run-and-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-R', f'bs-run-dependent-1000-{measure}@{depth}-{measure}@{depth}')]:
        for m in ['-upper-bound-0.01', '-upper-bound-0.05', '-lower-bound-0.01', '-lower-bound-0.05']:
            part_name = f'pbs{m}-{v}'
            measures += [(f'{k}-RMSE{m}-{measure}@{depth}', (f'depth-{depth}-incomplete', f'{part_name}-{part_name}'))]

    for k, v in [('PBS-P', f'bs-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-RP', f'bs-run-and-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-R', f'bs-run-dependent-1000-{measure}@{depth}-{measure}@{depth}')]:
        measures += [(f'{k}-ML-{measure}@{depth}', (f'depth-{depth}-incomplete', f'bs-ml-{v}-{v}-bs-ml-{v}-{v}'))]

    for k, v in [('PBS-P', f'bs-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-RP', f'bs-run-and-pool-dependent-1000-{measure}@{depth}-{measure}@{depth}'),
                 ('PBS-R', f'bs-run-dependent-1000-{measure}@{depth}-{measure}@{depth}')]:
        for m in ['75', '90', '95']:
            measures += [(f'{k}-{m}-{measure}@{depth}', (f'depth-{depth}-incomplete', f'pbs-{m}-{v}-{v}-pbs-{m}-{v}-{v}'))]

    for i in ['upper-bound-0.01', 'upper-bound-0.05', 'lower-bound-0.01', 'lower-bound-0.05']:
        measures += [(f'gsd-{i}-condensed-{measure}@{depth}', (f'depth-{depth}-incomplete',
                                                               f'gsd-{i}-condensed-{measure}@{depth}-condensed-{measure}@{depth}-gsd-{i}-condensed-{measure}@{depth}-condensed-{measure}@{depth}')),
                     (f'gsd-{i}-{measure}@{depth}', (f'depth-{depth}-incomplete',
                                                     f'gsd-{i}-{measure}@{depth}-{measure}@{depth}-gsd-{i}-{measure}@{depth}-{measure}@{depth}'))]

    for display_name, m in measures:
        try:
            tmp[display_name] = json.loads(df_row[m])
        except:
            raise ValueError(f'Can not handle "{m}". Got {df_row.keys()}')

    ret = []

    for topic in tmp[f'ground-truth-{measure}@{depth}']:
        entry = {'run': tmp['run'], 'topic': topic}
        for k, v in tmp.items():
            if k in ['run']:
                continue

            if topic in v:
                entry[k] = v[topic]
        ret += [entry]

    return ret


def create_aggregated_df(df, measure, depth, loc, runs_to_keep):
    if df.iloc[loc]['run'] not in runs_to_keep:
        return None
    ret = pd.DataFrame([dict(i) for i in report_for_row(df.iloc[loc], measure, depth)])
    ret = ret.sort_values(f'ground-truth-{measure}@{depth}', ascending=False).reset_index()
    del ret['index']
    return ret


def data_for_reconstruction_experiments(df, trec, failsave, runs_to_keep, min_unjudged=0, max_unjudged=None):
    ret = {}
    for run in tqdm(range(len(df['run'].unique()))):
        try:
            tmp = create_aggregated_df(df, 'ndcg', 10, run, runs_to_keep)
            if tmp is None:
                continue
        except Exception as e:
            if not failsave:
                raise e

            continue

        if min_unjudged is not None:
            tmp = tmp[tmp['unjudged'] > min_unjudged].dropna()
        if max_unjudged is not None:
            tmp = tmp[tmp['unjudged'] < max_unjudged].dropna()

        if len(tmp) <= 1:
            continue

        measures_to_report = [('Condensed', 'condensed-ndcg@10'), ('Min-Residual', 'min-residual-ndcg@10'),
                              ('Max-Residual', 'max-residual-ndcg@10'), ('Always 1', 'always-1'),
                              ('Always 0', 'always-0'),
                              ]

        for i in ['', '-upper-bound-0.01', '-upper-bound-0.05', '-lower-bound-0.01', '-lower-bound-0.05']:
            for p in ['P-', 'R-', 'RP-']:
                measures_to_report += [(f'PBS-{p}RMSE{i}', f'PBS-{p}RMSE{i}-ndcg@10')]

        for p in ['P', 'R', 'RP']:
            measures_to_report += [(f'PBS-{p}-ML', f'PBS-{p}-ML-ndcg@10')]

            for m in ['75', '90', '95']:
                measures_to_report += [(f'PBS-{p}-{m}', f'PBS-{p}-{m}-ndcg@10')]

            for m in ['0.8', '0.9', '0.95', '0.99']:
                measures_to_report += [(f'PBS-{p}-CL-{m}', f'PBS-{p}-CL-{m}-ndcg@10')]

        for i in ['upper-bound-0.01', 'upper-bound-0.05', 'lower-bound-0.01', 'lower-bound-0.05']:
            measures_to_report += [(f'GSD-Condensed-{i}', f'gsd-{i}-condensed-ndcg@10'),
                                   (f'GSD-{i}', f'gsd-{i}-ndcg@10')]

        for _, i in tmp.iterrows():
            to_add = {
                'topic': i['topic'],
                'system': i['run'],
                'ground_truth': i['ground-truth-ndcg@10']
            }

            for k, v in measures_to_report:
                to_add[k] = i[v]

            if i['topic'] not in ret:
                ret[i['topic']] = []

            ret[i['topic']] += [to_add]

    return ret


def load_df_reconstruction(trec, num_runs_to_keep=100000, failsave=True, min_unjudged=0, max_unjudged=None):
    runs_to_keep = pd.read_json('../resources/processed/ndcg-at-10-effectiveness.jsonl', lines=True)
    runs_to_keep = runs_to_keep[runs_to_keep['position'] < num_runs_to_keep]
    runs_to_keep = set(runs_to_keep['run'].unique())
    df = load_df(trec)
    d = data_for_reconstruction_experiments(df, trec, failsave, runs_to_keep, min_unjudged=min_unjudged, max_unjudged=max_unjudged)
    reconstruction_approaches = {
        'Residuals': DataConstruction('Min-Residual', 'Condensed', 'Max-Residual'),
        'MinResiduals': DataConstruction('Min-Residual', 'Min-Residual', 'Min-Residual'),
        'Condensed': DataConstruction('Condensed', 'Condensed', 'Condensed'),
        'Min-Condensed': DataConstruction('Min-Residual', 'Condensed', 'Condensed'),

        'PBS-P-CL-0.80': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-P-CL-0.8'),
        'PBS-P-CL-0.90': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-P-CL-0.9'),
        'PBS-P-CL-0.95': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-P-CL-0.95'),
        'PBS-P-CL-0.99': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-P-CL-0.99'),

        'PBS-R-CL-0.80': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-R-CL-0.8'),
        'PBS-R-CL-0.90': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-R-CL-0.9'),
        'PBS-R-CL-0.95': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-R-CL-0.95'),
        'PBS-R-CL-0.99': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-R-CL-0.99'),

        'PBS-RP-CL-0.80': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-RP-CL-0.8'),
        'PBS-RP-CL-0.90': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-RP-CL-0.9'),
        'PBS-RP-CL-0.95': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-RP-CL-0.95'),
        'PBS-RP-CL-0.99': DataConstruction('Min-Residual', 'Min-Residual', 'PBS-RP-CL-0.99'),

        'PBS-RP-RMSE': DataConstruction('PBS-RP-RMSE', 'PBS-RP-RMSE', 'PBS-RP-RMSE'),
        'PBS-R-RMSE': DataConstruction('PBS-R-RMSE', 'PBS-R-RMSE', 'PBS-R-RMSE'),
        'PBS-P-RMSE': DataConstruction('PBS-P-RMSE', 'PBS-P-RMSE', 'PBS-P-RMSE'),

        'PBS-RP-ML': DataConstruction('PBS-RP-ML', 'PBS-RP-ML', 'PBS-RP-ML'),
        'PBS-R-ML': DataConstruction('PBS-R-ML', 'PBS-R-ML', 'PBS-R-ML'),
        'PBS-P-ML': DataConstruction('PBS-P-ML', 'PBS-P-ML', 'PBS-P-ML'),

        'Min-PBS-R-75': DataConstruction('Min-Residual', 'PBS-R-75', 'PBS-R-75'),
        'Min-PBS-P-75': DataConstruction('Min-Residual', 'PBS-P-75', 'PBS-P-75'),
        'Min-PBS-RP-75': DataConstruction('Min-Residual', 'PBS-RP-75', 'PBS-RP-75'),

        'Min-PBS-R-90': DataConstruction('Min-Residual', 'PBS-R-90', 'PBS-R-90'),
        'Min-PBS-P-90': DataConstruction('Min-Residual', 'PBS-P-90', 'PBS-P-90'),
        'Min-PBS-RP-90': DataConstruction('Min-Residual', 'PBS-RP-90', 'PBS-RP-90'),

        'Min-PBS-R-95': DataConstruction('Min-Residual', 'PBS-R-95', 'PBS-R-95'),
        'Min-PBS-P-95': DataConstruction('Min-Residual', 'PBS-P-95', 'PBS-P-95'),
        'Min-PBS-RP-95': DataConstruction('Min-Residual', 'PBS-RP-95', 'PBS-RP-95'),

        'Min-PBS-RP-ML': DataConstruction('Min-Residual', 'PBS-RP-ML', 'PBS-RP-ML'),
        'Min-PBS-R-ML': DataConstruction('Min-Residual', 'PBS-R-ML', 'PBS-R-ML'),
        'Min-PBS-P-ML': DataConstruction('Min-Residual', 'PBS-P-ML', 'PBS-P-ML'),

        'PBS-RP-0.01': DataConstruction('PBS-RP-RMSE-lower-bound-0.01', 'PBS-RP-RMSE', 'PBS-RP-RMSE-upper-bound-0.01'),
        'PBS-RP-0.05': DataConstruction('PBS-RP-RMSE-lower-bound-0.05', 'PBS-RP-RMSE', 'PBS-RP-RMSE-upper-bound-0.05'),

        'PBS-R-0.01': DataConstruction('PBS-R-RMSE-lower-bound-0.01', 'PBS-R-RMSE', 'PBS-R-RMSE-upper-bound-0.01'),
        'PBS-R-0.05': DataConstruction('PBS-R-RMSE-lower-bound-0.05', 'PBS-R-RMSE', 'PBS-R-RMSE-upper-bound-0.05'),

        'PBS-P-0.01': DataConstruction('PBS-P-RMSE-lower-bound-0.01', 'PBS-P-RMSE', 'PBS-P-RMSE-upper-bound-0.01'),
        'PBS-P-0.05': DataConstruction('PBS-P-RMSE-lower-bound-0.05', 'PBS-P-RMSE', 'PBS-P-RMSE-upper-bound-0.05'),

        'Interp-0.75-min-cond-max': InterpolationDataConstruction('Min-Residual', 'Condensed', 'Max-Residual', 0.75),
        'Interp-1.05-min-cond-max': InterpolationDataConstruction('Min-Residual', 'Condensed', 'Max-Residual', 1.05),
        'Interp-1.2-min-cond-max': InterpolationDataConstruction('Min-Residual', 'Condensed', 'Max-Residual', 1.2),
    }

    df_reconstruction = []

    reconstruction_eval = ReconstructionEvaluation()

    for approach_name, approach in reconstruction_approaches.items():
        for topic, topic_data in approach.construct_data_for_reconstruction_evaluation(d).items():
            df_reconstruction += [{
                'approach': approach_name,
                'topic': topic,
                'precision': reconstruction_eval.precision(topic_data),
                'recall': reconstruction_eval.recall(topic_data),
                'topic_data': topic_data
            }]

    df_reconstruction = pd.DataFrame(df_reconstruction)
    df_reconstruction['f1'] = df_reconstruction.apply(
        lambda i: 0 if (i['precision'] + i['recall']) == 0 else 2 * (i['precision'] * i['recall']) / (
                    i['precision'] + i['recall']), axis=1)

    return df_reconstruction


def calculate_error(min_value, max_value, actual, predicted, normalized):
    actual = min(max(actual, min_value), max_value)
    predicted = min(max(predicted, min_value), max_value)
    ret = actual - predicted

    if max_value < min_value:
        raise ValueError('Some programming error.')

    if normalized is True and max_value-min_value == 0:
        return 0

    if ret > 0 and normalized is True:
        min_value = predicted
        ret = actual

        return (ret-min_value)/(max_value-min_value)
    elif ret < 0 and normalized is True:
        max_value = predicted-min_value
        min_value = 0
        ret = predicted - actual

        return -(ret-min_value)/(max_value-min_value)

    return ret
