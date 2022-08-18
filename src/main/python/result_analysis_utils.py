from glob import glob
import pandas as pd
import json
from tqdm import tqdm
from numpy import isnan

def load_evaluations(files):
    return load_raw_evaluations(files).groupby('run').apply(__process_df)


def __rename_pooling(pool):
    if pool.startswith('depth-10-pool-incomplete-for-'):
        return 'depth-10-incomplete'
    if pool == 'complete-pool-depth-10':
        return 'depth-10-complete'
    if pool == 'complete-pool-depth-20':
        return 'depth-20-complete'
    if pool.startswith('depth-20-pool-incomplete-for-'):
        return 'depth-20-incomplete'
    
    raise ValueError('I cant handle ' + str(pool))


def __rename_measure(m):
    if m == 'unjudged@10-UNJ@10':
        return 'unjudged@10'
    if m == 'condensed-ndcg@10-NDCG@10':
        return 'condensed-ndcg@10'
    if m == 'residual-ndcg@10-MIN-NDCG@10':
        return 'residual-ndcg@10-min'
    if m == 'residual-ndcg@10-MAX-NDCG@10':
        return 'residual-ndcg@10-max'
    if m == 'ndcg@10-NDCG@10':
        return 'ndcg@10'
    if m == 'unjudged@20-UNJ@20':
        return 'unjudged@20'
    if m == 'condensed-ndcg@20-NDCG@20':
        return 'condensed-ndcg@20'
    if m == 'residual-ndcg@20-MIN-NDCG@20':
        return 'residual-ndcg@20-min'
    if m == 'residual-ndcg@20-MAX-NDCG@20':
        return 'residual-ndcg@20-max'
    if m == 'ndcg@20-NDCG@20':
        return 'ndcg@20'
    if m == 'rbp@10-RBP@10':
        return 'rbp@10'
    if m == 'condensed-rbp@10-RBP@10':
        return 'condensed-rbp@10'
    if m == 'residual-rbp@10-MAX-RBP@10':
        return 'residual-rbp@10-max'
    if m == 'residual-rbp@10-MIN-RBP@10':
        return 'residual-rbp@10-min'
    if m == 'rbp@20-RBP@20':
        return 'rbp@20'
    if m == 'condensed-rbp@20-RBP@20':
        return 'condensed-rbp@20'
    if m == 'residual-rbp@20-MAX-RBP@20':
        return 'residual-rbp@20-max'
    if m == 'residual-rbp@20-MIN-RBP@20':
        return 'residual-rbp@20-min'
    if m.lower() in ['bs-1000-ndcg@10-mean', 'bs-1000-ndcg@10-q-01', 'bs-1000-ndcg@10-q-10', 'bs-1000-ndcg@10-q-15', 'bs-1000-ndcg@10-q-5', 'bs-1000-ndcg@10-q-25', 'bs-1000-ndcg@10-q-50', 'bs-1000-ndcg@10-q-75', 'bs-p-1000-ndcg@10-ndcg@10']:
        return m.lower()


def __process_row(df_row):
    df_row = df_row.to_dict()
    pool = __rename_pooling(df_row['pooling'])
    
    ret = {}
    for k, v in df_row.items():
        k = __rename_measure(k)
        if k is None:
            continue
            
        if type(v) is not str:
            continue

        k = (pool, k)
        assert k not in ret
        ret[k] = v

    return ret


def __process_df(df):
    ret = {}
    run = df.iloc[0]['run']
    for _, i in df.iterrows():
        assert i['run'] == run
        for k, v in __process_row(i).items():
            if k in ret:
                raise ValueError(f'Redundant data. I got multiple entries of {k}.')
            ret[k] = v

    ret['run'] = run
    return pd.DataFrame([ret])


def load_raw_evaluations(files):
    df = []

    for eval_file in files:
        df += [__load_eval_file(eval_file)]

    return pd.concat([i for i in df if i is not None])


def __load_eval_file(file_name, expected_queries=None, runs_to_skip=None):
    ret = []
    
    eval_result = json.load(open(file_name, 'r'))
    if runs_to_skip and eval_result['task']['run'] in RUNS_TO_SKIP:
        return None
    
    for pool_name, results in eval_result.items():
        if pool_name in ['task']:
            continue
        covered_queries = set()
        scores = {}
        
        for result in results:
            assert results[0]['run_file'] == result['run_file']
            for eval_measure in result.keys():
                if eval_measure in set(['run_file', 'query']):
                    continue
                measure_name = eval_result['task']['measure'] + '-' + eval_measure
                if measure_name not in scores:
                    scores[measure_name] = {}
                
                assert result['query'] not in scores[measure_name]
                covered_queries.add(result['query'])
                parsed_score = result[eval_measure]
                if type(parsed_score) is not list:
                    parsed_score = float(parsed_score)
                    parsed_score = parsed_score if not isnan(parsed_score) else 0
                
                scores[measure_name][result['query']] = parsed_score
        
        current_entry = {'run': results[0]['run_file'], 'pooling': pool_name}
        to_update = {k: json.dumps(v) for k,v in scores.items()}
        current_entry.update(to_update)
        
        ret += [current_entry]

    return pd.DataFrame(ret)

