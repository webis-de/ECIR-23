import pandas as pd
import json
from numpy import isnan
from copy import deepcopy
from io import StringIO


def load_cross_validation_results(input_data, depth, return_buffers=False):
    df = pd.read_json(input_data, lines=True, orient='records')
    
    ret = {}
    
    for _, i in df.iterrows():
        if i['model'] not in ret:
            ret[i['model']] = {}

        if i['run'] not in ret[i['model']]:
            ret[i['model']][i['run']] = {}

        if i['query'] in ret[i['model']][i['run']]:
            raise ValueError('Duplicated stuff')

        ret[i['model']][i['run']][i['query']] = {i['model'] + '-' + i['measures']['x']: i['y_prediction'], "run_file": i['run'], "query": i['query']}
        
    final_ret = []
    
    for model in ret:
        for run in ret[model]:
            measure = list(ret[model][run].values())[0]
            measure = [i for i in measure if i not in ['run_file', 'query']]
            assert len(measure) == 1
            measure = measure[0]
            
            final_ret += [{f"depth-{depth}-pool-incomplete-for-TBD": list(ret[model][run].values()), "task": {"run": run,  "measure": measure}}]      
    
    if not return_buffers:
        return final_ret
    else:
        return (StringIO(json.dumps(i)) for i in final_ret)

def run_cross_validation(ground_truth_data, model):
    ret = []
    
    for split in ground_truth_data['split'].unique():
        train = ground_truth_data[ground_truth_data['split'] != split]
        train_queries = train['query'].unique()
        test = ground_truth_data[ground_truth_data['split'] == split]
        
        model.fit(list(train['x']), list(train['y']))
        
        for _, i in test.iterrows():
            i = deepcopy(i.to_dict())
            if i['query'] in train_queries:
                raise ValueError('Invalid splits')
            
            i['y_prediction'] = model.predict([i['x']])
            assert len(i['y_prediction']) == 1
            i['y_prediction'] = i['y_prediction'][0]
            i['model'] = str(model)
            
            ret += [i]

    return pd.DataFrame(ret)


def load_ground_truth_data(df, ground_truth_measure, depth, input_measure, random_state=None, failsave=True):
    if type(df) == list:
        return load_ground_truth_data(load_evaluations(df), ground_truth_measure, depth, input_measure, random_state)
    
    if len(df) != 1:
        raise ValueError('I expect exactly one run for the construction of the ground-truth data. Got ' + str(len(df)))
    
    df = df.iloc[0].to_dict()

    if type(input_measure) == tuple:
        x_values = {}
        for measure in input_measure:
            x_value = json.loads(df[(f'depth-{depth}-incomplete', measure)])

            for k, v in x_value.items():
                if k not in x_values:
                    x_values[k] = []
                x_values[k] += [v]

        df = {
            'run': df['run'],
            'measures': {'x': input_measure, 'y': f'{ground_truth_measure}@{depth}'},
            'x': x_values,
            'y': json.loads(df[(f'depth-{depth}-complete', f'{ground_truth_measure}@{depth}')])
        }

        for k, v in df['x'].items():
            if not(failsave) and len(v) != len(input_measure):
                raise ValueError(f'Cant handle missing values for {k}: {v}')
    else:
        df = {
            'run': df['run'],
            'measures': {'x': input_measure, 'y': f'{ground_truth_measure}@{depth}'},
            'x': json.loads(df[(f'depth-{depth}-incomplete', input_measure)]),
            'y': json.loads(df[(f'depth-{depth}-complete', f'{ground_truth_measure}@{depth}')])
        }
    ret = []
   
    for query_id, y in df['y'].items():
        if isnan(y):
            continue
        if query_id not in df['x']:
            continue
        if type(df['x'][query_id]) is not list and isnan(df['x'][query_id]):
            continue
        if type(df['x'][query_id]) is list and any([type(i) is not list and isnan(i) for i in df['x'][query_id]]):
            continue
        
        ret += [{'run': df['run'], 'query': query_id, 'x': df['x'][query_id], 'y': y, 'measures': df['measures']}]
    
    ret = pd.DataFrame(ret)
    splits = __train_test_split(ret, random_state)
    ret['split'] = ret['query'].apply(lambda i: splits[i])
    
    return ret


def __train_test_split(df, random_state):
    from sklearn.model_selection import train_test_split
    split_1, split_2, _, _ = train_test_split(df['query'], [None]*len(df), test_size=0.5, random_state=random_state)
    
    ret = {}
    
    for q in split_1:
        ret[q] = 0
    
    for q in split_2:
        ret[q] = 1

    return ret


def load_evaluations(files):
    return load_raw_evaluations(files).groupby('run').apply(__process_df)


def __rename_pooling(pool):
    to_rename = {
        'depth-10-pool-incomplete-for-': 'depth-10-incomplete',
        'complete-pool-depth-10': 'depth-10-complete',
        'complete-pool-depth-20': 'depth-20-complete',
        'depth-20-pool-incomplete-for-': 'depth-20-incomplete',
        'complete-pool-depth-all': 'complete-pool'
    }
    
    if pool in to_rename.values():
        return pool
    if pool in to_rename:
        to_rename[pool]
    
    for k,v in to_rename.items():
        if pool.startswith(k):
            return v
    
    raise ValueError('I cant handle ' + str(pool))


def __rename_measure(m):
    identical_measures = [
        'rmse', 'bs-1000-ndcg@10-mean', 'bs-1000-ndcg@10-q-01', 'bs-1000-ndcg@10-q-10',
        'bs-1000-ndcg@10-q-15', 'bs-1000-ndcg@10-q-5', 'bs-1000-ndcg@10-q-25',
        'bs-1000-ndcg@10-q-50', 'bs-1000-ndcg@10-q-75', 'bs-p-1000-ndcg@10-ndcg@10', 
        'bs-run-and-pool-dependent-1000-ndcg@10-ndcg@10', 'bs-pool-dependent-1000-ndcg@10-ndcg@10',
        'bs-run-dependent-1000-ndcg@10-ndcg@10', 'bs-1000-ndcg@10-ndcg@10'
    ]
    
    to_rename = {i.lower():i.lower() for i in identical_measures}
    to_rename.update({
        'unjudged@10-UNJ@10': 'unjudged@10',
        'condensed-ndcg@10-NDCG@10': 'condensed-ndcg@10',
        'residual-ndcg@10-MIN-NDCG@10': 'residual-ndcg@10-min',
        'residual-ndcg@10-MAX-NDCG@10': 'residual-ndcg@10-max',
        'ndcg@10-NDCG@10': 'ndcg@10',
        'unjudged@20-UNJ@20': 'unjudged@20',
        'condensed-ndcg@20-NDCG@20': 'condensed-ndcg@20',
        'residual-ndcg@20-MIN-NDCG@20': 'residual-ndcg@20-min',
        'residual-ndcg@20-MAX-NDCG@20': 'residual-ndcg@20-max',
        'ndcg@20-NDCG@20': 'ndcg@20',
        'rbp@10-RBP@10': 'rbp@10',
        'condensed-rbp@10-RBP@10': 'condensed-rbp@10',
        'residual-rbp@10-MAX-RBP@10': 'residual-rbp@10-max',
        'residual-rbp@10-MIN-RBP@10': 'residual-rbp@10-min',
        'rbp@20-RBP@20': 'rbp@20',
        'condensed-rbp@20-RBP@20': 'condensed-rbp@20',
        'residual-rbp@20-MAX-RBP@20': 'residual-rbp@20-max',
        'residual-rbp@20-MIN-RBP@20': 'residual-rbp@20-min',
    })

    if m in to_rename.values():
        return m
    if m.lower() in to_rename.values():
        return m.lower()
    if m in to_rename:
        return to_rename[m]
    if m.lower() in to_rename:
        return to_rename[m.lower()]
    if 'rmse' in m.lower():
        return m.lower()
    if 'bs-run-and-pool-dependent2' in m.lower():
        return m.lower()
    if 'pbs-upper-bound' in m.lower():
        return m.lower()
    if 'pbs-lower-bound' in m.lower():
        return m.lower()
    if 'always-1' in m.lower():
        return 'always-1'
    if 'always-0' in m.lower():
        return 'always-0'
    if m.lower().startswith('gsd-'):
        return m.lower()
    if m.lower().startswith('always-'):
        return m.lower()
    if m.lower().startswith('bs-ci-'):
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
    
    if len(ret) == 0:
        raise ValueError('This can not happen. Worked on ' + json.dumps(df_row))
    
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

    ret = pd.concat([i for i in df if i is not None])
    ret['run'] = ret['run'].apply(lambda i: i.split('/')[-1].replace('input.', '').replace('.gz', ''))
    
    return ret


def __load_eval_file(file_name, expected_queries=None, runs_to_skip=None):
    ret = []
    
    if type(file_name) is str:
        eval_result = json.load(open(file_name, 'r'))
    else:
        eval_result = json.load(file_name)

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
                if eval_measure in {'run_file', 'query'}:
                    continue
                measure_name = eval_result['task']['measure']
                if 'rmse' not in measure_name:
                    measure_name +=  '-' + eval_measure
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

