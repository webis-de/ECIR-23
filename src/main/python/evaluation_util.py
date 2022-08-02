from trectools import TrecRun, TrecQrel, TrecEval
import pandas as pd
from run_file_processing import normalize_run


def evaluate_on_pools(run_file, qrel_file, pools, measure):
    run = TrecRun(run_file)
    qrels = TrecQrel(qrel_file)
    
    ret = {
        'complete-pool': __evaluate_run_on_pool(run, qrels, measure, None, run_file)
    }
    
    for pool_name, pool in pools.items():
        assert pool_name not in ret
        ret[pool_name] = __evaluate_run_on_pool(run, qrels, measure, pool, run_file)
    
    return ret


def __evaluate_run_on_pool(run, qrels, measure, pool, run_file_name):
    qrels = __adjust_qrels_to_pool(qrels, pool)
    trec_eval = TrecEval(run, qrels)
    ret = None
    depth = int(measure.split('@')[-1])
    
    if measure.startswith('unjudged@'):
        ret = trec_eval.get_unjudged(depth, per_query=True)
    elif measure.startswith('ndcg@'):
        ret = trec_eval.get_ndcg(depth, per_query=True, removeUnjudged=False)
    elif measure.startswith('condensed-ndcg@'):
        ret = trec_eval.get_ndcg(depth, per_query=True, removeUnjudged=True)
    else:
        raise ValueError('Can not handle measure "' + measure +'".')

    return list(normalize_eval_output(ret, run_file_name))


def normalize_eval_output(df, run_file_name):
    for q_id, i in df.iterrows():
        i['run_file'] = run_file_name
        i['query'] = q_id
        yield dict(i)


def __create_max_residual_qrel(run, qrels, depth):
    return __fully_judged_qrels_with_default_rel(run, qrels, depth, qrels.qrels_data['rel'].max())


def __create_min_residual_qrel(run, qrels, depth):
    return __fully_judged_qrels_with_default_rel(run, qrels, depth, 0)


def __fully_judged_qrels_with_default_rel(run, qrels, depth, rel):
    additional_qrels = []

    judged_docs = {}
    for _, i in qrels.qrels_data.iterrows():
        if i['query'] not in judged_docs:
            judged_docs[i['query']] = set([])
        
        judged_docs[i['query']].add(i['docid'])
    
    for _, i in normalize_run(run).run_data.iterrows():
        if i['query'] not in judged_docs or i['docid'] not in judged_docs[i['query']]:
            additional_qrels += [{'query': i['query'],'q0': '0', 'docid': i['docid'], 'rel': rel}]

    ret = TrecQrel()
    ret.qrels_data = pd.concat([qrels.qrels_data.copy(), pd.DataFrame(additional_qrels)])

    return ret


def __adjust_qrels_to_pool(qrels, pool):
    ret = TrecQrel()

    if pool is None:
        ret.qrels_data = qrels.qrels_data.copy()
        return ret

    new_qrels_data = []

    for _, i in qrels.qrels_data.iterrows():
        if i['query'] in pool and i['docid'] in pool[i['query']]:
            new_qrels_data += [i]

    ret.qrels_data = pd.DataFrame(new_qrels_data)
    
    return ret
    
