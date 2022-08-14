from trectools import TrecRun, TrecQrel, TrecEval
import pandas as pd
from run_file_processing import normalize_run


def evaluate_on_pools(run_file, qrel_file, pools, measure):
    run = TrecRun(run_file)
    qrels = TrecQrel(qrel_file)
    ret = {}
    
    for pool_name, pool in pools.items():
        assert pool_name not in ret
        ret[pool_name] = __evaluate_run_on_pool(run, qrels, measure, pool, run_file)
    
    return ret


def __qrels_to_ir_measures(qrels):
    return qrels.qrels_data.copy().rename(columns={'docid':'doc_id', 'query': 'query_id', 'rel': 'relevance'})


def __run_to_ir_measures(run):
    return run.run_data.copy().rename(columns={'docid':'doc_id', 'query': 'query_id'})


def __run_to_ir_measures_unjudged_removed(run, qrels, depth):
    run = run.copy()
    qrels = qrels.copy()

    onlyjudged = pd.merge(run, qrels[["query","docid","rel"]], how="left")
    onlyjudged = onlyjudged[~onlyjudged["rel"].isnull()]
    onlyjudged = onlyjudged[["query","q0","docid","rank","score","system"]]
    d = TrecRun()
    d.run_data = onlyjudged
    onlyjudged = normalize_run(d, depth).run_data

    return onlyjudged.rename(columns={'docid':'doc_id', 'query': 'query_id'})


def __eval_rbp(run, qrels, depth, removeUnjudged):
    import ir_measures
    from ir_measures import RBP

    if removeUnjudged:
        run = __run_to_ir_measures_unjudged_removed(run.run_data, qrels.qrels_data, depth)
    else:
        run = __run_to_ir_measures(normalize_run(run, depth))
    
    qrels = __qrels_to_ir_measures(qrels)
    
    ret = []
    for metric in ir_measures.iter_calc([RBP(rel=1, p=0.8)], qrels, run):
        ret += [{f"RBP@{depth}": metric.value, "query": metric.query_id}]
    
    ret = pd.DataFrame(ret)
    ret = ret.set_index('query')
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
    elif measure.startswith('rbp@'):
        ret = __eval_rbp(run, qrels, depth, removeUnjudged=False)
    elif measure.startswith('condensed-rbp@'):
        ret = __eval_rbp(run, qrels, depth, removeUnjudged=True)
    elif measure.startswith('condensed-ndcg@'):
        ret = trec_eval.get_ndcg(depth, per_query=True, removeUnjudged=True)
    elif measure.startswith('residual-ndcg@'):
        max_qrels = __create_max_residual_qrel(run, qrels, depth)
        min_qrels = __create_min_residual_qrel(run, qrels, depth)
        
        trec_eval = TrecEval(run, max_qrels)
        max_eval = trec_eval.get_ndcg(depth, per_query=True)
        
        trec_eval = TrecEval(run, min_qrels)
        min_eval = trec_eval.get_ndcg(depth, per_query=True)
        
        min_eval = min_eval.rename(columns={'NDCG@' + str(depth): 'MIN-NDCG@' + str(depth)}, errors='raise')
        max_eval = max_eval.rename(columns={'NDCG@' + str(depth): 'MAX-NDCG@' + str(depth)}, errors='raise')
        
        ret = min_eval.join(max_eval)
    elif measure.startswith('residual-rbp@'):
        max_qrels = __create_max_residual_qrel(run, qrels, depth)
        min_qrels = __create_min_residual_qrel(run, qrels, depth)
        
        max_eval = __eval_rbp(run, max_qrels, depth, removeUnjudged=False)
        
        min_eval =  __eval_rbp(run, min_qrels, depth, removeUnjudged=False)
        
        min_eval = min_eval.rename(columns={'RBP@' + str(depth): 'MIN-RBP@' + str(depth)}, errors='raise')
        max_eval = max_eval.rename(columns={'RBP@' + str(depth): 'MAX-RBP@' + str(depth)}, errors='raise')
        
        ret = min_eval.join(max_eval)
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
    
    for _, i in normalize_run(run, depth).run_data.iterrows():
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
    
