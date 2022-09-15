from trectools import TrecRun, TrecQrel, TrecEval
import pandas as pd
from run_file_processing import normalize_run
from copy import deepcopy
from probability_estimation import CountProbabilityEstimator, RunIndependentCountProbabilityEstimator, PoissonEstimator, RunAndPoolDependentProbabilityEstimator
from bootstrap_util import BootstrappEval, FullyIndependentBootstrappingStrategey
from pool_bootstrap_util import PoolAndRunIndependentBootstrappingStrategey, ProbabilityEstimatedBootstrappingStrategey


def evaluate_on_pools(run_file, qrel_file, pools, measure):
    run = TrecRun(run_file)
    qrels = TrecQrel(qrel_file)
    ret = {}
    
    for pool_name, pool in pools.items():
        assert pool_name not in ret
        ret[pool_name] = __evaluate_run_on_pool(run, qrels, measure, pool, run_file)
    
    return ret


def evaluate_on_original_pool_only(run_file, qrel_file, measure):
    run = TrecRun(run_file)
    qrels = TrecQrel(qrel_file)

    return {'complete-pool-depth-all': __evaluate_trec_eval_on_pool(TrecEval(run, qrels), measure, run_file)}


def __qrels_to_ir_measures(qrels):
    return qrels.qrels_data.copy().rename(columns={'docid': 'doc_id', 'query': 'query_id', 'rel': 'relevance'})


def __run_to_ir_measures(run):
    return run.run_data.copy().rename(columns={'docid': 'doc_id', 'query': 'query_id'})


def __run_to_ir_measures_unjudged_removed(run, qrels, depth):
    run = run.copy()
    qrels = qrels.copy()

    onlyjudged = pd.merge(run, qrels[["query", "docid", "rel"]], how="left")
    onlyjudged = onlyjudged[~onlyjudged["rel"].isnull()]
    onlyjudged = onlyjudged[["query", "q0", "docid", "rank", "score", "system"]]
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
    return __evaluate_trec_eval_on_pool(TrecEval(run, qrels), measure, run_file_name)


def __evaluate_trec_eval_on_pool(trec_eval,  measure, run_file_name):
    ret = None
    depth = int(measure.split('@')[-1])
        
    if measure.startswith('unjudged@'):
        ret = trec_eval.get_unjudged(depth, per_query=True)
    elif measure.startswith('ndcg@'):
        ret = trec_eval.get_ndcg(depth, per_query=True, removeUnjudged=False)
    elif measure.startswith('rbp@'):
        ret = __eval_rbp(trec_eval.run, trec_eval.qrels, depth, removeUnjudged=False)
    elif measure.startswith('condensed-rbp@'):
        ret = __eval_rbp(trec_eval.run, trec_eval.qrels, depth, removeUnjudged=True)
    elif measure.startswith('condensed-ndcg@'):
        ret = trec_eval.get_ndcg(depth, per_query=True, removeUnjudged=True)
    elif measure.startswith('residual-ndcg@'):
        max_eval = __create_residual_trec_eval(trec_eval.run, trec_eval.qrels, depth, residual_type='max', adjust_idcg=False).get_ndcg(depth, per_query=True)
        min_eval = __create_residual_trec_eval(trec_eval.run, trec_eval.qrels, depth, residual_type='min', adjust_idcg=False).get_ndcg(depth, per_query=True)
        
        min_eval = min_eval.rename(columns={'NDCG@' + str(depth): 'MIN-NDCG@' + str(depth)}, errors='raise')
        max_eval = max_eval.rename(columns={'NDCG@' + str(depth): 'MAX-NDCG@' + str(depth)}, errors='raise')
        
        ret = min_eval.join(max_eval)
    elif measure.startswith('residual-rbp@'):
        max_eval = __create_residual_trec_eval(trec_eval.run, trec_eval.qrels, depth, residual_type='max', adjust_idcg=True)
        min_qrels = __create_residual_trec_eval(trec_eval.run, trec_eval.qrels, depth, residual_type='min', adjust_idcg=True)
        
        max_eval = __eval_rbp(max_eval.run, max_eval.qrels, depth, removeUnjudged=False)
        
        min_eval =  __eval_rbp(min_qrels.run, min_qrels.qrels, depth, removeUnjudged=False)
        
        min_eval = min_eval.rename(columns={'RBP@' + str(depth): 'MIN-RBP@' + str(depth)}, errors='raise')
        max_eval = max_eval.rename(columns={'RBP@' + str(depth): 'MAX-RBP@' + str(depth)}, errors='raise')
        
        ret = min_eval.join(max_eval)
    elif measure.startswith('bs-1000-ndcg@'):
        bs_eval = BootstrappEval(f'ndcg@{depth}', FullyIndependentBootstrappingStrategey(trec_eval.qrels))
        ret = bs_eval.bootstrap(trec_eval.run, trec_eval.qrels, f'ndcg@{depth}', repeat=1000, seed=None)
    elif measure.startswith('bs-p-1000-ndcg@10'):
        bs_eval = BootstrappEval(f'ndcg@{depth}', PoolAndRunIndependentBootstrappingStrategey(trec_eval.qrels))
        ret = bs_eval.bootstrap(trec_eval.run, trec_eval.qrels, f'ndcg@{depth}', repeat=1000, seed=None)
    elif measure.startswith('bs-pool-dependent-1000-ndcg@10'):
        bs_strategy = ProbabilityEstimatedBootstrappingStrategey(trec_eval.qrels, CountProbabilityEstimator())
        bs_eval = BootstrappEval(f'ndcg@{depth}', bs_strategy)
        ret = bs_eval.bootstrap(trec_eval.run, trec_eval.qrels, f'ndcg@{depth}', repeat=1000, seed=None)
    elif measure.startswith('bs-run-dependent-1000-ndcg@10'):
        bs_strategy = ProbabilityEstimatedBootstrappingStrategey(
            trec_eval.qrels, RunIndependentCountProbabilityEstimator()
        )
        bs_eval = BootstrappEval(f'ndcg@{depth}', bs_strategy)
        ret = bs_eval.bootstrap(trec_eval.run, trec_eval.qrels, f'ndcg@{depth}', repeat=1000, seed=None)
    elif measure.startswith('bs-run-and-pool-dependent-1000-ndcg@10'):
        bs_strategy = ProbabilityEstimatedBootstrappingStrategey(
            trec_eval.qrels,
            PoissonEstimator(to_add=1, p_add_from_pool_given_unjudged=0.05, lower_p=0.025, upper_p=0.075)
        )
        bs_eval = BootstrappEval(f'ndcg@{depth}', bs_strategy)
        ret = bs_eval.bootstrap(trec_eval.run, trec_eval.qrels, f'ndcg@{depth}', repeat=1000, seed=None)
    elif measure.startswith('bs-run-and-pool-dependent2-1000-ndcg@10'):
        bs_strategy = ProbabilityEstimatedBootstrappingStrategey(
            trec_eval.qrels,
            RunAndPoolDependentProbabilityEstimator()
        )
        bs_eval = BootstrappEval(f'ndcg@{depth}', bs_strategy)
        ret = bs_eval.bootstrap(trec_eval.run, trec_eval.qrels, f'ndcg@{depth}', repeat=1000, seed=None)
    
    else:
        raise ValueError('Can not handle measure "' + measure +'".')

    return list(normalize_eval_output(ret, run_file_name))


def __docs_for_max_residual_trec_eval(run_for_topic, qrels_for_topic, min_score):
    pools = PoolAndRunIndependentBootstrappingStrategey(qrels_for_topic)\
        .available_qrels_for_topic(run_for_topic, run_for_topic.iloc[0]['query'])
    ret = []
    last_score = 1000000
    
    for k, v in pools.items():
        if k < min_score:
            continue
        if last_score < k:
            raise ValueError('Wrong sorting...')
        
        ret += v
    
    return ret


def __create_residual_trec_eval(run, qrels, depth, residual_type, adjust_idcg):
    if residual_type not in ['max', 'min']:
        raise ValueError('Invalid type ' + residual_type)
    
    run = normalize_run(run, depth)
    
    if residual_type == 'min':
        return TrecEval(run, qrels)
    
    new_run = []
    additional_qrels = []
    
    for topic in run.topics():
        run_for_topic = run.run_data[run.run_data['query'] == topic]
        pool_bs = FullyIndependentBootstrappingStrategey(qrels.qrels_data)
        unjudged_documents = set(pool_bs.unjudged_documents(run_for_topic))
        qrels_for_topic = qrels.qrels_data[qrels.qrels_data['query'] == topic]
        
        min_score = 1
        
        if adjust_idcg:
            min_score = qrels_for_topic['rel'].max()
        
        remaining_positive_docs = __docs_for_max_residual_trec_eval(run_for_topic, qrels_for_topic, min_score)

        for _, i in run_for_topic.iterrows():
            i = deepcopy(dict(i))
            if i['docid'] in unjudged_documents and len(remaining_positive_docs) == 0 and adjust_idcg:
                additional_qrels += [{'query': topic, 'q0': 0, 'docid': i['docid'], 'rel': min_score}]
            elif i['docid'] in unjudged_documents and len(remaining_positive_docs) > 0:
                i['docid'] = remaining_positive_docs.pop()
            
            new_run += [i]

    additional_qrels = pd.concat([qrels.qrels_data, pd.DataFrame(additional_qrels)])
    qrels = TrecQrel()
    qrels.qrels_data = additional_qrels

    run = TrecRun()
    run.run_data = pd.DataFrame(new_run)

    return TrecEval(run, qrels)


def normalize_eval_output(df, run_file_name):
    if type(df) is dict:
        for query, eval_results in df.items():
            for k, v in eval_results.items():
                yield {'run_file': run_file_name, 'query': query, k: v}
        return
            
    for q_id, i in df.iterrows():
        i['run_file'] = run_file_name
        i['query'] = q_id
        yield dict(i)


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
    
