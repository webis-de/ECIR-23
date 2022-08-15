from run_file_processing import normalize_run
import pandas as pd
import json
from copy import deepcopy
from trectools import TrecQrel, TrecEval, TrecRun


def __substitute_pools_for_topic(run_for_topic, qrels_for_topic):
    unjudged = pd.merge(run_for_topic, qrels_for_topic[["query","docid","rel"]], how="left")
    
    unjudged = unjudged[unjudged["rel"].isnull()]
    unjudged = set(list(unjudged["docid"].unique()))
    
    qrels = set(qrels_for_topic['rel'].unique())
    ret = []
    for unjudged_doc in unjudged:
        tmp_ret = []
        previous_lists = [{}]
        if len(ret) > 0:
            previous_lists = deepcopy(ret)
        for qrel in qrels:
            for prev in previous_lists:
                prev = deepcopy(prev)
                prev[unjudged_doc] = int(qrel)
                
                tmp_ret += [prev]
        ret = tmp_ret
    
    return sorted([json.dumps(i, sort_keys=True) for i in ret])


def __create_qrels_for_topic(qrels, topic, substitute_pool):
    substitute_qrels_for_topic = pd.DataFrame([{'query': topic, 'q0': 0, 'docid': k, 'rel': v} for k,v in json.loads(substitute_pool).items()])
    qrels_for_topic = qrels.qrels_data[qrels.qrels_data['query'] == topic].copy()
    
    ret = TrecQrel()
    ret.qrels_data = pd.concat([qrels_for_topic, substitute_qrels_for_topic])
    
    return ret

def substitate_pools_with_effectivenes_scores(run, qrels, measure):
    if not measure.startswith('ndcg@'):
        raise ValueError(f'Can not handle measure "{measure}".')
    
    depth = int(measure.split('@')[-1])
    run = normalize_run(run, depth)
    ret = {}
    
    for topic, substitute_pools in create_substitute_pools(run, qrels, depth).items():
        assert topic not in ret
        ret[topic] = {}
        run_for_topic = TrecRun()
        run_for_topic.run_data = run.run_data[run.run_data['query'] == topic]
        
        for substitute_pool in substitute_pools:
            qrels_for_topic = __create_qrels_for_topic(qrels, topic, substitute_pool)
            assert substitute_pool not in ret[topic]
            ret[topic][substitute_pool] = __extract_single_topic(TrecEval(run_for_topic, qrels_for_topic).get_ndcg(depth, per_query=True, removeUnjudged=False), topic)
            
    return ret


def __extract_single_topic(df, topic):
    assert len(df) == 1
    df = df.iloc[0]
    assert topic == df.name
    df = dict(df)
    df = df.values()
    assert len(df) == 1
    
    return list(df)[0]


def create_substitute_pools(run, qrels, depth):
    ret = {}
    run = normalize_run(run, depth)
    
    for topic in qrels.qrels_data['query'].unique():
        qrels_for_topic = qrels.qrels_data[qrels.qrels_data['query'] == topic]
        run_for_topic = run.run_data[run.run_data['query'] == topic]
        ret[topic] = __substitute_pools_for_topic(run_for_topic, qrels_for_topic)
    
    return ret

