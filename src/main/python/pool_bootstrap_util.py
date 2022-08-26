from run_file_processing import normalize_run
import pandas as pd
import json
from copy import deepcopy
from numpy import isnan
from tqdm import tqdm
from trectools import TrecQrel, TrecEval, TrecRun
from fast_ndcg_implementation import MemorizedNdcg


def __unjudged_documents(run, qrels):
    ret = pd.merge(run, qrels[["query","docid","rel"]], how="left")
    ret = ret[ret["rel"].isnull()]
    
    return set(list(ret["docid"].unique()))


def __single_bootstrap(representative_documents, unjudged_docs, rand):
    ret = {}
    representative_documents = rand.sample(representative_documents, len(unjudged_docs))
    
    for doc in sorted(unjudged_docs):
        for i in representative_documents.pop():
            if i not in ret.values():
                ret[doc] = i
                break
    
    return json.dumps(ret, sort_keys=True)


def __rels_for_topic(run, qrels):
    topic_id = run['query'].unique()
    assert len(topic_id) == 1
    topic_id = topic_id[0]
    unjudged = __unjudged_documents(run, qrels)
    qrels = __available_qrels_for_topic(run, qrels[qrels['query'] == topic_id])
    qrels = {k: v[:len(unjudged)] for k, v in qrels.items()}
    
    ret = []
    for v in qrels.values():
        for i in v:
            ret += [v]
    
    return sorted(ret)


def __bootstraps_for_topic(run, qrels, repeat, seed=None):
    from random import Random
    
    rand = Random() if seed is None else Random(seed)
    
    rels = list(__rels_for_topic(run, qrels))
    unjudged_docs = __unjudged_documents(run, qrels)
    
    ret = []
    for i in range(repeat):
        ret += [__single_bootstrap(rels, unjudged_docs, rand)]
        
    return ret


def __available_qrels_for_topic(run_for_topic, qrels_for_topic):
    qrels_for_topic = qrels_for_topic[~qrels_for_topic['docid'].isin(run_for_topic['docid'].unique())]
    qrels_for_topic = qrels_for_topic.groupby('rel').aggregate({'docid': lambda x: sorted(list(x))}).reset_index()
    qrels = {i['rel']: i['docid'] for _, i in qrels_for_topic.iterrows()}

    return {k: qrels[k] for k in sorted(qrels.keys())}


def __substitute_pools_for_topic(run_for_topic, qrels_for_topic):
    unjudged = sorted(__unjudged_documents(run_for_topic, qrels_for_topic))
    qrels = __available_qrels_for_topic(run_for_topic, qrels_for_topic)

    if unjudged is None or len(unjudged) == 0:
        return ['{}']

    if sum([len(v) for v in qrels.values()]) < len(unjudged):
        raise ValueError('too few judged documents in pool. Expected ' + str(len(unjudged)) + ' but have only ' + str(sum([len(v) for v in qrels.values()])) + '.')

    ret = []
    for unjudged_doc in unjudged:
        tmp_ret = []
        previous_lists = [{}]
        if len(ret) > 0:
            previous_lists = deepcopy(ret)
        for _, qrel in qrels.items():
            for prev in previous_lists:
                prev = deepcopy(prev)
                already_used = prev.values()
                available = [i for i in qrel if i not in already_used]
                if len(available) == 0:
                    continue
                prev[unjudged_doc] = available[0]
                
                tmp_ret += [prev]
        ret = tmp_ret
    
    return sorted([json.dumps(i, sort_keys=True) for i in ret])


def __create_qrels_for_topic(qrels, topic):
    ret = TrecQrel()
    ret.qrels_data = qrels.qrels_data[qrels.qrels_data['query'] == topic].copy()
    
    return ret


def __create_run_for_topic(run, topic, substitute_pool):
    ret_df = []
    substitute_pool = json.loads(substitute_pool)
    
    for _, i in run.run_data.iterrows():
        i = deepcopy(dict(i))
        
        if int(topic) != int(i['query']):
            continue
        if i['docid'] in substitute_pool:
            i['docid'] = substitute_pool[i['docid']]
        ret_df += [i]

    ret = TrecRun()
    ret.run_data = pd.DataFrame(ret_df)
    
    if len(ret_df) == 0:
        ret.run_data = pd.DataFrame([{'query': topic, 'q0': 'Q0', 'docid': 'DUMMY-DOCUMENT-UNJUDGED', 'rank': 0, 'score': 3000, 'system': 'a'}])

    return ret


def substitate_pools_with_effectivenes_scores(run, qrels, measure):
    if not measure.startswith('ndcg@'):
        raise ValueError(f'Can not handle measure "{measure}".')
    
    depth = int(measure.split('@')[-1])
    run = normalize_run(run, depth)
    ret = {}
    memorized_ndg_scores = MemorizedNdcg(depth).get_ndcg(run, qrels, depth)

    for topic, substitute_pools in tqdm(create_substitute_pools(run, qrels, depth).items()):
        assert topic not in ret
        incomplete_ndcg = memorized_ndg_scores[topic]
        doc_to_qrel = {}
        for _, i in qrels.qrels_data[qrels.qrels_data['query'] == topic].iterrows():
            if i['docid'] in doc_to_qrel:
                raise ValueError('I do not know how to handle duplicates in qrels')

            doc_to_qrel[i['docid']] = i['rel']

        ret[topic] = {}
        
        for substitute_pool in substitute_pools:
            assert substitute_pool not in ret[topic]
            ret[topic][substitute_pool] = incomplete_ndcg.calculate(substitute_pool, doc_to_qrel)

    return ret


def __extract_single_topic(df, topic):
    if len(df) != 1:
        raise ValueError(f'I can not handle topic {topic} for df: {str(df)}')
    df = df.iloc[0]
    assert topic == df.name
    df = dict(df)
    df = df.values()
    assert len(df) == 1
    
    ret = list(df)[0]
    
    return 0 if isnan(ret) else ret


def create_substitute_pools(run, qrels, depth):
    ret = {}
    run = normalize_run(run, depth)
    
    for topic in qrels.qrels_data['query'].unique():
        qrels_for_topic = qrels.qrels_data[qrels.qrels_data['query'] == topic]
        run_for_topic = run.run_data[run.run_data['query'] == topic]
        ret[topic] = __substitute_pools_for_topic(run_for_topic, qrels_for_topic)
    
    return ret


def evaluate_bootstrap(run, qrels, measure, repeat=5, seed=1):
    print('Prepare substitute pools for bootstrapping')
    substitute_pools = substitate_pools_with_effectivenes_scores(run, qrels, measure)
    
    depth = int(measure.split('@')[-1])
    run = normalize_run(run, depth)
    ret = {}
    print('Evaluate with Bootstrapping')
    for topic in tqdm(substitute_pools.keys()):
        assert topic not in ret
        ret[topic] = {measure: []}
        run_for_topic = run.run_data[run.run_data['query'] == topic]
        for bootstrap in __bootstraps_for_topic(run_for_topic, qrels.qrels_data, seed=seed, repeat=repeat):
            if bootstrap not in substitute_pools[topic]:
                raise ValueError(f'Can not find \'{bootstrap}\' for topic {topic}.')
            ret[topic][measure] += [substitute_pools[topic][bootstrap]]

    return ret

