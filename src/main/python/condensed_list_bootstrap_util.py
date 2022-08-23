from run_file_processing import normalize_run
import json
from pool_bootstrap_util import __unjudged_documents, __available_qrels_for_topic, __extract_single_topic,\
    __create_qrels_for_topic
from copy import deepcopy
from tqdm import tqdm
from trectools import TrecRun, TrecEval, TrecQrel
import pandas as pd
import numpy as np


def substitute_pools_for_condensed_lists_with_effectivenes_scores(run, qrels, measure):
    if not measure.startswith('ndcg@'):
        raise ValueError(f'Can not handle measure "{measure}".')

    depth = int(measure.split('@')[-1])
    ret = {}

    for topic, substitute_pools in tqdm(create_substitute_pools_for_condensed_lists(run, qrels, depth).items()):
        assert topic not in ret
        ret[topic] = {}
        qrels_for_topic = __create_qrels_for_topic(qrels, topic)
        run_for_topic = TrecRun()
        run_for_topic.run_data = run.run_data[run.run_data['query'] == topic]

        for substitute_pool in substitute_pools:
            assert substitute_pool not in ret[topic]
            if len(substitute_pools) > 200:
                ret[topic][substitute_pool] = float('NaN')
            else:
                ret[topic][substitute_pool] = __extract_single_topic(
                    TrecEval(__create_run_for_topic(run, topic, substitute_pool), qrels_for_topic)
                    .get_ndcg(depth, per_query=True, removeUnjudged=True), topic
                )

    return ret


def __single_bootstrap(representative_documents, unjudged_docs, rand):
    ret = {}
    representative_documents = rand.sample(representative_documents, len(unjudged_docs))

    for doc in sorted(unjudged_docs):
        for i in representative_documents.pop():
            if i == 'REMOVE-THIS-DOCUMENT' or i not in ret.values():
                ret[doc] = i
                break

    return json.dumps(ret, sort_keys=True)

def evaluate_bootstrap(run, qrels, measure, repeat=5, seed=1):
    substitute_pools = substitute_pools_for_condensed_lists_with_effectivenes_scores(run, qrels, measure)

    depth = int(measure.split('@')[-1])
    run = normalize_run(run, depth)
    ret = {}

    for topic in substitute_pools.keys():
        assert topic not in ret
        ret[topic] = {measure: []}
        run_for_topic = run.run_data[run.run_data['query'] == topic]
        for bootstrap in __bootstraps_for_topic(run_for_topic, qrels.qrels_data, seed=seed, depth=depth, repeat=repeat):
            ret[topic][measure] += [substitute_pools[topic][bootstrap]]

    return ret


def __bootstraps_for_topic(run, qrels, repeat, depth, seed=None):
    from random import Random

    rand = Random() if seed is None else Random(seed)

    rels = __rels_for_topic(run, qrels, depth)
    unjudged_docs = __unjudged_documents(run, qrels)

    ret = []
    for i in range(repeat):
        ret += [__single_bootstrap(rels, unjudged_docs, rand)]

    return ret


def create_substitute_pools_for_condensed_lists(run, qrels, depth):
    ret = {}
    run = normalize_run(run, depth)

    for topic in qrels.qrels_data['query'].unique():
        qrels_for_topic = qrels.qrels_data[qrels.qrels_data['query'] == topic]
        run_for_topic = run.run_data[run.run_data['query'] == topic]
        ret[topic] = __substitute_pools_for_topic(run_for_topic, qrels_for_topic)

    return ret


def __rels_for_topic(run, qrels, depth):
    topic_id = run['query'].unique()
    assert len(topic_id) == 1
    topic_id = topic_id[0]
    unjudged = __unjudged_documents(run, qrels)
    qrels = __available_qrels_for_topic(run, qrels[qrels['query'] == topic_id])
    qrels = {k: v[:len(unjudged)] for k, v in qrels.items()}

    add_for_unjudged = ['REMOVE-THIS-DOCUMENT']*len(unjudged)
    print('--->' + str(len(add_for_unjudged)))
    for i in range(int(np.ceil(depth*(len(unjudged)/depth)))):
        qrels[f'unj-{i}'] = deepcopy(add_for_unjudged)

    ret = []
    for v in qrels.values():
        for i in v:
            ret += [v]

    return sorted(ret)


def __create_run_for_topic(run, topic, substitute_pool):
    ret_df = []
    substitute_pool = json.loads(substitute_pool)

    for _, i in run.run_data.iterrows():
        i = deepcopy(dict(i))

        if int(topic) != int(i['query']):
            continue
        if i['docid'] in substitute_pool:
            if substitute_pool[i['docid']] == 'REMOVE-THIS-DOCUMENT':
                continue
            i['docid'] = substitute_pool[i['docid']]
        ret_df += [i]

    ret = TrecRun()
    ret.run_data = pd.DataFrame(ret_df)

    if len(ret_df) == 0:
        ret.run_data = pd.DataFrame(
            [{'query': topic, 'q0': 'Q0', 'docid': 'DUMMY-DOCUMENT-UNJUDGED', 'rank': 0, 'score': 3000, 'system': 'a'}])

    return ret


def __substitute_pools_for_topic(run_for_topic, qrels_for_topic):
    unjudged = sorted(__unjudged_documents(run_for_topic, qrels_for_topic))
    qrels = __available_qrels_for_topic(run_for_topic, qrels_for_topic)

    if unjudged is None or len(unjudged) == 0:
        return ['{}']

    if sum([len(v) for v in qrels.values()]) < len(unjudged):
        raise ValueError('too few judged documents in pool. Expected ' + str(len(unjudged)) + ' but have only ' + str(
            sum([len(v) for v in qrels.values()])) + '.')

    qrels['REMOVE'] = []
    for i in range(len(unjudged)):
        qrels['REMOVE'] += ['REMOVE-THIS-DOCUMENT']

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
                available = [i for i in qrel if i == 'REMOVE-THIS-DOCUMENT' or i not in already_used]
                if len(available) == 0:
                    continue
                prev[unjudged_doc] = available[0]

                tmp_ret += [prev]
        ret = tmp_ret

    return sorted([json.dumps(i, sort_keys=True) for i in ret])

