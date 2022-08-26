from pool_bootstrap_util import substitate_pools_with_effectivenes_scores, __rels_for_topic, __bootstraps_for_topic, evaluate_bootstrap, __single_bootstrap
from trectools import TrecRun, TrecQrel
from evaluation_util import normalize_eval_output
import json
import pandas as pd


def test_substitate_pools_with_effectivenes_scores_for_some_unjudged_documents_with_many_different_qrels_huge_pool_01():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'c-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'e-0', 'rel': 0},
        
        {'query': '301', 'q0': 0, 'docid': 'b-1', 'rel': 1},
        
        
        {'query': '301', 'q0': 0, 'docid': 'a-2', 'rel': 2},
        
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    # Unjudged are: doc-juru-01, shared-doc-01
    expected = {'301': {
        json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'b-0'}, sort_keys=True): 0.3354350434265105,
        json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'a-2'}, sort_keys=True): 0.6012610260559063,
        json.dumps({'doc-juru-01': 'b-1', 'shared-doc-01': 'a-2'}, sort_keys=True): 0.8670870086853021,
        json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'b-1'}, sort_keys=True): 0.4683480347412084,
        json.dumps({'doc-juru-01': 'b-1', 'shared-doc-01': 'a-0'}, sort_keys=True): 0.6012610260559063,
        json.dumps({'doc-juru-01': 'a-2', 'shared-doc-01': 'a-0'}, sort_keys=True): 0.8670870086853021,
        json.dumps({'doc-juru-01': 'a-2', 'shared-doc-01': 'b-1'}, sort_keys=True): 1.0,
    }, '302': {'{}': 0.0}}
    
    actual = substitate_pools_with_effectivenes_scores(run, qrels, 'ndcg@10')
    actual = {qid: {k: actual[qid][k] for k in expected[qid]} for qid in expected}
    print(json.dumps(actual))
    print(json.dumps(expected))
    
    assert expected == actual


def test_substitate_pools_with_effectivenes_scores_for_some_unjudged_documents_with_many_different_qrels_huge_pool_02():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'unjudged', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'c-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'e-0', 'rel': 0},
        
        {'query': '301', 'q0': 0, 'docid': 'b-1', 'rel': 1},
        
        
        {'query': '301', 'q0': 0, 'docid': 'a-2', 'rel': 2},
        
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    # Unjudged are: doc-juru-01, shared-doc-01
    expected = {'301': {
        json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'b-0'}, sort_keys=True): 0.3354350434265105,
        json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'a-2'}, sort_keys=True): 0.6012610260559063,
        json.dumps({'doc-juru-01': 'b-1', 'shared-doc-01': 'a-2'}, sort_keys=True): 0.8670870086853021,
        json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'b-1'}, sort_keys=True): 0.4683480347412084,
        json.dumps({'doc-juru-01': 'b-1', 'shared-doc-01': 'a-0'}, sort_keys=True): 0.6012610260559063,
        json.dumps({'doc-juru-01': 'a-2', 'shared-doc-01': 'a-0'}, sort_keys=True): 0.8670870086853021,
        json.dumps({'doc-juru-01': 'a-2', 'shared-doc-01': 'b-1'}, sort_keys=True): 1.0,
    }, '302': {
        json.dumps({'unjudged': 'doc-0'}, sort_keys=True): 0.0,
        json.dumps({'unjudged': 'doc-1'}, sort_keys=True): 0.38009376671593426,
        json.dumps({'unjudged': 'doc-2'}, sort_keys=True): 0.7601875334318685,
    }}
    
    actual = substitate_pools_with_effectivenes_scores(run, qrels, 'ndcg@10')
    actual = {qid: {k: actual[qid][k] for k in expected[qid]} for qid in expected}
    print(json.dumps(actual))
    print(json.dumps(expected))
    
    assert expected == actual


def test_substitate_pools_with_effectivenes_scores_for_some_unjudged_documents_with_many_different_qrels_huge_pool_03():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'unjudged', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'c-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd-0', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'e-0', 'rel': 0},
        
        {'query': '301', 'q0': 0, 'docid': 'b-1', 'rel': 1},
        
        
        {'query': '301', 'q0': 0, 'docid': 'a-2', 'rel': 2},
        
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected = {'301': {
        json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'b-0'}, sort_keys=True): 0.3354350434265105,
        json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'a-2'}, sort_keys=True): 0.6012610260559063,
        json.dumps({'doc-juru-01': 'b-1', 'shared-doc-01': 'a-2'}, sort_keys=True): 0.8670870086853021,
        json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'b-1'}, sort_keys=True): 0.4683480347412084,
        json.dumps({'doc-juru-01': 'b-1', 'shared-doc-01': 'a-0'}, sort_keys=True): 0.6012610260559063,
        json.dumps({'doc-juru-01': 'a-2', 'shared-doc-01': 'a-0'}, sort_keys=True): 0.8670870086853021,
        json.dumps({'doc-juru-01': 'a-2', 'shared-doc-01': 'b-1'}, sort_keys=True): 1.0,
    }, '302': {
        json.dumps({'unjudged': 'doc-0'}, sort_keys=True): 0.0,
        json.dumps({'unjudged': 'doc-1'}, sort_keys=True): 1.0,
    }}
    
    actual = substitate_pools_with_effectivenes_scores(run, qrels, 'ndcg@10')
    actual = {qid: {k: actual[qid][k] for k in expected[qid]} for qid in expected}
    print(json.dumps(actual))
    print(json.dumps(expected))
    
    assert expected == actual


def test_rels_for_topic_for_single_judged_doc():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = [['doc-wdo-01'], ['doc-wdo-02']]
    actual = __rels_for_topic(run, qrels)
    print(actual)
    
    assert expected == actual


def test_rels_for_topic_for_multiple_judged_doc_01():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'shared-doc-01', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = []

    actual = __rels_for_topic(run, qrels)
    print(actual)
        
    assert expected == actual


def test_rels_for_topic_for_multiple_judged_doc_02():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = [['doc-wdo-01']]
    actual = __rels_for_topic(run, qrels)
    
    assert sorted(expected) == sorted(actual)    


def test_rels_for_topic_for_multiple_judged_doc_03():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-03', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = [['doc-wdo-01']]
    actual = __rels_for_topic(run, qrels)
    
    assert sorted(expected) == sorted(actual) 


def test_rels_for_topic_for_multiple_judged_doc_04():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-03', 'rel': 0},
        
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 1},
        
        {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = [['doc-wdo-01'], ['a'], ['c']]
    actual = __rels_for_topic(run, qrels)
    
    assert sorted(expected) == sorted(actual)


def test_rels_for_topic_for_multiple_judged_doc_05():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-03', 'rel': 0},
        
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'i', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'j', 'rel': 1},
        
        {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'k', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'l', 'rel': 2},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = [['doc-wdo-01', 'doc-wdo-02'], ['doc-wdo-01', 'doc-wdo-02'], ['a', 'b'],  ['a', 'b'], ['c', 'd'],
                ['c', 'd']]
    actual = __rels_for_topic(run, qrels)
    
    assert sorted(expected) == sorted(actual) 


def test_bootstraps_for_only_judged_documents():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'shared-doc-01', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = ['{}', '{}', '{}', '{}', '{}']
    actual = __bootstraps_for_topic(run, qrels, seed=0, repeat=5)
    
    assert expected == actual


def test_bootstraps_for_single_unjudged_document_only_irrelevant():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'shared-doc-01', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = [json.dumps({'doc-juru-02': 'doc-wdo-01'}, sort_keys=True)]*5
    actual = __bootstraps_for_topic(run, qrels, seed=0, repeat=5)
    
    print(expected)
    print(actual)
    assert expected == actual


def test_bootstraps_for_two_unjudged_document_only_relevant():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
    ])
    
    try:
        __bootstraps_for_topic(run, qrels, seed=0, repeat=5)
    except:
        return
        
    assert False


def test_bootstraps_for_two_unjudged_document_only_relevant_01():
    for i in range(10):
        run = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
    
        qrels = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 1},
        
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        ])
    
        expected = ['{"doc-juru-02": "a", "shared-doc-01": "b"}', '{"doc-juru-02": "a", "shared-doc-01": "b"}', '{"doc-juru-02": "a", "shared-doc-01": "b"}', '{"doc-juru-02": "a", "shared-doc-01": "b"}', '{"doc-juru-02": "a", "shared-doc-01": "b"}']
        actual = __bootstraps_for_topic(run, qrels, seed=0, repeat=5)
    
        print(i)
        print(expected)
        print(actual)
        assert expected == actual


def test_bootstraps_for_two_unjudged_document_only_relevant_02():
    for i in range(10):
        run = pd.DataFrame([
         {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
         {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
         {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
    
        qrels = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 1},
        
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        ])
    
        expected = ['{"doc-juru-02": "a", "shared-doc-01": "b"}', '{"doc-juru-02": "a", "shared-doc-01": "b"}', '{"doc-juru-02": "a", "shared-doc-01": "b"}', '{"doc-juru-02": "a", "shared-doc-01": "b"}', '{"doc-juru-02": "a", "shared-doc-01": "b"}']
        actual = __bootstraps_for_topic(run, qrels, seed=0, repeat=5)
    
        print(i)
        print(expected)
        print(actual)
        assert expected == actual


def test_bootstraps_for_single_unjudged_document_some_relevant_seed_0():
    for i in range(10):
        run = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
    
        qrels = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'a-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'a-1', 'rel': 1},
        
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        ])
    
        expected = ['{"shared-doc-01": "a-1"}', '{"shared-doc-01": "a-1"}', '{"shared-doc-01": "a-0"}', '{"shared-doc-01": "a-1"}', '{"shared-doc-01": "a-1"}']
        actual = __bootstraps_for_topic(run, qrels, seed=0, repeat=5)
    
        print(i)
        print(expected)
        print(actual)
        assert expected == actual


def test_bootstraps_for_two_unjudged_document_only_relevant_02_dummy():
    from random import Random
    
    for i in range(1000):
        r = Random(0)
        r = __single_bootstrap(['a', 'b', 'c'], ['e', 'f'], r) + '|' + __single_bootstrap(['a', 'b', 'c'], ['e', 'f'], r) + '|' + __single_bootstrap(['a', 'b', 'c'], ['e', 'f'], r) + '|' + __single_bootstrap(['a', 'b', 'c'], ['e', 'f'], r) + '|' + __single_bootstrap(['a', 'b', 'c'], ['e', 'f'], r)
        print('ASASAS: ' + r)
        assert '{"e": "c", "f": "b"}|{"e": "b", "f": "a"}|{"e": "b", "f": "c"}|{"e": "c", "f": "b"}|{"e": "c", "f": "b"}' == r


def test_bootstrap_end_to_end_all_judged_01():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-2', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-3', 'rel': 1},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        {'query': '302', 'q0': 0, 'docid': 'doc-3', 'rel': 1},
    ])

    expected = {'301': {'ndcg@10': [0.9197207891481876]*5}, '302': {'ndcg@10': [0.66967181649423]*5}}
    actual = evaluate_bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)
    
    print(actual)
    assert expected == actual


def test_bootstrap_end_to_end_all_judged_02():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-2', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-3', 'rel': 1},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 2},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        {'query': '302', 'q0': 0, 'docid': 'doc-3', 'rel': 1},
    ])

    expected = {'301': {'ndcg@10': [1.0]*5}, '302': {'ndcg@10': [1.0]*5}}
    actual = evaluate_bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)
    
    print(actual)
    assert expected == actual


def test_bootstrap_end_to_end_all_judged_03():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-2', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-20', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-3', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},        
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-20', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-3', 'rel': 0},
    ])

    expected = {'301': {'ndcg@10': [0.0]*5}, '302': {'ndcg@10': [0.0]*5}}
    actual = evaluate_bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)
    
    print(actual)
    assert expected == actual


def test_bootstrap_with_some_relevant_and_some_irrelevant_with_normalize_output():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-2', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-4', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-5', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        {'query': '302', 'q0': 0, 'docid': 'doc-4', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-5', 'rel': 0},
    ])

    expected = [
        {'run_file': 'a', 'query': '301', 'ndcg@10': [0.9197207891481876, 0.9197207891481876, 0.6131471927654584, 0.9197207891481876, 0.6131471927654584]},
        {'run_file': 'a', 'query': '302', 'ndcg@10': [0.66967181649423, 0.66967181649423, 0.4796249331362629, 0.66967181649423, 0.4796249331362629]}
    ]
    actual = list(normalize_eval_output(evaluate_bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1), 'a'))
    
    print(actual)
    print(expected)
    
    assert expected == actual


def test_bootstrap_with_some_relevant_and_some_irrelevant_with_normalize_output_and_missing_topic():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])

    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-2', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-4', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-5', 'rel': 0},

        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        {'query': '302', 'q0': 0, 'docid': 'doc-4', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-5', 'rel': 0},
    ])

    expected = [
        {'run_file': 'a', 'query': '301',
         'ndcg@10': [0.9197207891481876, 0.9197207891481876, 0.6131471927654584,
                     0.9197207891481876, 0.6131471927654584]
         },
        {'run_file': 'a', 'query': '302',
         'ndcg@10': [0.0, 0.0, 0.0, 0.0, 0.0]
         }
    ]
    actual = list(normalize_eval_output(evaluate_bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1), 'a'))

    print(actual)
    print(expected)

    assert expected == actual


def test_dasa():
    for i in range(1):
        from run_file_processing import IncompletePools
        from evaluation_util import __adjust_qrels_to_pool
        run = TrecRun('src/test/resources/sample-robust-04-run-for-topic-306.txt')
        qrels = TrecQrel('src/test/resources/sample-robust-04-qrels-for-topic-306.txt')
        pooling = IncompletePools(pool_per_run_file='src/main/resources/processed/pool-documents-per-run-trec-system-runs-trec13-robust.json.gz')
        pool = {k: v for k, v in pooling.create_incomplete_pools_for_run('src/main/resources/processed/normalized-runs/trec-system-runs/trec13/robust/input.pircRB04td2.gz')}['depth-10-pool-incomplete-for-pirc']
    
        expected = [{'run_file': 'a', 'query': '306', 'ndcg@10': [0.3822360008387524, 0.3159817773843634, 0.3159817773843634, 0.42602766053340346, 0.42602766053340346, 0.3159817773843634, 0.3159817773843634, 0.4922818839877925, 0.3822360008387524, 0.42602766053340346, 0.3822360008387524, 0.3159817773843634, 0.3822360008387524, 0.3822360008387524, 0.3159817773843634, 0.3822360008387524, 0.3822360008387524, 0.3822360008387524, 0.4922818839877925, 0.3822360008387524, 0.3822360008387524, 0.3822360008387524, 0.42602766053340346, 0.4922818839877925, 0.3822360008387524]}]
    
        actual = list(normalize_eval_output(evaluate_bootstrap(run, __adjust_qrels_to_pool(qrels, pool), 'ndcg@10', repeat=25, seed=1), 'a'))
    
        print(actual)
        print(expected)
    
        assert expected == actual

