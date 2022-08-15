from bootstrap_util import create_substitute_pools, substitate_pools_with_effectivenes_scores, __rels_for_topic, __bootstraps_for_topic
from trectools import TrecRun, TrecQrel
import json
import pandas as pd

def test_creation_of_substitute_pools_for_no_unjudged_documents():
    run = TrecRun('src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt')
    qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
    expected = {'301': []}
    actual = create_substitute_pools(run, qrels, 10)
    
    assert expected == actual


def test_creation_of_substitute_pools_for_some_unjudged_documents():
    run = TrecRun('src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt')
    qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
    
    #doc-juru-02 0
    #doc-wdo-01 2
    #=> It should create bootstraps with rel-values of 0 and 2
    qrels.qrels_data = qrels.qrels_data[(qrels.qrels_data['docid'] == 'doc-juru-02') | (qrels.qrels_data['docid'] == 'doc-wdo-01')]
    
    #Unjudged are: doc-juru-01, shared-doc-01
    expected = {'301': sorted([
        json.dumps({'doc-juru-01': 0, 'shared-doc-01': 0}, sort_keys=True),
        json.dumps({'doc-juru-01': 2, 'shared-doc-01': 0}, sort_keys=True),
        json.dumps({'doc-juru-01': 0, 'shared-doc-01': 2}, sort_keys=True),
        json.dumps({'doc-juru-01': 2, 'shared-doc-01': 2}, sort_keys=True),
    ])}
    actual = create_substitute_pools(run, qrels, 10)
    print(json.dumps(actual))
    
    assert expected == actual
    

def test_creation_of_substitute_pools_for_some_unjudged_documents_with_many_different_qrels():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-0', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-unjudged', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = {'301': sorted([
        json.dumps({'doc-juru-01': 0, 'shared-doc-01': 0}, sort_keys=True),
        json.dumps({'doc-juru-01': 1, 'shared-doc-01': 0}, sort_keys=True),
        json.dumps({'doc-juru-01': 2, 'shared-doc-01': 0}, sort_keys=True),
        json.dumps({'doc-juru-01': 2, 'shared-doc-01': 1}, sort_keys=True),
        json.dumps({'doc-juru-01': 0, 'shared-doc-01': 1}, sort_keys=True),
        json.dumps({'doc-juru-01': 0, 'shared-doc-01': 2}, sort_keys=True),
        json.dumps({'doc-juru-01': 2, 'shared-doc-01': 2}, sort_keys=True),
        json.dumps({'doc-juru-01': 1, 'shared-doc-01': 1}, sort_keys=True),
        json.dumps({'doc-juru-01': 1, 'shared-doc-01': 2}, sort_keys=True),
    ]),
    '302': sorted([
        json.dumps({'doc-unjudged': 0}, sort_keys=True),
        json.dumps({'doc-unjudged': 1}, sort_keys=True),
        json.dumps({'doc-unjudged': 2}, sort_keys=True),
    ])}
    actual = create_substitute_pools(run, qrels, 10)
    
    assert expected == actual

def test_creation_of_substitute_pools_for_only_unjudged_documents():
    run = TrecRun('src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt')
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 3},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = {'301': sorted([
        json.dumps({'doc-juru-01': 0, 'doc-juru-02': 0, 'shared-doc-01': 0}, sort_keys=True),
        json.dumps({'doc-juru-01': 0, 'doc-juru-02': 3, 'shared-doc-01': 0}, sort_keys=True),
        json.dumps({'doc-juru-01': 3, 'doc-juru-02': 0, 'shared-doc-01': 0}, sort_keys=True),
        json.dumps({'doc-juru-01': 0, 'doc-juru-02': 0, 'shared-doc-01': 3}, sort_keys=True),
        json.dumps({'doc-juru-01': 0, 'doc-juru-02': 3, 'shared-doc-01': 3}, sort_keys=True),
        json.dumps({'doc-juru-01': 3, 'doc-juru-02': 0, 'shared-doc-01': 3}, sort_keys=True),
        json.dumps({'doc-juru-01': 3, 'doc-juru-02': 3, 'shared-doc-01': 0}, sort_keys=True),
        json.dumps({'doc-juru-01': 3, 'doc-juru-02': 3, 'shared-doc-01': 3}, sort_keys=True),
    ]), '302': []}
    actual = create_substitute_pools(run, qrels, 10)
    
    print(actual)
    print(expected)
    assert expected == actual


def test_substitate_pools_with_effectivenes_scores_for_some_unjudged_documents_with_many_different_qrels():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-0', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '302', 'q0': 'Q0', 'docid': 'doc-unjudged', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = {'301': 
        {
        json.dumps({'doc-juru-01': 0, 'shared-doc-01': 0}, sort_keys=True): 0.4796249331362629,
        json.dumps({'doc-juru-01': 1, 'shared-doc-01': 0}, sort_keys=True): 0.7224242270408039,
        json.dumps({'doc-juru-01': 2, 'shared-doc-01': 0}, sort_keys=True): 0.8670870086853021,
        json.dumps({'doc-juru-01': 2, 'shared-doc-01': 1}, sort_keys=True): 0.8972754076830647,
        json.dumps({'doc-juru-01': 0, 'shared-doc-01': 1}, sort_keys=True): 0.5627272554209044,
        json.dumps({'doc-juru-01': 0, 'shared-doc-01': 2}, sort_keys=True): 0.6012610260559063,
        json.dumps({'doc-juru-01': 2, 'shared-doc-01': 2}, sort_keys=True): 0.9082209380838205,
        json.dumps({'doc-juru-01': 1, 'shared-doc-01': 1}, sort_keys=True): 0.7754533391612871,
        json.dumps({'doc-juru-01': 1, 'shared-doc-01': 2}, sort_keys=True): 0.7780158492147935,
    },
    '302': {
        json.dumps({'doc-unjudged': 0}, sort_keys=True): 0.23981246656813146,
        json.dumps({'doc-unjudged': 1}, sort_keys=True): 0.36121211352040195,
        json.dumps({'doc-unjudged': 2}, sort_keys=True): 0.43354350434265104,
    }}

    actual = substitate_pools_with_effectivenes_scores(run, qrels, 'ndcg@10')
    print(actual)
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
    
    expected = [2]
    actual = __rels_for_topic(run, qrels)
    
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
    
    expected = [0,0,0]
    actual = __rels_for_topic(run, qrels)
    
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
        {'query': '301', 'q0': 0, 'docid': 'shared-doc-01', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
    ])
    
    expected = [0,1,2]
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
    
    expected = ['all-judged', 'all-judged', 'all-judged', 'all-judged', 'all-judged']
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
    
    expected = [json.dumps({'doc-juru-02': 0}, sort_keys=True)]*5
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
    
    expected = [json.dumps({'doc-juru-02': 1, 'shared-doc-01': 1}, sort_keys=True)]*5
    actual = __bootstraps_for_topic(run, qrels, seed=0, repeat=5)
    
    print(expected)
    print(actual)
    assert expected == actual


def test_bootstraps_for_single_unjudged_document_some_relevant_seed_0():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
    ])
    
    expected = [{'shared-doc-01': 0}, {'shared-doc-01': 0}, {'shared-doc-01': 1}, {'shared-doc-01': 0}, {'shared-doc-01': 0},]
    expected = [json.dumps(i) for i in expected]
    actual = __bootstraps_for_topic(run, qrels, seed=0, repeat=5)
    
    print(expected)
    print(actual)
    assert expected == actual



def test_bootstraps_for_single_unjudged_document_some_relevant_seed_1():
    run = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
    ])
    
    qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
    ])
    
    expected = [{'shared-doc-01': 1}, {'shared-doc-01': 1}, {'shared-doc-01': 0}, {'shared-doc-01': 1}, {'shared-doc-01': 0},]
    expected = [json.dumps(i) for i in expected]
    actual = __bootstraps_for_topic(run, qrels, seed=1, repeat=5)
    
    print(expected)
    print(actual)
    assert expected == actual
