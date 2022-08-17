from trectools import TrecRun, TrecQrel
from evaluation_util import normalize_eval_output, __evaluate_run_on_pool, __create_residual_trec_eval
import pandas as pd


def __to_string(data):
    if type(data) is TrecQrel:
        return __to_string(data.qrels_data)
    elif type(data) is TrecRun:
        return __to_string(data.run_data)
    else:
        ret = []
        for _, i in data.iterrows():
            if 'rel' in i:
                ret += [{'query': i['query'], 'docid': i['docid'], 'rel': i['rel']}]
            else:
                ret += [{'query': i['query'], 'docid': i['docid'], 'rank': i['rank']}]
        return ret


def test_create_max_residual_eval_for_target_ndcg_of_05():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'a', 'rank': 1, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'b', 'rank': 2, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'c', 'rank': 3, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected_run = pd.DataFrame([
        {'query': '301', 'docid': 'a', 'rank': 1},
        {'query': '301', 'docid': 'b', 'rank': 2},
        {'query': '301', 'docid': 'd', 'rank': 3},
    ])
    
    actual = __create_residual_trec_eval(run, qrels, depth=10, residual_type='max', adjust_idcg=False)
    
    print(__to_string(actual.qrels))
    assert __to_string(actual.qrels) == __to_string(qrels)
    
    print(__to_string(actual.run))
    assert __to_string(actual.run) == __to_string(expected_run)
    
    print(actual.get_ndcg())
    assert actual.get_ndcg() == 0.5


def test_create_max_residual_eval_for_target_ndcg_of_05_adjust_idcg():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'a', 'rank': 1, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'b', 'rank': 2, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'c', 'rank': 3, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected_run = pd.DataFrame([
        {'query': '301', 'docid': 'a', 'rank': 1},
        {'query': '301', 'docid': 'b', 'rank': 2},
        {'query': '301', 'docid': 'd', 'rank': 3},
    ])
    
    actual = __create_residual_trec_eval(run, qrels, depth=10, residual_type='max', adjust_idcg=True)
    
    print(__to_string(actual.qrels))    
    print(__to_string(qrels))
    assert __to_string(actual.qrels) == __to_string(qrels)
    
    print(__to_string(actual.run))
    assert __to_string(actual.run) == __to_string(expected_run)
    
    print(actual.get_ndcg())
    assert actual.get_ndcg() == 0.5


def test_create_max_residual_eval_for_target_ndcg_of_09_adjust_idcg():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'a', 'rank': 1, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'b', 'rank': 2, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'c', 'rank': 3, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 0},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected_run = pd.DataFrame([
        {'query': '301', 'docid': 'a', 'rank': 1},
        {'query': '301', 'docid': 'b', 'rank': 2},
        {'query': '301', 'docid': 'c', 'rank': 3},
    ])
    
    expected_qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 1},
        
    ])
    
    actual = __create_residual_trec_eval(run, qrels, depth=10, residual_type='max', adjust_idcg=True)
    
    print(__to_string(actual.qrels))
    print(__to_string(expected_qrels))
    assert __to_string(actual.qrels) == __to_string(expected_qrels)
    
    print(__to_string(actual.run))
    assert __to_string(actual.run) == __to_string(expected_run)
    
    print(actual.get_ndcg())
    assert actual.get_ndcg() == 0.9197207891481876


def test_create_max_residual_eval_for_target_ndcg_of_077_adjust_idcg():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'a', 'rank': 1, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'b', 'rank': 2, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'c', 'rank': 3, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 1},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected_run = pd.DataFrame([
        {'query': '301', 'docid': 'a', 'rank': 1},
        {'query': '301', 'docid': 'b', 'rank': 2},
        {'query': '301', 'docid': 'c', 'rank': 3},
    ])
    
    expected_qrels = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 2},
        
    ])
    
    actual = __create_residual_trec_eval(run, qrels, depth=10, residual_type='max', adjust_idcg=True)
    
    print(__to_string(actual.qrels))
    print(__to_string(expected_qrels))
    assert __to_string(actual.qrels) == __to_string(expected_qrels)
    
    print(__to_string(actual.run))
    assert __to_string(actual.run) == __to_string(expected_run)
    
    print(actual.get_ndcg())
    assert actual.get_ndcg() == 0.7780158492147935

def test_create_min_residual_eval_for_target_ndcg_of_0():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'a', 'rank': 1, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'b', 'rank': 2, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'c', 'rank': 3, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected_run = pd.DataFrame([
        {'query': '301', 'docid': 'a', 'rank': 1},
        {'query': '301', 'docid': 'b', 'rank': 2},
        {'query': '301', 'docid': 'c', 'rank': 3},
    ])
    
    actual = __create_residual_trec_eval(run, qrels, depth=10, residual_type='min', adjust_idcg=False)
    
    print(__to_string(actual.qrels))
    assert __to_string(actual.qrels) == __to_string(qrels)
    
    print(__to_string(actual.run))
    assert __to_string(actual.run) == __to_string(expected_run)
    
    print(actual.get_ndcg())
    assert actual.get_ndcg() == 0.0


def test_create_max_residual_eval_for_target_ndcg_of_1():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'a', 'rank': 1, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'b', 'rank': 2, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'c', 'rank': 3, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected_run = pd.DataFrame([
        {'query': '301', 'docid': 'd', 'rank': 1},
        {'query': '301', 'docid': 'b', 'rank': 2},
        {'query': '301', 'docid': 'c', 'rank': 3},
    ])
    
    actual = __create_residual_trec_eval(run, qrels, depth=10, residual_type='max', adjust_idcg=False)
    
    print(__to_string(actual.qrels))
    assert __to_string(actual.qrels) == __to_string(qrels)
    
    print(__to_string(actual.run))
    assert __to_string(actual.run) == __to_string(expected_run)
    
    print(actual.get_ndcg())
    assert actual.get_ndcg() == 1.0


def test_create_min_residual_eval_for_target_ndcg_of_0_v2():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'a', 'rank': 1, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'b', 'rank': 2, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'c', 'rank': 3, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected_run = pd.DataFrame([
        {'query': '301', 'docid': 'a', 'rank': 1},
        {'query': '301', 'docid': 'b', 'rank': 2},
        {'query': '301', 'docid': 'c', 'rank': 3},
    ])
    
    actual = __create_residual_trec_eval(run, qrels, depth=10, residual_type='min', adjust_idcg=False)
    
    print(__to_string(actual.qrels))
    assert __to_string(actual.qrels) == __to_string(qrels)
    
    print(__to_string(actual.run))
    assert __to_string(actual.run) == __to_string(expected_run)
    
    print(actual.get_ndcg())
    assert actual.get_ndcg() == 0.0


def test_create_max_residual_eval_for_target_ndcg_of_1_two_unjudged_documents():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'a', 'rank': 1, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'b', 'rank': 2, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'c', 'rank': 3, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'e', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected_run = pd.DataFrame([
        {'query': '301', 'docid': 'd', 'rank': 1},
        {'query': '301', 'docid': 'b', 'rank': 2},
        {'query': '301', 'docid': 'c', 'rank': 3},
    ])
    
    actual = __create_residual_trec_eval(run, qrels, depth=10, residual_type='max', adjust_idcg=False)
    
    print(__to_string(actual.qrels))
    assert __to_string(actual.qrels) == __to_string(qrels)
    
    print(__to_string(actual.run))
    assert __to_string(actual.run) == __to_string(expected_run)
    
    print(actual.get_ndcg())
    assert actual.get_ndcg() == 1.0


def test_create_max_residual_eval_for_target_ndcg_of_095_two_unjudged_documents():
    run = TrecRun()
    run.run_data = pd.DataFrame([
        {'query': '301', 'q0': 'Q0', 'docid': 'a', 'rank': 1, 'score': 3000, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'b', 'rank': 2, 'score': 2999, 'system': 'a'},
        {'query': '301', 'q0': 'Q0', 'docid': 'c', 'rank': 3, 'score': 2998, 'system': 'a'},
    ])
    qrels = TrecQrel()
    qrels.qrels_data = pd.DataFrame([
        {'query': '301', 'q0': 0, 'docid': 'e', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
        {'query': '301', 'q0': 0, 'docid': 'f', 'rel': 1},
        
        # Some noise
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
    ])
    
    expected_run = pd.DataFrame([
        {'query': '301', 'docid': 'd', 'rank': 1},
        {'query': '301', 'docid': 'b', 'rank': 2},
        {'query': '301', 'docid': 'f', 'rank': 3},
    ])
    
    actual = __create_residual_trec_eval(run, qrels, depth=10, residual_type='max', adjust_idcg=False)
    
    print(__to_string(actual.qrels))
    assert __to_string(actual.qrels) == __to_string(qrels)
    
    print(__to_string(actual.run))
    assert __to_string(actual.run) == __to_string(expected_run)
    
    print(actual.get_ndcg())
    assert actual.get_ndcg() == 0.9502344167898356
