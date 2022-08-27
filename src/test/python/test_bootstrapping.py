from bootstrap_util import BootstrappEval, FullyIndependentBootstrappingStrategey
from trectools import TrecRun, TrecQrel
from evaluation_util import normalize_eval_output
import pandas as pd


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

    bs_strategy = FullyIndependentBootstrappingStrategey(qrels)
    bs_eval = BootstrappEval('ndcg@10', bs_strategy, False)

    expected = {'301': {'ndcg@10': [0.9197207891481876]*5}, '302': {'ndcg@10': [0.66967181649423]*5}}
    actual = bs_eval.bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)
    
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

    bs_strategy = FullyIndependentBootstrappingStrategey(qrels)
    bs_eval = BootstrappEval('ndcg@10', bs_strategy, False)

    expected = {'301': {'ndcg@10': [1.0]*5}, '302': {'ndcg@10': [1.0]*5}}
    actual = bs_eval.bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)
    
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
        {'query': '301', 'q0': 0, 'docid': 'doc-20', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-3', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-20', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-3', 'rel': 0},
    ])

    bs_strategy = FullyIndependentBootstrappingStrategey(qrels)
    bs_eval = BootstrappEval('ndcg@10', bs_strategy, False)

    # Only one relevant document_remaining
    expected = {'301': {'ndcg@10': [0.6309297535714575]*5},
                '302': {'ndcg@10': [0.6309297535714575]*5}}
    actual = bs_eval.bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)
    
    print(actual)
    assert expected == actual


def test_bootstrap_end_to_end_all_judged_04():
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
        {'query': '301', 'q0': 0, 'docid': 'doc-2', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-3', 'rel': 0},

        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-3', 'rel': 0},
    ])

    bs_strategy = FullyIndependentBootstrappingStrategey(qrels)
    bs_eval = BootstrappEval('ndcg@10', bs_strategy, False)

    # Only one relevant document_remaining
    expected = {'301': {'ndcg@10': [0.6309297535714575] * 5},
                '302': {'ndcg@10': [0.6309297535714575] * 5}}
    actual = bs_eval.bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)

    print(actual)
    assert expected == actual


def test_bootstrap_end_to_end_all_judged_05():
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
        {'query': '301', 'q0': 0, 'docid': 'doc-20', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-21', 'rel': 0},
        {'query': '301', 'q0': 0, 'docid': 'doc-3', 'rel': 0},

        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-20', 'rel': 1},
        {'query': '302', 'q0': 0, 'docid': 'doc-21', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-3', 'rel': 0},
    ])

    bs_strategy = FullyIndependentBootstrappingStrategey(qrels)
    bs_eval = BootstrappEval('ndcg@10', bs_strategy, False)

    expected = {
        '301': {'ndcg@10': [0.6309297535714575, 0.6309297535714575, 0.0, 0.6309297535714575, 0.0]},
        '302': {'ndcg@10': [0.6309297535714575, 0.6309297535714575, 0.0, 0.6309297535714575, 0.0]}
    }

    actual = bs_eval.bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)

    print(actual)
    assert expected == actual


def test_bootstrap_with_some_relevant_and_some_irrelevant():
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

        {'query': '301', 'q0': 0, 'docid': 'doc-11', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-22', 'rel': 0},
        
        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        {'query': '302', 'q0': 0, 'docid': 'doc-11', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-22', 'rel': 2},
    ])

    bs_strategy = FullyIndependentBootstrappingStrategey(qrels)
    bs_eval = BootstrappEval('ndcg@10', bs_strategy, False)

    expected = {
        '301': {'ndcg@10': [0.9197207891481876, 0.9197207891481876, 0.6131471927654584, 0.9197207891481876, 0.6131471927654584]},
        '302': {'ndcg@10':  [0.38685280723454163, 0.38685280723454163, 0.6934264036172708, 0.38685280723454163, 0.6934264036172708]}}
    actual = bs_eval.bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)
    
    print(actual)
    print(expected)
    
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

        {'query': '301', 'q0': 0, 'docid': 'doc-11', 'rel': 1},
        {'query': '301', 'q0': 0, 'docid': 'doc-22', 'rel': 0},

        {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        {'query': '302', 'q0': 0, 'docid': 'doc-11', 'rel': 0},
        {'query': '302', 'q0': 0, 'docid': 'doc-22', 'rel': 2},
    ])

    expected = [
        {'run_file': 'a', 'query': '301', 'ndcg@10': [0.9197207891481876, 0.9197207891481876, 0.6131471927654584, 0.9197207891481876, 0.6131471927654584]},
        {'run_file': 'a', 'query': '302', 'ndcg@10': [0.38685280723454163, 0.38685280723454163, 0.6934264036172708, 0.38685280723454163, 0.6934264036172708]}
    ]

    bs_strategy = FullyIndependentBootstrappingStrategey(qrels)
    bs_eval = BootstrappEval('ndcg@10', bs_strategy, False)

    actual = list(normalize_eval_output(bs_eval.bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1), 'a'))
    
    print(actual)
    print(expected)
    
    assert expected == actual

