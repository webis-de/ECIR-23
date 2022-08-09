from test_creation_of_incomplete_pools import EXPECTED_POOL_ROBUST04_SAMPLES
from evaluation_util import evaluate_on_pools, __adjust_qrels_to_pool, __create_max_residual_qrel, __create_min_residual_qrel
import json
from trectools import TrecQrel, TrecRun

def test_adjustment_of_pools_juru_incomplete():
    qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
    expected = set(['doc-wdo-01', 'doc-wdo-02', 'shared-doc-01'])
    actual = set(__adjust_qrels_to_pool(qrels, EXPECTED_POOL_ROBUST04_SAMPLES['depth-10-pool-incomplete-for-juru']).qrels_data['docid'].unique())
    
    assert expected == actual

def test_adjustment_of_pools_wdo_incomplete():
    qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
    expected = set(['doc-juru-01', 'doc-juru-02', 'shared-doc-01'])
    actual = set(__adjust_qrels_to_pool(qrels, EXPECTED_POOL_ROBUST04_SAMPLES['depth-10-pool-incomplete-for-wdo']).qrels_data['docid'].unique())
    
    assert expected == actual


def __jsonify_df(df):
    return json.dumps([dict(i) for _, i in df.iterrows()])


def test_max_qrels_for_wdo_with_complete_pools():
    run = TrecRun('src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt')
    qrels = __adjust_qrels_to_pool(TrecQrel('src/test/resources/dummy-qrels-robust04.txt'), EXPECTED_POOL_ROBUST04_SAMPLES['depth-10-pool-incomplete-for-wdo'])
    
    expected = __jsonify_df(qrels.qrels_data)
    actual = __create_max_residual_qrel(run, qrels, 3)
    
    assert expected == __jsonify_df(actual.qrels_data)


def test_min_qrels_for_wdo_with_complete_pools():
    run = TrecRun('src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt')
    qrels = __adjust_qrels_to_pool(TrecQrel('src/test/resources/dummy-qrels-robust04.txt'), EXPECTED_POOL_ROBUST04_SAMPLES['depth-10-pool-incomplete-for-wdo'])
    
    expected = __jsonify_df(qrels.qrels_data)
    actual = __create_min_residual_qrel(run, qrels, 3)
    
    assert expected == __jsonify_df(actual.qrels_data)


def test_max_qrels_for_wdo_with_incomplete_pools():
    run = TrecRun('src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt')
    qrels = __adjust_qrels_to_pool(TrecQrel('src/test/resources/dummy-qrels-robust04.txt'), EXPECTED_POOL_ROBUST04_SAMPLES['depth-10-pool-incomplete-for-juru'])
    
    expected = '[{"query": "301", "q0": "0", "docid": "shared-doc-01", "rel": 1}, {"query": "301", "q0": "0", "docid": "doc-wdo-02", "rel": 0}, {"query": "301", "q0": "0", "docid": "doc-wdo-01", "rel": 2}, {"query": "301", "q0": "0", "docid": "doc-juru-01", "rel": 2}, {"query": "301", "q0": "0", "docid": "doc-juru-02", "rel": 2}]'

    actual = __create_max_residual_qrel(run, qrels, 3)
    
    print(__jsonify_df(actual.qrels_data))
    assert expected == __jsonify_df(actual.qrels_data)


def test_min_qrels_for_wdo_with_incomplete_pools():
    run = TrecRun('src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt')
    qrels = __adjust_qrels_to_pool(TrecQrel('src/test/resources/dummy-qrels-robust04.txt'), EXPECTED_POOL_ROBUST04_SAMPLES['depth-10-pool-incomplete-for-juru'])
    
    expected = '[{"query": "301", "q0": "0", "docid": "shared-doc-01", "rel": 1}, {"query": "301", "q0": "0", "docid": "doc-wdo-02", "rel": 0}, {"query": "301", "q0": "0", "docid": "doc-wdo-01", "rel": 2}, {"query": "301", "q0": "0", "docid": "doc-juru-01", "rel": 0}, {"query": "301", "q0": "0", "docid": "doc-juru-02", "rel": 0}]'

    actual = __create_min_residual_qrel(run, qrels, 3)
    
    print(__jsonify_df(actual.qrels_data))
    assert expected == __jsonify_df(actual.qrels_data)


def test_unjudged_at_3_for_juru_run():
    run_file = 'src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt'
    qrel_file = 'src/test/resources/dummy-qrels-robust04.txt'
    measure = 'unjudged@3'
    
    expected = {
        "complete-pool": [{"UNJ@3": 0.0, "query": "301", 'run_file': run_file}],
        "depth-10-pool-incomplete-for-juru": [{"UNJ@3": 0.6666666666666666, "query": "301", 'run_file': run_file}],
        "depth-10-pool-incomplete-for-wdo": [{"UNJ@3": 0.0, "query": "301", 'run_file': run_file}],
        "depth-20-pool-incomplete-for-juru": [{"UNJ@3": 0.6666666666666666, "query": "301", 'run_file': run_file}],
        "depth-20-pool-incomplete-for-wdo": [{"UNJ@3": 0.0, "query": "301", 'run_file': run_file}]
    }

    actual = evaluate_on_pools(run_file, qrel_file, EXPECTED_POOL_ROBUST04_SAMPLES, measure)
    print(json.dumps(actual))

    assert expected == actual


def test_unjudged_at_3_for_wdo_run():
    run_file = 'src/test/resources/dummy-run-files-robust04/input.wdo-dummy-01.txt'
    qrel_file = 'src/test/resources/dummy-qrels-robust04.txt'
    measure = 'unjudged@3'
    
    expected = {
        "complete-pool": [{"UNJ@3": 0.0, "query": "301", 'run_file': run_file}],
        "depth-10-pool-incomplete-for-juru": [{"UNJ@3": 0.0, "query": "301", 'run_file': run_file}],
        "depth-10-pool-incomplete-for-wdo": [{"UNJ@3": 0.6666666666666666, "query": "301", 'run_file': run_file}],
        "depth-20-pool-incomplete-for-juru": [{"UNJ@3": 0.0, "query": "301", 'run_file': run_file}],
        "depth-20-pool-incomplete-for-wdo": [{"UNJ@3": 0.6666666666666666, "query": "301", 'run_file': run_file}]
    }

    actual = evaluate_on_pools(run_file, qrel_file, EXPECTED_POOL_ROBUST04_SAMPLES, measure)
    print(json.dumps(actual))

    assert expected == actual


def test_ndcg_at_3_for_juru_run():
    run_file = 'src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt'
    qrel_file = 'src/test/resources/dummy-qrels-robust04.txt'
    measure = 'ndcg@3'
    
    expected = {
        "complete-pool": [{"NDCG@3": 0.6645649565734895, "query": "301", 'run_file': run_file}],
        "depth-10-pool-incomplete-for-juru": [{"NDCG@3": 0.19004688335796713, "query": "301", 'run_file': run_file}],
        "depth-10-pool-incomplete-for-wdo": [{"NDCG@3": 0.9502344167898356, "query": "301", 'run_file': run_file}],
        "depth-20-pool-incomplete-for-juru": [{"NDCG@3": 0.19004688335796713, "query": "301", 'run_file': run_file}],
        "depth-20-pool-incomplete-for-wdo": [{"NDCG@3": 0.9502344167898356, "query": "301", 'run_file': run_file}]
    }

    actual = evaluate_on_pools(run_file, qrel_file, EXPECTED_POOL_ROBUST04_SAMPLES, measure)
    print(json.dumps(actual))

    assert expected == actual


def test_condensed_ndcg_at_3_for_juru_run():
    run_file = 'src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt'
    qrel_file = 'src/test/resources/dummy-qrels-robust04.txt'
    measure = 'condensed-ndcg@3'
    
    expected = {
        "complete-pool": [{"NDCG@3": 0.6645649565734895, "query": "301", 'run_file': run_file}],
        "depth-10-pool-incomplete-for-juru": [{"NDCG@3": 0.38009376671593426, "query": "301", 'run_file': run_file}],
        "depth-10-pool-incomplete-for-wdo": [{"NDCG@3": 0.9502344167898356, "query": "301", 'run_file': run_file}],
        "depth-20-pool-incomplete-for-juru": [{"NDCG@3": 0.38009376671593426, "query": "301", 'run_file': run_file}],
        "depth-20-pool-incomplete-for-wdo": [{"NDCG@3": 0.9502344167898356, "query": "301", 'run_file': run_file}]
    }

    actual = evaluate_on_pools(run_file, qrel_file, EXPECTED_POOL_ROBUST04_SAMPLES, measure)
    print(json.dumps(actual))

    assert expected == actual


def test_condensed_ndcg_at_3_for_juru_run():
    run_file = 'src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt'
    qrel_file = 'src/test/resources/dummy-qrels-robust04.txt'
    measure = 'residual-ndcg@3'
    
    expected = {"complete-pool": [{"MIN-NDCG@3": 0.6645649565734895, "MAX-NDCG@3": 0.6645649565734895, "run_file": "src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt", "query": "301"}], "depth-10-pool-incomplete-for-juru": [{"MIN-NDCG@3": 0.19004688335796713, "MAX-NDCG@3": 0.8826803184943108, "run_file": "src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt", "query": "301"}], "depth-10-pool-incomplete-for-wdo": [{"MIN-NDCG@3": 0.9502344167898356, "MAX-NDCG@3": 0.9502344167898356, "run_file": "src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt", "query": "301"}], "depth-20-pool-incomplete-for-juru": [{"MIN-NDCG@3": 0.19004688335796713, "MAX-NDCG@3": 0.8826803184943108, "run_file": "src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt", "query": "301"}], "depth-20-pool-incomplete-for-wdo": [{"MIN-NDCG@3": 0.9502344167898356, "MAX-NDCG@3": 0.9502344167898356, "run_file": "src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt", "query": "301"}]}

    actual = evaluate_on_pools(run_file, qrel_file, EXPECTED_POOL_ROBUST04_SAMPLES, measure)
    print(json.dumps(actual))

    assert expected == actual

