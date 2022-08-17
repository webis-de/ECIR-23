from trectools import TrecRun, TrecQrel
from evaluation_util import __evaluate_run_on_pool, __adjust_qrels_to_pool
from run_file_processing import IncompletePools


def test_example_for_which_min_ndcg_is_above_max_ndcg():
    run = TrecRun('src/main/resources/processed/normalized-runs/trec-system-runs/trec13/robust/input.JuruTitDes.gz')
    run.run_data = run.run_data[run.run_data['query'] == '690']
    qrels = TrecQrel('src/main/resources/unprocessed/topics-and-qrels/qrels.robust04.txt')
    qrels.qrels_data = qrels.qrels_data[qrels.qrels_data['query'] == '690']
    pooling = IncompletePools(pool_per_run_file='src/main/resources/processed/pool-documents-per-run-trec-system-runs-trec13-robust.json.gz')
        
    pool = {k:v for k,v in pooling.create_incomplete_pools_for_run('src/main/resources/processed/normalized-runs/trec-system-runs/trec13/robust/input.JuruTitDes.gz')}
    
    # This is a special case in which the minimum nDCG can be higher than the actual nDCG
    actual = __evaluate_run_on_pool(run, qrels, 'residual-ndcg@10', pool['depth-10-pool-incomplete-for-juru'], 'tbd')
    print(actual)
    assert len(actual) == 1
    assert actual[0]['MIN-NDCG@10'] == 0.09109240322345806
    
    # 
    actual = __evaluate_run_on_pool(run, qrels, 'ndcg@10', pool['complete-pool-depth-10'], 'tbd')
    print(actual)
    assert len(actual) == 1
    assert actual[0]['NDCG@10'] == 0.08274602130532002


def test_example_for_which_max_ndcg_is_wrong_if_idcg_is_changed():
    run = TrecRun('src/main/resources/processed/normalized-runs/trec-system-runs/trec13/robust/input.JuruDes.gz')
    run.run_data = run.run_data[run.run_data['query'] == '348']
    qrels = TrecQrel('src/main/resources/unprocessed/topics-and-qrels/qrels.robust04.txt')
    qrels.qrels_data = qrels.qrels_data[qrels.qrels_data['query'] == '348']
    pooling = IncompletePools(pool_per_run_file='src/main/resources/processed/pool-documents-per-run-trec-system-runs-trec13-robust.json.gz')
        
    pool = {k:v for k,v in pooling.create_incomplete_pools_for_run('src/main/resources/processed/normalized-runs/trec-system-runs/trec13/robust/input.JuruDes.gz')}
    
    actual = __evaluate_run_on_pool(run, qrels, 'residual-ndcg@10', pool['depth-10-pool-incomplete-for-juru'], 'tbd')
    print(actual)
    assert len(actual) == 1
    assert actual[0]['MIN-NDCG@10'] == 0.8954442351097147
    assert actual[0]['MAX-NDCG@10'] == 0.8954442351097147
     
    actual = __evaluate_run_on_pool(run, qrels, 'ndcg@10', pool['complete-pool-depth-10'], 'tbd')
    print(actual)
    assert len(actual) == 1
    assert actual[0]['NDCG@10'] == 0.8954442351097147

