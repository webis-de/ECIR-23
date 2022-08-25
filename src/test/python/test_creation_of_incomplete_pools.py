from run_file_processing import IncompletePools
import json

EXPECTED_POOL_ROBUST04_SAMPLES = {
    'complete-pool-depth-10': {'301': ['doc-juru-01', 'doc-juru-02', 'doc-wdo-01', 'doc-wdo-02', 'shared-doc-01']},
    'complete-pool-depth-20': {'301': ['doc-juru-01', 'doc-juru-02', 'doc-wdo-01', 'doc-wdo-02', 'shared-doc-01']},
    'depth-10-pool-incomplete-for-juru': {'301': ['doc-wdo-01', 'doc-wdo-02', 'shared-doc-01']},
    'depth-20-pool-incomplete-for-juru': {'301': ['doc-wdo-01', 'doc-wdo-02', 'shared-doc-01']},
    'depth-10-pool-incomplete-for-wdo': {'301': ['doc-juru-01', 'doc-juru-02', 'shared-doc-01']},
    'depth-20-pool-incomplete-for-wdo': {'301': ['doc-juru-01', 'doc-juru-02', 'shared-doc-01']},
}

EXPECTED_POOLS_PER_RUN_ON_ROBUST04_SAMPLES = {'pool_entries': {
    10: {
        'src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt': {'301': ['doc-juru-01', 'doc-juru-02', 'shared-doc-01']},
        'src/test/resources/dummy-run-files-robust04/input.wdo-dummy-01.txt': {'301': ['doc-wdo-01', 'doc-wdo-02', 'shared-doc-01']}
    },
    20: {
        'src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt': {'301': ['doc-juru-01', 'doc-juru-02', 'shared-doc-01']},
        'src/test/resources/dummy-run-files-robust04/input.wdo-dummy-01.txt': {'301': ['doc-wdo-01', 'doc-wdo-02', 'shared-doc-01']}
    }
},
'groups': {'juru': ['src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt'], 'wdo': ['src/test/resources/dummy-run-files-robust04/input.wdo-dummy-01.txt']}}


def test_creation_of_incomplete_pools_for_small_robust04_samples():
    pooling = IncompletePools('src/test/resources/dummy-run-files-robust04', 'src/main/resources/processed/trec-system-runs-groups.json', 'trec-system-runs/trec13/robust')
    
    actual = pooling.create_all_incomplete_pools()
    
    assert json.dumps(EXPECTED_POOL_ROBUST04_SAMPLES, sort_keys=True) == json.dumps(actual, sort_keys=True)


def test_creation_of_pool_per_runs():
    pooling = IncompletePools('src/test/resources/dummy-run-files-robust04', 'src/main/resources/processed/trec-system-runs-groups.json', 'trec-system-runs/trec13/robust')
    
    actual = pooling.pool_per_runs()

    assert json.dumps(EXPECTED_POOLS_PER_RUN_ON_ROBUST04_SAMPLES, sort_keys=True) == json.dumps(actual, sort_keys=True)


def test_creation_of_pool_for_run():
    pooling = IncompletePools('src/test/resources/dummy-run-files-robust04', 'src/main/resources/processed/trec-system-runs-groups.json', 'trec-system-runs/trec13/robust')
    
    actual = {k: v for (k, v) in pooling.create_incomplete_pools_for_run('src/test/resources/dummy-run-files-robust04/input.wdo-dummy-01.txt')}
    expected = {
        'complete-pool-depth-10': {'301': ['doc-juru-01', 'doc-juru-02', 'doc-wdo-01', 'doc-wdo-02', 'shared-doc-01']},
        'complete-pool-depth-20': {'301': ['doc-juru-01', 'doc-juru-02', 'doc-wdo-01', 'doc-wdo-02', 'shared-doc-01']},
        'depth-10-pool-incomplete-for-wdo': {'301': ['doc-juru-01', 'doc-juru-02', 'shared-doc-01']},
        'depth-20-pool-incomplete-for-wdo': {'301': ['doc-juru-01', 'doc-juru-02', 'shared-doc-01']},
    }
    
    print(actual)
    assert expected == actual
