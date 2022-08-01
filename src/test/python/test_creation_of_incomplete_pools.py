from run_file_processing import IncompletePools

def test_creation_of_incomplete_pools_for_small_robust04_samples():
    pooling = IncompletePools('src/test/resources/dummy-run-files-robust04', 'src/main/resources/processed/trec-system-runs-groups.json', 'trec-system-runs/trec13/robust')
    
    expected = {
        'depth-10-pool-incomplete-for-juru': {'301': {'doc-wdo-01', 'doc-wdo-02'}},
        'depth-10-pool-incomplete-for-wdo': {'301': {'doc-juru-01', 'doc-juru-02'}},
        
        'depth-20-pool-incomplete-for-juru': {'301': {'doc-wdo-01', 'doc-wdo-02'}},
        'depth-20-pool-incomplete-for-wdo': {'301': {'doc-juru-01', 'doc-juru-02'}},
    }
    actual = pooling.create_all_incomplete_pools()
    
    assert expected == actual

