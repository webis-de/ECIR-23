from run_file_processing import RunFileGroups


def test_assignment_of_runs_to_groups_on_robust04():
    run_file_groups = RunFileGroups('src/main/resources/processed/trec-system-runs-groups.json', 'trec-system-runs/trec13/robust')
    
    expected = {'juru': ['trec-system-runs/trec13/robust/input.Juru.asd.gz']}
    actual = run_file_groups.assign_runs_to_groups(['trec-system-runs/trec13/robust/input.Juru.asd.gz'])
    
    assert expected == actual


def test_assignment_of_runs_to_groups_fails_on_unknown_runs_for_robust04():
    run_file_groups = RunFileGroups('src/main/resources/processed/trec-system-runs-groups.json', 'trec-system-runs/trec13/robust')
    
    try:
        run_file_groups.assign_runs_to_groups(['trec-system-runs/trec13/robust/input.Juhu.asd.gz'])
    except:
        return
    
    assert False

