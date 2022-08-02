import json
from trectools import TrecPool, TrecRun, TrecPoolMaker
from pathlib import Path
from tqdm import tqdm

class RunFileGroups():
    def __init__(self, group_definition_file, trec_identifier):
        self.__groups = json.load(open(group_definition_file))[trec_identifier]

    def assign_runs_to_groups(self, runs):
        ret = {}
        for run in runs:
            group = self.assign_run_to_group_or_return_none(run)
            assert group is not None
            
            if group not in ret:
                ret[group] = []
            
            ret[group] += [run]
        return ret

    def assign_run_to_group_or_return_none(self, run):
        if not self.__groups or not 'items' in dir(self.__groups) or not self.__groups.items():
            return None
        
        ret = []
        for group_name, group_definition in self.__groups.items():
            assert set(['prefix']) == group_definition.keys() # Currently, we only support group definitions via prefixes
            if ('/' + group_definition['prefix']) in run:
                ret += [group_name]

        return ret[0] if len(ret) == 1 else None

def load_all_runs(run_dir):
    run_dir = Path(str(run_dir))
    ret = {}
    print('Load runs: ')
    for run in tqdm(list(run_dir.glob('*'))):
        ret[str(run)] = TrecRun(run)
    return ret

def normalize_run(run, depth=100):
    """
    Break score ties, etc, as in trec_eval.
    As implemented in trectools: https://github.com/joaopalotti/trectools/blob/master/trectools/trec_eval.py#L230
    """

    trecformat = run.run_data.copy().sort_values(["query", "score", "docid"], ascending=[True,False,False]).reset_index()
    topX = trecformat.groupby("query")[["query","docid","score"]].head(depth)

    # Make sure that rank position starts by 1
    topX["rank"] = 1
    topX["rank"] = topX.groupby("query")["rank"].cumsum()
    
    ret = TrecRun()
    ret.run_data = topX
    
    return ret


def make_top_x_pool(list_of_runs, depth):
    pool_documents = {}

    for run in list_of_runs:
        run = normalize_run(run, depth)
        
        for _, i in run.run_data.iterrows():
            topic = str(i['query'])
            doc = str(i['docid'])
            
            if topic not in pool_documents:
                pool_documents[topic] = set([])
            
            pool_documents[topic].add(doc)

    return TrecPool(pool_documents)


class IncompletePools():
    def __init__(self, run_dir, group_definition_file, trec_identifier):
        self.__runs = load_all_runs(run_dir)
        self.__run_file_groups = RunFileGroups(group_definition_file, trec_identifier).assign_runs_to_groups(self.__runs.keys())

    def create_all_incomplete_pools(self):
        ret = {}
        for depth in [10, 20]:
            print('Create incomplete judgment pools at depth ' + str(depth))
            for exclusion_group_name, runs_to_skip in tqdm(self.__run_file_groups.items()):
                ret['depth-' + str(depth) + '-pool-incomplete-for-' + exclusion_group_name] = self.incomplete_pools(set(runs_to_skip), depth)
        return ret

    def incomplete_pools(self, runs_to_skip, depth):
        skipped_runs = 0
        runs_for_pooling = []
        
        for run_name, run in self.__runs.items():
            if run_name in runs_to_skip:
                skipped_runs += 1
                continue
            runs_for_pooling += [run]
        
        assert skipped_runs == skipped_runs
        return make_top_x_pool(runs_for_pooling, depth).pool

