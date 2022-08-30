import json
from trectools import TrecPool, TrecRun
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import gzip


class RunFileGroups:
    def __init__(self, group_definition_file, trec_identifier):
        self.__groups = json.load(open(group_definition_file))[trec_identifier]

    def assign_runs_to_groups(self, runs):
        ret = {}
        for run in runs:
            group = self.assign_run_to_group_or_return_none(run)

            if not group:
                group = self.assign_run_to_group_or_return_none(run, case_insensitive=True)

            if not group:
                raise ValueError('I cant determine a group for ' + run)
            
            if group not in ret:
                ret[group] = []
            
            ret[group] += [run]
        return ret

    def assign_run_to_group_or_return_none(self, run, case_insensitive=False):
        if not self.__groups or not 'items' in dir(self.__groups) or not self.__groups.items():
            return None
        
        ret = []
        for group_name, group_definition in self.__groups.items():
            # Currently, we only support group definitions via prefixes
            assert {'prefix'} == group_definition.keys()
            if ('/' + group_definition['prefix']) in run:
                ret += [group_name]
            if case_insensitive and ('/' + group_definition['prefix']).lower() in run.lower():
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

    trecformat = run.run_data.copy().sort_values(["query", "score", "docid"], ascending=[True, False, False]).reset_index()
    topX = trecformat.groupby("query")[["query", "q0", "docid", "rank", "score", "system"]].head(depth)

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


class IncompletePools:
    def __init__(self, run_dir=None, group_definition_file=None, trec_identifier=None, pool_per_run_file=None):
        self.__runs = load_all_runs(run_dir) if run_dir else None
        self.__run_file_groups = RunFileGroups(group_definition_file, trec_identifier).assign_runs_to_groups(self.__runs.keys()) if group_definition_file and trec_identifier else None
        self.pool_per_run_file = pool_per_run_file

    def pool_per_runs(self):
        if self.pool_per_run_file:
            ret = json.load(gzip.open(self.pool_per_run_file))
            self.__run_file_groups = ret['groups']
            return ret

        ret = {'pool_entries': {'10': {}, '20': {}}, 'groups': self.__run_file_groups}
        
        print('Create Pool Entries for all Runs.')
        for run_name, run in tqdm(self.__runs.items()):
            for depth in ret['pool_entries'].keys():
                ret['pool_entries'][depth][run_name] = self.incomplete_pools([i for i in self.__runs.keys() if i != run_name], int(depth))

        return ret

    def create_all_incomplete_pools(self):
        ret = {}
        all_runs = set()
        for _, runs_to_skip in self.__run_file_groups.items():
            all_runs = all_runs.union(runs_to_skip)

        for run in all_runs:
            for pool_name, pool in self.create_incomplete_pools_for_run(run):
                ret[pool_name] = pool

        return ret

    def create_incomplete_pools_for_run(self, run):
        if not hasattr(self, 'pool_per_run'):
            self.pool_per_run = self.pool_per_runs()

        for depth in ['10', '20']:
            yield ('complete-pool-depth-' + str(depth), self.complete_pool_at_depth(depth))
            for exclusion_group_name, runs_to_skip in self.__run_file_groups.items():
                if run not in runs_to_skip:
                    continue
                
                pool = None
                
                for run_name, run_pool in self.pool_per_run['pool_entries'][depth].items():
                    if run_name in runs_to_skip:
                        continue
                    if pool is None:
                        pool = deepcopy(run_pool)
                    
                    for topic, topic_pool in run_pool.items():
                        if topic not in pool:
                            pool[topic] = []
                        pool[topic] = pool[topic] + topic_pool
                
                pool = {k:sorted(list(set(v))) for k,v in pool.items()}
                yield ('depth-' + str(depth) + '-pool-incomplete-for-' + exclusion_group_name, pool)

    def complete_pool_at_depth(self, depth):
        if not hasattr(self, 'pool_per_run'):
            self.pool_per_run = self.pool_per_runs()

        pool = None
        for run_name, run_pool in self.pool_per_run['pool_entries'][depth].items():
            if pool is None:
                pool = deepcopy(run_pool)
                    
            for topic, topic_pool in run_pool.items():
                if topic not in pool:
                    pool[topic] = []
                pool[topic] = pool[topic] + topic_pool

        return {k: sorted(list(set(v))) for k, v in pool.items()}

    def incomplete_pools(self, runs_to_skip, depth):
        skipped_runs = 0
        runs_for_pooling = []
        
        for run_name, run in self.__runs.items():
            if run_name in runs_to_skip:
                skipped_runs += 1
                continue
            runs_for_pooling += [run]
        
        assert skipped_runs == len(runs_to_skip)
        return {k: sorted(list(v)) for k,v in make_top_x_pool(runs_for_pooling, depth).pool.items()}

