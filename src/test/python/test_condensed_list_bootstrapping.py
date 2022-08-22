from unittest import TestCase
from condensed_list_bootstrap_util import create_substitute_pools_for_condensed_lists, \
    substitute_pools_for_condensed_lists_with_effectivenes_scores
from trectools import TrecRun, TrecQrel
import pandas as pd
import json


class TestCondensedListBootstrapping(TestCase):

    def test_creation_of_substitute_pools_for_no_unjudged_documents(self):
        run = TrecRun('src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt')
        qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
        expected = {'301': ['{}']}
        actual = create_substitute_pools_for_condensed_lists(run, qrels, 10)

        assert expected == actual

    def test_too_small_pool_fails(self):
        run = TrecRun('src/test/resources/dummy-run-files-robust04/input.Juru-dummy-01.txt')

        qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},

            # Some noise
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        try:
            create_substitute_pools_for_condensed_lists(run, qrels, 10)
        except:
            return

        assert False

    def test_creation_of_substitute_pools_for_some_unjudged_documents(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
        qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},

            # Some noise
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        # Unjudged are: doc-juru-01, shared-doc-01
        expected = {'301': sorted([
            json.dumps({'doc-juru-01': 'doc-wdo-01', 'shared-doc-01': 'doc-wdo-02'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'REMOVE-THIS-DOCUMENT', 'shared-doc-01': 'doc-wdo-01'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'doc-wdo-01', 'shared-doc-01': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'REMOVE-THIS-DOCUMENT', 'shared-doc-01': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True),
        ]), '302': ['{}']}
        actual = create_substitute_pools_for_condensed_lists(run, qrels, 10)
        print(json.dumps(actual))
        print(json.dumps(expected))

        assert expected == actual

    def test_creation_of_substitute_pools_for_some_unjudged_documents_02(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
        qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'a-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'b-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},

            # Some noise
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        expected = {'301': sorted([
            json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'b-0'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'REMOVE-THIS-DOCUMENT', 'shared-doc-01': 'a-0'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'REMOVE-THIS-DOCUMENT', 'shared-doc-01': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True),
        ]), '302': ['{}']}
        actual = create_substitute_pools_for_condensed_lists(run, qrels, 10)
        print(json.dumps(actual))
        print(json.dumps(expected))

        assert expected == actual

    def test_creation_of_substitute_pools_for_some_unjudged_documents_03(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
        qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'a-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'b-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'c-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'd-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'e-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},

            # Some noise
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        # Unjudged are: doc-juru-01, shared-doc-01
        expected = {'301': sorted([
            json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'b-0'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'REMOVE-THIS-DOCUMENT', 'shared-doc-01': 'a-0'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'REMOVE-THIS-DOCUMENT', 'shared-doc-01': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True),
        ]), '302': ['{}']}
        actual = create_substitute_pools_for_condensed_lists(run, qrels, 10)
        print(json.dumps(actual))
        print(json.dumps(expected))

        assert expected == actual

    def test_creation_of_substitute_pools_for_some_unjudged_documents_with_many_different_qrels_large_pool(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
        qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'a-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'b-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'c-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'd-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'e-0', 'rel': 0},

            {'query': '301', 'q0': 0, 'docid': 'a-1', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'b-1', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'c-1', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'd-1', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'e-1', 'rel': 1},

            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},

            # Some noise
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        # Unjudged are: doc-juru-01, shared-doc-01
        expected = {'301': sorted([

            json.dumps({'doc-juru-01': 'REMOVE-THIS-DOCUMENT', 'shared-doc-01': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'REMOVE-THIS-DOCUMENT', 'shared-doc-01': 'a-0'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'b-0'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'a-1', 'shared-doc-01': 'b-1'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'a-0', 'shared-doc-01': 'a-1'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'a-1', 'shared-doc-01': 'a-0'}, sort_keys=True),

            json.dumps({'doc-juru-01': 'a-1', 'shared-doc-01': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True),
            json.dumps({'doc-juru-01': 'REMOVE-THIS-DOCUMENT', 'shared-doc-01': 'a-1'}, sort_keys=True),
        ]), '302': ['{}']}

        actual = create_substitute_pools_for_condensed_lists(run, qrels, 10)
        print(json.dumps(actual))
        print(json.dumps(expected))

        assert expected == actual


    def test_substitate_pools_with_effectivenes_scores_for_single_unjudged_document_with_single_qrels_huge_pool_02(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
            {'query': '302', 'q0': 'Q0', 'docid': 'unjudged', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
        qrels = TrecQrel('src/test/resources/dummy-qrels-robust04.txt')
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'a-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'b-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'c-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'd-0', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'e-0', 'rel': 0},

            {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'shared-doc-01', 'rel': 1},

             # Some noise
             {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        # Unjudged are: doc-juru-01, shared-doc-01
        expected = {'301': {
            json.dumps({'doc-juru-02': 'a-0'}, sort_keys=True): 0.9197207891481876,
            json.dumps({'doc-juru-02': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True): 1.0,
        }, '302': {
            json.dumps({'unjudged': 'doc-0'}, sort_keys=True): 0.0,
            json.dumps({'unjudged': 'doc-1'}, sort_keys=True): 0.38009376671593426,
            json.dumps({'unjudged': 'doc-2'}, sort_keys=True): 0.7601875334318685,
            json.dumps({'unjudged': 'REMOVE-THIS-DOCUMENT'}, sort_keys=True): 0.0,
        }}

        actual = substitute_pools_for_condensed_lists_with_effectivenes_scores(run, qrels, 'ndcg@10')
        print(json.dumps(actual))
        print(json.dumps(expected))

        assert expected == actual
