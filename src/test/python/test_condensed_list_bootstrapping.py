from unittest import TestCase
from condensed_list_bootstrap_util import create_substitute_pools_for_condensed_lists, \
    substitute_pools_for_condensed_lists_with_effectivenes_scores
from condensed_list_bootstrap_util import __rels_for_topic as tmp___rels_for_topic, evaluate_bootstrap
from trectools import TrecRun, TrecQrel
import pandas as pd
from evaluation_util import normalize_eval_output
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

    def test_rels_for_topic_for_multiple_judged_doc_01(self):
        run = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])

        qrels = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'shared-doc-01', 'rel': 0},

            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        expected = []

        actual = tmp___rels_for_topic(run, qrels, 3)
        print(actual)

        assert expected == actual

    def test_further_rels_example_used_in_bootstrap_with_some_relevant_and_some_irrelevant_with_normalize_output(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '302', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '302', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '302', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
            {'query': '302', 'q0': 'Q0', 'docid': 'doc-14', 'rank': 3, 'score': 2998, 'system': 'a'},
        ])

        qrels = TrecQrel()
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
            {'query': '301', 'q0': 0, 'docid': 'doc-5', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-3', 'rel': 0},

            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-3', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-5', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-6', 'rel': 0},
        ])

        expected = {'301': ['{}'], '302': ['{"doc-2": "REMOVE-THIS-DOCUMENT"}', '{"doc-2": "doc-5"}']}

        actual = create_substitute_pools_for_condensed_lists(run, qrels, 3)
        print(actual)

        assert expected == actual

    def test_rels_for_topic_for_multiple_judged_doc_02(self):
        run = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])

        qrels = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},

            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        # 33% are unjudged, so with uprounding I have to include one "remove this" entry
        expected = [['doc-wdo-01'], ['REMOVE-THIS-DOCUMENT']]
        actual = tmp___rels_for_topic(run, qrels, 3)
        print(actual)
        assert sorted(expected) == sorted(actual)

    def test_rels_for_topic_for_multiple_judged_doc_03(self):
        run = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])

        qrels = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-03', 'rel': 0},

            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        # 33% are unjudged, so with uprounding I have to include one "remove this" entry
        expected = [['doc-wdo-01'], ['REMOVE-THIS-DOCUMENT']]
        actual = tmp___rels_for_topic(run, qrels, 3)

        assert sorted(expected) == sorted(actual)

    def test_rels_for_topic_for_multiple_judged_doc_04(self):
        run = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])

        qrels = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-02', 'rel': 2},
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-03', 'rel': 0},

            {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 1},

            {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 2},
            {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},

            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        expected = [['doc-wdo-01'], ['a'], ['c'], ['REMOVE-THIS-DOCUMENT']]
        actual = tmp___rels_for_topic(run, qrels, 3)

        assert sorted(expected) == sorted(actual)

    def test_rels_for_topic_for_multiple_judged_doc_05(self):
        run = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-01', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-juru-02', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'shared-doc-01', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])

        qrels = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-01', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-juru-01', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-02', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-wdo-03', 'rel': 0},

            {'query': '301', 'q0': 0, 'docid': 'a', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'b', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'i', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'j', 'rel': 1},

            {'query': '301', 'q0': 0, 'docid': 'c', 'rel': 2},
            {'query': '301', 'q0': 0, 'docid': 'd', 'rel': 2},
            {'query': '301', 'q0': 0, 'docid': 'k', 'rel': 2},
            {'query': '301', 'q0': 0, 'docid': 'l', 'rel': 2},

            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        # 33% are unjudged, so with uprounding I have to include four "remove this" entry
        expected = [['doc-wdo-01', 'doc-wdo-02'], ['doc-wdo-01', 'doc-wdo-02'], ['a', 'b'], ['a', 'b'], ['c', 'd'],
                    ['c', 'd'], ['REMOVE-THIS-DOCUMENT', 'REMOVE-THIS-DOCUMENT'],
                    ['REMOVE-THIS-DOCUMENT', 'REMOVE-THIS-DOCUMENT'], ['REMOVE-THIS-DOCUMENT', 'REMOVE-THIS-DOCUMENT'],
                    ['REMOVE-THIS-DOCUMENT', 'REMOVE-THIS-DOCUMENT']]
        print(expected)
        actual = tmp___rels_for_topic(run, qrels, 3)
        print(actual)

        assert sorted(expected) == sorted(actual)

    def test_bootstrap_end_to_end_all_judged_01(self):
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

        expected = {'301': {'ndcg@10': [0.9197207891481876] * 5}, '302': {'ndcg@10': [0.66967181649423] * 5}}
        actual = evaluate_bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)

        print(actual)
        assert expected == actual

    def test_bootstrap_end_to_end_all_judged_02(self):
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

        expected = {'301': {'ndcg@10': [1.0] * 5}, '302': {'ndcg@10': [1.0] * 5}}
        actual = evaluate_bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)

        print(actual)
        assert expected == actual

    def test_bootstrap_end_to_end_all_judged_03(self):
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
            {'query': '301', 'q0': 0, 'docid': 'doc-2', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-20', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'doc-3', 'rel': 0},

            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-20', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-3', 'rel': 0},
        ])

        expected = {'301': {'ndcg@10': [0.0] * 5}, '302': {'ndcg@10': [0.0] * 5}}
        actual = evaluate_bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1)

        print(actual)
        assert expected == actual

    def test_bootstrap_with_some_relevant_and_some_irrelevant_with_normalize_output(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
            {'query': '302', 'q0': 'Q0', 'docid': 'doc-1', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '302', 'q0': 'Q0', 'docid': 'doc-2', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '302', 'q0': 'Q0', 'docid': 'doc-3', 'rank': 2, 'score': 2998, 'system': 'a'},
            {'query': '302', 'q0': 'Q0', 'docid': 'doc-4', 'rank': 3, 'score': 2997, 'system': 'a'},
        ])

        qrels = TrecQrel()
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
            {'query': '301', 'q0': 0, 'docid': 'doc-5', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'doc-3', 'rel': 0},

            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-3', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-5', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-6', 'rel': 0},
        ])

        expected = [
            {'run_file': 'a', 'query': '301',
             'ndcg@10': [1.0, 1.0, 0.6309297535714575, 1.0, 0.6309297535714575]},
            {'run_file': 'a', 'query': '302',
             'ndcg@10': [0.9197207891481876, 1.0, 1.0, 0.9197207891481876, 1.0]}
        ]
        actual = list(normalize_eval_output(evaluate_bootstrap(run, qrels, 'ndcg@10', repeat=5, seed=1), 'a'))

        print(actual)
        print(expected)

        assert expected == actual
