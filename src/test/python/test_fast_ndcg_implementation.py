from unittest import TestCase
from trectools import TrecEval, TrecRun, TrecQrel

from fast_ndcg_implementation import MemorizedNdcg

from error_analysis_approaches import create_run, create_qrels, ql, rl

QRELS_FEW_RELEVANTS_MANY_IRRELEVANT = create_qrels([
        ql('r-1', 1), ql('r-2', 1), ql('r-3', 1), ql('r-4', 1),

        ql('nr-1', 0), ql('nr-2', 0), ql('nr-3', 0), ql('nr-4', 0),
        ql('nr-5', 0), ql('nr-6', 0), ql('nr-7', 0), ql('nr-8', 0),
        ql('nr-9', 0), ql('nr-10', 0), ql('nr-11', 0), ql('nr-12', 0),
    ])

QRELS_MANY_RELEVANTS_FEW_IRRELEVANT = create_qrels([
        ql('r-1', 1), ql('r-2', 1), ql('r-3', 1), ql('r-4', 1),
        ql('r-5', 1), ql('r-6', 1), ql('r-7', 1), ql('r-8', 1),

        ql('nr-1', 0), ql('nr-2', 0), ql('nr-3', 0), ql('nr-4', 0),
        ql('nr-5', 0), ql('nr-6', 0), ql('nr-7', 0), ql('nr-8', 0),
        ql('nr-9', 0), ql('nr-10', 0)
    ])

memorized_ndcg_trec_eval = MemorizedNdcg(10, trec_eval=True)
memorized_ndcg_no_trec_eval = MemorizedNdcg(10, trec_eval=False)


def trectools_ndcg_eval(run, qrels, trec_eval):
    te = TrecEval(run, qrels)

    return te.get_ndcg(10, trec_eval=trec_eval)


def fast_ndcg_eval(run, qrels, trec_eval):
    if trec_eval:
        ret = memorized_ndcg_trec_eval.get_ndcg(run, qrels, 10, trec_eval)
    else:
        ret = memorized_ndcg_no_trec_eval.get_ndcg(run, qrels, 10, trec_eval)

    if len(ret) != 1:
        raise ValueError('Wrong')

    return ret['1'].calculate(None)


class TestFastNdcgImplementation(TestCase):

    def test_ndcg_of_0_qrels_few_relevant_many_irrelant(self):
        run = create_run([
            rl('nr-1', 1), rl('nr-2', 2), rl('nr-3', 3), rl('nr-4', 4),
            rl('nr-5', 5), rl('nr-6', 6), rl('nr-7', 7), rl('nr-8', 8),
            rl('nr-9', 9), rl('nr-10', 10)
        ])

        self.assertEquals(0.0, trectools_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, True))
        self.assertEquals(0.0, trectools_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, False))

        self.assertEquals(0.0, fast_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, True))
        self.assertEquals(0.0, fast_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, False))

    def test_ndcg_of_0_qrels_many_relevant_few_irrelant(self):
        run = create_run([
            rl('nr-1', 1), rl('nr-2', 2), rl('nr-3', 3), rl('nr-4', 4),
            rl('nr-5', 5), rl('nr-6', 6), rl('nr-7', 7), rl('nr-8', 8),
            rl('nr-9', 9), rl('nr-10', 10)
        ])

        self.assertEquals(0.0, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True))
        self.assertEquals(0.0, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False))

        self.assertEquals(0.0, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True))
        self.assertEquals(0.0, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False))

    def test_ndcg_of_1_many_relevant_few_irrelant(self):
        run = create_run([
            rl('r-1', 1), rl('r-2', 2), rl('r-3', 3), rl('r-4', 4),
            rl('r-5', 5), rl('r-6', 6), rl('r-7', 7), rl('r-8', 8),
            rl('nr-9', 9), rl('nr-10', 10)
        ])

        self.assertEquals(1.0, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True))
        self.assertEquals(1.0, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False))

        self.assertEquals(1.0, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True))
        self.assertEquals(1.0, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False))

    def test_ndcg_of_06_many_relevant_few_irrelant(self):
        run = create_run([
            rl('r-1', 1), rl('r-2', 2), rl('r-3', 3), rl('r-4', 4),
            rl('nr-5', 5), rl('nr-6', 6), rl('nr-7', 7), rl('nr-8', 8),
            rl('nr-9', 9), rl('nr-10', 10)
        ])

        self.assertAlmostEqual(0.64793, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True), 4)
        self.assertAlmostEqual(0.64793, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False), 4)

        self.assertAlmostEqual(0.64793, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True), 4)
        self.assertAlmostEqual(0.64793, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False), 4)

    def test_ndcg_of_1_few_relevant_many_irrelant(self):
        run = create_run([
            rl('r-1', 1), rl('r-2', 2), rl('r-3', 3), rl('r-4', 4),
            rl('nr-5', 5), rl('nr-6', 6), rl('nr-7', 7), rl('nr-8', 8),
            rl('nr-9', 9), rl('nr-10', 10)
        ])

        self.assertEquals(1.0, trectools_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, True))
        self.assertEquals(1.0, trectools_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, False))

        self.assertEquals(1.0, fast_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, True))
        self.assertEquals(1.0, fast_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, False))

    def test_ndcg_of_05_few_relevant_many_irrelant(self):
        run = create_run([
            rl('r-1', 1), rl('nr-2', 2), rl('r-2', 3), rl('nr-4', 4),
            rl('r-3', 5), rl('nr-6', 6), rl('r-4', 7), rl('nr-8', 8),
            rl('nr-9', 9), rl('nr-10', 10)
        ])

        self.assertAlmostEqual(0.8667, trectools_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, True), 4)
        self.assertAlmostEqual(0.8667, trectools_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, False), 4)

        self.assertAlmostEqual(0.8667, fast_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, True), 4)
        self.assertAlmostEqual(0.8667, fast_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, False), 4)

    def test_ndcg_of_05_many_relevant_few_irrelant(self):
        run = create_run([
            rl('r-1', 1), rl('nr-2', 2), rl('r-2', 3), rl('nr-4', 4),
            rl('r-3', 5), rl('nr-6', 6), rl('r-4', 7), rl('nr-8', 8),
            rl('nr-9', 9), rl('nr-10', 10)
        ])

        self.assertAlmostEqual(0.56157, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True), 4)
        self.assertAlmostEqual(0.56157, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False), 4)

        self.assertAlmostEqual(0.56157, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True), 4)
        self.assertAlmostEqual(0.56157, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False), 4)

    def test_empty_ranking_few_relevant_many_irrelant(self):
        run = create_run([])

        # self.assertAlmostEqual(0.0, trectools_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, True), 4)
        # self.assertAlmostEqual(0.0, trectools_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, False), 4)

        self.assertAlmostEqual(0.0, fast_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, True), 4)
        self.assertAlmostEqual(0.0, fast_ndcg_eval(run, QRELS_FEW_RELEVANTS_MANY_IRRELEVANT, False), 4)

    def test_empty_ranking_many_relevant_few_irrelant(self):
        run = create_run([])

        # self.assertAlmostEqual(0.0, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True), 4)
        # self.assertAlmostEqual(0.0, trectools_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False), 4)

        self.assertAlmostEqual(0.0, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, True), 4)
        self.assertAlmostEqual(0.0, fast_ndcg_eval(run, QRELS_MANY_RELEVANTS_FEW_IRRELEVANT, False), 4)

    def test_permutation_test(self):
        raise ValueError('ToDo Implement')

    def test_permutation_with_gains(self):
        raise ValueError('ToDo Implement')
