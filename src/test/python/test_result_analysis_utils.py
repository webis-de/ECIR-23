from unittest import TestCase
from result_analysis_utils import *
from glob import glob

class TestResultAnalysisUtils(TestCase):
    def test_loading_of_raw_results(self):
        df = load_raw_evaluations(glob('src/test/resources/dummy-eval-results/*.jsonl')).to_json(lines=True, orient='records')

        print(df)
        expected = """{"run":"a","pooling":"complete-pool-depth-10","ndcg@10-NDCG@10":"{\\"1\\": 1.0, \\"2\\": 0.0, \\"3\\": 1.0, \\"4\\": 0.0}","bs-p-1000-ndcg@10-NDCG@10":null}
{"run":"a","pooling":"depth-10-pool-incomplete-for-a","ndcg@10-NDCG@10":"{\\"1\\": -1.0, \\"2\\": -1.0, \\"3\\": -1.0, \\"4\\": -1.0}","bs-p-1000-ndcg@10-NDCG@10":null}
{"run":"a","pooling":"complete-pool-depth-10","ndcg@10-NDCG@10":null,"bs-p-1000-ndcg@10-NDCG@10":"{\\"1\\": [1, 1, 1, 1], \\"2\\": [0, 0, 0, 0], \\"3\\": [1, 1, 1, 1], \\"4\\": [0, 0, 0, 0]}"}
{"run":"a","pooling":"depth-10-pool-incomplete-for-a","ndcg@10-NDCG@10":null,"bs-p-1000-ndcg@10-NDCG@10":"{\\"1\\": [0, 0, 0, 1], \\"2\\": [0, 1, 1, 1], \\"3\\": [0, 0, 0, 1], \\"4\\": [0, 1, 1, 1]}"}"""
    
        self.assertEquals(df, expected)


    def test_loading_of_results(self):
        df = load_evaluations(glob('src/test/resources/dummy-eval-results/*.jsonl')).to_json(lines=True, orient='records')

        expected = """{"('depth-10-complete', 'ndcg@10')":"{\\"1\\": 1.0, \\"2\\": 0.0, \\"3\\": 1.0, \\"4\\": 0.0}","('depth-10-incomplete', 'ndcg@10')":"{\\"1\\": -1.0, \\"2\\": -1.0, \\"3\\": -1.0, \\"4\\": -1.0}","('depth-10-complete', 'bs-p-1000-ndcg@10-ndcg@10')":"{\\"1\\": [1, 1, 1, 1], \\"2\\": [0, 0, 0, 0], \\"3\\": [1, 1, 1, 1], \\"4\\": [0, 0, 0, 0]}","('depth-10-incomplete', 'bs-p-1000-ndcg@10-ndcg@10')":"{\\"1\\": [0, 0, 0, 1], \\"2\\": [0, 1, 1, 1], \\"3\\": [0, 0, 0, 1], \\"4\\": [0, 1, 1, 1]}","run":"a"}"""
    
        self.assertEquals(df, expected)


    def test_loading_redundant_data_fails_01(self):
        with self.assertRaises(ValueError) as context:
            load_evaluations(['src/test/resources/dummy-eval-results/dummy-results-1.jsonl', 'src/test/resources/dummy-eval-results/dummy-results-2.jsonl', 'src/test/resources/dummy-eval-results/dummy-results-2.jsonl'])


    def test_loading_redundant_data_fails_02(self):
        with self.assertRaises(ValueError) as context:
            load_evaluations(['src/test/resources/dummy-eval-results/dummy-results-1.jsonl', 'src/test/resources/dummy-eval-results/dummy-results-2.jsonl', 'src/test/resources/dummy-eval-results/dummy-results-1.jsonl'])


    def test_loading_redundant_data_fails_03(self):
        with self.assertRaises(ValueError) as context:
            load_evaluations(['src/test/resources/dummy-eval-results/dummy-results-1.jsonl', 'src/test/resources/dummy-eval-results/dummy-results-1.jsonl'])

