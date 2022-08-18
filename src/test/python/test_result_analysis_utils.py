from unittest import TestCase
from result_analysis_utils import *
from glob import glob
from parametrized_bootstrapping_model import ParametrizedBootstrappingModel
from io import StringIO


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

    def test_loading_ground_truth_data(self):
        expected = """{"run":"a","query":"1","x":[0,0,0,1],"y":1.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":0}
{"run":"a","query":"2","x":[0,1,1,1],"y":0.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":1}
{"run":"a","query":"3","x":[0,0,0,1],"y":1.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":0}
{"run":"a","query":"4","x":[0,1,1,1],"y":0.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":1}"""

        # I want that this method can use multiple inputs
        for inp in [glob('src/test/resources/dummy-eval-results/*.jsonl'), load_evaluations(glob('src/test/resources/dummy-eval-results/*.jsonl'))]:
            actual = load_ground_truth_data(inp, 'ndcg', 10, 'bs-p-1000-ndcg@10-ndcg@10', random_state=3).to_json(lines=True, orient='records')
            print(actual)
            self.assertEquals(expected, actual)


    def test_cross_validation_with_model_returning_always_1_assuring_splits_are_correct(self):
        class Tmp():
            def fit(self, x, y):
                print(x)
                print(y)
                if x == [[0,0,0,1], [0,0,0,1]] and y == [1.0, 1.0]:
                    return
                
                if x == [[0,1,1,1], [0,1,1,1]] and y == [0.0, 0.0]:
                    return
                
                raise ValueError('Invalid Input')
                
            def predict(self, X):
                return [1]*len(X)

            def __str__(self):
                return 'tmp'

        expected = """{"run":"a","query":"1","x":[0,0,0,1],"y":1.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":0,"y_prediction":1,"model":"tmp"}
{"run":"a","query":"3","x":[0,0,0,1],"y":1.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":0,"y_prediction":1,"model":"tmp"}
{"run":"a","query":"2","x":[0,1,1,1],"y":0.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":1,"y_prediction":1,"model":"tmp"}
{"run":"a","query":"4","x":[0,1,1,1],"y":0.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":1,"y_prediction":1,"model":"tmp"}"""

        # I want that this method can use multiple inputs
        for inp in [glob('src/test/resources/dummy-eval-results/*.jsonl'), load_evaluations(glob('src/test/resources/dummy-eval-results/*.jsonl'))]:
            ground_truth_data = load_ground_truth_data(inp, 'ndcg', 10, 'bs-p-1000-ndcg@10-ndcg@10', random_state=3)
            actual = run_cross_validation(ground_truth_data, model=Tmp())
            actual = actual.to_json(lines=True, orient='records')
            print(actual)
            self.assertEquals(expected, actual)


    def test_cross_validation_with_model_returning_always_0_assuring_splits_are_correct(self):
        class Tmp():
            def fit(self, x, y):
                print(x)
                print(y)
                if x == [[0,0,0,1], [0,0,0,1]] and y == [1.0, 1.0]:
                    return
                
                if x == [[0,1,1,1], [0,1,1,1]] and y == [0.0, 0.0]:
                    return
                
                raise ValueError('Invalid Input')
                
            def predict(self, X):
                return [0]*len(X)
            
            def __str__(self):
                return 'tmp'

        expected = """{"run":"a","query":"1","x":[0,0,0,1],"y":1.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":0,"y_prediction":0,"model":"tmp"}
{"run":"a","query":"3","x":[0,0,0,1],"y":1.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":0,"y_prediction":0,"model":"tmp"}
{"run":"a","query":"2","x":[0,1,1,1],"y":0.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":1,"y_prediction":0,"model":"tmp"}
{"run":"a","query":"4","x":[0,1,1,1],"y":0.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":1,"y_prediction":0,"model":"tmp"}"""

        # I want that this method can use multiple inputs
        for inp in [glob('src/test/resources/dummy-eval-results/*.jsonl'), load_evaluations(glob('src/test/resources/dummy-eval-results/*.jsonl'))]:
            ground_truth_data = load_ground_truth_data(inp, 'ndcg', 10, 'bs-p-1000-ndcg@10-ndcg@10', random_state=3)
            actual = run_cross_validation(ground_truth_data, model=Tmp())
            actual = actual.to_json(lines=True, orient='records')
            print(actual)
            self.assertEquals(expected, actual)


    def test_cross_validation_with_parametrized_bootstrapping_model_optimizing_rmse_lower_bound(self):
        model = ParametrizedBootstrappingModel('rmse[1,0]', [10, 25, 50, 75, 90])
        expected = """{"run":"a","query":"1","x":[0,0,0,1],"y":1.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":0,"y_prediction":0.0,"model":"pbs-rmse[1,0]"}
{"run":"a","query":"3","x":[0,0,0,1],"y":1.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":0,"y_prediction":0.0,"model":"pbs-rmse[1,0]"}
{"run":"a","query":"2","x":[0,1,1,1],"y":0.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":1,"y_prediction":1.0,"model":"pbs-rmse[1,0]"}
{"run":"a","query":"4","x":[0,1,1,1],"y":0.0,"measures":{"x":"bs-p-1000-ndcg@10-ndcg@10","y":"ndcg@10"},"split":1,"y_prediction":1.0,"model":"pbs-rmse[1,0]"}"""

        # I want that this method can use multiple inputs
        for inp in [glob('src/test/resources/dummy-eval-results/*.jsonl'), load_evaluations(glob('src/test/resources/dummy-eval-results/*.jsonl'))]:
            ground_truth_data = load_ground_truth_data(inp, 'ndcg', 10, 'bs-p-1000-ndcg@10-ndcg@10', random_state=3)
            actual = run_cross_validation(ground_truth_data, model=model)
            actual = actual.to_json(lines=True, orient='records')
            print(actual)
            self.assertEquals(expected, actual)

    def test_results_are_correct_persisted(self):
        model = ParametrizedBootstrappingModel('rmse[1,0]', [10, 25, 50, 75, 90])
        expected = json.dumps([{'depth-10-pool-incomplete-for-TBD': [{'pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10': 0, 'run_file': 'a', 'query': 1}, {'pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10': 0, 'run_file': 'a', 'query': 3}, {'pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10': 1, 'run_file': 'a', 'query': 2}, {'pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10': 1, 'run_file': 'a', 'query': 4}], 'task': {'run': 'a', 'measure': 'pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10'}}])
        
        inp = load_evaluations(glob('src/test/resources/dummy-eval-results/*.jsonl'))
        ground_truth_data = load_ground_truth_data(inp, 'ndcg', 10, 'bs-p-1000-ndcg@10-ndcg@10', random_state=3)
        cross_validation_results = StringIO(run_cross_validation(ground_truth_data, model=model).to_json(lines=True, orient='records'))
        
        actual = json.dumps(load_cross_validation_results(cross_validation_results, depth=10))
        print(actual)
        
        self.assertEquals(actual, expected)


    def test_multiple_results_are_correct_persisted(self):
        expected = json.dumps([{"depth-10-pool-incomplete-for-TBD": [{"pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10": 0, "run_file": "a", "query": 1}, {"pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10": 0, "run_file": "a", "query": 3}, {"pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10": 1, "run_file": "a", "query": 2}, {"pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10": 1, "run_file": "a", "query": 4}], "task": {"run": "a", "measure": "pbs-rmse[1,0]-bs-p-1000-ndcg@10-ndcg@10"}}, {"depth-10-pool-incomplete-for-TBD": [{"pbs-rmse[1,1]-bs-p-1000-ndcg@10-ndcg@10": 0, "run_file": "a", "query": 1}, {"pbs-rmse[1,1]-bs-p-1000-ndcg@10-ndcg@10": 0, "run_file": "a", "query": 3}, {"pbs-rmse[1,1]-bs-p-1000-ndcg@10-ndcg@10": 1, "run_file": "a", "query": 2}, {"pbs-rmse[1,1]-bs-p-1000-ndcg@10-ndcg@10": 1, "run_file": "a", "query": 4}], "task": {"run": "a", "measure": "pbs-rmse[1,1]-bs-p-1000-ndcg@10-ndcg@10"}}])
        
        inp = load_evaluations(glob('src/test/resources/dummy-eval-results/*.jsonl'))
        
        ground_truth_data = load_ground_truth_data(inp, 'ndcg', 10, 'bs-p-1000-ndcg@10-ndcg@10', random_state=3)
        inptut_str = run_cross_validation(ground_truth_data, model=ParametrizedBootstrappingModel('rmse[1,0]', [10, 25, 50, 75, 90])).to_json(lines=True, orient='records')
        inptut_str += '\n' + run_cross_validation(ground_truth_data, model=ParametrizedBootstrappingModel('rmse[1,1]', [10, 25, 50, 75, 90])).to_json(lines=True, orient='records')
        
        actual = json.dumps(load_cross_validation_results(StringIO(inptut_str), depth=10))
        print(actual)
        
        self.assertEquals(actual, expected)

