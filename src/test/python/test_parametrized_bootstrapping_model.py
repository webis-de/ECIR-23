from parametrized_bootstrapping_model import ParametrizedBootstrappingModel
from unittest import TestCase
import json


class TestParametrizedBootstrappingModel(TestCase):
    def test_optimization_loss_for_rmse_perfect(self):
        y = [1,1,0,0]
        y_pred = [1,1,0,0]
        expected = 0.0
        
        actual = ParametrizedBootstrappingModel('rmse').loss(y, y_pred)
        self.assertEquals(expected, actual)
        
        actual = ParametrizedBootstrappingModel('rmse[1,1]').loss(y, y_pred)
        self.assertEquals(expected, actual)

    def test_optimization_loss_for_rmse_worst_case(self):
        y = [1,1,0,0]
        y_pred = [0,0,1,1]
        expected = 1.0
        
        actual = ParametrizedBootstrappingModel('rmse').loss(y, y_pred)
        self.assertEquals(expected, actual)


    def test_optimization_loss_for_rmse_only_upper_bound_perfect(self):
        y = [1,1,0,0]
        y_pred = [1,1,1,1]
        expected = 0.0
        
        actual = ParametrizedBootstrappingModel('rmse[0,1]').loss(y, y_pred)
        
        self.assertEquals(expected, actual)


    def test_optimization_loss_for_rmse_only_upper_bound_not_perfect_case_01(self):
        # still perfect, as no upper bound is hurt
        actual = ParametrizedBootstrappingModel('rmse[0,1]').loss([1,1,0,0], [1,1,0.5,0.5])
        self.assertEquals(0.0, actual)


    def test_optimization_loss_for_rmse_only_upper_bound_not_perfect_case_02(self):
        # not perfect, but only small violations against the upper bound
        actual = ParametrizedBootstrappingModel('rmse[0,1]').loss([1,1,0,0], [.9,.9,0.5,0.5])
        self.assertEquals(0.07071067811865474, actual)


    def test_optimization_loss_for_rmse_only_upper_bound_not_perfect_case_02(self):
        # not perfect, but only small violations against the upper bound
        actual = ParametrizedBootstrappingModel('rmse[0,1]').loss([1,1,0,0], [.8,.8,0.5,0.5])
        self.assertEquals(0.14142135623730948, actual)


    def test_optimization_loss_for_rmse_only_lower_bound_perfect(self):
        y = [1,1,0,0]
        y_pred = [0,0,0,0]
        expected = 0.0
        
        actual = ParametrizedBootstrappingModel('rmse[1,0]').loss(y, y_pred)
        
        self.assertEquals(expected, actual)


    def test_optimization_loss_for_rmse_only_lower_bound_not_perfect_case_01(self):
        # still perfect, as no lower bound is hurt
        actual = ParametrizedBootstrappingModel('rmse[1,0]').loss([1,1,0,0], [0.5,0.5,0,0])
        self.assertEquals(0.0, actual)


    def test_optimization_loss_for_rmse_only_lower_bound_not_perfect_case_02(self):
        # not perfect, but only small violations against the lower bound
        actual = ParametrizedBootstrappingModel('rmse[1,0]').loss([1,1,0,0], [0.9,0.9,0.1,0.1])
        self.assertEquals(0.07071067811865477, actual)


    def test_optimization_loss_for_rmse_only_lower_bound_not_perfect_case_03(self):
        # not perfect, larger violations against the lower bound
        actual = ParametrizedBootstrappingModel('rmse[1,0]').loss([1,1,0,0], [0.9,0.9,0.2,0.2])
        self.assertEquals(0.14142135623730953, actual)


    def test_fitting_for_all_perfect_values_case_01(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse[1,0]": 0.0, "quantile": 90, "search_space_size": 5})
        model = ParametrizedBootstrappingModel('rmse[1,0]', [10, 25, 50, 75, 90])
        
        model.fit([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [1.0, 1.0, 0, 0])
        
        self.assertEquals(expected, model.parameters())


    def test_fitting_for_all_perfect_values_case_02(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse[0,1]": 0.7071067811865476, "quantile": 10, "search_space_size": 5})
        model = ParametrizedBootstrappingModel('rmse[0,1]', [10, 25, 50, 75, 90])
        
        # all is perfect and we put weight on the lower bound, so we expect k=90
        model.fit([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [1.0, 1.0, 0, 0])
        
        self.assertEquals(expected, model.parameters())


    def test_fitting_for_all_perfect_values_case_03(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse[1,1]": 0.7071067811865476, "quantile": 50, "search_space_size": 5})
        model = ParametrizedBootstrappingModel('rmse[1,1]', [10, 25, 50, 75, 90])
        
        model.fit([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [1.0, 1.0, 0, 0])
        
        self.assertEquals(expected, str(model))


    def test_fitting_for_all_perfect_values_case_03(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse": 0.7071067811865476, "quantile": 50, "search_space_size": 5})
        model = ParametrizedBootstrappingModel('rmse', [10, 25, 50, 75, 90])
        
        model.fit([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [1.0, 1.0, 0, 0])
        
        self.assertEquals(expected, model.parameters())


    def test_fitting_for_single_perfect_score(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse": 0.0, "quantile": 50, "search_space_size": 5})
        
        model = ParametrizedBootstrappingModel('rmse', [10, 25, 50, 75, 90])
        
        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [1.0, 1.0, 1.0, 1.0])
        
        self.assertEquals(expected, model.parameters())


    def test_fitting_for_single_almost_perfect_score_case_01(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse": 0.25, "quantile": 25, "search_space_size": 5})
        
        model = ParametrizedBootstrappingModel('rmse', [10, 25, 50, 75, 90])
        
        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [0.75, 0.75, 0.75, 0.75])
        
        self.assertEquals(expected, model.parameters())


    def test_fitting_for_single_almost_perfect_score_case_02(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse": 0.04999999999999999, "quantile": 10, "search_space_size": 5})
        
        model = ParametrizedBootstrappingModel('rmse', [10, 25, 50, 75, 90])
        
        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [0.25, 0.25, 0.25, 0.25])
        
        self.assertEquals(expected, model.parameters())
