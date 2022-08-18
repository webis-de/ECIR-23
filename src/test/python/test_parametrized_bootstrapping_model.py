from parametrized_bootstrapping_model import ParametrizedBootstrappingModel
from unittest import TestCase


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

