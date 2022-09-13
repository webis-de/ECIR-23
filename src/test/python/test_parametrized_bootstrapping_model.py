from parametrized_bootstrapping_model import ParametrizedBootstrappingModel, UpperBoundFixedBudgetBootstrappingModel,\
    LowerBoundFixedBudgetBootstrappingModel, ReturnAlways1Model, ReturnAlways0Model,\
    UpperBoundDeltaModel, LowerBoundDeltaModel
from unittest import TestCase
import json


class TestParametrizedBootstrappingModel(TestCase):
    def test_optimization_loss_for_rmse_perfect(self):
        y = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        expected = 0.0
        
        actual = ParametrizedBootstrappingModel('rmse').loss(y, y_pred)
        self.assertEquals(expected, actual)
        
        actual = ParametrizedBootstrappingModel('rmse[1,1]').loss(y, y_pred)
        self.assertEquals(expected, actual)

    def test_optimization_loss_for_rmse_worst_case(self):
        y = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        expected = 1.0
        
        actual = ParametrizedBootstrappingModel('rmse').loss(y, y_pred)
        self.assertEquals(expected, actual)

    def test_optimization_loss_for_rmse_only_upper_bound_perfect(self):
        y = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 1]
        expected = 0.0
        
        actual = ParametrizedBootstrappingModel('rmse[0,1]').loss(y, y_pred)
        
        self.assertEquals(expected, actual)

    def test_optimization_loss_for_rmse_only_upper_bound_not_perfect_case_01(self):
        # still perfect, as no upper bound is hurt
        actual = ParametrizedBootstrappingModel('rmse[0,1]').loss([1, 1, 0, 0], [1, 1, 0.5, 0.5])
        self.assertEquals(0.0, actual)

    def test_optimization_loss_for_rmse_only_upper_bound_not_perfect_case_02(self):
        # not perfect, but only small violations against the upper bound
        actual = ParametrizedBootstrappingModel('rmse[0,1]').loss([1, 1, 0, 0], [.9, .9, 0.5, 0.5])
        self.assertEquals(0.07071067811865474, actual)

    def test_optimization_loss_for_rmse_only_upper_bound_not_perfect_case_03(self):
        # not perfect, but only small violations against the upper bound
        actual = ParametrizedBootstrappingModel('rmse[0,1]').loss([1, 1, 0, 0], [.8, .8, 0.5, 0.5])
        self.assertEquals(0.14142135623730948, actual)

    def test_optimization_loss_for_rmse_only_lower_bound_perfect(self):
        y = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        expected = 0.0
        
        actual = ParametrizedBootstrappingModel('rmse[1,0]').loss(y, y_pred)
        
        self.assertEquals(expected, actual)

    def test_optimization_loss_for_rmse_only_lower_bound_not_perfect_case_01(self):
        # still perfect, as no lower bound is hurt
        actual = ParametrizedBootstrappingModel('rmse[1,0]').loss([1, 1, 0, 0], [0.5, 0.5, 0, 0])
        self.assertEquals(0.0, actual)

    def test_optimization_loss_for_rmse_only_lower_bound_not_perfect_case_02(self):
        # not perfect, but only small violations against the lower bound
        actual = ParametrizedBootstrappingModel('rmse[1,0]').loss([1, 1, 0, 0], [0.9, 0.9, 0.1, 0.1])
        self.assertEquals(0.07071067811865477, actual)

    def test_optimization_loss_for_rmse_only_lower_bound_not_perfect_case_03(self):
        # not perfect, larger violations against the lower bound
        actual = ParametrizedBootstrappingModel('rmse[1,0]').loss([1, 1, 0, 0], [0.9, 0.9, 0.2, 0.2])
        self.assertEquals(0.14142135623730953, actual)

    def test_fitting_for_all_perfect_values_case_01(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse[1,0]": 0.0,
                               "quantile": 90, "search_space_size": 5})
        model = ParametrizedBootstrappingModel('rmse[1,0]', [10, 25, 50, 75, 90])
        
        model.fit([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [1.0, 1.0, 0, 0])
        
        self.assertEquals(expected, model.parameters())

    def test_fitting_for_all_perfect_values_case_02(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse[0,1]": 0.7071067811865476,
                               "quantile": 10, "search_space_size": 5})
        model = ParametrizedBootstrappingModel('rmse[0,1]', [10, 25, 50, 75, 90])
        
        # all is perfect and we put weight on the lower bound, so we expect k=90
        model.fit([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [1.0, 1.0, 0, 0])
        
        self.assertEquals(expected, model.parameters())

    def test_fitting_for_all_perfect_values_case_03(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse[1,1]": 0.7071067811865476,
                               "quantile": 50, "search_space_size": 5})
        model = ParametrizedBootstrappingModel('rmse[1,1]', [10, 25, 50, 75, 90])
        
        model.fit([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [1.0, 1.0, 0, 0])
        
        self.assertEquals(expected, model.parameters())

    def test_fitting_for_all_perfect_values_case_04(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse": 0.7071067811865476,
                               "quantile": 50, "search_space_size": 5})
        model = ParametrizedBootstrappingModel('rmse', [10, 25, 50, 75, 90])
        
        model.fit([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [1.0, 1.0, 0, 0])
        
        self.assertEquals(expected, model.parameters())

    def test_fitting_for_single_perfect_score(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse": 0.0,
                               "quantile": 50, "search_space_size": 5})
        
        model = ParametrizedBootstrappingModel('rmse', [10, 25, 50, 75, 90])
        
        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [1.0, 1.0, 1.0, 1.0])
        
        self.assertEquals(expected, model.parameters())

    def test_fitting_for_single_almost_perfect_score_case_01(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse": 0.25,
                               "quantile": 25, "search_space_size": 5})
        
        model = ParametrizedBootstrappingModel('rmse', [10, 25, 50, 75, 90])
        
        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [0.75, 0.75, 0.75, 0.75])
        
        self.assertEquals(expected, model.parameters())

    def test_fitting_for_single_almost_perfect_score_case_02(self):
        expected = json.dumps({"model": "ParametrizedBootstrappingModel", "rmse": 0.04999999999999999,
                               "quantile": 10, "search_space_size": 5})
        
        model = ParametrizedBootstrappingModel('rmse', [10, 25, 50, 75, 90])
        
        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [0.25, 0.25, 0.25, 0.25])
        
        self.assertEquals(expected, model.parameters())

    def test_upper_bound_on_dataset_with_selection_of_default_solution(self):
        expected = json.dumps({"model": "FixedBudgetBootstrappingModel", "rmse[0,1]": 0.19999999999999996,
                               "quantile": 90, "search_space_size": 5, "budget_type": "upper-bound",
                               "budget": 0.01})

        model = UpperBoundFixedBudgetBootstrappingModel(0.01, [10, 25, 50, 75, 90])

        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [2, 2, 2, 2])

        self.assertEquals(expected, model.parameters())

    def test_upper_bound_on_dataset_with_selection_of_non_default_solution_small_budget(self):
        expected = json.dumps({"model": "FixedBudgetBootstrappingModel", "rmse[0,1]": 0.5,
                               "quantile": 75, "search_space_size": 5, "budget_type": "upper-bound",
                               "budget": 0.6})

        model = UpperBoundFixedBudgetBootstrappingModel(0.6, [10, 25, 50, 75, 90])

        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [2, 2, 2, 2])

        self.assertEquals(expected, model.parameters())
        self.assertEquals('pbs-upper-bound-0.6', str(model))

    def test_upper_bound_on_dataset_with_selection_of_non_default_solution_larger_budget(self):
        expected = json.dumps({"model": "FixedBudgetBootstrappingModel", "rmse[0,1]": 1.0,
                               "quantile": 50, "search_space_size": 5, "budget_type": "upper-bound",
                               "budget": 1.2})

        model = UpperBoundFixedBudgetBootstrappingModel(1.2, [10, 25, 50, 75, 90])

        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [2, 2, 2, 2])

        self.assertEquals(expected, model.parameters())

    def test_lower_bound_on_dataset_with_selection_of_default_solution(self):
        expected = json.dumps({"model": "FixedBudgetBootstrappingModel", "rmse[1,0]": 0.2,
                               "quantile": 10, "search_space_size": 5, "budget_type": "lower-bound",
                               "budget": 0.01})

        model = LowerBoundFixedBudgetBootstrappingModel(0.01, [10, 25, 50, 75, 90])

        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [0, 0, 0, 0])

        self.assertEquals(expected, model.parameters())

    def test_lower_bound_on_dataset_with_selection_of_non_default_solution_small_budget(self):
        expected = json.dumps({"model": "FixedBudgetBootstrappingModel", "rmse[1,0]": 0.5,
                               "quantile": 25, "search_space_size": 5, "budget_type": "lower-bound",
                               "budget": 0.6})

        model = LowerBoundFixedBudgetBootstrappingModel(0.6, [10, 25, 50, 75, 90])

        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [0, 0, 0, 0])

        self.assertEquals(expected, model.parameters())

    def test_lower_bound_on_dataset_with_selection_of_non_default_solution_larger_budget(self):
        expected = json.dumps({"model": "FixedBudgetBootstrappingModel", "rmse[1,0]": 1.0,
                               "quantile": 50, "search_space_size": 5, "budget_type": "lower-bound",
                               "budget": 1.2})

        model = LowerBoundFixedBudgetBootstrappingModel(1.2, [10, 25, 50, 75, 90])

        model.fit([[0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0], [0, 1.0, 2.0]], [0, 0, 0, 0])

        self.assertEquals(expected, model.parameters())

    def test_predict_always_1_model(self):
        model = ReturnAlways1Model()
        expected = [1, 1, 1, 1, 1]

        self.assertEquals(expected, model.predict([0, 0, 0, 0, 0]))
        self.assertEquals(expected, model.predict([1, 1, 1, 1, 1]))

    def test_predict_always_0_model(self):
        model = ReturnAlways0Model()
        expected = [0, 0, 0, 0, 0]

        self.assertEquals(expected, model.predict([0, 0, 0, 0, 0]))
        self.assertEquals(expected, model.predict([1, 1, 1, 1, 1]))

    def test_upper_bound_of_zero(self):
        model = UpperBoundDeltaModel(0.01, 'm')

        model.fit([0.5, 0.5, 0.5], [0.4, 0.4, 0.4])

        self.assertEquals([0, 0, 0, 0], model.predict([0, 0, 0, 0]))
        self.assertEquals([0.3, 0.3, 0.3, 0.3], model.predict([0.3, 0.3, 0.3, 0.3]))
        self.assertEquals([0.6, 0.6, 0.6, 0.6], model.predict([0.6, 0.6, 0.6, 0.6]))

        expected = {"model": "GridSearchDeltaModel", "rmse[0,1]": 0.0, "relative_delta": 0.0, "search_space_size": 101}
        print(model.parameters())
        self.assertEquals(json.dumps(expected), model.parameters())

    def test_upper_bound_of_larger_than_001(self):
        model = UpperBoundDeltaModel(0.01, 'm')

        model.fit([0.4, 0.4, 0.4], [0.5, 0.5, 0.5])

        self.assertEquals([0, 0, 0, 0], model.predict([0, 0, 0, 0]))
        self.assertEquals([0.369, 0.369, 0.369, 0.369], model.predict([0.3, 0.3, 0.3, 0.3]))
        self.assertEquals([0.738, 0.738, 0.738, 0.738], model.predict([0.6, 0.6, 0.6, 0.6]))

        expected = {"model": "GridSearchDeltaModel", "rmse[0,1]": 0.007999999999999952, "relative_delta": 0.23,
                    "search_space_size": 101}
        print(model.parameters())
        self.assertEquals(json.dumps(expected), model.parameters())

    def test_upper_bound_of_larger_than_005(self):
        model = UpperBoundDeltaModel(0.05, 'm')

        model.fit([0.4, 0.4, 0.4], [0.5, 0.5, 0.5])

        self.assertEquals([0, 0, 0, 0], model.predict([0, 0, 0, 0]))
        self.assertEquals([0.33899999999999997, 0.33899999999999997, 0.33899999999999997, 0.33899999999999997],
                          model.predict([0.3, 0.3, 0.3, 0.3]))
        self.assertEquals([0.6779999999999999, 0.6779999999999999, 0.6779999999999999, 0.6779999999999999],
                          model.predict([0.6, 0.6, 0.6, 0.6]))

        expected = {"model": "GridSearchDeltaModel", "rmse[0,1]": 0.04799999999999999, "relative_delta": 0.13,
                    "search_space_size": 101}
        print(model.parameters())
        self.assertEquals(json.dumps(expected), model.parameters())

    def test_lower_bound_of_zero(self):
        model = LowerBoundDeltaModel(0.01, 'm')

        model.fit([0.4, 0.4, 0.4], [0.5, 0.5, 0.5])

        self.assertEquals([0, 0, 0, 0], model.predict([0, 0, 0, 0]))
        self.assertEquals([0.3, 0.3, 0.3, 0.3], model.predict([0.3, 0.3, 0.3, 0.3]))
        self.assertEquals([0.6, 0.6, 0.6, 0.6], model.predict([0.6, 0.6, 0.6, 0.6]))

        expected = {"model": "GridSearchDeltaModel", "rmse[1,0]": 0.0, "relative_delta": 0.0, "search_space_size": 101}
        print(model.parameters())
        self.assertEquals(json.dumps(expected), model.parameters())

    def test_lower_bound_of_larger_than_001(self):
        model = LowerBoundDeltaModel(0.01, 'm')

        model.fit([0.5, 0.5, 0.5], [0.4, 0.4, 0.4])

        self.assertEquals([0, 0, 0, 0], model.predict([0, 0, 0, 0]))
        self.assertEquals([0.243, 0.243, 0.243, 0.243], model.predict([0.3, 0.3, 0.3, 0.3]))
        self.assertEquals([0.486, 0.486, 0.486, 0.486], model.predict([0.6, 0.6, 0.6, 0.6]))

        expected = {"model": "GridSearchDeltaModel", "rmse[1,0]": 0.0050000000000000044, "relative_delta": -0.19,
                    "search_space_size": 101}
        print(model.parameters())
        self.assertEquals(json.dumps(expected), model.parameters())

    def test_lower_bound_of_larger_than_005(self):
        model = LowerBoundDeltaModel(0.05, 'm')

        model.fit([0.4, 0.4, 0.4], [0.3, 0.3, 0.3])

        self.assertEquals([0, 0, 0, 0], model.predict([0, 0, 0, 0]))
        self.assertEquals([0.261, 0.261, 0.261, 0.261],
                          model.predict([0.3, 0.3, 0.3, 0.3]))
        self.assertEquals([0.522, 0.522, 0.522, 0.522],
                          model.predict([0.6, 0.6, 0.6, 0.6]))

        expected = {"model": "GridSearchDeltaModel", "rmse[1,0]": 0.04800000000000004, "relative_delta": -0.13,
                    "search_space_size": 101}
        print(model.parameters())
        self.assertEquals(json.dumps(expected), model.parameters())
