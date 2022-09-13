from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from statistics import median
import json


class FixedDeltaModel:
    def __init__(self):
        self.absolute_delta = None
        self.relative_delta = None

    def predict(self, X, relative_delta=None):
        if relative_delta is not None:
            self.relative_delta = relative_delta

        ret = []
        for x in X:
            if (type(x) is not float and type(x) is not int) or np.isnan(x):
                raise ValueError(f'I can only work with numbers as input. Got {x}.')

            ret += [max(min(self.__apply_delta(x), 1), 0)]

        return ret

    def fit(self, x, y):
        pass

    def __apply_delta(self, x):
        if self.absolute_delta is None and self.relative_delta is None:
            raise ValueError('The model is not fitted. I expect either an absolute_delta or an relative_delta.')
        if self.absolute_delta is not None and self.relative_delta is not None:
            raise ValueError('The model contains both, absolute_delta and relative_delta, but I can handle only one.')

        if self.absolute_delta is not None:
            return x + self.absolute_delta

        if self.relative_delta is not None:
            return x + (x * self.relative_delta)

        raise ValueError(f'Can not handle {x}')


class ReturnAlways1Model(FixedDeltaModel):
    def __init__(self):
        super().__init__()
        self.absolute_delta = 2
    
    def __str__(self):
        return 'always-1'


class ReturnAlways0Model(FixedDeltaModel):
    def __init__(self):
        super().__init__()
        self.absolute_delta = -2
    
    def __str__(self):
        return 'always-0'


class ReturnAlwaysX(FixedDeltaModel):
    def __init__(self, original_measure):
        super().__init__()
        self.absolute_delta = 0
        self.original_measure = original_measure

    def __str__(self):
        return 'always-' + self.original_measure


class GridSearchDeltaModel(FixedDeltaModel):
    def __init__(self, budget, budget_type, original_measure):
        super().__init__()
        if budget_type == 'upper-bound':
            self.optimize_for = 'rmse[0,1]'
            self.__search_space = list(i/100 for i in range(101))
        elif budget_type == 'lower-bound':
            self.optimize_for = 'rmse[1,0]'
            self.__search_space = list(-i/100 for i in range(101))
        else:
            raise ValueError(f'Can not handle {budget_type}')

        self.__budget_type = budget_type
        self.__selection = min if budget_type == 'upper-bound' else max
        self.__default_value = max if budget_type == 'upper-bound' else min
        self.__budget = budget
        self.training_loss = None
        self.original_measure = original_measure

    def loss(self, y, y_pred):
        if self.optimize_for == 'rmse':
            return mean_squared_error(y, y_pred, squared=False)
        elif self.optimize_for.startswith('rmse['):
            lower_bound_weight, upper_bound_weight = self.optimize_for.split('[')[1].split(']')[0].split(',')

            rmse_lower_bound = mean_squared_error(y, np.maximum(y_pred, y), squared=False)
            rmse_upper_bound = mean_squared_error(y, np.minimum(y_pred, y), squared=False)

            return (float(lower_bound_weight) * rmse_lower_bound) + (float(upper_bound_weight) * rmse_upper_bound)

        raise ValueError('can not handle ' + self.optimize_for)

    def fit(self, x, y):
        ret = [self.__default_value(self.__search_space)]
        grid_search_results = {}

        for relative_delta in self.__search_space:
            y_pred = self.predict(x, relative_delta)
            grid_search_results[relative_delta] = self.loss(y, y_pred)
            if grid_search_results[relative_delta] <= self.__budget:
                ret += [relative_delta]

        self.relative_delta = self.__selection(ret)
        self.training_loss = grid_search_results[self.relative_delta]

    def parameters(self):
        return json.dumps({'model': 'GridSearchDeltaModel', self.optimize_for: self.training_loss,
                           'relative_delta': self.relative_delta, 'search_space_size': len(self.__search_space)})

    def __str__(self):
        return 'gsd-' + self.__budget_type + '-' + str(self.__budget) + '-' + self.original_measure


class UpperBoundDeltaModel(GridSearchDeltaModel):
    def __init__(self, budget, original_measure):
        super().__init__(budget, 'upper-bound', original_measure)


class LowerBoundDeltaModel(GridSearchDeltaModel):
    def __init__(self, budget, original_measure):
        super().__init__(budget, 'lower-bound', original_measure)


class ParametrizedBootstrappingModel:
    def __init__(self, optimize_for, search_space=(i for i in range(101))):
        self.optimize_for = optimize_for
        self.search_space = set(search_space)
        self.quantile = None
        self.training_loss = None

    def fit(self, x, y):
        grid_search_results = {}
        
        for quantile in self.search_space:
            y_pred = self.predict(x, quantile)
            grid_search_results[quantile] = self.loss(y, y_pred)
        
        min_loss = min(grid_search_results.values())
        
        optimal_solutions = sorted([k for k, v in grid_search_results.items() if v < (min_loss + 0.00000001)], key=lambda i: int(i))
        
        self.quantile = self.__select_closest_quantile_to_loss_optimization(optimal_solutions)
        self.training_loss = grid_search_results[self.quantile]

    def predict(self, x, quantile=None):
        if quantile is None:
            quantile = self.quantile
        
        if quantile is None:
            raise ValueError('Call fit first.')
        
        quantile = float(quantile)/100.0
        
        ret = []
        for l in x:
            l = pd.DataFrame(l)
            if str(l.columns) != 'RangeIndex(start=0, stop=1, step=1)':
                raise ValueError('Unknown columns ' + str(l.columns))
                
            ret += [l[0].quantile(quantile)]

        return ret

    def loss(self, y, y_pred):
        if self.optimize_for == 'rmse':
            return mean_squared_error(y, y_pred, squared=False)
        elif self.optimize_for.startswith('rmse['):
            lower_bound_weight, upper_bound_weight = self.optimize_for.split('[')[1].split(']')[0].split(',')
            
            rmse_lower_bound = mean_squared_error(y, np.maximum(y_pred, y), squared=False)
            rmse_upper_bound = mean_squared_error(y, np.minimum(y_pred, y), squared=False)
                        
            return (float(lower_bound_weight) * rmse_lower_bound) + (float(upper_bound_weight) * rmse_upper_bound)
        
        raise ValueError('can not handle ' + self.optimize_for)

    def __select_closest_quantile_to_loss_optimization(self, optimal_solutions):
        if len(optimal_solutions) == 1:
            return optimal_solutions[0]
    
        if self.optimize_for == 'rmse':
            if len(optimal_solutions) == 2:
                return optimal_solutions[0]

            ret = median(optimal_solutions)
            return max([i for i in optimal_solutions if i <= ret])
        elif self.optimize_for.startswith('rmse['):
            lower_bound_weight, upper_bound_weight = self.optimize_for.split('[')[1].split(']')[0].split(',')
            
            if lower_bound_weight < upper_bound_weight:
                return optimal_solutions[0]
            if lower_bound_weight > upper_bound_weight:
                return optimal_solutions[-1]

            if len(optimal_solutions) == 2:
                return optimal_solutions[0]

            ret = median(optimal_solutions)
            return max([i for i in optimal_solutions if i <= ret])
            
        raise ValueError('can not handle ' + self.optimize_for)

    def __str__(self):
        return 'pbs-' + self.optimize_for

    def parameters(self):
        return json.dumps({'model': 'ParametrizedBootstrappingModel', self.optimize_for: self.training_loss,
                           'quantile': self.quantile, 'search_space_size': len(self.search_space)})


class FixedBudgetBootstrappingModel(ParametrizedBootstrappingModel):
    def __init__(self, budget, budget_type, search_space=(i for i in range(101))):
        if budget_type == 'upper-bound':
            optimize_for = 'rmse[0,1]'
        elif budget_type == 'lower-bound':
            optimize_for = 'rmse[1,0]'
        else:
            raise ValueError(f'Can not handle {budget_type}')

        super().__init__(optimize_for, search_space=search_space)
        self.__budget_type = budget_type
        self.__selection = min if budget_type == 'upper-bound' else max
        self.__default_value = max if budget_type == 'upper-bound' else min
        self.__budget = budget

    def fit(self, x, y):
        ret = [self.__default_value(self.search_space)]
        grid_search_results = {}

        for quantile in self.search_space:
            y_pred = self.predict(x, quantile)
            grid_search_results[quantile] = self.loss(y, y_pred)
            if grid_search_results[quantile] <= self.__budget:
                ret += [quantile]

        self.quantile = self.__selection(ret)
        self.training_loss = grid_search_results[self.quantile]

    def __str__(self):
        return 'pbs-' + self.__budget_type + '-' + str(self.__budget)

    def parameters(self):
        return json.dumps({'model': 'FixedBudgetBootstrappingModel', self.optimize_for: self.training_loss,
                           'quantile': self.quantile, 'search_space_size': len(self.search_space),
                           'budget_type': self.__budget_type, 'budget': self.__budget})


class UpperBoundFixedBudgetBootstrappingModel(FixedBudgetBootstrappingModel):
    def __init__(self, budget, search_space=(i for i in range(101))):
        super().__init__(budget, 'upper-bound', search_space)


class LowerBoundFixedBudgetBootstrappingModel(FixedBudgetBootstrappingModel):
    def __init__(self, budget, search_space=(i for i in range(101))):
        super().__init__(budget, 'lower-bound', search_space)
