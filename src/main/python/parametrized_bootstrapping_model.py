from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from statistics import median
import json


class ReturnAlways1Model():
    def fit(self, x, y):
        pass

    def predict(self, X):
        return [1]*len(X)
    
    def __str__(self):
        return 'always-1'


class ReturnAlways0Model():
    def fit(self, x, y):
        pass

    def predict(self, X):
        return [0]*len(X)
    
    def __str__(self):
        return 'always-0'


class ParametrizedBootstrappingModel():
    def __init__(self, optimize_for, search_space=list(range(101))):
        self.__optimize_for = optimize_for
        self.__search_space = set(search_space)
        self.__quantile = None
        self.__training_loss = None

    def fit(self, x, y):
        grid_search_results = {}
        
        for quantile in self.__search_space:
            y_pred = self.predict(x, quantile)
            grid_search_results[quantile] = self.loss(y, y_pred)
        
        min_loss = min(grid_search_results.values())
        
        optimal_solutions = sorted([k for k, v in grid_search_results.items() if v <  (min_loss + 0.00000001)], key=lambda i: int(i))
        
        self.__quantile = self.__select_closest_quantile_to_loss_optimization(optimal_solutions)    
        self.__training_loss = grid_search_results[self.__quantile]


    def predict(self, x, quantile=None):
        if quantile is None:
            quantile = self.__quantile
        
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
        if self.__optimize_for == 'rmse':
            return mean_squared_error(y, y_pred, squared=False)
        elif self.__optimize_for.startswith('rmse['):
            lower_bound_weight, upper_bound_weight = self.__optimize_for.split('[')[1].split(']')[0].split(',')
            
            rmse_lower_bound = mean_squared_error(y, np.maximum(y_pred, y), squared=False)
            rmse_upper_bound = mean_squared_error(y, np.minimum(y_pred, y), squared=False)
                        
            return (float(lower_bound_weight) * rmse_lower_bound) + (float(upper_bound_weight) * rmse_upper_bound)
        
        raise ValueError('can not handle ' + self.__optimize_for)


    def __select_closest_quantile_to_loss_optimization(self, optimal_solutions):
        if len(optimal_solutions) == 1:
            return optimal_solutions[0]
    
        if self.__optimize_for == 'rmse':
            if len(optimal_solutions) == 2:
                return optimal_solutions[0]
            return median(optimal_solutions)
        elif self.__optimize_for.startswith('rmse['):
            lower_bound_weight, upper_bound_weight = self.__optimize_for.split('[')[1].split(']')[0].split(',')
            
            if lower_bound_weight < upper_bound_weight:
                return optimal_solutions[0]
            if lower_bound_weight > upper_bound_weight:
                return optimal_solutions[-1]

            if len(optimal_solutions) == 2:
                return optimal_solutions[0]
                
            return median(optimal_solutions)
            
        raise ValueError('can not handle ' + self.__optimize_for)

    def __str__(self):
        return 'pbs-' + self.__optimize_for

    def parameters(self):
        return json.dumps({'model': 'ParametrizedBootstrappingModel', self.__optimize_for: self.__training_loss, 'quantile': self.__quantile, 'search_space_size': len(self.__search_space)})
    
