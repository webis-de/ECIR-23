from sklearn.metrics import mean_squared_error
import numpy as np


class ParametrizedBootstrappingModel():
    def __init__(self, optimize_for, search_space=list(range(101))):
        self.__optimize_for = optimize_for
        self.__search_space = search_space

    def loss(self, y, y_pred):
        if self.__optimize_for == 'rmse':
            return mean_squared_error(y, y_pred, squared=False)
        elif self.__optimize_for.startswith('rmse['):
            lower_bound_weight, upper_bound_weight = self.__optimize_for.split('[')[1].split(']')[0].split(',')
            
            rmse_lower_bound = mean_squared_error(y, np.maximum(y_pred, y), squared=False)
            rmse_upper_bound = mean_squared_error(y, np.minimum(y_pred, y), squared=False)
                        
            return (float(lower_bound_weight) * rmse_lower_bound) + (float(upper_bound_weight) * rmse_upper_bound)
        
        raise ValueError('can not handle ' + self.__optimize_for)
    
