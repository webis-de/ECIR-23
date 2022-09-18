from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from statistics import median
from copy import deepcopy
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


class BootstrappingInducedByCondensedLists:
    def __init__(self, anchor_quantile, original_measure):
        anchor_quantile = float(anchor_quantile)
        if anchor_quantile < 0.0 or anchor_quantile > 1.0:
            raise ValueError('Expect the quantile between 0 (for 0%) and 1 (for 100%)')
        self.anchor_quantile = anchor_quantile
        self.original_measure = original_measure
        self.minimum_items = 1

    def predict(self, X):
        ret = []
        for x in X:
            if len(x) == 2:
                bootstrap_values, target_x = x
            else:
                bootstrap_values, target_x = x[0], 0.0

            if (type(target_x) is not float and type(target_x) is not int) or np.isnan(target_x):
                raise ValueError(f'I can only work with numbers as targets. Got {x}.')
            if type(bootstrap_values) is not list:
                raise ValueError(f'I can only work with lists as bootstrap_values. Got {bootstrap_values}.')
            if not all(type(i) is float or type(i) is int for i in bootstrap_values):
                raise ValueError(f'I can only work with lists as bootstrap_values. Got {bootstrap_values}.')

            tmp_log = {'bootstrap_max_before': max(bootstrap_values), 'target_x': target_x}
            bootstrap_values = self.adjust_to_target(bootstrap_values, target_x)
            tmp_log['bootstrap_max_after'] = max(min(max(bootstrap_values), 1), 0)
            tmp_log['bootstrap_length'] = len(bootstrap_values)

            # print(json.dumps(tmp_log))

            ret += [max(min(max(bootstrap_values), 1), 0)]

        return ret

    def adjust_to_target(self, bootstrap_values, target_x):
        bootstrap_values = sorted(deepcopy(bootstrap_values), reverse=False)
        ret = []
        count_below_target = 0
        count_above_target = 0

        for i in bootstrap_values:
            if i > target_x:
                count_above_target += 1
            else:
                count_below_target += 1

            quantile_at_target = count_below_target/(count_above_target + count_below_target)
            if (count_above_target + count_below_target) > self.minimum_items and quantile_at_target < self.anchor_quantile:
                break
            else:
                ret += [i]

        if max(ret) < target_x and self.anchor_quantile <= 1.0:
            # the distribution is expected to return values that contain the target_x, so we have to add
            # it manually in case it is not extracted from the distribution
            ret += [target_x]

        return ret

    def fit(self, x, y):
        pass

    def __str__(self):
        return f'bs-ci-{self.anchor_quantile}-{self.original_measure}'


class BootstrappingBySelectingMostLikelyDataPoint:
    def __init__(self, bootstrap_field):
        self.bootstrap_field = bootstrap_field
        self.cluster_borders = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    def fit(self, x, y):
        pass

    def predict(self, x):
        ret = []
        if type(x) is not list and type(x) is not tuple:
            raise ValueError(f'Invalid input. Got {x}')

        for l in x:
            if type(l) is not list and type(l) is not tuple and not(all(type(i) is int or type(i) is float for i in l)):
                raise ValueError(f'Invalid input. Got {l}')

            ret += [self.select_element(l)]

        return ret

    def select_element(self, data_points):
        cluster = self.cluster_datapoints(data_points)
        cluster = self.select_max_cluster(cluster)
        data_points = []
        for v in cluster.values():
            data_points += v

        cluster = self.cluster_datapoints(data_points)
        cluster = self.select_max_cluster(cluster)
        data_points = []
        for v in cluster.values():
            data_points += list(v)

        return max(data_points)

    def cluster_datapoints(self, data_points):
        min_x = min(data_points)
        max_x = max(data_points)
        if (max_x - min_x) <= 0.0001:
            return {0: data_points}
        normalized_data_points = [(i - min_x) / (max_x - min_x) for i in data_points]
        clusters = {}

        for i in range(len(data_points)):

            normalized_data_point = normalized_data_points[i]
            data_point = data_points[i]
            for cluster in self.cluster_borders:
                if normalized_data_point <= cluster:
                    if cluster not in clusters:
                        clusters[cluster] = []
                    clusters[cluster] += [data_point]
                    break

                if cluster >= 1:
                    raise ValueError('Can not happen: It should be aleady asigned to a cluster. ' +
                                     f'But element {normalized_data_point} did not found a cluster.')

        return clusters

    @staticmethod
    def select_max_cluster(clusters):
        max_elements = max([len(i) for i in clusters.values()])

        ret = {}
        for k, v in clusters.items():
            if len(v) < max_elements:
                continue
            ret[k] = v

        return ret

    def __str__(self):
        return f'bs-ml-{self.bootstrap_field}'
