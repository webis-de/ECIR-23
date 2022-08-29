class DataConstruction:
    def __init__(self, lower, actual, upper):
        self.lower = lower
        self.actual = actual
        self.upper = upper

    def construct_data_for_reconstruction_evaluation(self, data_per_row):
        ret = {}
        for topic in data_per_row.keys():
            assert topic not in ret

            ret[topic] = [self.transform(i) for i in data_per_row[topic]]

        return ret

    def transform(self, i):
        return {
            'system': i['system'],
            'ground_truth': i['ground_truth'],
            'prediction': {'lower': i[self.lower], 'actual': i[self.actual], 'upper': i[self.upper]}
        }



class ReconstructionEvaluation:
    def __init__(self, threshold=0):
        self.__threshold = threshold

    def precision(self, systems):
        ground_truth = set(self.ground_truth_pairs(systems))
        predicted = set(self.predicted_pairs(systems))

        if len(predicted) == 0:
            return 0

        return len(ground_truth.intersection(predicted))/len(predicted)

    def recall(self, systems):
        ground_truth = set(self.ground_truth_pairs(systems))
        predicted = set(self.predicted_pairs(systems))

        if len(ground_truth) == 0:
            return 1

        return len(ground_truth.intersection(predicted))/len(ground_truth)

    def ground_truth_pairs(self, systems):
        def to_ground_truth(i):
            return i['ground_truth']

        return self.__all_pairs(systems, to_ground_truth, to_ground_truth)

    def predicted_pairs(self, systems):
        def lower_bound(i):
            return min(i['prediction'].values())

        def upper_bound(i):
            return max(i['prediction'].values())

        return self.__all_pairs(systems, lower_bound, upper_bound)

    def __all_pairs(self, systems, lower_bound, upper_bound):
        ret = set()

        for s1 in systems:
            for s2 in systems:
                diff = lower_bound(s1) - upper_bound(s2)

                if s1['system'] != s2['system'] and diff > self.__threshold:
                    ret.add((s1['system'], s2['system']))

        return ret
