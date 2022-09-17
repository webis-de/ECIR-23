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
        for k in [self.upper, self.actual, self.lower]:
            if k not in i:
                raise ValueError(f'I cand find the expected key {k}. Available are: {i.keys()}.')

        return {
            'system': i['system'],
            'ground_truth': i['ground_truth'],
            'prediction': {'lower': i[self.lower], 'actual': i[self.actual], 'upper': i[self.upper]}
        }


class InterpolationDataConstruction:
    def __init__(self, lower, actual, upper, interpolation):
        self.lower = lower
        self.actual = actual
        self.upper = upper
        self.interpolation = interpolation - 1.0

    def construct_data_for_reconstruction_evaluation(self, data_per_row):
        ret = {}
        for topic in data_per_row.keys():
            assert topic not in ret

            ret[topic] = [self.transform(i) for i in data_per_row[topic]]

        return ret

    def transform(self, i):
        interpolation_start = min(i[self.actual], i[self.upper])
        interpolation_upper = max(i[self.actual], i[self.upper])

        if self.interpolation >= 0:
            upper = interpolation_start + ((interpolation_upper - interpolation_start) * self.interpolation)
        else:
            interpolation_start = max(i[self.actual], i[self.lower])
            interpolation_lower = min(i[self.actual], i[self.lower])
            upper = interpolation_start + ((interpolation_start - interpolation_lower) * self.interpolation)

        return {
            'system': i['system'],
            'ground_truth': i['ground_truth'],
            'prediction': {'lower': i[self.lower], 'actual': upper, 'upper': upper}
        }


def to_ground_truth(i):
    return i['ground_truth']


def lower_bound(i):
    return min(i['prediction'].values())


def upper_bound(i):
    return max(i['prediction'].values())


class ReconstructionEvaluation:
    def __init__(self, threshold=0):
        self.__threshold = threshold

    def construct_reconstruction_scenarious(self, systems):
        ret = []

        for post_hoc_system in systems:
            reconstruction_scenario = {
                'post_hoc': {'system': post_hoc_system['system'], 'prediction': post_hoc_system['prediction']},
                'existing': [{'system': i['system'], 'ground_truth': i['ground_truth']}
                             for i in systems if i['system'] != post_hoc_system['system']]
            }

            reconstruction_scenario['expected_pairs'] = self.ground_truth_ranking(
                post_hoc_system, reconstruction_scenario['existing']
            )
            reconstruction_scenario['predicted_pairs'] = self.predicted_ranking(
                reconstruction_scenario['post_hoc'], reconstruction_scenario['existing']
            )
            ret += [reconstruction_scenario]

        return ret

    def precision(self, systems):
        reconstruction_scenarious = self.construct_reconstruction_scenarious(systems)
        predicted = 0
        correct_predicted = 0

        for scenario in reconstruction_scenarious:
            predicted += len(scenario['predicted_pairs'])
            correct_predicted += len(set(scenario['predicted_pairs']).intersection(scenario['expected_pairs']))

        if predicted == 0:
            return 0

        return correct_predicted/predicted

    def recall(self, systems):
        reconstruction_scenarious = self.construct_reconstruction_scenarious(systems)
        correct_predicted = 0
        expected = 0

        for scenario in reconstruction_scenarious:
            expected += len(scenario['expected_pairs'])
            correct_predicted += len(set(scenario['predicted_pairs']).intersection(scenario['expected_pairs']))

        if expected == 0:
            return 0

        return correct_predicted/expected

    def predicted_ranking(self, a, systems):
        ret = []

        for b in systems:
            diff_a_wins = lower_bound(a) - to_ground_truth(b)

            diff_b_wins = to_ground_truth(b) - upper_bound(a)

            if diff_a_wins >= self.__threshold:
                ret += [(a['system'], b['system'])]
            elif diff_b_wins >= self.__threshold:
                ret += [(b['system'], a['system'])]

        return set(ret)

    def ground_truth_ranking(self, a, systems):
        ret = []

        for b in systems:
            diff = to_ground_truth(a) - to_ground_truth(b)

            if a['system'] == b['system']:
                continue
            if diff >= self.__threshold:
                ret += [(a['system'], b['system'])]
            elif diff <= -1 * self.__threshold:
                ret += [(b['system'], a['system'])]

        return set(ret)


class AllApproachesDidNotParticipateInPoolingReconstructionEvaluation:
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
        return self.__all_pairs(systems, to_ground_truth, to_ground_truth)

    def predicted_pairs(self, systems):
        return self.__all_pairs(systems, lower_bound, upper_bound)

    def __all_pairs(self, systems, lower, upper):
        ret = set()

        for s1 in systems:
            for s2 in systems:
                diff = lower(s1) - upper(s2)

                if s1['system'] != s2['system'] and diff > self.__threshold:
                    ret.add((s1['system'], s2['system']))

        return ret
