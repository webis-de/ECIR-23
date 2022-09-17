from unittest import TestCase
from reconstruction_evaluation import InterpolationDataConstruction


class TestInterpolationDataConstruction(TestCase):
    def test_interpolation_of_1(self):
        model = InterpolationDataConstruction('l', 'a', 'u', 1.0)
        expected = {
            '1': [{'ground_truth': 0.9, 'prediction': {'actual': 0.4, 'lower': 0.3, 'upper': 0.4}, 'system': 'a'}],
            '2': [{'ground_truth': 0.9, 'prediction': {'actual': 0.5, 'lower': 0.3, 'upper': 0.5}, 'system': 'a'}]
        }
        actual = model.construct_data_for_reconstruction_evaluation({
            '1': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.4, 'u': 0.5}],
            '2': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.5, 'u': 0.6}]
        })

        self.assertEquals(expected, actual)

    def test_interpolation_of_0_9(self):
        model = InterpolationDataConstruction('l', 'a', 'u', 0.9)
        expected = {
            '1': [{'ground_truth': 0.9, 'prediction': {'actual': 0.39, 'lower': 0.3, 'upper': 0.39}, 'system': 'a'}],
            '2': [{'ground_truth': 0.9, 'prediction': {'actual': 0.48, 'lower': 0.3, 'upper': 0.48}, 'system': 'a'}]
        }
        actual = model.construct_data_for_reconstruction_evaluation({
            '1': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.4, 'u': 0.5}],
            '2': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.5, 'u': 0.6}]
        })

        self.assertEquals(expected, actual)

    def test_interpolation_of_0_8(self):
        model = InterpolationDataConstruction('l', 'a', 'u', 0.8)
        expected = {
            '1': [{'ground_truth': 0.9, 'prediction': {'actual': 0.38, 'lower': 0.3, 'upper': 0.38}, 'system': 'a'}],
            '2': [{'ground_truth': 0.9, 'prediction': {'actual': 0.46, 'lower': 0.3, 'upper': 0.46}, 'system': 'a'}]
        }
        actual = model.construct_data_for_reconstruction_evaluation({
            '1': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.4, 'u': 0.5}],
            '2': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.5, 'u': 0.6}]
        })

        self.assertEquals(expected, actual)

    def test_interpolation_of_1_2(self):
        model = InterpolationDataConstruction('l', 'a', 'u', 1.2)
        expected = {
            '1': [{'ground_truth': 0.9, 'prediction': {'actual': 0.42000000000000004, 'lower': 0.3,
                                                       'upper': 0.42000000000000004}, 'system': 'a'}],
            '2': [{'ground_truth': 0.9, 'prediction': {'actual': 0.52, 'lower': 0.3, 'upper': 0.52}, 'system': 'a'}]
        }
        actual = model.construct_data_for_reconstruction_evaluation({
            '1': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.4, 'u': 0.5}],
            '2': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.5, 'u': 0.6}]
        })

        self.assertEquals(expected, actual)

    def test_interpolation_of_1_1(self):
        model = InterpolationDataConstruction('l', 'a', 'u', 1.1)
        expected = {
            '1': [{'ground_truth': 0.9, 'prediction': {'actual': 0.41000000000000003, 'lower': 0.3,
                                                       'upper': 0.41000000000000003}, 'system': 'a'}],
            '2': [{'ground_truth': 0.9, 'prediction': {'actual': 0.51, 'lower': 0.3, 'upper': 0.51}, 'system': 'a'}]
        }
        actual = model.construct_data_for_reconstruction_evaluation({
            '1': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.4, 'u': 0.5}],
            '2': [{'system': 'a', 'ground_truth': 0.9, 'l': 0.3, 'a': 0.5, 'u': 0.6}]
        })

        self.assertEquals(expected, actual)