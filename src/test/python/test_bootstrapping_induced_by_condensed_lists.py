from unittest import TestCase
from parametrized_bootstrapping_model import BootstrappingInducedByCondensedLists


class TestBootstrappingInducedByCondensedLists(TestCase):
    def test_no_changes_for_100_quantile(self):
        model = BootstrappingInducedByCondensedLists(1, 'm')

        actual = model.predict([
            [[0, 0, 0, 0], 0],
            [[0, 1, 1, 1], 1],
            [[0.1, 0.3, 0.3], 0.3],
            [[0.1, 0.2, 0.3, 0.5], 0.5],
            [[0, 0, 0, 1, 1], 1]
        ])
        self.assertEquals([0, 1, 0.3, 0.5, 1], actual)

        actual = model.predict([
            [[0, 1, 2], 0],
            [[0, 1, 2], 1],
            [[0.1, 0.3, 1], 0.3],
            [[0.1, 0.5, 0.9, 1.5], 0.5],
            [[2, 1, 0], 1]
        ])

        self.assertEquals([0, 1, 0.3, 0.5, 1], actual)

    def test_no_changes_for_70_quantile(self):
        model = BootstrappingInducedByCondensedLists(0.7, 'm')

        actual = model.predict([
            [[0, 0, 0, 0], 0],
            [[0, 1, 1, 1], 1],
            [[0.1, 0.2, 0.2, 0.3, 0.4], 0.2],
            [[0.1, 0.2, 0.3, 0.5], 0.5],
            [[0, 0.09, 0.09, 0.2, 1, 1], 0.1]
        ])
        self.assertEquals([0, 1, 0.3, 0.5, 0.2], actual)

        actual = model.predict([
            [[0, 0, 0, 1, 2], 0],
            [[0, 0.1, 0.01, 0.5, 1], .2],
            [[0.1, 0.29, 0.2, 1], 0.3],
            [[0.1, 0.49, 0.49, 0.9, 1.5], 0.5],
            [[2, 1, 0], 1]
        ])

        self.assertEquals([1, .5, 1, 0.9, 1], actual)
