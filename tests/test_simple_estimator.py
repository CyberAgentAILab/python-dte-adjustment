import unittest
import numpy as np
from dte_adj import SimpleDistributionEstimator


class TestSimpleEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = SimpleDistributionEstimator()
        self.confoundings = np.zeros((20, 5))
        self.treatment_arms = np.hstack([np.zeros(10), np.ones(10)])
        self.outcomes = np.arange(20)
        self.estimator.fit(self.confoundings, self.treatment_arms, self.outcomes)

    def test_predict(self):
        # Arrange
        treatment_arms_test = np.array([0, 1])
        locations_test = np.array([3, 6])
        expected_output = np.array([0.4, 0])

        # Act
        output = self.estimator.predict(treatment_arms_test, locations_test)

        # Assert
        np.testing.assert_array_almost_equal(output, expected_output, decimal=2)
