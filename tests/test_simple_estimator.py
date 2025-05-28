import unittest
import numpy as np
from dte_adj import SimpleDistributionEstimator


class TestSimpleEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = SimpleDistributionEstimator()
        self.covariates = np.zeros((20, 5))
        self.treatment_arms = np.hstack([np.zeros(10), np.ones(10)])
        self.outcomes = np.arange(20)
        self.estimator.fit(self.covariates, self.treatment_arms, self.outcomes)

    def test_predict(self):
        # Arrange
        treatment_arm_test = 0
        locations_test = np.array([3, 6])
        expected_output = np.array([0.4, 0.7])

        # Act
        output = self.estimator.predict(treatment_arm_test, locations_test)

        # Assert
        np.testing.assert_array_almost_equal(output, expected_output, decimal=2)

    def test_fit_invalid_shapes(self):
        # Arrange
        covariates_invalid = np.array([[1, 2], [3, 4]])
        treatment_arms_invalid = np.array([0, 1])
        outcomes_invalid = np.array([0.5, 0.7])

        # Assert
        with self.assertRaises(ValueError):
            self.estimator.fit(covariates_invalid, self.treatment_arms, self.outcomes)

        with self.assertRaises(ValueError):
            self.estimator.fit(self.covariates, treatment_arms_invalid, self.outcomes)

        with self.assertRaises(ValueError):
            self.estimator.fit(self.covariates, self.treatment_arms, outcomes_invalid)
