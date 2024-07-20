import unittest
import numpy as np
from dte_adj import SimpleDistributionEstimator
import numpy.testing as npt


class TestSimpleEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = SimpleDistributionEstimator()
        self.confoundings = np.zeros((20, 5))
        self.treatment_arms = np.hstack([np.zeros(10), np.ones(10)])
        self.outcomes = np.arange(20)
        self.estimator.fit(self.confoundings, self.treatment_arms, self.outcomes)

    def test_fit(self):
        self.assertTrue(np.array_equal(self.estimator.confoundings, self.confoundings))
        self.assertTrue(
            np.array_equal(self.estimator.treatment_arms, self.treatment_arms)
        )
        self.assertTrue(np.array_equal(self.estimator.outcomes, self.outcomes))

    def test_fit_invalid_shapes(self):
        # Arrange
        confoundings_invalid = np.array([[1, 2], [3, 4]])
        treatment_arms_invalid = np.array([0, 1])
        outcomes_invalid = np.array([0.5, 0.7])

        # Assert
        with self.assertRaises(ValueError):
            self.estimator.fit(confoundings_invalid, self.treatment_arms, self.outcomes)

        with self.assertRaises(ValueError):
            self.estimator.fit(self.confoundings, treatment_arms_invalid, self.outcomes)

        with self.assertRaises(ValueError):
            self.estimator.fit(self.confoundings, self.treatment_arms, outcomes_invalid)

    def test_predict(self):
        # Arrange
        treatment_arms_test = np.array([0, 1])
        locations_test = np.array([3, 6])
        expected_output = np.array([0.4, 0])

        # Act
        output = self.estimator.predict(treatment_arms_test, locations_test)

        # Assert
        np.testing.assert_array_almost_equal(output, expected_output, decimal=2)

    def test_prediction_fail_before_fit(self):
        # Arrange
        treatment_arms_test = np.array([0, 1])
        locations_test = np.array([3, 6])
        subject = SimpleDistributionEstimator()

        # Act, Assert
        with self.assertRaises(ValueError) as cm:
            subject.predict(treatment_arms_test, locations_test)
        self.assertEqual(
            str(cm.exception),
            "This estimator has not been trained yet. Please call fit first",
        )

    def test_prediction_fail_invalid_arm(self):
        # Arrange
        treatment_arms_invalid = np.array([2])
        locations_test = np.array([3, 6])

        # Act, Assert
        with self.assertRaises(ValueError) as cm:
            self.estimator.predict(treatment_arms_invalid, locations_test)
        self.assertEqual(
            str(cm.exception),
            "This treatment_arms argument contains arms not included in the training data: {2}",
        )
