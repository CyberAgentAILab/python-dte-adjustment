import unittest
import numpy as np
from dte_adj import AdjustedDistributionEstimator
from unittest.mock import MagicMock


class TestAdjustedEstimator(unittest.TestCase):
    def test_prediction_success(self):
        # TODO!
        return

    def test_prediction_fail_before_fit(self):
        # Arrange
        D = np.zeros(20)
        D[:10] = 1
        Y = np.arange(20)
        base_model = MagicMock()
        subject = AdjustedDistributionEstimator(base_model)

        # Act, Assert
        with self.assertRaises(ValueError) as cm:
            subject.predict(D, Y)
        self.assertEqual(
            str(cm.exception),
            "This estimator has not been trained yet. Please call fit first",
        )

    def test_fit_fail_invalid_input(self):
        # Arrange
        X = np.arange(20)
        D = np.zeros(10)
        D[:10] = 1
        Y = np.arange(20)
        base_model = MagicMock()
        subject = AdjustedDistributionEstimator(base_model)

        # Act, Assert
        with self.assertRaises(ValueError) as cm:
            subject.fit(X, D, Y)
        self.assertEqual(
            str(cm.exception),
            "The shape of confounding and treatment_arm should be same",
        )
