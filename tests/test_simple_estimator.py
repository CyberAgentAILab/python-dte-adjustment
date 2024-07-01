import unittest
import numpy as np
from dte_adj import SimpleDistributionEstimator
import numpy.testing as npt


class TestSimpleEstimator(unittest.TestCase):
    def test_prediction_success(self):
        # Arrange
        X = np.arange(20)
        D = np.zeros(20)
        D[:10] = 1
        Y = np.arange(20)
        subject = SimpleDistributionEstimator()
        subject.fit(X, D, Y)

        # Act
        actual = subject.predict(D, Y)

        # Assert
        expected = np.array(
            [0.1 * i for i in range(1, 11)] + [0.1 * i for i in range(1, 11)]
        )
        npt.assert_allclose(actual, expected)

    def test_prediction_fail_before_fit(self):
        # Arrange
        D = np.zeros(20)
        D[:10] = 1
        Y = np.arange(20)
        subject = SimpleDistributionEstimator()

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
        subject = SimpleDistributionEstimator()

        # Act, Assert
        with self.assertRaises(ValueError) as cm:
            subject.fit(X, D, Y)
        self.assertEqual(
            str(cm.exception),
            "The shape of confounding and treatment_arm should be same",
        )
