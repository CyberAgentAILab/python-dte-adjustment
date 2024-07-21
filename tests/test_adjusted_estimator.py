import unittest
import numpy as np
from dte_adj import AdjustedDistributionEstimator
from unittest.mock import MagicMock


class TestAdjustedEstimator(unittest.TestCase):
    def setUp(self):
        base_model = MagicMock()
        base_model.predict_proba.side_effect = lambda x, y: x
        self.estimator = AdjustedDistributionEstimator(base_model, folds=1)
        self.confoundings = np.zeros((20, 5))
        self.treatment_arms = np.hstack([np.zeros(10), np.ones(10)])
        self.outcomes = np.arange(20)
        self.estimator.fit(self.confoundings, self.treatment_arms, self.outcomes)

    def test_init_fail_incorrect_base_model(self):
        # Act, Assert
        with self.assertRaises(ValueError) as cm:
            AdjustedDistributionEstimator("dummy")
        self.assertEqual(
            str(cm.exception),
            "Base model should implement either predict_proba or predict",
        )

    def test_predict_fail_before_fit(self):
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

    def test_compute_cumulative_distribution(self):
        # Arrange
        mock_model = self.estimator.base_model
        mock_model.predict_proba.side_effect = lambda x: np.ones((x.shape[0], 2)) * 0.5
        target_treatment_arms = np.zeros(10)
        locations = np.arange(10)

        # Act
        cumulative_distribution, superset_prediction = (
            self.estimator._compute_cumulative_distribution(
                target_treatment_arms,
                locations,
                self.confoundings,
                self.treatment_arms,
                self.outcomes,
            )
        )

        # Assert
        self.assertEqual(cumulative_distribution.shape, (10,))
        self.assertEqual(superset_prediction.shape, (20, 10))

        for i in range(10):
            self.assertAlmostEqual(cumulative_distribution[i], (i + 1) / 10, places=2)

        for i in range(20):
            for j in range(10):
                self.assertAlmostEqual(superset_prediction[i, j], 0.5, places=2)
