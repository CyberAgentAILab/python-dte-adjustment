import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from dte_adj import DistributionEstimatorBase


def compute_cumulative_distribution(
    target_treatment_arms: np.ndarray,
    locations: np.ndarray,
    covariates: np.ndarray,
    treatment_arms: np.ndarray,
    outcomes: np.array,
) -> np.ndarray:
    """Mock implementation for testing purposes."""
    return (
        np.linspace(0, 0.9, locations.shape[0]) + target_treatment_arms * 0.1,
        np.zeros((outcomes.shape[0], locations.shape[0])),
        np.zeros((outcomes.shape[0], locations.shape[0])),
    )


class MockDistributionEstimator(DistributionEstimatorBase):
    """
    Mock class to implement _compute_cumulative_distribution for testing.
    """

    def __init__(
        self, mock_compute_cumulative_distribution=compute_cumulative_distribution
    ):
        super().__init__()
        self.compute_cumulative_distribution = MagicMock()
        self.compute_cumulative_distribution.side_effect = (
            mock_compute_cumulative_distribution
        )

    def fit(self, covariates, treatment_arms, outcomes):
        """Mock fit method to set attributes."""
        self.covariates = covariates
        self.treatment_arms = treatment_arms
        self.outcomes = outcomes

    def _compute_cumulative_distribution(
        self,
        target_treatment_arms: np.ndarray,
        locations: np.ndarray,
        covariates: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.array,
    ) -> np.ndarray:
        return self.compute_cumulative_distribution(
            target_treatment_arms, locations, covariates, treatment_arms, outcomes
        )


def compute_confidence_intervals(*args, **kwargs):
    """Mock function for compute_confidence_intervals."""
    size = len(kwargs["vec_loc"])
    lower_bound = np.full(size, 0.1)
    upper_bound = np.full(size, 0.9)
    return lower_bound, upper_bound


class TestDistributionEstimatorBase(unittest.TestCase):
    def setUp(self):
        self.estimator = MockDistributionEstimator()
        self.covariates = np.zeros((20, 5))
        self.treatment_arms = np.hstack([np.zeros(10), np.ones(10)])
        self.outcomes = np.arange(20)
        self.estimator.fit(self.covariates, self.treatment_arms, self.outcomes)

    def test_initialization(self):
        # Arrange
        base_estimator = MockDistributionEstimator()

        # Assert
        self.assertIsNone(base_estimator.covariates)
        self.assertIsNone(base_estimator.treatment_arms)
        self.assertIsNone(base_estimator.outcomes)

    @patch(
        "dte_adj.compute_confidence_intervals", side_effect=compute_confidence_intervals
    )
    def test_predict_dte(self, mock_compute_confidence_intervals):
        # Arrange
        target_treatment_arm = 1
        control_treatment_arm = 0
        locations = np.arange(20)

        # Act
        dte, lower_bound, upper_bound = self.estimator.predict_dte(
            target_treatment_arm, control_treatment_arm, locations
        )

        # Assert
        np.testing.assert_array_almost_equal(dte, np.full(locations.shape, 0.1))
        np.testing.assert_array_almost_equal(lower_bound, np.full(locations.shape, 0.1))
        np.testing.assert_array_almost_equal(upper_bound, np.full(locations.shape, 0.9))
        self.estimator.compute_cumulative_distribution.assert_called()

    @patch(
        "dte_adj.compute_confidence_intervals", side_effect=compute_confidence_intervals
    )
    def test_predict_pte(self, mock_compute_confidence_intervals):
        # Arrange
        target_treatment_arm = 1
        control_treatment_arm = 0
        locations = np.arange(20)
        width = 0.1

        # Act
        pte, lower_bound, upper_bound = self.estimator.predict_pte(
            target_treatment_arm, control_treatment_arm, width, locations
        )

        # Assert
        np.testing.assert_array_almost_equal(pte, np.full(locations.shape, 0))
        np.testing.assert_array_almost_equal(lower_bound, np.full(locations.shape, 0.1))
        np.testing.assert_array_almost_equal(upper_bound, np.full(locations.shape, 0.9))
        self.estimator.compute_cumulative_distribution.assert_called()

    def test_predict_qte(self):
        # Arrange
        target_treatment_arm = 1
        control_treatment_arm = 0
        quantiles = np.array([0.1 * i for i in range(1, 10)])
        expected_qte = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Act
        qte, lower_bound, upper_bound = self.estimator.predict_qte(
            target_treatment_arm, control_treatment_arm, quantiles, n_bootstrap=5
        )

        # Assert
        np.testing.assert_array_almost_equal(qte, expected_qte)
        np.testing.assert_array_almost_equal(lower_bound.shape, quantiles.shape)
        np.testing.assert_array_almost_equal(upper_bound.shape, quantiles.shape)
        self.estimator.compute_cumulative_distribution.assert_called()

    def test_fit_success(self):
        # Assert
        self.assertTrue(np.array_equal(self.estimator.covariates, self.covariates))
        self.assertTrue(
            np.array_equal(self.estimator.treatment_arms, self.treatment_arms)
        )
        self.assertTrue(np.array_equal(self.estimator.outcomes, self.outcomes))

    def test_predict_success(self):
        # Arrange
        treatment_arm_test = 0
        locations_test = np.array([3, 6])

        # Act
        self.estimator.predict(treatment_arm_test, locations_test)

        # Assert
        self.estimator.compute_cumulative_distribution.assert_called_once()

    def test_predict_fail_before_fit(self):
        # Arrange
        treatment_arm_test = 0
        locations_test = np.array([3, 6])
        subject = MockDistributionEstimator()

        # Act, Assert
        with self.assertRaises(ValueError) as cm:
            subject.predict(treatment_arm_test, locations_test)
        self.assertEqual(
            str(cm.exception),
            "This estimator has not been trained yet. Please call fit first",
        )

    def test_predict_fail_invalid_arm(self):
        # Arrange
        treatment_arm_invalid = 2
        locations_test = np.array([3, 6])

        # Act, Assert
        with self.assertRaises(ValueError) as cm:
            self.estimator.predict(treatment_arm_invalid, locations_test)
        self.assertEqual(
            str(cm.exception),
            "This target treatment arm was not included in the training data: 2",
        )
