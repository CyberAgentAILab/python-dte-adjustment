import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from dte_adj import SimpleLocalDistributionEstimator, AdjustedLocalDistributionEstimator


class TestLocalEstimators(unittest.TestCase):
    def setUp(self):
        # Set up test data
        np.random.seed(42)
        n_samples = 100
        n_features = 3

        # Generate covariates
        self.covariates = np.random.randn(n_samples, n_features)

        # Generate strata
        self.strata = np.random.choice([0, 1], size=n_samples)

        # Generate treatment assignment and indicator (both binary in this case)
        self.treatment_arms = np.random.choice([0, 1], size=n_samples)
        self.treatment_indicator = self.treatment_arms.copy()  # Same for simple case

        # Generate outcomes
        self.outcomes = np.random.randn(n_samples) + 0.5 * self.treatment_indicator

        # Note: weights are now computed internally from strata and treatment assignment

        # Test locations
        self.locations = np.array([-1.0, 0.0, 1.0])

    def test_simple_local_estimator_fit(self):
        """Test that SimpleLocalDistributionEstimator can be fitted."""
        estimator = SimpleLocalDistributionEstimator()
        fitted_estimator = estimator.fit(
            self.covariates,
            self.treatment_arms,
            self.treatment_indicator,
            self.outcomes,
            self.strata,
        )

        # Check that the estimator is fitted
        self.assertIsNotNone(fitted_estimator.covariates)
        self.assertIsNotNone(fitted_estimator.treatment_arms)
        self.assertIsNotNone(fitted_estimator.outcomes)
        self.assertIsNotNone(fitted_estimator.strata)
        self.assertIsNotNone(fitted_estimator.treatment_indicator)

    def test_simple_local_estimator_predict_ldte(self):
        """Test that SimpleLocalDistributionEstimator can predict LDTE."""
        estimator = SimpleLocalDistributionEstimator()
        estimator.fit(
            self.covariates,
            self.treatment_arms,
            self.treatment_indicator,
            self.outcomes,
            self.strata,
        )

        # Predict LDTE
        beta, lower_bound, upper_bound = estimator.predict_ldte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            locations=self.locations,
            alpha=0.05,
        )

        # Check output shapes
        self.assertEqual(beta.shape, self.locations.shape)
        self.assertEqual(lower_bound.shape, self.locations.shape)
        self.assertEqual(upper_bound.shape, self.locations.shape)

        # Check that confidence intervals are properly ordered
        self.assertTrue(np.all(lower_bound <= upper_bound))
        self.assertTrue(np.all(lower_bound <= beta))
        self.assertTrue(np.all(beta <= upper_bound))

    def test_adjusted_local_estimator_fit(self):
        """Test that AdjustedLocalDistributionEstimator can be fitted."""
        base_model = LogisticRegression(random_state=42)
        estimator = AdjustedLocalDistributionEstimator(base_model=base_model)
        fitted_estimator = estimator.fit(
            self.covariates,
            self.treatment_arms,
            self.treatment_indicator,
            self.outcomes,
            self.strata,
        )

        # Check that the estimator is fitted
        self.assertIsNotNone(fitted_estimator.covariates)
        self.assertIsNotNone(fitted_estimator.treatment_arms)
        self.assertIsNotNone(fitted_estimator.outcomes)
        self.assertIsNotNone(fitted_estimator.strata)
        self.assertIsNotNone(fitted_estimator.treatment_indicator)

    def test_adjusted_local_estimator_predict_ldte(self):
        """Test that AdjustedLocalDistributionEstimator can predict LDTE."""
        base_model = LogisticRegression(random_state=42)
        estimator = AdjustedLocalDistributionEstimator(base_model=base_model)
        estimator.fit(
            self.covariates,
            self.treatment_arms,
            self.treatment_indicator,
            self.outcomes,
            self.strata,
        )

        # Predict LDTE
        beta, lower_bound, upper_bound = estimator.predict_ldte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            locations=self.locations,
            alpha=0.05,
        )

        # Check output shapes
        self.assertEqual(beta.shape, self.locations.shape)
        self.assertEqual(lower_bound.shape, self.locations.shape)
        self.assertEqual(upper_bound.shape, self.locations.shape)

        # Check that confidence intervals are properly ordered
        self.assertTrue(np.all(lower_bound <= upper_bound))
        self.assertTrue(np.all(lower_bound <= beta))
        self.assertTrue(np.all(beta <= upper_bound))

    def test_invalid_shapes(self):
        """Test that estimators raise errors for invalid input shapes."""
        estimator = SimpleLocalDistributionEstimator()

        # Test with mismatched shapes
        with self.assertRaises(ValueError):
            estimator.fit(
                self.covariates[:50],
                self.treatment_arms,
                self.treatment_indicator,
                self.outcomes,
                self.strata,
            )

        with self.assertRaises(ValueError):
            estimator.fit(
                self.covariates,
                self.treatment_arms,
                self.treatment_indicator,
                self.outcomes[:50],
                self.strata,
            )

    def test_different_alpha_values(self):
        """Test that different alpha values produce different confidence intervals."""
        estimator = SimpleLocalDistributionEstimator()
        estimator.fit(
            self.covariates,
            self.treatment_arms,
            self.treatment_indicator,
            self.outcomes,
            self.strata,
        )

        # Test with different alpha values
        beta1, lower1, upper1 = estimator.predict_ldte(1, 0, self.locations, alpha=0.05)
        beta2, lower2, upper2 = estimator.predict_ldte(1, 0, self.locations, alpha=0.1)

        # Beta should be the same
        np.testing.assert_array_almost_equal(beta1, beta2)

        # Confidence intervals should be different (narrower for higher alpha)
        self.assertTrue(np.all(lower2 >= lower1))
        self.assertTrue(np.all(upper2 <= upper1))

    def test_simple_local_estimator_predict_lpte(self):
        """Test that SimpleLocalDistributionEstimator can predict LPTE."""
        estimator = SimpleLocalDistributionEstimator()
        estimator.fit(
            self.covariates,
            self.treatment_arms,
            self.treatment_indicator,
            self.outcomes,
            self.strata,
        )

        # Predict LPTE (note: need more than one location for intervals)
        interval_locations = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        beta, lower_bound, upper_bound = estimator.predict_lpte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            locations=interval_locations,
            alpha=0.05,
        )

        # Check output shapes (should be len(locations) - 1 for intervals)
        expected_shape = (len(interval_locations) - 1,)
        self.assertEqual(beta.shape, expected_shape)
        self.assertEqual(lower_bound.shape, expected_shape)
        self.assertEqual(upper_bound.shape, expected_shape)

        # Check that confidence intervals are properly ordered
        self.assertTrue(np.all(lower_bound <= upper_bound))
        self.assertTrue(np.all(lower_bound <= beta))
        self.assertTrue(np.all(beta <= upper_bound))

    def test_adjusted_local_estimator_predict_lpte(self):
        """Test that AdjustedLocalDistributionEstimator can predict LPTE."""
        base_model = LogisticRegression(random_state=42)
        estimator = AdjustedLocalDistributionEstimator(base_model=base_model)
        estimator.fit(
            self.covariates,
            self.treatment_arms,
            self.treatment_indicator,
            self.outcomes,
            self.strata,
        )

        # Predict LPTE (note: need more than one location for intervals)
        interval_locations = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        beta, lower_bound, upper_bound = estimator.predict_lpte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            locations=interval_locations,
            alpha=0.05,
        )

        # Check output shapes (should be len(locations) - 1 for intervals)
        expected_shape = (len(interval_locations) - 1,)
        self.assertEqual(beta.shape, expected_shape)
        self.assertEqual(lower_bound.shape, expected_shape)
        self.assertEqual(upper_bound.shape, expected_shape)

        # Check that confidence intervals are properly ordered
        self.assertTrue(np.all(lower_bound <= upper_bound))
        self.assertTrue(np.all(lower_bound <= beta))
        self.assertTrue(np.all(beta <= upper_bound))
