import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from dte_adj import SimpleLocalDistributionEstimator, AdjustedLocalDistributionEstimator

np.random.seed(123)


def generate_data(n=1000, S=4):
    # Generate W ~ U(0,1)
    W = np.random.uniform(0, 1, n)

    # Assign strata based on W
    strata = np.digitize(W, np.linspace(0, 1, S + 1)[1:])

    # Generate X ~ N(0, I_20)
    X = np.random.randn(n, 20)

    # Treatment assignment Z ~ Bernoulli(0.5) within each stratum
    Z = np.zeros(n)
    for s in range(S):
        indices = np.where(strata == s)[0]
        Z[indices] = np.random.binomial(1, 0.5, size=len(indices))

    # Define functions b(X, W) and c(X, W)
    def b(X, W):
        return (
            np.sin(np.pi * X[:, 0] * X[:, 1])
            + 2 * (X[:, 2] - 0.5) ** 2
            + X[:, 3]
            + 0.5 * X[:, 4]
            + 0.1 * W
        )

    def c(X, W):
        return 0.1 * (X[:, 0] + np.log(1 + np.exp(X[:, 1])) + W)

    # Define parameters
    a1, a0 = 4, 1
    b1, b0 = 1, -1
    c1, c0 = 3, 3

    # Generate errors
    epsilon = np.random.randn(n)

    # Compute Y(d)
    Y0 = a0 + b(X, W) + epsilon
    Y1 = a1 + b(X, W) + epsilon

    # Compute D(0) and D(1)
    D0 = (b0 + c(X, W) > c0 * epsilon).astype(int)
    D1 = np.where(D0 == 0, (b1 + c(X, W) > c1 * epsilon).astype(int), 1)

    # Compute observed D and Y
    D = D1 * Z + D0 * (1 - Z)
    Y = Y1 * D + Y0 * (1 - D)

    # discrete
    Y = np.random.poisson(np.abs(Y))

    return {
        "W": W,
        "X": X,
        "Z": Z,
        "D": D,
        "Y": Y,
        "strata": strata,
    }


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


class TestE2E(unittest.TestCase):
    def test_e2e(self):
        # Arrange
        data = generate_data(n=3000)
        X, D, Y, Z, S = data["X"], data["W"], data["Y"], data["Z"], data["strata"]
        locations = np.array([np.percentile(Y, p) for p in range(10, 91, 10)])
        simple_estimator = SimpleLocalDistributionEstimator()
        adjusted_estimator = AdjustedLocalDistributionEstimator(LinearRegression())

        # Act
        simple_estimator.fit(X, Z, D, Y, S)
        adjusted_estimator.fit(X, Z, D, Y, S)

        simple_dte, simple_lower_bound, simple_upper_bound = (
            simple_estimator.predict_dte(1, 0, locations)
        )
        adjusted_dte, adjusted_lower_bound, adjusted_upper_bound = (
            adjusted_estimator.predict_dte(1, 0, locations)
        )

        # Assert
        np.testing.assert_(np.all(simple_dte < 0), "Not all values are negative")
        np.testing.assert_(np.all(adjusted_dte < 0), "Not all values are negative")
        np.testing.assert_(
            np.all(simple_lower_bound < simple_upper_bound),
            "Upper bound is less than lower bound",
        )
        np.testing.assert_(
            np.all(adjusted_lower_bound < adjusted_upper_bound),
            "Upper bound is less than lower bound",
        )
        np.testing.assert_(
            np.all(
                adjusted_upper_bound - adjusted_lower_bound
                < simple_upper_bound - simple_lower_bound
            ),
            "Adjusted estimator does not have narrower intervals",
        )
