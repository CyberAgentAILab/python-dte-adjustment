import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from dte_adj import SimpleDistributionEstimator, AdjustedDistributionEstimator


def generate_data(n, d_x=100, rho=0.5):
    """
    Generate data according to the described data generating process (DGP).

    Args:
    n (int): Number of samples.
    d_x (int): Number of covariates. Default is 100.
    rho (float): Success probability for the Bernoulli distribution. Default is 0.5.

    Returns:
    X (np.ndarray): Covariates matrix of shape (n, d_x).
    D (np.ndarray): Treatment variable array of shape (n,).
    Y (np.ndarray): Outcome variable array of shape (n,).
    """
    # Generate covariates X from a uniform distribution on (0, 1)
    X = np.random.uniform(0, 1, (n, d_x))

    # Generate treatment variable D from a Bernoulli distribution with success probability rho
    D = np.random.binomial(1, rho, n)

    # Define beta_j and gamma_j according to the problem statement
    beta = np.zeros(d_x)
    gamma = np.zeros(d_x)

    # Set the first 50 values of beta and gamma to 1
    beta[:50] = 1
    gamma[:50] = 1

    # Compute the outcome Y
    U = np.random.normal(0, 1, n)  # Error term
    linear_term = np.dot(X, beta)
    quadratic_term = np.dot(X**2, gamma)

    # Outcome equation
    Y = 5 * D + linear_term + quadratic_term + U

    return X, D, Y


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


class TestAdjustedEstimator(unittest.TestCase):
    def setUp(self):
        base_model = MagicMock()
        base_model.predict_proba.side_effect = lambda x, y: x
        self.estimator = AdjustedDistributionEstimator(base_model, folds=2)
        self.covariates = np.zeros((20, 5))
        self.treatment_arms = np.hstack([np.zeros(10), np.ones(10)])
        self.outcomes = np.arange(20)
        self.estimator.fit(self.covariates, self.treatment_arms, self.outcomes)

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
            "The shape of covariates and treatment_arm should be same",
        )

    def test_compute_cumulative_distribution(self):
        # Arrange
        mock_model = self.estimator.base_model
        mock_model.predict_proba.side_effect = lambda x: np.ones((len(x), 2)) * 0.5
        target_treatment_arm = 0
        locations = np.arange(10)

        # Act
        with patch(
            "numpy.random.randint",
            return_value=np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5),
        ):
            cumulative_distribution, _, superset_prediction = (
                self.estimator._compute_cumulative_distribution(
                    target_treatment_arm,
                    locations,
                    self.covariates,
                    self.treatment_arms,
                    self.outcomes,
                )
            )

        # Assert
        self.assertEqual(cumulative_distribution.shape, (10,))
        self.assertEqual(superset_prediction.shape, (20, 10))

        for i in range(10):
            self.assertAlmostEqual(cumulative_distribution[i], (i + 1) / 10, places=2)

        expected_result = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(
            superset_prediction, expected_result, decimal=2
        )


class TestE2E(unittest.TestCase):
    def test_e2e(self):
        # Arrange
        X, D, Y = generate_data(n=1000)
        locations = np.array([np.percentile(Y, p) for p in range(10, 91, 10)])
        simple_estimator = SimpleDistributionEstimator()
        adjusted_estimator = AdjustedDistributionEstimator(LogisticRegression())

        # Act
        simple_estimator.fit(X, D, Y)
        adjusted_estimator.fit(X, D, Y)

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
