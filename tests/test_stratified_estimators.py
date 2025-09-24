import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from dte_adj import (
    SimpleStratifiedDistributionEstimator,
    AdjustedStratifiedDistributionEstimator,
)


def generate_data(n=1000, S=4, d=2, discrete=False):
    d = 20

    Z = np.random.uniform(0, 1, n)

    S_i = np.digitize(Z, np.linspace(0, 1, S + 1)[1:-1])

    X = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n)

    W = np.zeros(n, dtype=int)
    unique_strata = np.unique(S_i)
    for s in unique_strata:
        idx = np.where(S_i == s)[0]
        n_s = len(idx)
        W[idx[: n_s // 2]] = 1
        np.random.shuffle(W[idx])

    b_X = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    c_X = 0.1 * (X[:, 0] + np.log(1 + np.exp(X[:, 1])))

    gamma = 0.1
    u = np.random.normal(0, 1, n)

    Y = b_X + c_X * W + gamma * Z + u
    if discrete:
        Y = np.random.poisson(0.2 * np.abs(Y))

    return {"W": W, "X": X, "Z": Z, "Y": Y, "strata": S_i}


class TestStratifiedEstimators(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        data = generate_data(n=1000, S=4, d=20, discrete=False)
        self.X = data["X"]
        self.W = data["W"]
        self.Y = data["Y"]
        self.strata = data["strata"]
        self.locations = np.linspace(self.Y.min(), self.Y.max(), 20)

    def test_simple_stratified_estimator_fit(self):
        estimator = SimpleStratifiedDistributionEstimator()
        result = estimator.fit(self.X, self.W, self.Y, self.strata)

        self.assertIsInstance(result, SimpleStratifiedDistributionEstimator)
        self.assertTrue(np.array_equal(estimator.covariates, self.X))
        self.assertTrue(np.array_equal(estimator.treatment_arms, self.W))
        self.assertTrue(np.array_equal(estimator.outcomes, self.Y))
        self.assertTrue(np.array_equal(estimator.strata, self.strata))

    def test_simple_stratified_estimator_predict_dte(self):
        estimator = SimpleStratifiedDistributionEstimator()
        estimator.fit(self.X, self.W, self.Y, self.strata)

        dte, lower_bound, upper_bound = estimator.predict_dte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            locations=self.locations,
            alpha=0.05,
        )

        self.assertEqual(dte.shape, self.locations.shape)
        self.assertEqual(lower_bound.shape, self.locations.shape)
        self.assertEqual(upper_bound.shape, self.locations.shape)
        self.assertTrue(np.all(lower_bound <= dte))
        self.assertTrue(np.all(dte <= upper_bound))

    def test_simple_stratified_estimator_predict_pte(self):
        estimator = SimpleStratifiedDistributionEstimator()
        estimator.fit(self.X, self.W, self.Y, self.strata)

        pte, lower_bound, upper_bound = estimator.predict_pte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            locations=self.locations,
            alpha=0.05,
        )

        expected_length = len(self.locations) - 1
        self.assertEqual(pte.shape, (expected_length,))
        self.assertEqual(lower_bound.shape, (expected_length,))
        self.assertEqual(upper_bound.shape, (expected_length,))
        self.assertTrue(np.all(lower_bound <= upper_bound))

    def test_simple_stratified_estimator_predict_qte(self):
        estimator = SimpleStratifiedDistributionEstimator()
        estimator.fit(self.X, self.W, self.Y, self.strata)

        quantiles = np.array([0.25, 0.5, 0.75])
        qte, lower_bound, upper_bound = estimator.predict_qte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            quantiles=quantiles,
            n_bootstrap=50,
        )

        self.assertEqual(qte.shape, quantiles.shape)
        self.assertEqual(lower_bound.shape, quantiles.shape)
        self.assertEqual(upper_bound.shape, quantiles.shape)
        self.assertTrue(np.all(lower_bound <= upper_bound))

    def test_adjusted_stratified_estimator_fit(self):
        base_model = LogisticRegression(max_iter=1000, random_state=42)
        estimator = AdjustedStratifiedDistributionEstimator(base_model, folds=3)
        result = estimator.fit(self.X, self.W, self.Y, self.strata)

        self.assertIsInstance(result, AdjustedStratifiedDistributionEstimator)
        self.assertTrue(np.array_equal(estimator.covariates, self.X))
        self.assertTrue(np.array_equal(estimator.treatment_arms, self.W))
        self.assertTrue(np.array_equal(estimator.outcomes, self.Y))
        self.assertTrue(np.array_equal(estimator.strata, self.strata))
        self.assertEqual(estimator.folds, 3)

    def test_adjusted_stratified_estimator_predict_dte(self):
        base_model = LogisticRegression(max_iter=1000, random_state=42)
        estimator = AdjustedStratifiedDistributionEstimator(base_model, folds=3)
        estimator.fit(self.X, self.W, self.Y, self.strata)

        dte, lower_bound, upper_bound = estimator.predict_dte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            locations=self.locations,
            alpha=0.05,
            variance_type="moment",
        )

        self.assertEqual(dte.shape, self.locations.shape)
        self.assertEqual(lower_bound.shape, self.locations.shape)
        self.assertEqual(upper_bound.shape, self.locations.shape)
        self.assertTrue(np.all(lower_bound <= dte))
        self.assertTrue(np.all(dte <= upper_bound))

    def test_adjusted_stratified_estimator_predict_pte(self):
        base_model = LogisticRegression(max_iter=1000, random_state=42)
        estimator = AdjustedStratifiedDistributionEstimator(base_model, folds=3)
        estimator.fit(self.X, self.W, self.Y, self.strata)

        pte, lower_bound, upper_bound = estimator.predict_pte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            locations=self.locations,
            alpha=0.05,
            variance_type="moment",
        )

        expected_length = len(self.locations) - 1
        self.assertEqual(pte.shape, (expected_length,))
        self.assertEqual(lower_bound.shape, (expected_length,))
        self.assertEqual(upper_bound.shape, (expected_length,))
        self.assertTrue(np.all(lower_bound <= upper_bound))

    def test_adjusted_stratified_estimator_predict_qte(self):
        base_model = LogisticRegression(max_iter=1000, random_state=42)
        estimator = AdjustedStratifiedDistributionEstimator(base_model, folds=3)
        estimator.fit(self.X, self.W, self.Y, self.strata)

        quantiles = np.array([0.25, 0.5, 0.75])
        qte, lower_bound, upper_bound = estimator.predict_qte(
            target_treatment_arm=1,
            control_treatment_arm=0,
            quantiles=quantiles,
            n_bootstrap=50,
        )

        self.assertEqual(qte.shape, quantiles.shape)
        self.assertEqual(lower_bound.shape, quantiles.shape)
        self.assertEqual(upper_bound.shape, quantiles.shape)
        self.assertTrue(np.all(lower_bound <= upper_bound))

    def test_discrete_outcomes(self):
        data = generate_data(n=1000, S=4, d=20, discrete=True)

        estimator = SimpleStratifiedDistributionEstimator()
        estimator.fit(data["X"], data["W"], data["Y"], data["strata"])

        locations = np.arange(0, data["Y"].max() + 1)
        dte, lower, upper = estimator.predict_dte(1, 0, locations)

        self.assertEqual(dte.shape, locations.shape)
        self.assertTrue(np.all(lower <= upper))

    def test_invalid_input_shapes(self):
        estimator = SimpleStratifiedDistributionEstimator()

        X_wrong = self.X[:-10]

        with self.assertRaises(ValueError):
            estimator.fit(X_wrong, self.W, self.Y, self.strata)

    def test_different_alpha_values(self):
        estimator = SimpleStratifiedDistributionEstimator()
        estimator.fit(self.X, self.W, self.Y, self.strata)

        locations = self.locations[:10]

        _, lower_005, upper_005 = estimator.predict_dte(1, 0, locations, alpha=0.05)
        _, lower_010, upper_010 = estimator.predict_dte(1, 0, locations, alpha=0.10)

        width_005 = upper_005 - lower_005
        width_010 = upper_010 - lower_010

        self.assertTrue(np.all(width_010 < width_005))
