import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression
from dte_adj import (
    SimpleDistributionEstimator,
    AdjustedDistributionEstimator,
    SimpleStratifiedDistributionEstimator,
    AdjustedStratifiedDistributionEstimator,
    SimpleLocalDistributionEstimator,
    AdjustedLocalDistributionEstimator,
)


class TestPandasInputSimple(unittest.TestCase):
    """Test that Simple/Adjusted DistributionEstimator accept pandas inputs."""

    def setUp(self):
        np.random.seed(42)
        n = 20
        self.covariates_df = pd.DataFrame(np.zeros((n, 5)), columns=[f"x{i}" for i in range(5)])
        self.treatment_arms_series = pd.Series(np.hstack([np.zeros(10), np.ones(10)]))
        self.outcomes_series = pd.Series(np.arange(n, dtype=float))

    def test_simple_estimator_with_dataframe_and_series(self):
        estimator = SimpleDistributionEstimator()
        result = estimator.fit(
            self.covariates_df, self.treatment_arms_series, self.outcomes_series
        )

        self.assertIsInstance(result.covariates, np.ndarray)
        self.assertIsInstance(result.treatment_arms, np.ndarray)
        self.assertIsInstance(result.outcomes, np.ndarray)

    def test_simple_estimator_predict_after_pandas_fit(self):
        estimator = SimpleDistributionEstimator()
        estimator.fit(self.covariates_df, self.treatment_arms_series, self.outcomes_series)

        output = estimator.predict(0, np.array([3, 6]))
        expected = np.array([0.4, 0.7])
        np.testing.assert_array_almost_equal(output, expected, decimal=2)

    def test_adjusted_estimator_with_dataframe_and_series(self):
        base_model = MagicMock()
        base_model.predict_proba.side_effect = lambda x, y: x
        estimator = AdjustedDistributionEstimator(base_model, folds=2)
        result = estimator.fit(
            self.covariates_df, self.treatment_arms_series, self.outcomes_series
        )

        self.assertIsInstance(result.covariates, np.ndarray)
        self.assertIsInstance(result.treatment_arms, np.ndarray)
        self.assertIsInstance(result.outcomes, np.ndarray)


class TestPandasInputStratified(unittest.TestCase):
    """Test that Stratified estimators accept pandas inputs."""

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.covariates_df = pd.DataFrame(
            np.random.randn(n, 5), columns=[f"x{i}" for i in range(5)]
        )
        self.treatment_arms_series = pd.Series(np.random.choice([0, 1], size=n))
        self.outcomes_series = pd.Series(np.random.randn(n))
        self.strata_series = pd.Series(np.random.choice([0, 1, 2], size=n))

    def test_simple_stratified_with_pandas(self):
        estimator = SimpleStratifiedDistributionEstimator()
        result = estimator.fit(
            self.covariates_df,
            self.treatment_arms_series,
            self.outcomes_series,
            self.strata_series,
        )

        self.assertIsInstance(result.covariates, np.ndarray)
        self.assertIsInstance(result.treatment_arms, np.ndarray)
        self.assertIsInstance(result.outcomes, np.ndarray)
        self.assertIsInstance(result.strata, np.ndarray)

    def test_adjusted_stratified_with_pandas(self):
        base_model = LogisticRegression(random_state=42)
        estimator = AdjustedStratifiedDistributionEstimator(base_model, folds=2)
        result = estimator.fit(
            self.covariates_df,
            self.treatment_arms_series,
            self.outcomes_series,
            self.strata_series,
        )

        self.assertIsInstance(result.covariates, np.ndarray)
        self.assertIsInstance(result.treatment_arms, np.ndarray)
        self.assertIsInstance(result.outcomes, np.ndarray)
        self.assertIsInstance(result.strata, np.ndarray)


class TestPandasInputLocal(unittest.TestCase):
    """Test that Local estimators accept pandas inputs."""

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.covariates_df = pd.DataFrame(
            np.random.randn(n, 3), columns=[f"x{i}" for i in range(3)]
        )
        self.treatment_arms_series = pd.Series(np.random.choice([0, 1], size=n))
        self.treatment_indicator_series = pd.Series(np.random.choice([0, 1], size=n))
        self.outcomes_series = pd.Series(np.random.randn(n))
        self.strata_series = pd.Series(np.random.choice([0, 1], size=n))

    def test_simple_local_with_pandas(self):
        estimator = SimpleLocalDistributionEstimator()
        result = estimator.fit(
            self.covariates_df,
            self.treatment_arms_series,
            self.treatment_indicator_series,
            self.outcomes_series,
            self.strata_series,
        )

        self.assertIsInstance(result.covariates, np.ndarray)
        self.assertIsInstance(result.treatment_arms, np.ndarray)
        self.assertIsInstance(result.treatment_indicator, np.ndarray)
        self.assertIsInstance(result.outcomes, np.ndarray)
        self.assertIsInstance(result.strata, np.ndarray)

    def test_adjusted_local_with_pandas(self):
        base_model = LogisticRegression(random_state=42)
        estimator = AdjustedLocalDistributionEstimator(base_model=base_model)
        result = estimator.fit(
            self.covariates_df,
            self.treatment_arms_series,
            self.treatment_indicator_series,
            self.outcomes_series,
            self.strata_series,
        )

        self.assertIsInstance(result.covariates, np.ndarray)
        self.assertIsInstance(result.treatment_arms, np.ndarray)
        self.assertIsInstance(result.treatment_indicator, np.ndarray)
        self.assertIsInstance(result.outcomes, np.ndarray)
        self.assertIsInstance(result.strata, np.ndarray)


class TestNumpyInputStillWorks(unittest.TestCase):
    """Verify that np.ndarray inputs continue to work as before."""

    def test_simple_estimator_with_numpy(self):
        estimator = SimpleDistributionEstimator()
        covariates = np.zeros((20, 5))
        treatment_arms = np.hstack([np.zeros(10), np.ones(10)])
        outcomes = np.arange(20, dtype=float)

        result = estimator.fit(covariates, treatment_arms, outcomes)

        self.assertIsInstance(result.covariates, np.ndarray)
        self.assertIsInstance(result.treatment_arms, np.ndarray)
        self.assertIsInstance(result.outcomes, np.ndarray)
