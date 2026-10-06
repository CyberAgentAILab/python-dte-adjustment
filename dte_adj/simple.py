from __future__ import annotations

import numpy as np
from dte_adj.stratified import (
    SimpleStratifiedDistributionEstimator,
    AdjustedStratifiedDistributionEstimator,
)
from dte_adj.util import ArrayLike, _prepare_fit_inputs


class SimpleDistributionEstimator(SimpleStratifiedDistributionEstimator):
    """
    A class for computing the empirical distribution function and distributional treatment effects
    using simple (unadjusted) estimation methods.

    This estimator computes Distribution Treatment Effects (DTE), Probability Treatment Effects (PTE),
    and Quantile Treatment Effects (QTE) without using machine learning models for adjustment.
    It provides a baseline approach for randomized experiments where covariate adjustment
    is not needed.

    Example:
        ```python
        import numpy as np
        from dte_adj import SimpleDistributionEstimator

        # Generate sample data
        X = np.random.randn(1000, 5)
        D = np.random.binomial(1, 0.5, 1000)  # Random treatment
        Y = X[:, 0] + 2 * D + np.random.randn(1000)

        # Fit simple estimator
        estimator = SimpleDistributionEstimator()
        estimator.fit(X, D, Y)

        # Compute treatment effects
        locations = np.linspace(Y.min(), Y.max(), 20)
        dte, lower, upper = estimator.predict_dte(1, 0, locations)
        pte, pte_lower, pte_upper = estimator.predict_pte(1, 0, locations)
        ```
    """

    def __init__(self):
        """Initializes the SimpleDistributionEstimator.

        Returns:
            SimpleDistributionEstimator: An instance of the estimator.
        """
        super().__init__()

    def fit(
        self, covariates: ArrayLike, treatment_arms: ArrayLike, outcomes: ArrayLike
    ) -> SimpleDistributionEstimator:
        """
        Set parameters.

        Args:
            covariates: Pre-treatment covariates.
            treatment_arms: The index of the treatment arm.
            outcomes: Scalar-valued observed outcome.

        Returns:
            SimpleDistributionEstimator: The fitted estimator.
        """
        covariates, treatment_arms, outcomes, strata = _prepare_fit_inputs(
            covariates, treatment_arms, outcomes
        )

        self.covariates = covariates
        self.treatment_arms = treatment_arms
        self.outcomes = outcomes
        self.strata = strata

        return self


class AdjustedDistributionEstimator(AdjustedStratifiedDistributionEstimator):
    """
    A class for computing distribution treatment effects using machine learning adjustment.

    This estimator uses cross-fitting with ML models of the conditional distribution given
    covariates to compute Distribution Treatment Effects (DTE), Probability Treatment Effects
    (PTE), and Quantile Treatment Effects (QTE) with reduced variance. It is designed for
    randomized experiments, where treatment is assigned independently of the covariates: in
    that setting the estimator stays consistent regardless of how well the ML model fits, and
    a good fit yields tighter confidence intervals than ``SimpleDistributionEstimator``.

    Note:
        The method targets efficiency, not bias correction. It does not correct for
        confounding in observational data; if treatment assignment depends on covariates, the
        estimates are valid only under selection on observables, and ``fit`` assumes the
        assignment probability is constant across observations (no propensity weighting).

    Example:
        ```python
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from dte_adj import AdjustedDistributionEstimator

        # Generate data from a randomized experiment: treatment is independent of X,
        # while the outcome depends on X (this is what the ML adjustment exploits)
        X = np.random.randn(1000, 5)
        D = np.random.binomial(1, 0.5, 1000)
        Y = X.sum(axis=1) + 2 * D + np.random.randn(1000)

        # Fit adjusted estimator
        base_model = RandomForestClassifier(n_estimators=100)
        estimator = AdjustedDistributionEstimator(base_model, folds=3)
        estimator.fit(X, D, Y)

        # Compute adjusted treatment effects
        locations = np.linspace(Y.min(), Y.max(), 20)
        dte, lower, upper = estimator.predict_dte(1, 0, locations, variance_type="moment")
        ```
    """

    def fit(
        self, covariates: ArrayLike, treatment_arms: ArrayLike, outcomes: ArrayLike
    ) -> AdjustedDistributionEstimator:
        """
        Set parameters.

        Args:
            covariates: Pre-treatment covariates.
            treatment_arms: The index of the treatment arm.
            outcomes: Scalar-valued observed outcome.

        Returns:
            AdjustedDistributionEstimator: The fitted estimator.
        """
        covariates, treatment_arms, outcomes, strata = _prepare_fit_inputs(
            covariates, treatment_arms, outcomes
        )

        self.covariates = covariates
        self.treatment_arms = treatment_arms
        self.outcomes = outcomes
        self.strata = strata

        return self
