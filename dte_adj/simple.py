import numpy as np
from .stratified import (
    SimpleStratifiedDistributionEstimator,
    AdjustedStratifiedDistributionEstimator,
)


class SimpleDistributionEstimator(SimpleStratifiedDistributionEstimator):
    """
    A class for computing the empirical distribution function and distributional treatment effects
    using simple (unadjusted) estimation methods.

    This estimator computes Distribution Treatment Effects (DTE), Probability Treatment Effects (PTE),
    and Quantile Treatment Effects (QTE) without using machine learning models for adjustment.
    It provides a baseline approach suitable when treatment assignment is random or when
    covariate adjustment is not needed.

    Example:
        .. code-block:: python

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
    """

    def __init__(self):
        """Initializes the SimpleDistributionEstimator.

        Returns:
            SimpleDistributionEstimator: An instance of the estimator.
        """
        super().__init__()

    def fit(
        self, covariates: np.ndarray, treatment_arms: np.ndarray, outcomes: np.ndarray
    ) -> "SimpleDistributionEstimator":
        """
        Set parameters.

        Args:
            covariates (np.ndarray): Pre-treatment covariates.
            treatment_arms (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar-valued observed outcome.

        Returns:
            SimpleDistributionEstimator: The fitted estimator.
        """
        if covariates.shape[0] != treatment_arms.shape[0]:
            raise ValueError("The shape of covariates and treatment_arm should be same")

        if covariates.shape[0] != outcomes.shape[0]:
            raise ValueError("The shape of covariates and outcome should be same")

        self.covariates = covariates
        self.treatment_arms = treatment_arms
        self.outcomes = outcomes
        self.strata = np.zeros(len(self.covariates))

        return self


class AdjustedDistributionEstimator(AdjustedStratifiedDistributionEstimator):
    """
    A class for computing distribution treatment effects using machine learning adjustment.

    This estimator uses cross-fitting with ML models to adjust for confounding when computing
    Distribution Treatment Effects (DTE), Probability Treatment Effects (PTE), and
    Quantile Treatment Effects (QTE). It provides more precise estimates when treatment
    assignment depends on observed covariates.

    Example:
        .. code-block:: python

            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from dte_adj import AdjustedDistributionEstimator

            # Generate confounded data
            X = np.random.randn(1000, 5)
            treatment_prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1])))
            D = np.random.binomial(1, treatment_prob, 1000)
            Y = X.sum(axis=1) + 2 * D + np.random.randn(1000)

            # Fit adjusted estimator
            base_model = RandomForestClassifier(n_estimators=100)
            estimator = AdjustedDistributionEstimator(base_model, folds=3)
            estimator.fit(X, D, Y)

            # Compute adjusted treatment effects
            locations = np.linspace(Y.min(), Y.max(), 20)
            dte, lower, upper = estimator.predict_dte(1, 0, locations, variance_type="moment")
    """

    def fit(
        self, covariates: np.ndarray, treatment_arms: np.ndarray, outcomes: np.ndarray
    ) -> "AdjustedDistributionEstimator":
        """
        Set parameters.

        Args:
            covariates (np.ndarray): Pre-treatment covariates.
            treatment_arms (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar-valued observed outcome.

        Returns:
            AdjustedDistributionEstimator: The fitted estimator.
        """
        if covariates.shape[0] != treatment_arms.shape[0]:
            raise ValueError("The shape of covariates and treatment_arm should be same")

        if covariates.shape[0] != outcomes.shape[0]:
            raise ValueError("The shape of covariates and outcome should be same")

        self.covariates = covariates
        self.treatment_arms = treatment_arms
        self.outcomes = outcomes
        self.strata = np.zeros(len(self.covariates))

        return self
