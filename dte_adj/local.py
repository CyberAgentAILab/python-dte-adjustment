from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from dte_adj.stratified import (
    SimpleStratifiedDistributionEstimator,
    AdjustedStratifiedDistributionEstimator,
)
from dte_adj.util import (
    ArrayLike,
    compute_ldte,
    compute_lpte,
    _convert_to_ndarray,
    _infer_default_locations,
)


class SimpleLocalDistributionEstimator(SimpleStratifiedDistributionEstimator):
    """
    A class for computing Local Distribution Treatment Effects (LDTE) and Local Probability
    Treatment Effects (LPTE) using simple empirical estimation.

    This estimator computes treatment effects that are weighted by treatment propensity
    within each stratum, providing estimates that are locally robust to treatment assignment
    heterogeneity across strata. It uses empirical methods without ML adjustment.
    """

    def __init__(self):
        """
        Initializes the SimpleLocalDistributionEstimator.

        Returns:
            SimpleLocalDistributionEstimator: An instance of the estimator.
        """
        super().__init__()

    def fit(
        self,
        covariates: ArrayLike,
        treatment_arms: ArrayLike,
        treatment_indicator: ArrayLike,
        outcomes: ArrayLike,
        strata: ArrayLike,
    ) -> SimpleLocalDistributionEstimator:
        """
        Train the SimpleLocalDistributionEstimator.

        Args:
            covariates: Pre-treatment covariates.
            treatment_arms: Treatment assignment variable (Z).
            treatment_indicator: Treatment indicator variable (D).
            outcomes: Scalar-valued observed outcome.
            strata: Stratum indicators.

        Returns:
            SimpleLocalDistributionEstimator: The fitted estimator.
        """
        treatment_indicator = _convert_to_ndarray(treatment_indicator)
        super().fit(covariates, treatment_arms, outcomes, strata)
        self.treatment_indicator = treatment_indicator

        return self

    def predict_ldte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        locations: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        display_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Local Distribution Treatment Effects (LDTE).

        LDTE measures the difference in cumulative distribution functions between treatment groups
        weighted by treatment propensity within each stratum. This provides estimates that are
        locally robust to treatment assignment heterogeneity across strata.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            locations (np.ndarray, optional): Scalar values to be used for computing the cumulative
                distribution. If None, evenly-spaced locations spanning the observed outcome range
                are generated automatically. The number of points is determined from data size and
                distribution via ``np.histogram_bin_edges(outcomes, bins='auto')``. The actual
                array used is stored on ``self.last_locations``.
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            display_progress (bool, optional): Whether to display a progress bar. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected LDTEs (np.ndarray): Local treatment effect estimates at each location
                - Lower bounds (np.ndarray): Lower confidence interval bounds
                - Upper bounds (np.ndarray): Upper confidence interval bounds

        Example:
            ```python
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from dte_adj import AdjustedLocalDistributionEstimator

            # Generate sample data with strata
            np.random.seed(42)
            X = np.random.randn(1000, 5)
            strata = np.random.choice([0, 1], size=1000)  # Binary strata
            D = np.random.binomial(1, 0.3 + 0.4 * strata, 1000)  # Treatment depends on strata
            Y = X[:, 0] + 2 * D + strata + np.random.randn(1000)

            # Fit local estimator
            base_model = LogisticRegression()
            estimator = AdjustedLocalDistributionEstimator(base_model)
            estimator.fit(X, D, D, Y, strata)  # treatment_arms = treatment_indicator for binary case

            # Compute LDTE
            locations = np.linspace(Y.min(), Y.max(), 20)
            ldte, lower, upper = estimator.predict_ldte(
                target_treatment_arm=1,
                control_treatment_arm=0,
                locations=locations
            )

            print(f"LDTE shape: {ldte.shape}")  # Should match locations.shape
            print(f"Average LDTE: {ldte.mean():.3f}")
            ```
        """
        if locations is None:
            locations = _infer_default_locations(self.outcomes, for_intervals=False)
        self.last_locations = locations
        return compute_ldte(
            self,
            target_treatment_arm,
            control_treatment_arm,
            locations,
            alpha,
            display_progress,
        )

    def predict_lpte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        locations: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        display_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Local Probability Treatment Effects (LPTE).

        LPTE measures the difference in probability mass between treatment groups for intervals
        defined by consecutive location pairs, weighted by treatment propensity within each stratum.
        This provides locally robust estimates of treatment effects on interval probabilities.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            locations (np.ndarray, optional): Scalar values defining interval boundaries for
                probability computation. For each interval (locations[i], locations[i+1]], the LPTE
                is computed. If None, boundaries spanning the observed outcome range are generated
                automatically with the left endpoint placed just below ``outcomes.min()``. The
                number of boundaries is determined from data size and distribution via
                ``np.histogram_bin_edges(outcomes, bins='auto')``. The actual array used is stored
                on ``self.last_locations``.
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            display_progress (bool, optional): Whether to display a progress bar. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected LPTEs (np.ndarray): Local treatment effect estimates for each interval,
                  shape (len(locations)-1,)
                - Lower bounds (np.ndarray): Lower confidence interval bounds
                - Upper bounds (np.ndarray): Upper confidence interval bounds

        Example:
            ```python
            import numpy as np
            from dte_adj import SimpleLocalDistributionEstimator

            # Generate sample data with strata
            np.random.seed(42)
            X = np.random.randn(1000, 5)
            strata = np.random.choice([0, 1], size=1000)  # Binary strata
            Z = np.random.binomial(1, 0.5, 1000)  # Treatment assignment
            D = np.random.binomial(1, 0.3 + 0.4 * Z, 1000)  # Treatment receipt
            Y = X[:, 0] + 2 * D + strata + np.random.randn(1000)

            # Fit local estimator
            estimator = SimpleLocalDistributionEstimator()
            estimator.fit(X, Z, D, Y, strata)

            # Define interval boundaries
            locations = np.array([-2, -1, 0, 1, 2])  # Creates intervals: (-2,-1], (-1,0], (0,1], (1,2]

            # Compute LPTE
            lpte, lower, upper = estimator.predict_lpte(
                target_treatment_arm=1,
                control_treatment_arm=0,
                locations=locations
            )

            print(f"LPTE shape: {lpte.shape}")  # Should be (4,) for 4 intervals
            print(f"Interval effects: {lpte}")
            ```
        """
        if locations is None:
            locations = _infer_default_locations(self.outcomes, for_intervals=True)
        self.last_locations = locations
        return compute_lpte(
            self,
            target_treatment_arm,
            control_treatment_arm,
            locations,
            alpha,
            display_progress,
        )


class AdjustedLocalDistributionEstimator(AdjustedStratifiedDistributionEstimator):
    """
    A class for computing Local Distribution Treatment Effects (LDTE) and Local Probability
    Treatment Effects (LPTE) using machine learning adjustment.

    This estimator combines the benefits of ML adjustment with local treatment effect estimation,
    providing precise estimates of treatment effects that are weighted by treatment propensity
    within each stratum. It uses cross-fitting to avoid overfitting issues.
    """

    def fit(
        self,
        covariates: ArrayLike,
        treatment_arms: ArrayLike,
        treatment_indicator: ArrayLike,
        outcomes: ArrayLike,
        strata: ArrayLike,
    ) -> AdjustedLocalDistributionEstimator:
        """
        Train the AdjustedLocalDistributionEstimator.

        Args:
            covariates: Pre-treatment covariates.
            treatment_arms: Treatment assignment variable (Z).
            treatment_indicator: Treatment indicator variable (D).
            outcomes: Scalar-valued observed outcome.
            strata: Stratum indicators.

        Returns:
            AdjustedLocalDistributionEstimator: The fitted estimator.
        """
        treatment_indicator = _convert_to_ndarray(treatment_indicator)
        super().fit(covariates, treatment_arms, outcomes, strata)
        self.treatment_indicator = treatment_indicator

        return self

    def predict_ldte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        locations: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        display_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Local Distribution Treatment Effects (LDTE) using ML adjustment.

        This method combines machine learning adjustment with local treatment effect estimation
        to provide precise, locally robust estimates of distributional treatment effects.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            locations (np.ndarray, optional): Scalar values to be used for computing the cumulative
                distribution. If None, evenly-spaced locations spanning the observed outcome range
                are generated automatically. The number of points is determined from data size and
                distribution via ``np.histogram_bin_edges(outcomes, bins='auto')``. The actual
                array used is stored on ``self.last_locations``.
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            display_progress (bool, optional): Whether to display a progress bar. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected LDTEs (np.ndarray): Local treatment effect estimates at each location
                - Lower bounds (np.ndarray): Lower confidence interval bounds
                - Upper bounds (np.ndarray): Upper confidence interval bounds

        Example:
            ```python
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from dte_adj import AdjustedLocalDistributionEstimator

            # Generate confounded data with strata
            np.random.seed(42)
            X = np.random.randn(1000, 5)
            strata = np.random.choice([0, 1], size=1000)
            # Treatment assignment depends on covariates
            Z_prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] + strata)))
            Z = np.random.binomial(1, Z_prob, 1000)
            D = np.random.binomial(1, 0.3 + 0.4 * Z, 1000)
            Y = X.sum(axis=1) + 2 * D + strata + np.random.randn(1000)

            # Fit adjusted local estimator
            base_model = RandomForestClassifier(n_estimators=100)
            estimator = AdjustedLocalDistributionEstimator(base_model, folds=3)
            estimator.fit(X, Z, D, Y, strata)

            # Compute LDTE with ML adjustment
            locations = np.linspace(Y.min(), Y.max(), 20)
            ldte, lower, upper = estimator.predict_ldte(
                target_treatment_arm=1,
                control_treatment_arm=0,
                locations=locations
            )

            print(f"Adjusted LDTE: {ldte.mean():.3f}")
            ```
        """
        if locations is None:
            locations = _infer_default_locations(self.outcomes, for_intervals=False)
        self.last_locations = locations
        return compute_ldte(
            self,
            target_treatment_arm,
            control_treatment_arm,
            locations,
            alpha,
            display_progress,
        )

    def predict_lpte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        locations: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        display_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Local Probability Treatment Effects (LPTE) using ML adjustment.

        This method combines machine learning adjustment with local treatment effect estimation
        to provide precise estimates of treatment effects on interval probabilities.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            locations (np.ndarray, optional): Scalar values defining interval boundaries for
                probability computation. For each interval (locations[i], locations[i+1]], the LPTE
                is computed. If None, boundaries spanning the observed outcome range are generated
                automatically with the left endpoint placed just below ``outcomes.min()``. The
                number of boundaries is determined from data size and distribution via
                ``np.histogram_bin_edges(outcomes, bins='auto')``. The actual array used is stored
                on ``self.last_locations``.
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            display_progress (bool, optional): Whether to display a progress bar. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected LPTEs (np.ndarray): Local treatment effect estimates for each interval,
                  shape (len(locations)-1,)
                - Lower bounds (np.ndarray): Lower confidence interval bounds
                - Upper bounds (np.ndarray): Upper confidence interval bounds

        Example:
            ```python
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from dte_adj import AdjustedLocalDistributionEstimator

            # Generate confounded data with strata
            np.random.seed(42)
            X = np.random.randn(1000, 5)
            strata = np.random.choice([0, 1], size=1000)
            # Treatment assignment depends on covariates
            Z_prob = 1 / (1 + np.exp(-(X[:, 0] + strata)))
            Z = np.random.binomial(1, Z_prob, 1000)
            D = np.random.binomial(1, 0.3 + 0.4 * Z, 1000)
            Y = X.sum(axis=1) + 2 * D + strata + np.random.randn(1000)

            # Fit adjusted local estimator
            base_model = LogisticRegression()
            estimator = AdjustedLocalDistributionEstimator(base_model, folds=3)
            estimator.fit(X, Z, D, Y, strata)

            # Define interval boundaries
            locations = np.array([-2, -1, 0, 1, 2])

            # Compute LPTE with ML adjustment
            lpte, lower, upper = estimator.predict_lpte(
                target_treatment_arm=1,
                control_treatment_arm=0,
                locations=locations
            )

            print(f"Adjusted LPTE: {lpte}")
            ```
        """
        if locations is None:
            locations = _infer_default_locations(self.outcomes, for_intervals=True)
        self.last_locations = locations
        return compute_lpte(
            self,
            target_treatment_arm,
            control_treatment_arm,
            locations,
            alpha,
            display_progress,
        )
