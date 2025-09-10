import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm
from abc import ABC
import dte_adj


class DistributionEstimatorBase(ABC):
    """A mixin including several convenience functions to compute and display distribution functions."""

    def __init__(self):
        """
        Initializes the DistributionFunctionMixin.

        Returns:
            DistributionFunctionMixin: An instance of the estimator.
        """
        self.covariates = None
        self.outcomes = None
        self.treatment_arms = None

    def predict_dte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        locations: np.ndarray,
        alpha: float = 0.05,
        variance_type="moment",
        n_bootstrap=500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Distribution Treatment Effects (DTE) based on the estimator for the distribution function.

        The DTE measures the difference in cumulative distribution functions between treatment groups
        at specified locations. It quantifies how treatment affects the probability of observing
        outcomes below each threshold.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            variance_type (str, optional): Variance type to be used to compute confidence intervals.
                Available values are "moment", "simple", and "uniform". Defaults to "moment".
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected DTEs (np.ndarray): Treatment effect estimates at each location
                - Lower bounds (np.ndarray): Lower confidence interval bounds
                - Upper bounds (np.ndarray): Upper confidence interval bounds

        Example:
            .. code-block:: python

                import numpy as np
                from dte_adj import SimpleDistributionEstimator

                # Generate sample data
                X = np.random.randn(1000, 5)
                D = np.random.binomial(1, 0.5, 1000)
                Y = X[:, 0] + 2 * D + np.random.randn(1000)

                # Fit estimator
                estimator = SimpleDistributionEstimator()
                estimator.fit(X, D, Y)

                # Compute DTE
                locations = np.linspace(Y.min(), Y.max(), 20)
                dte, lower, upper = estimator.predict_dte(
                    target_treatment_arm=1,
                    control_treatment_arm=0,
                    locations=locations,
                    variance_type="moment"
                )

                print(f"DTE shape: {dte.shape}")  # Should match locations.shape
                print(f"Average DTE: {dte.mean():.3f}")
        """
        return self._compute_dtes(
            target_treatment_arm,
            control_treatment_arm,
            locations,
            alpha,
            variance_type,
            n_bootstrap,
        )

    def predict_pte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        locations: np.ndarray,
        alpha: float = 0.05,
        variance_type="moment",
        n_bootstrap=500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Probability Treatment Effects (PTE) based on the estimator for the distribution function.

        The PTE measures the difference in probability mass between treatment groups for intervals
        defined by consecutive location pairs. It quantifies how treatment affects the probability
        of observing outcomes within specific ranges.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            locations (np.ndarray): Scalar values defining interval boundaries for probability computation.
                For each interval (locations[i], locations[i+1]], the PTE is computed.
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            variance_type (str, optional): Variance type to be used to compute confidence intervals.
                Available values are "moment", "simple", and "uniform". Defaults to "moment".
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected PTEs (np.ndarray): Treatment effect estimates for each interval,
                  shape (len(locations)-1,)
                - Lower bounds (np.ndarray): Lower confidence interval bounds
                - Upper bounds (np.ndarray): Upper confidence interval bounds

        Example:
            .. code-block:: python

                import numpy as np
                from dte_adj import SimpleDistributionEstimator

                # Generate sample data
                X = np.random.randn(1000, 5)
                D = np.random.binomial(1, 0.5, 1000)
                Y = X[:, 0] + 2 * D + np.random.randn(1000)

                # Fit estimator
                estimator = SimpleDistributionEstimator()
                estimator.fit(X, D, Y)

                # Define interval boundaries
                locations = np.array([-2, -1, 0, 1, 2])  # Creates intervals: (-2,-1], (-1,0], (0,1], (1,2]

                # Compute PTE
                pte, lower, upper = estimator.predict_pte(
                    target_treatment_arm=1,
                    control_treatment_arm=0,
                    locations=locations,
                    variance_type="moment"
                )

                print(f"PTE shape: {pte.shape}")  # Should be (4,) for 4 intervals
                print(f"Interval effects: {pte}")
        """
        return self._compute_ptes(
            target_treatment_arm,
            control_treatment_arm,
            locations,
            alpha,
            variance_type,
            n_bootstrap,
        )

    def predict_qte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        quantiles: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap=500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Quantile Treatment Effects (QTE) based on the estimator for the distribution function.

        The QTE measures the difference in quantiles between treatment groups, providing insights
        into how treatment affects different parts of the outcome distribution. For stratified
        estimators, the computation properly accounts for strata.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            quantiles (np.ndarray, optional): Quantiles used for QTE. Defaults to [0.1, 0.2, ..., 0.9].
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected QTEs (np.ndarray): Treatment effect estimates at each quantile
                - Lower bounds (np.ndarray): Lower confidence interval bounds
                - Upper bounds (np.ndarray): Upper confidence interval bounds

        Example:
            .. code-block:: python

                import numpy as np
                from dte_adj import SimpleStratifiedDistributionEstimator

                # Generate stratified sample data
                X = np.random.randn(1000, 5)
                strata = np.random.choice([0, 1, 2], size=1000)
                D = np.random.binomial(1, 0.5, 1000)
                Y = X[:, 0] + 2 * D + 0.5 * strata + np.random.randn(1000)

                # Fit stratified estimator
                estimator = SimpleStratifiedDistributionEstimator()
                estimator.fit(X, D, Y, strata)

                # Compute QTE at specific quantiles
                quantiles = np.array([0.25, 0.5, 0.75])  # 25th, 50th, 75th percentiles
                qte, lower, upper = estimator.predict_qte(
                    target_treatment_arm=1,
                    control_treatment_arm=0,
                    quantiles=quantiles,
                    n_bootstrap=100
                )

                print(f"QTE at quantiles {quantiles}: {qte}")
                print(f"Median effect (50th percentile): {qte[1]:.3f}")
        """
        qte = self._compute_qtes(
            target_treatment_arm,
            control_treatment_arm,
            quantiles,
            self.covariates,
            self.treatment_arms,
            self.outcomes,
            self.strata,
        )
        n_obs = len(self.outcomes)
        indexes = np.arange(n_obs)

        qtes = np.zeros((n_bootstrap, qte.shape[0]))
        for b in range(n_bootstrap):
            bootstrap_indexes = np.random.choice(indexes, size=n_obs, replace=True)

            qtes[b] = self._compute_qtes(
                target_treatment_arm,
                control_treatment_arm,
                quantiles,
                self.covariates[bootstrap_indexes],
                self.treatment_arms[bootstrap_indexes],
                self.outcomes[bootstrap_indexes],
                self.strata[bootstrap_indexes],
            )

        qte_var = qtes.var(axis=0)

        qte_lower = qte + norm.ppf(alpha / 2) * np.sqrt(qte_var)
        qte_upper = qte + norm.ppf(1 - alpha / 2) * np.sqrt(qte_var)

        return qte, qte_lower, qte_upper

    def _compute_dtes(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        locations: np.ndarray,
        alpha: float,
        variance_type: str,
        n_bootstrap: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected DTEs."""
        treatment_cdf, treatment_cdf_mat, _ = self._compute_cumulative_distribution(
            target_treatment_arm,
            locations,
            self.covariates,
            self.treatment_arms,
            self.outcomes,
        )
        control_cdf, control_cdf_mat, _ = self._compute_cumulative_distribution(
            control_treatment_arm,
            locations,
            self.covariates,
            self.treatment_arms,
            self.outcomes,
        )

        dte = treatment_cdf - control_cdf

        mat_indicator = (self.outcomes[:, np.newaxis] <= locations).astype(int)

        lower_bound, upper_bound = dte_adj.compute_confidence_intervals(
            vec_y=self.outcomes,
            vec_d=self.treatment_arms,
            vec_loc=locations,
            mat_y_u=mat_indicator,
            vec_prediction_target=treatment_cdf,
            vec_prediction_control=control_cdf,
            mat_entire_predictions_target=treatment_cdf_mat,
            mat_entire_predictions_control=control_cdf_mat,
            ind_target=target_treatment_arm,
            ind_control=control_treatment_arm,
            alpha=alpha,
            variance_type=variance_type,
            n_bootstrap=n_bootstrap,
        )

        return (
            dte,
            lower_bound,
            upper_bound,
        )

    def _compute_ptes(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        locations: np.ndarray,
        alpha: float,
        variance_type: str,
        n_bootstrap: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected PTEs."""
        treatment_pdf, treatment_pdf_mat, _ = self._compute_interval_probability(
            target_treatment_arm,
            locations,
            self.covariates,
            self.treatment_arms,
            self.outcomes,
        )
        control_pdf, control_pdf_mat, _ = self._compute_interval_probability(
            control_treatment_arm,
            locations,
            self.covariates,
            self.treatment_arms,
            self.outcomes,
        )

        pte = treatment_pdf - control_pdf

        # Compute interval indicators for confidence intervals
        mat_indicator = (self.outcomes[:, np.newaxis] <= locations).astype(int)
        mat_interval_indicator = mat_indicator[:, 1:] - mat_indicator[:, :-1]

        lower_bound, upper_bound = dte_adj.compute_confidence_intervals(
            vec_y=self.outcomes,
            vec_d=self.treatment_arms,
            vec_loc=locations[:-1],  # Use interval boundaries
            mat_y_u=mat_interval_indicator,
            vec_prediction_target=treatment_pdf,
            vec_prediction_control=control_pdf,
            mat_entire_predictions_target=treatment_pdf_mat,
            mat_entire_predictions_control=control_pdf_mat,
            ind_target=target_treatment_arm,
            ind_control=control_treatment_arm,
            alpha=alpha,
            variance_type=variance_type,
            n_bootstrap=n_bootstrap,
        )

        return (
            pte,
            lower_bound,
            upper_bound,
        )

    def _compute_qtes(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        quantiles: np.ndarray,
        covariates: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.array,
        strata: np.ndarray,
    ) -> np.ndarray:
        """Compute expected QTEs."""
        locations = np.sort(outcomes)

        def find_quantile(quantile, arm):
            low, high = 0, locations.shape[0] - 1
            result = -1
            while low <= high:
                mid = (low + high) // 2
                # Temporarily store original strata and use the provided strata
                original_strata = self.strata
                self.strata = strata

                val, _, _ = self._compute_cumulative_distribution(
                    arm,
                    np.full((1), locations[mid]),
                    covariates,
                    treatment_arms,
                    outcomes,
                )

                # Restore original strata
                self.strata = original_strata

                if val[0] <= quantile:
                    result = locations[mid]
                    low = mid + 1
                else:
                    high = mid - 1
            return result

        result = np.zeros(quantiles.shape)
        for i, q in enumerate(quantiles):
            result[i] = find_quantile(q, target_treatment_arm) - find_quantile(
                q, control_treatment_arm
            )

        return result

    def predict(self, treatment_arm: int, locations: np.ndarray) -> np.ndarray:
        """
        Compute cumulative distribution values.

        Args:
            treatment_arm (int): The index of the treatment arm.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.

        Returns:
            np.ndarray: Estimated cumulative distribution values for the input.
        """
        if self.outcomes is None:
            raise ValueError(
                "This estimator has not been trained yet. Please call fit first"
            )

        if treatment_arm not in self.treatment_arms:
            raise ValueError(
                f"This target treatment arm was not included in the training data: {treatment_arm}"
            )

        return self._compute_cumulative_distribution(
            treatment_arm,
            locations,
            self.covariates,
            self.treatment_arms,
            self.outcomes,
        )[0]

    def _compute_cumulative_distribution(
        self,
        target_treatment_arm: int,
        locations: np.ndarray,
        covariates: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the cumulative distribution values.

        Args:
            target_treatment_arm (int): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            covariates: (np.ndarray): An array of covariates variables in the observed data.
            treatment_arms (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Estimated cumulative distribution values, prediction for each observation, and superset prediction for each observation.
        """
        raise NotImplementedError()
