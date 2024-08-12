import numpy as np
from typing import Tuple
from scipy.stats import norm
from copy import deepcopy
from abc import ABC
from .util import compute_confidence_intervals

__all__ = ["SimpleDistributionEstimator", "AdjustedDistributionEstimator"]


class DistributionEstimatorBase(ABC):
    """A mixin including several convenience functions to compute and display distribution functions."""

    def __init__(self):
        """Initializes the DistributionFunctionMixin.

        Returns:
            DistributionFunctionMixin: An instance of the estimator.
        """
        self.confoundings = None
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
        """Compute DTE based on the estimator for the distribution function.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            variance_type (str, optional): Variance type to be used to compute confidence intervals. Available values are moment, simple, and uniform.
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected DTEs
                - Upper bounds
                - Lower bounds
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
        width: float,
        locations: np.ndarray,
        alpha: float = 0.05,
        variance_type="moment",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute PTE based on the estimator for the distribution function.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            width (float): The width of each outcome interval.
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            variance_type (str, optional): Variance type to be used to compute confidence intervals. Available values are moment, simple, and uniform.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected PTEs
                - Upper bounds
                - Lower bounds
        """
        return self._compute_ptes(
            target_treatment_arm,
            control_treatment_arm,
            locations,
            width,
            alpha,
            variance_type,
        )

    def predict_qte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        quantiles: np.ndarray = np.array(
            [0.1 * i for i in range(1, 10)], dtype=np.float32
        ),
        alpha: float = 0.05,
        n_bootstrap=500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute QTE based on the estimator for the distribution function.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            quantiles (np.ndarray, optional): Quantiles used for QTE. Defaults to [0.1 * i for i in range(1, 10)].
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected QTEs
                - Upper bounds
                - Lower bounds
        """
        qte = self._compute_qtes(
            target_treatment_arm,
            control_treatment_arm,
            quantiles,
            self.confoundings,
            self.treatment_arms,
            self.outcomes,
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
                self.confoundings[bootstrap_indexes],
                self.treatment_arms[bootstrap_indexes],
                self.outcomes[bootstrap_indexes],
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
        treatment_cdf, treatment_cdf_mat = self._compute_cumulative_distribution(
            target_treatment_arm,
            locations,
            self.confoundings,
            self.treatment_arms,
            self.outcomes,
        )
        control_cdf, control_cdf_mat = self._compute_cumulative_distribution(
            control_treatment_arm,
            locations,
            self.confoundings,
            self.treatment_arms,
            self.outcomes,
        )

        dte = treatment_cdf - control_cdf

        mat_indicator = (self.outcomes[:, np.newaxis] <= locations).astype(int)

        lower_bound, upper_bound = compute_confidence_intervals(
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
        width: float,
        alpha: float,
        variance_type: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected PTEs."""
        treatment_cumulative_pre, treatment_cdf_mat_pre = (
            self._compute_cumulative_distribution(
                target_treatment_arm,
                locations,
                self.confoundings,
                self.treatment_arms,
                self.outcomes,
            )
        )
        treatment_cumulative_post, treatment_cdf_mat_post = (
            self._compute_cumulative_distribution(
                target_treatment_arm,
                locations + width,
                self.confoundings,
                self.treatment_arms,
                self.outcomes,
            )
        )
        treatment_pdf = treatment_cumulative_post - treatment_cumulative_pre
        control_cumulative_pre, control_cdf_mat_pre = (
            self._compute_cumulative_distribution(
                control_treatment_arm,
                locations,
                self.confoundings,
                self.treatment_arms,
                self.outcomes,
            )
        )
        control_cumulative_post, control_cdf_mat_post = (
            self._compute_cumulative_distribution(
                control_treatment_arm,
                locations + width,
                self.confoundings,
                self.treatment_arms,
                self.outcomes,
            )
        )
        control_pdf = control_cumulative_post - control_cumulative_pre

        pte = treatment_pdf - control_pdf

        mat_indicator_pre = (self.outcomes[:, np.newaxis] <= locations).astype(int)
        mat_indicator_post = (self.outcomes[:, np.newaxis] <= locations + width).astype(
            int
        )

        lower_bound, upper_bound = compute_confidence_intervals(
            vec_y=self.outcomes,
            vec_d=self.treatment_arms,
            vec_loc=locations,
            mat_y_u=mat_indicator_post - mat_indicator_pre,
            vec_prediction_target=treatment_pdf,
            vec_prediction_control=control_pdf,
            mat_entire_predictions_target=treatment_cdf_mat_post
            - treatment_cdf_mat_pre,
            mat_entire_predictions_control=control_cdf_mat_post - control_cdf_mat_pre,
            ind_target=target_treatment_arm,
            ind_control=control_treatment_arm,
            alpha=alpha,
            variance_type=variance_type,
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
        confoundings: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected QTEs."""
        locations = np.sort(outcomes)

        def find_quantile(quantile, arm):
            low, high = 0, locations.shape[0] - 1
            result = -1
            while low <= high:
                mid = (low + high) // 2
                val, _ = self._compute_cumulative_distribution(
                    arm,
                    np.full((1), locations[mid]),
                    confoundings,
                    treatment_arms,
                    outcomes,
                )
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

    def fit(
        self, confoundings: np.ndarray, treatment_arms: np.ndarray, outcomes: np.ndarray
    ) -> "DistributionEstimatorBase":
        """Train the DistributionEstimatorBase.

        Args:
            confoundings (np.ndarray): Pre-treatment covariates.
            treatment_arms (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar-valued observed outcome.

        Returns:
            DistributionEstimatorBase: The fitted estimator.
        """
        if confoundings.shape[0] != treatment_arms.shape[0]:
            raise ValueError(
                "The shape of confounding and treatment_arm should be same"
            )

        if confoundings.shape[0] != outcomes.shape[0]:
            raise ValueError("The shape of confounding and outcome should be same")

        self.confoundings = confoundings
        self.treatment_arms = treatment_arms
        self.outcomes = outcomes

        return self

    def predict(self, treatment_arm: int, locations: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution values.

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
            self.confoundings,
            self.treatment_arms,
            self.outcomes,
        )[0]

    def _compute_cumulative_distribution(
        self,
        target_treatment_arm: int,
        locations: np.ndarray,
        confoundings: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.array,
    ) -> np.ndarray:
        """Compute the cumulative distribution values."""
        raise NotImplementedError()


class SimpleDistributionEstimator(DistributionEstimatorBase):
    """A class for computing the empirical distribution function and the distributional parameters
    based on the distribution function.
    """

    def __init__(self):
        """Initializes the SimpleDistributionEstimator.

        Returns:
            SimpleDistributionEstimator: An instance of the estimator.
        """
        super().__init__()

    def _compute_cumulative_distribution(
        self,
        target_treatment_arm: int,
        locations: np.ndarray,
        confoundings: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.array,
    ) -> np.ndarray:
        """Compute the cumulative distribution values.

        Args:
            target_treatment_arm (int): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            confoundings: (np.ndarray): An array of confounding variables in the observed data.
            treatment_arms (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data.

        Returns:
            np.ndarray: Estimated cumulative distribution values.
        """
        unique_treatment_arm = np.unique(treatment_arms)
        d_confounding = {}
        d_outcome = {}
        n_obs = outcomes.shape[0]
        n_loc = locations.shape[0]
        for arm in unique_treatment_arm:
            selected_confounding = confoundings[treatment_arms == arm]
            selected_outcome = outcomes[treatment_arms == arm]
            sorted_indices = np.argsort(selected_outcome)
            d_confounding[arm] = selected_confounding[sorted_indices]
            d_outcome[arm] = selected_outcome[sorted_indices]
        cumulative_distribution = np.zeros(locations.shape)
        for i, outcome in enumerate(locations):
            cumulative_distribution[i] = (
                np.searchsorted(d_outcome[target_treatment_arm], outcome, side="right")
            ) / len(d_outcome[target_treatment_arm])
        return cumulative_distribution, np.zeros((n_obs, n_loc))


class AdjustedDistributionEstimator(DistributionEstimatorBase):
    """A class is for estimating the adjusted distribution function and computing the Distributional parameters based on the trained conditional estimator."""

    def __init__(self, base_model, folds=3, is_multi_task=False):
        """Initializes the AdjustedDistributionEstimator.

        Args:
            base_model (scikit-learn estimator): The base model implementing used for conditional distribution function estimators. The model should implement fit(data, targets) and predict_proba(data).
            folds (int): The number of folds for cross-fitting.
            is_multi_task(bool): Whether to use multi-task learning. If True, your base model needs to support multi-task prediction (n_samples, n_features) -> (n_samples, n_targets).

        Returns:
            AdjustedDistributionEstimator: An instance of the estimator.
        """
        if (not hasattr(base_model, "predict")) and (
            not hasattr(base_model, "predict_proba")
        ):
            raise ValueError(
                "Base model should implement either predict_proba or predict"
            )
        self.base_model = base_model
        self.folds = folds
        self.is_multi_task = is_multi_task
        super().__init__()

    def _compute_cumulative_distribution(
        self,
        target_treatment_arm: int,
        locations: np.ndarray,
        confoundings: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.array,
    ) -> np.ndarray:
        """Compute the cumulative distribution values.

        Args:
            target_treatment_arm (int): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            confoundings: (np.ndarray): An array of confounding variables in the observed data.
            treatment_arm (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data

        Returns:
            np.ndarray: Estimated cumulative distribution values.
        """
        n_records = outcomes.shape[0]
        n_loc = locations.shape[0]
        cumulative_distribution = np.zeros(n_loc)
        superset_prediction = np.zeros((n_records, n_loc))
        treatment_mask = treatment_arms == target_treatment_arm
        if self.is_multi_task:
            confounding_in_arm = confoundings[treatment_mask]
            n_records_in_arm = len(confounding_in_arm)
            outcome_in_arm = outcomes[treatment_mask]  # (n_records)
            subset_prediction = np.zeros(
                (n_records_in_arm, n_loc)
            )  # (n_records_in_arm, n_loc)
            binominal = (outcomes.reshape(-1, 1) <= locations) * 1  # (n_records, n_loc)
            cdf = binominal[treatment_mask].mean(axis=0)  # (n_loc)
            for fold in range(self.folds):
                superset_mask = np.arange(n_records) % self.folds == fold
                subset_mask = superset_mask & treatment_mask
                subset_mask_inner = superset_mask[treatment_mask]
                confounding_train = confoundings[~subset_mask]
                confounding_fit = confoundings[subset_mask]
                binominal_train = binominal[~subset_mask]
                model = deepcopy(self.base_model)
                model.fit(confounding_train, binominal_train)
                subset_prediction[subset_mask_inner] = self._compute_model_prediction(
                    model, confounding_fit
                )
                superset_prediction[superset_mask] = self._compute_model_prediction(
                    model, confoundings[superset_mask]
                )
            cumulative_distribution = (
                cdf - subset_prediction.mean(axis=0) + superset_prediction.mean(axis=0)
            )  # (n_loc)
        else:
            for i, location in enumerate(locations):
                confounding_in_arm = confoundings[treatment_mask]
                outcome_in_arm = outcomes[treatment_mask]
                subset_prediction = np.zeros(outcome_in_arm.shape[0])
                binominal = (outcomes <= location) * 1  # (n_records)
                cdf = binominal[treatment_mask].mean()
                for fold in range(self.folds):
                    superset_mask = np.arange(n_records) % self.folds == fold
                    subset_mask = superset_mask & treatment_mask
                    subset_mask_inner = superset_mask[treatment_mask]
                    confounding_train = confoundings[~subset_mask]
                    confounding_fit = confoundings[subset_mask]
                    binominal_train = binominal[~subset_mask]
                    if len(np.unique(binominal_train)) == 1:
                        subset_prediction[subset_mask_inner] = binominal_train[0]
                        superset_prediction[superset_mask, i] = binominal_train[0]
                        continue
                    model = deepcopy(self.base_model)
                    model.fit(confounding_train, binominal_train)
                    subset_prediction[subset_mask_inner] = (
                        self._compute_model_prediction(model, confounding_fit)
                    )
                    superset_prediction[superset_mask, i] = (
                        self._compute_model_prediction(
                            model, confoundings[superset_mask]
                        )
                    )
                cumulative_distribution[i] = (
                    cdf - subset_prediction.mean() + superset_prediction[:, i].mean()
                )
        return cumulative_distribution, superset_prediction

    def _compute_model_prediction(self, model, confoundings: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            if self.is_multi_task:
                return model.predict_proba(confoundings)
            return model.predict_proba(confoundings)[:, 1]
        else:
            return model.predict(confoundings)
