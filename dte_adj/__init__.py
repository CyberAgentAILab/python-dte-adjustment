import numpy as np
from typing import Tuple
from scipy.stats import norm
import math
from copy import deepcopy


class DistributionFunctionMixin(object):
    """A mixin including several convenience functions to compute and display distribution functions."""

    def __init__(self):
        """Initializes the DistributionFunctionMixin.

        Returns:
            DistributionFunctionMixin: An instance of the estimator.
        """
        self.confounding = None
        self.outcome = None
        self.treatment_arm = None

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
            alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.
            variance_type (str, optional): Variance type to be used to compute confidence intervals. Available values are moment, simple, and uniform.
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected DTEs
                - Upper bands
                - Lower bands
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
            alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.
            variance_type (str, optional): Variance type to be used to compute confidence intervals. Available values are moment, simple, and uniform.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected PTEs
                - Upper bands
                - Lower bands
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
            alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected QTEs
                - Upper bands
                - Lower bands
        """
        qte = self._compute_qtes(
            target_treatment_arm,
            control_treatment_arm,
            quantiles,
            self.confounding,
            self.treatment_arm,
            self.outcome,
        )
        n_obs = len(self.outcome)
        indexes = np.arange(n_obs)

        qtes = np.zeros((n_bootstrap, qte.shape[0]))
        for b in range(n_bootstrap):
            bootstrap_indexes = np.random.choice(indexes, size=n_obs, replace=True)
            qtes[b] = self._compute_qtes(
                target_treatment_arm,
                control_treatment_arm,
                quantiles,
                self.confounding[bootstrap_indexes],
                self.treatment_arm[bootstrap_indexes],
                self.outcome[bootstrap_indexes],
            )

        qte_var = qtes.var(axis=0)

        qte_lower = qte + norm.ppf(alpha / 2) / np.sqrt(qte_var)
        qte_upper = qte + norm.ppf(1 - alpha / 2) / np.sqrt(qte_var)

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
            np.full(locations.shape, target_treatment_arm),
            locations,
            self.confounding,
            self.treatment_arm,
            self.outcome,
        )
        control_cdf, control_cdf_mat = self._compute_cumulative_distribution(
            np.full(locations.shape, control_treatment_arm),
            locations,
            self.confounding,
            self.treatment_arm,
            self.outcome,
        )

        dte = treatment_cdf - control_cdf

        mat_indicator = (self.outcome[:, np.newaxis] <= locations).astype(int)

        lower_band, upper_band = compute_confidence_intervals(
            vec_y=self.outcome,
            vec_d=self.treatment_arm,
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
            lower_band,
            upper_band,
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
                np.full(locations.shape, target_treatment_arm),
                locations,
                self.confounding,
                self.treatment_arm,
                self.outcome,
            )
        )
        treatment_cumulative_post, treatment_cdf_mat_post = (
            self._compute_cumulative_distribution(
                np.full(locations.shape, target_treatment_arm),
                locations + width,
                self.confounding,
                self.treatment_arm,
                self.outcome,
            )
        )
        treatment_pdf = treatment_cumulative_post - treatment_cumulative_pre
        control_cumulative_pre, control_cdf_mat_pre = (
            self._compute_cumulative_distribution(
                np.full(locations.shape, control_treatment_arm),
                locations,
                self.confounding,
                self.treatment_arm,
                self.outcome,
            )
        )
        control_cumulative_post, control_cdf_mat_post = (
            self._compute_cumulative_distribution(
                np.full(locations.shape, control_treatment_arm),
                locations + width,
                self.confounding,
                self.treatment_arm,
                self.outcome,
            )
        )
        control_pdf = control_cumulative_post - control_cumulative_pre

        pte = treatment_pdf - control_pdf

        mat_indicator_pre = (self.outcome[:, np.newaxis] <= locations).astype(int)
        mat_indicator_post = (self.outcome[:, np.newaxis] <= locations + width).astype(
            int
        )

        lower_band, upper_band = compute_confidence_intervals(
            vec_y=self.outcome,
            vec_d=self.treatment_arm,
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
            lower_band,
            upper_band,
        )

    def _compute_qtes(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        quantiles: np.ndarray,
        confounding: np.ndarray,
        treatment_arm: np.ndarray,
        outcome: np.array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected QTEs."""
        treatment_cumulative, _ = self._compute_cumulative_distribution(
            np.full(outcome.shape, target_treatment_arm),
            outcome,
            confounding,
            treatment_arm,
            outcome,
        )
        control_cumulative, _ = self._compute_cumulative_distribution(
            np.full(outcome.shape, control_treatment_arm),
            outcome,
            confounding,
            treatment_arm,
            outcome,
        )
        result = np.zeros(quantiles.shape)
        for i, q in enumerate(quantiles):
            treatment_idx = find_le(treatment_cumulative, q)
            control_idx = find_le(control_cumulative, q)
            result[i] = outcome[treatment_idx] - outcome[control_idx]

        return result

    def predict(self, treatment_arm: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution values.

        Args:
            treatment_arm (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.

        Returns:
            np.ndarray: Estimated cumulative distribution values for the input.
        """
        raise NotImplementedError()

    def _compute_cumulative_distribution(
        self,
        target_treatment_arm: np.ndarray,
        locations: np.ndarray,
        confounding: np.ndarray,
        treatment_arm: np.ndarray,
        outcome: np.array,
    ) -> np.ndarray:
        """Compute the cumulative distribution values."""
        raise NotImplementedError()


def find_le(array: np.ndarray, threshold):
    """Find the rightmost value less than or equal to threshold in a sorted array

    Args:
        array (np.ndarray): The sorted array to search in.
        threshold (float): The threshold value.

    Returns:
        int: The index where the value first exceeds the threshold.
    """
    low, high = 0, array.shape[0] - 1
    result = -1
    while low <= high:
        mid = (low + high) // 2
        if array[mid] <= threshold:
            result = mid
            low = mid + 1
        else:
            high = mid - 1
    return result


class SimpleDistributionEstimator(DistributionFunctionMixin):
    """A class for computing the empirical distribution function and the distributional parameters
    based on the distribution function.
    """

    def __init__(self):
        """Initializes the SimpleDistributionEstimator.

        Returns:
            SimpleDistributionEstimator: An instance of the estimator.
        """
        super().__init__()

    def fit(
        self, confounding: np.ndarray, treatment_arm: np.ndarray, outcome: np.ndarray
    ) -> "SimpleDistributionEstimator":
        """Train the SimpleDistributionEstimator.

        Args:
            confounding (np.ndarray): Pre-treatment covariates.
            treatment_arm (np.ndarray): The index of the treatment arm.
            outcome (np.ndarray): Scalar-valued observed outcome.

        Returns:
            SimpleDistributionEstimator: The fitted estimator.
        """
        if confounding.shape[0] != treatment_arm.shape[0]:
            raise RuntimeError(
                "The shape of confounding and treatment_arm should be same"
            )

        if confounding.shape[0] != outcome.shape[0]:
            raise RuntimeError("The shape of confounding and outcome should be same")

        self.confounding = confounding
        self.treatment_arm = treatment_arm
        self.outcome = outcome

        return self

    def predict(self, treatment_arm: np.ndarray, locations: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution values.

        Args:
            treatment_arm (np.ndarray): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.

        Returns:
            np.ndarray: Estimated cumulative distribution values for the input.
        """
        if self.outcome is None:
            raise RuntimeError(
                "This estimator has not been trained yet. Please call fit first"
            )

        return self._compute_cumulative_distribution(
            treatment_arm, locations, self.confounding, self.treatment_arm, self.outcome
        )[0]

    def _compute_cumulative_distribution(
        self,
        target_treatment_arm: np.ndarray,
        locations: np.ndarray,
        confounding: np.ndarray,
        treatment_arm: np.ndarray,
        outcome: np.array,
    ) -> np.ndarray:
        """Compute the cumulative distribution values.

        Args:
            target_treatment_arm (np.ndarray): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            confounding: (np.ndarray): An array of confounding variables in the observed data
            treatment_arm (np.ndarray): An array of treatment arms in the observed data
            outcome (np.ndarray): An array of outcomes in the observed data

        Returns:
            np.ndarray: Estimated cumulative distribution values.
        """
        unique_treatment_arm = np.unique(treatment_arm)
        d_confounding = {}
        d_outcome = {}
        n_obs = outcome.shape[0]
        n_loc = locations.shape[0]
        for arm in unique_treatment_arm:
            selected_confounding = confounding[treatment_arm == arm]
            selected_outcome = outcome[treatment_arm == arm]
            sorted_indices = np.argsort(selected_outcome)
            d_confounding[arm] = selected_confounding[sorted_indices]
            d_outcome[arm] = selected_outcome[sorted_indices]
        cumulative_distribution = np.zeros(locations.shape)
        for i, (outcome, arm) in enumerate(zip(locations, target_treatment_arm)):
            cumulative_distribution[i] = (
                find_le(d_outcome[arm], outcome) + 1
            ) / d_outcome[arm].shape[0]
        return cumulative_distribution, np.zeros((n_obs, n_loc))


class AdjustedDistributionEstimator(DistributionFunctionMixin):
    """A class is for estimating the adjusted distribution function and computing the Distributional parameters based on the trained conditional estimator."""

    def __init__(self, base_model, folds=3):
        """Initializes the AdjustedDistributionEstimator.

        Args:
            base_model (scikit-learn estimator): The base model implementing used for conditional distribution function estimators. The model should implement scikit-learn interface: https://scikit-learn.org/stable/developers/develop.html.
            folds (int): The number of folds for cross-fitting.

        Returns:
            AdjustedDistributionEstimator: An instance of the estimator.
        """
        self.base_model = base_model
        self.folds = folds
        super().__init__()

    def fit(
        self, confounding: np.ndarray, treatment_arm: np.ndarray, outcome: np.ndarray
    ) -> "AdjustedDistributionEstimator":
        """Train the AdjustedDistributionEstimator.

        Args:
            confounding (np.ndarray): Pre-treatment covariates.
            treatment_arm (np.ndarray): The index of the treatment arm.
            outcome (np.ndarray): Scalar-valued observed outcome.

        Returns:
            AdjustedDistributionEstimator: The fitted estimator.
        """
        if confounding.shape[0] != treatment_arm.shape[0]:
            raise RuntimeError(
                "The shape of confounding and treatment_arm should be same"
            )

        if confounding.shape[0] != outcome.shape[0]:
            raise RuntimeError("The shape of confounding and outcome should be same")

        self.confounding = confounding
        self.treatment_arm = treatment_arm
        self.outcome = outcome

        return self

    def predict(self, treatment_arm: np.ndarray, locations: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution values.

        Args:
            treatment_arm (np.ndarray): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.

        Returns:
            np.ndarray: Estimated cumulative distribution values for the input.
        """
        if self.outcome is None:
            raise RuntimeError(
                "This estimator has not been trained yet. Please call fit first"
            )

        return self._compute_cumulative_distribution(
            treatment_arm, locations, self.confounding, self.treatment_arm, self.outcome
        )[0]

    def _compute_cumulative_distribution(
        self,
        target_treatment_arm: np.ndarray,
        locations: np.ndarray,
        confounding: np.ndarray,
        treatment_arm: np.ndarray,
        outcome: np.array,
    ) -> np.ndarray:
        """Compute the cumulative distribution values.

        Args:
            target_treatment_arm (np.ndarray): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            confounding: (np.ndarray): An array of confounding variables in the observed data
            treatment_arm (np.ndarray): An array of treatment arms in the observed data
            outcome (np.ndarray): An array of outcomes in the observed data

        Returns:
            np.ndarray: Estimated cumulative distribution values.
        """
        n_obs = outcome.shape[0]
        n_loc = locations.shape[0]
        cumulative_distribution = np.zeros(locations.shape)
        superset_prediction = np.zeros((n_obs, n_loc))
        for i, (location, arm) in enumerate(zip(locations, target_treatment_arm)):
            confounding_in_arm = confounding[treatment_arm == arm]
            outcome_in_arm = outcome[treatment_arm == arm]
            subset_prediction = np.zeros(outcome_in_arm.shape[0])
            binominal = (outcome_in_arm <= location) * 1
            cdf = binominal.mean()
            for fold in range(self.folds):
                subset_mask = (
                    np.arange(confounding_in_arm.shape[0]) % self.folds == fold
                )
                confounding_train = confounding_in_arm[~subset_mask]
                confounding_fit = confounding_in_arm[subset_mask]
                binominal_train = binominal[~subset_mask]
                superset_mask = np.arange(self.outcome.shape[0]) % self.folds == fold
                if np.unique(binominal_train).shape[0] == 1:
                    subset_prediction[subset_mask] = binominal_train[0]
                    superset_prediction[superset_mask, i] = binominal_train[0]
                    continue
                model = deepcopy(self.base_model)
                model.fit(confounding_train, binominal_train)
                subset_prediction[subset_mask] = model.predict_proba(confounding_fit)[
                    :, 1
                ]
                superset_prediction[superset_mask, i] = model.predict_proba(
                    confounding[superset_mask]
                )[:, 1]
            cumulative_distribution[i] = (
                cdf - subset_prediction.mean() + superset_prediction[:, i].mean()
            )
        return cumulative_distribution, superset_prediction


def compute_confidence_intervals(
    vec_y: np.ndarray,
    vec_d: np.ndarray,
    vec_loc: np.ndarray,
    mat_y_u: np.ndarray,
    vec_prediction_target: np.ndarray,
    vec_prediction_control: np.ndarray,
    mat_entire_predictions_target: np.ndarray,
    mat_entire_predictions_control: np.ndarray,
    ind_target: int,
    ind_control: int,
    alpha: 0.05,
    variance_type="moment",
    n_bootstrap=500,
):
    """Computes the confidence intervals of distribution parameters.

    Args:
        vec_y (np.ndarray): Outcome variable vector.
        vec_d (np.ndarray): Treatment indicator vector.
        vec_loc (np.ndarray): Locations where the distribution parameters are estimated.
        mat_y_u (np.ndarray): Indicator function for 1{Y⩽y}. Shape is n_obs * n_loc.
        vec_prediction_target (np.ndarray): Estimated values from the conditional model for the treatment group.
        vec_prediction_control (np.ndarray): Estimated values from the conditional model for the control group.
        mat_entire_predictions_target (np.ndarray): Prediction of the conditional distribution estimator for target group.
        mat_entire_predictions_control (np.ndarray): Prediction of the conditional distribution estimator for control group.
        ind_target (int): Index of the target treatment indicator.
        ind_control (int): Index of the control treatment indicator.
        alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.
        variance_type (str, optional): Variance type to be used to compute confidence intervals. Available values are moment, simple, and uniform.
        n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: lower band.
            - np.ndarray: upper band.
    """
    num_obs = vec_y.shape[0]
    n_loc = vec_loc.shape[0]
    mat_d = np.tile(vec_d, (n_loc, 1)).T
    vec_dte = vec_prediction_target - vec_prediction_control
    mat_dte = np.tile(vec_dte, (num_obs, 1))

    num_target = (vec_d == ind_target).sum()
    num_control = (vec_d == ind_control).sum()
    influence_function = (
        num_obs
        / num_target
        * (mat_d == ind_target)
        * (mat_y_u - mat_entire_predictions_target)
        + mat_entire_predictions_target
        - num_obs
        / num_control
        * (mat_d == ind_control)
        * (mat_y_u - mat_entire_predictions_control)
        - mat_entire_predictions_control
        - mat_dte
    )

    omega = (influence_function**2).mean(axis=0)

    if variance_type == "moment":
        vec_dte_lower_moment = vec_dte + norm.ppf(alpha / 2) * np.sqrt(omega / num_obs)
        vec_dte_upper_moment = vec_dte + norm.ppf(1 - alpha / 2) * np.sqrt(
            omega / num_obs
        )
        return vec_dte_lower_moment, vec_dte_upper_moment
    elif variance_type == "uniform":
        tstats = np.zeros((n_bootstrap, len(vec_loc)))
        boot_draw = np.zeros((n_bootstrap, len(vec_loc)))

        for b in range(n_bootstrap):
            eta1 = np.random.normal(0, 1, num_obs)
            eta2 = np.random.normal(0, 1, num_obs)
            xi = eta1 / np.sqrt(2) + (eta2**2 - 1) / 2

            boot_draw[b, :] = (
                1 / num_obs * np.sum(xi[:, np.newaxis] * influence_function, axis=0)
            )

        tstats = np.abs(boot_draw)[:, :-1] / np.sqrt(omega[:-1] / num_obs)
        max_tstats = np.max(tstats, axis=1)
        quantile_max_tstats = np.quantile(max_tstats, 1 - alpha)

        vec_dte_lower_boot = vec_dte - quantile_max_tstats * np.sqrt(omega / num_obs)
        vec_dte_upper_boot = vec_dte + quantile_max_tstats * np.sqrt(omega / num_obs)
        return vec_dte_lower_boot, vec_dte_upper_boot
    elif variance_type == "simple":
        w_target = num_obs / num_target
        w_control = num_obs / num_control
        vec_dte_var = w_target * (
            vec_prediction_target * (1 - vec_prediction_target)
        ) + w_control * vec_prediction_control * (1 - vec_prediction_control)

        vec_dte_lower_simple = vec_dte + norm.ppf(alpha / 2) / np.sqrt(
            num_obs
        ) * np.sqrt(vec_dte_var)
        vec_dte_upper_simple = vec_dte + norm.ppf(1 - alpha / 2) / np.sqrt(
            num_obs
        ) * np.sqrt(vec_dte_var)

        return vec_dte_lower_simple, vec_dte_upper_simple
    else:
        raise RuntimeError(f"Invalid variance type was speficied: {variance_type}")
