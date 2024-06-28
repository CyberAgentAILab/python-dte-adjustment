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
        outcomes: np.ndarray,
        alpha: float = 0.05,
        variance_type="moment",
        n_bootstrap=500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute DTE based on the estimator for the distribution function.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.
            variance_type (str, optional): Variance type to be used to compute confidence intervals. Available values are moment, analytic, and uniform.
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
            outcomes,
            alpha,
            variance_type,
            n_bootstrap,
        )

    def predict_pte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        width: float,
        outcomes: np.ndarray,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute PTE based on the estimator for the distribution function.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            width (float): The width of each outcome interval.
            alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected PTEs
                - Upper bands
                - Lower bands
        """
        return self._compute_ptes(
            target_treatment_arm, control_treatment_arm, outcomes, width, alpha
        )

    def predict_qte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        quantiles: np.ndarray = np.array(
            [0.1 * i for i in range(1, 10)], dtype=np.float32
        ),
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute QTE based on the estimator for the distribution function.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            quantiles (np.ndarray, optional): Quantiles used for QTE. Defaults to [0.1 * i for i in range(1, 10)].
            alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected QTEs
                - Upper bands
                - Lower bands
        """
        return self._compute_expected_qtes(
            target_treatment_arm, control_treatment_arm, quantiles, alpha
        )

    def _compute_dtes(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        outcomes: np.ndarray,
        alpha: float,
        variance_type: str,
        n_bootstrap: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected DTEs."""
        treatment_cdf, treatment_cdf_mat = self._compute_cumulative_distribution(
            np.full(outcomes.shape, target_treatment_arm), outcomes
        )
        control_cdf, control_cdf_mat = self._compute_cumulative_distribution(
            np.full(outcomes.shape, control_treatment_arm), outcomes
        )

        dte = treatment_cdf - control_cdf

        lower_band, upper_band = compute_dte_confidence_intervals(
            vec_y=self.outcome,
            vec_d=self.treatment_arm,
            vec_loc=outcomes,
            vec_cdf_target=treatment_cdf,
            vec_cdf_control=control_cdf,
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
        outcomes: np.ndarray,
        width: float,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected PTEs."""
        treatment_cumulative_pre = self.predict(
            np.full(outcomes.shape, target_treatment_arm), outcomes
        )
        treatment_cumulative_post = self.predict(
            np.full(outcomes.shape, target_treatment_arm), outcomes + width
        )
        treatment_pdf = treatment_cumulative_post - treatment_cumulative_pre
        control_cumulative_pre = self.predict(
            np.full(outcomes.shape, control_treatment_arm), outcomes
        )
        control_cumulative_post = self.predict(
            np.full(outcomes.shape, control_treatment_arm), outcomes + width
        )
        control_pdf = control_cumulative_post - control_cumulative_pre

        pte = treatment_pdf - control_pdf

        lower_band, upper_band = compute_pte_confidence_intervals(
            vec_d=self.treatment_arm,
            vec_pdf_target=treatment_pdf,
            vec_pdf_control=control_pdf,
            ind_target=target_treatment_arm,
            ind_control=control_treatment_arm,
            alpha=alpha,
        )

        return (
            pte,
            lower_band,
            upper_band,
        )

    def _compute_expected_qtes(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        quantiles: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected QTEs."""
        result = np.zeros(quantiles.shape)
        treatment_cumulative = self.predict(
            np.full(self.outcome.shape, target_treatment_arm), self.outcome
        )
        control_cumulative = self.predict(
            np.full(self.outcome.shape, control_treatment_arm), self.outcome
        )
        for i, q in enumerate(quantiles):
            treatment_quantile = treatment_cumulative[
                math.floor(treatment_cumulative.shape[0] * q)
            ]
            control_quantile = control_cumulative[
                math.floor(control_cumulative.shape[0] * q)
            ]
            result[i] = treatment_quantile - control_quantile

        # TODO: compute the right upperband and lowerband of QTE
        return result, result, result

    def _compute_bernoulli_upper_bands(
        self,
        estimates: np.ndarray,
        alpha: float,
        target_treatment_arm: int,
        control_treatment_arm: int,
    ) -> np.ndarray:
        """Compute upper confidence bands."""
        return estimates + np.sqrt(estimates * (1 - estimates)) * norm.ppf(alpha / 2)

    def _compute_bernoulli_lower_bands(
        self,
        estimates: np.ndarray,
        alpha: float,
        target_treatment_arm: int,
        control_treatment_arm: int,
    ) -> np.ndarray:
        """Compute lower confidence bands."""
        return estimates - np.sqrt(estimates * (1 - estimates)) * norm.ppf(alpha / 2)

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
        self, treatment_arm: np.ndarray, outcomes: np.ndarray
    ) -> np.ndarray:
        """Compute the cumulative distribution values."""
        raise NotImplementedError()

    def _get_num_data(self, treatment_arm: int) -> int:
        """Get the number of records with the specified treatment arm."""
        return ((self.treatment_arm == treatment_arm) * 1).sum()


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

    def predict(self, treatment_arm: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution values.

        Args:
            treatment_arm (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.

        Returns:
            np.ndarray: Estimated cumulative distribution values for the input.
        """
        if self.outcome is None:
            raise RuntimeError(
                "This estimator has not been trained yet. Please call fit first"
            )

        return self._compute_cumulative_distribution(treatment_arm, outcomes)[0]

    def _compute_cumulative_distribution(
        self, treatment_arm: np.ndarray, outcomes: np.ndarray
    ) -> np.ndarray:
        """Compute the cumulative distribution values.

        Args:
            treatment_arm (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.

        Returns:
            np.ndarray: Estimated cumulative distribution values.
        """
        unique_treatment_arm = np.unique(treatment_arm)
        d_confounding = {}
        d_outcome = {}
        n_obs = self.outcome.shape[0]
        n_loc = outcomes.shape[0]
        for arm in unique_treatment_arm:
            selected_confounding = self.confounding[self.treatment_arm == arm]
            selected_outcome = self.outcome[self.treatment_arm == arm]
            sorted_indices = np.argsort(selected_outcome)
            d_confounding[arm] = selected_confounding[sorted_indices]
            d_outcome[arm] = selected_outcome[sorted_indices]
        cumulative_distribution = np.zeros(outcomes.shape)
        for i, (outcome, arm) in enumerate(zip(outcomes, treatment_arm)):
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
            outcomes (np.ndarray): The number of folds for cross-fitting.

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

    def predict(self, treatment_arm: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution values.

        Args:
            treatment_arm (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.

        Returns:
            np.ndarray: Estimated cumulative distribution values for the input.
        """
        if self.outcome is None:
            raise RuntimeError(
                "This estimator has not been trained yet. Please call fit first"
            )

        return self._compute_cumulative_distribution(treatment_arm, outcomes)[0]

    def _compute_cumulative_distribution(
        self, treatment_arm: np.ndarray, outcomes: np.ndarray
    ) -> np.ndarray:
        """Compute the cumulative distribution values.

        Args:
            treatment_arm (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.

        Returns:
            np.ndarray: Estimated cumulative distribution values.
        """
        n_obs = self.outcome.shape[0]
        n_loc = outcomes.shape[0]
        cumulative_distribution = np.zeros(outcomes.shape)
        superset_prediction = np.zeros((n_obs, n_loc))
        for i, (outcome, arm) in enumerate(zip(outcomes, treatment_arm)):
            confounding_in_arm = self.confounding[self.treatment_arm == arm]
            outcome_in_arm = self.outcome[self.treatment_arm == arm]
            subset_prediction = np.zeros(outcome_in_arm.shape[0])
            binominal = (outcome_in_arm <= outcome) * 1
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
                    self.confounding[superset_mask]
                )[:, 1]
            cumulative_distribution[i] = (
                cdf - subset_prediction.mean() + superset_prediction[:, i].mean()
            )
        return cumulative_distribution, superset_prediction


def compute_dte_confidence_intervals(
    vec_y: np.ndarray,
    vec_d: np.ndarray,
    vec_loc: np.ndarray,
    vec_cdf_target: np.ndarray,
    vec_cdf_control: np.ndarray,
    mat_entire_predictions_target: np.ndarray,
    mat_entire_predictions_control: np.ndarray,
    ind_target: int,
    ind_control: int,
    alpha: 0.05,
    variance_type="moment",
    n_bootstrap=500,
):
    """Computes the confidence intervals of DTE.

    Args:
        vec_y (np.ndarray): Outcome variable vector.
        vec_d (np.ndarray): Treatment indicator vector.
        vec_loc (np.ndarray): Locations where the DTE is estimated.
        vec_cdf_target (np.ndarray): Estimated Cumulative Distributional Function of treatment group.
        vec_cdf_control (np.ndarray): Estimated Cumulative Distributional Function of control group.
        mat_entire_predictions_target (np.ndarray): Prediction of the conditional distribution estimator for target group.
        mat_entire_predictions_control (np.ndarray): Prediction of the conditional distribution estimator for control group.
        ind_target (int): Index of the target treatment indicator.
        ind_control (int): Index of the control treatment indicator.
        alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.
        variance_type (str, optional): Variance type to be used to compute confidence intervals. Available values are moment, analytic, and uniform.
        n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: lower band.
            - np.ndarray: upper band.
    """
    num_obs = vec_y.shape[0]
    mat_y_u = (vec_y[:, np.newaxis] <= vec_loc).astype(int)
    n_loc = vec_loc.shape[0]
    mat_d = np.tile(vec_d, (n_loc, 1)).T
    vec_dte = vec_cdf_target - vec_cdf_control
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
    elif variance_type == "analytic":
        w_target = num_obs / num_target
        w_control = num_obs / num_control
        vec_dte_var = w_target * (
            vec_cdf_target * (1 - vec_cdf_target)
        ) + w_control * vec_cdf_control * (1 - vec_cdf_control)

        vec_dte_lower_analytic = vec_dte + norm.ppf(alpha / 2) / np.sqrt(
            num_obs
        ) * np.sqrt(vec_dte_var)
        vec_dte_upper_analytic = vec_dte + norm.ppf(1 - alpha / 2) / np.sqrt(
            num_obs
        ) * np.sqrt(vec_dte_var)

        return vec_dte_lower_analytic, vec_dte_upper_analytic
    else:
        raise RuntimeError(f"Invalid variance type was speficied: {variance_type}")


def compute_pte_confidence_intervals(
    vec_d: np.ndarray,
    vec_pdf_target: np.ndarray,
    vec_pdf_control: np.ndarray,
    ind_target: int,
    ind_control: int,
    alpha: 0.05,
):
    """
    Compute the confidence interval of PTE.

    Args:
        vec_d (np.ndarray): Treatment indicator vector.
        vec_pdf_target (np.ndarray): Estimated Probability Density Function of treatment group.
        vec_pdf_control (np.ndarray): Estimated Probability Density Function of control group.
        ind_target (int): Index of the target treatment indicator.
        ind_control (int): Index of the control treatment indicator.
        alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: lower band.
            - np.ndarray: upper band.
    """
    num_obs = vec_d.shape[0]
    num_target = (vec_d == ind_target).sum()
    num_control = (vec_d == ind_control).sum()
    w_target = num_obs / num_target
    w_control = num_obs / num_control
    vec_pdf_var_target = vec_pdf_target * (1 - vec_pdf_target)
    vec_pdf_var_control = vec_pdf_control * (1 - vec_pdf_control)
    vec_pte = vec_pdf_target - vec_pdf_control

    vec_pte_var = (w_target * vec_pdf_var_target) + (w_control * vec_pdf_var_control)

    vec_pte_lower_analytic = vec_pte + norm.ppf(alpha / 2) / np.sqrt(num_obs) * np.sqrt(
        vec_pte_var
    )
    vec_pte_upper_analytic = vec_pte + norm.ppf(1 - alpha / 2) / np.sqrt(
        num_obs
    ) * np.sqrt(vec_pte_var)

    return vec_pte_lower_analytic, vec_pte_upper_analytic
