import numpy as np
from typing import Tuple
from scipy.stats import norm
import math


class DistributionFunctionMixin(object):
    """A mixin including several convenience functions to compute and display distribution functions."""

    def __init__(self):
        """Initializes the DistributionFunctionMixin.

        Returns:
            DistributionFunctionMixin: An instance of the estimator.
        """
        self.confounding = {}
        self.outcome = {}

    def predict_dte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        outcomes: np.ndarray,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute DTE based on the estimator for the distribution function.

        Args:
            target_treatment_arm (int): The index of the treatment arm of the target group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            alpha (float, optional): Significance level of the confidence band. Defaults to 0.05.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected DTEs
                - Upper bands
                - Lower bands
        """
        return self._compute_dtes(
            target_treatment_arm, control_treatment_arm, outcomes, alpha
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
            target_treatment_arm (int): The index of the treatment arm of the target group.
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
            target_treatment_arm (int): The index of the treatment arm of the target group.
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

    def _compute_outcomes_from_training_data(
        self, target_treatment_arm: int
    ) -> np.ndarray:
        """Compute outcomes from training data."""
        return self.outcome[target_treatment_arm]

    def _compute_dtes(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        outcomes: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected DTEs."""
        target_cumulative = self.predict(
            np.full(outcomes.shape, target_treatment_arm), outcomes
        )
        control_cumulative = self.predict(
            np.full(outcomes.shape, control_treatment_arm), outcomes
        )

        n_target = self._get_num_data(target_treatment_arm)
        n_control = self._get_num_data(control_treatment_arm)

        target_variance = target_cumulative * (1 - target_cumulative)
        control_variance = control_cumulative * (1 - control_cumulative)

        deviation = np.sqrt(
            (target_variance * n_target + control_variance * n_control)
            / (n_target * n_control)
        )

        dte = target_cumulative - control_cumulative

        return (
            dte,
            dte - deviation * norm.ppf(alpha / 2),
            dte + deviation * norm.ppf(alpha / 2),
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
        target_cumulative_pre = self.predict(
            np.full(outcomes.shape, target_treatment_arm), outcomes
        )
        target_cumulative_post = self.predict(
            np.full(outcomes.shape, target_treatment_arm), outcomes + width
        )
        target_likelihood = target_cumulative_post - target_cumulative_pre
        control_cumulative_pre = self.predict(
            np.full(outcomes.shape, control_treatment_arm), outcomes
        )
        control_cumulative_post = self.predict(
            np.full(outcomes.shape, control_treatment_arm), outcomes + width
        )
        control_likelihood = control_cumulative_post - control_cumulative_pre

        n_target = self._get_num_data(target_treatment_arm)
        n_control = self._get_num_data(control_treatment_arm)

        target_variance = target_cumulative_pre * (
            1 - target_cumulative_pre
        ) + target_cumulative_post * (1 - target_cumulative_post)
        control_variance = control_cumulative_pre * (
            1 - control_cumulative_pre
        ) + control_cumulative_post * (1 - control_cumulative_post)

        deviation = np.sqrt(
            (target_variance * n_target + control_variance * n_control)
            / (n_target * n_control)
        )

        pte = target_likelihood - control_likelihood

        return (
            pte,
            pte - deviation * norm.ppf(alpha / 2),
            pte + deviation * norm.ppf(alpha / 2),
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
        for i, q in enumerate(quantiles):
            target_quantile = self.outcome[target_treatment_arm][
                math.floor(self.outcome[i].shape[0] * q)
            ]
            control_quantile = self.outcome[control_treatment_arm][
                math.floor(self.outcome[i].shape[0] * q)
            ]
            result[i] = target_quantile = control_quantile

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

    def _get_num_data(self, treatment_arm: int) -> int:
        """Get the number of records with the specified treatment arm."""
        return self.confounding[treatment_arm].shape[0]


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

        unique_treatment_arm = np.unique(treatment_arm)
        for arm in unique_treatment_arm:
            selected_confounding = confounding[treatment_arm == arm]
            selected_outcome = outcome[treatment_arm == arm]
            sorted_indices = np.argsort(selected_outcome)
            self.confounding[arm] = selected_confounding[sorted_indices]
            self.outcome[arm] = selected_outcome[sorted_indices]

        return self

    def predict(self, treatment_arm: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution values.

        Args:
            treatment_arm (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar values to be used for computing the cumulative distribution.

        Returns:
            np.ndarray: Estimated cumulative distribution values for the input.
        """
        if self.outcome == {}:
            raise RuntimeError(
                "This estimator has not been trained yet. Please call fit first"
            )

        return self._compute_cumulative_distribution(treatment_arm, outcomes)

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
        cumulative_distribution = np.zeros(outcomes.shape)
        for i, (outcome, arm) in enumerate(zip(outcomes, treatment_arm)):
            cumulative_distribution[i] = (
                find_le(self.outcome[arm], outcome) + 1
            ) / self.outcome[arm].shape[0]
        return cumulative_distribution
