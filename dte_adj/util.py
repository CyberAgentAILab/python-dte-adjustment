from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from dte_adj.local import (
        SimpleStratifiedDistributionEstimator,
        AdjustedLocalDistributionEstimator,
    )

ArrayLike = Union[
    np.ndarray,
    list,
    tuple,
    "pd.DataFrame",
    "pd.Series",
    "pl.DataFrame",
    "pl.Series",
]

def _convert_to_ndarray(data: ArrayLike) -> np.ndarray:
    """Convert array-like data to np.ndarray if needed."""
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "to_numpy"):
        return data.to_numpy()
    return np.asarray(data)


def _infer_default_locations(
    outcomes: np.ndarray,
    for_intervals: bool = False,
) -> np.ndarray:
    """Generate evenly-spaced default locations from observed outcomes.

    The number of points is determined from data size and distribution using
    ``np.histogram_bin_edges(outcomes, bins='auto')`` (which combines the
    Sturges and Freedman-Diaconis rules).

    Args:
        outcomes (np.ndarray): Observed outcomes used to determine the range.
        for_intervals (bool, optional): If True, the left endpoint is placed
            slightly below ``outcomes.min()`` so that observations equal to the
            minimum fall inside the first interval ``(loc[0], loc[1]]``. Set
            this for PTE/LPTE estimation. Defaults to False.

    Returns:
        np.ndarray: Evenly-spaced locations array.
    """
    n_locations = len(np.histogram_bin_edges(outcomes, bins="auto"))

    y_min = float(outcomes.min())
    y_max = float(outcomes.max())

    if for_intervals:
        # Place the left endpoint strictly below y_min so that the smallest
        # observation falls inside the first interval (loc[0], loc[1]]. The
        # offset scales with the magnitude of the data so that ``y_min - eps``
        # is representable even when the outcome range is zero.
        scale = max(y_max - y_min, abs(y_min), abs(y_max), 1.0)
        eps = scale * 1e-9
        return np.linspace(y_min - eps, y_max, n_locations)
    return np.linspace(y_min, y_max, n_locations)


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the confidence intervals of distribution parameters.

    Args:
        vec_y (np.ndarray): Outcome variable vector.
        vec_d (np.ndarray): Treatment indicator vector.
        vec_loc (np.ndarray): Locations where the distribution parameters are estimated.
        mat_y_u (np.ndarray): Indicator function for 1{Y⩽y}. Shape is n_obs * n_loc.
        vec_prediction_target (np.ndarray): Unconditional estimated distributional effects for the treatment group.
        vec_prediction_control (np.ndarray): Unconditional estimated distributional effects for the control group.
        mat_entire_predictions_target (np.ndarray): Conditional stimated distributional effects for each observation.
        mat_entire_predictions_control (np.ndarray): Conditional stimated distributional effects for each observation.
        ind_target (int): Index of the target treatment indicator.
        ind_control (int): Index of the control treatment indicator.
        alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
        variance_type (str, optional): Variance type to be used to compute confidence intervals. Available values are moment, simple, and uniform.
        n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: lower bound.
            - np.ndarray: upper bound.
    """
    num_obs = vec_y.shape[0]
    vec_dte = vec_prediction_target - vec_prediction_control

    num_target = (vec_d == ind_target).sum()
    num_control = (vec_d == ind_control).sum()

    influence_function = (
        mat_entire_predictions_target - mat_entire_predictions_target.mean(axis=0)
    ) - (mat_entire_predictions_control - mat_entire_predictions_control.mean(axis=0))

    omega = (influence_function**2).mean(axis=0)

    if variance_type == "moment":
        vec_dte_lower_moment = vec_dte + norm.ppf(alpha / 2) * np.sqrt(omega / num_obs)
        vec_dte_upper_moment = vec_dte + norm.ppf(1 - alpha / 2) * np.sqrt(
            omega / num_obs
        )
        return vec_dte_lower_moment, vec_dte_upper_moment
    elif variance_type in ["uniform", "multiplier"]:
        tstats = np.zeros((n_bootstrap, len(vec_loc)))
        boot_draw = np.zeros((n_bootstrap, len(vec_loc)))

        for b in range(n_bootstrap):
            eta1 = np.random.normal(0, 1, num_obs)
            eta2 = np.random.normal(0, 1, num_obs)
            xi = eta1 / np.sqrt(2) + (eta2**2 - 1) / 2

            boot_draw[b, :] = (
                1 / num_obs * np.sum(xi[:, np.newaxis] * influence_function, axis=0)
            )

        if variance_type == "uniform":
            tstats = np.abs(boot_draw)[:, :-1] / np.sqrt(omega[:-1] / num_obs)
            max_tstats = np.max(tstats, axis=1)
            quantile_max_tstats = np.quantile(max_tstats, 1 - alpha)

            se = (
                np.quantile(boot_draw, 0.75, axis=0)
                - np.quantile(boot_draw, 0.25, axis=0)
            ) / (norm.ppf(0.75) - norm.ppf(0.25))

            vec_dte_lower_boot = vec_dte - quantile_max_tstats * se
            vec_dte_upper_boot = vec_dte + quantile_max_tstats * se
            return vec_dte_lower_boot, vec_dte_upper_boot
        else:
            se = np.std(boot_draw, axis=0)

            vec_dte_lower_boot = vec_dte + se * norm.ppf(alpha / 2)
            vec_dte_upper_boot = vec_dte + se * norm.ppf(1 - alpha / 2)
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
        raise ValueError(f"Invalid variance type was specified: {variance_type}")


def _compute_local_treatment_effects_core(
    estimator: "SimpleStratifiedDistributionEstimator | AdjustedLocalDistributionEstimator",
    target_treatment_arm: int,
    control_treatment_arm: int,
    locations: np.ndarray,
    alpha: float,
    use_intervals: bool = False,
    display_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core computation logic shared between LDTE and LPTE.

    Args:
        estimator: The fitted estimator instance with required attributes
        target_treatment_arm (int): The index of the treatment arm of the treatment group.
        control_treatment_arm (int): The index of the treatment arm of the control group.
        locations (np.ndarray): Scalar values to be used for computing the distribution.
        alpha (float): Significance level of the confidence bound.
        use_intervals (bool): If True, compute interval probabilities (LPTE), else cumulative (LDTE).
        display_progress (bool): Whether to display a progress bar.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - Expected effects (beta)
            - Lower bounds
            - Upper bounds
    """
    X = estimator.covariates
    Z = estimator.treatment_arms
    D = estimator.treatment_indicator
    S = estimator.strata
    Y = estimator.outcomes
    s_list = np.unique(S)

    # Compute weights
    weights = {
        s: np.sum((S == s) & (Z == target_treatment_arm)) / np.sum(S == s)
        for s in s_list
    }

    # Compute treatment propensity (probability of treatment)
    d_t_prediction, d_t_psi, d_t_eta = estimator._compute_cumulative_distribution(
        target_treatment_arm, np.zeros(1), X, Z, 1 - (target_treatment_arm == D)
    )
    d_c_prediction, d_c_psi, d_c_eta = estimator._compute_cumulative_distribution(
        control_treatment_arm, np.zeros(1), X, Z, 1 - (target_treatment_arm == D)
    )

    # Compute outcome distributions (different for LDTE vs LPTE)
    if use_intervals:
        y_t_prediction, y_t_psi, y_t_mu = estimator._compute_interval_probability(
            target_treatment_arm, locations, X, Z, Y, display_progress=display_progress
        )
        y_c_prediction, y_c_psi, y_c_mu = estimator._compute_interval_probability(
            control_treatment_arm, locations, X, Z, Y, display_progress=display_progress
        )
        output_size = len(locations) - 1
    else:
        y_t_prediction, y_t_psi, y_t_mu = estimator._compute_cumulative_distribution(
            target_treatment_arm, locations, X, Z, Y, display_progress=display_progress
        )
        y_c_prediction, y_c_psi, y_c_mu = estimator._compute_cumulative_distribution(
            control_treatment_arm, locations, X, Z, Y, display_progress=display_progress
        )
        output_size = len(locations)

    psi_b = d_t_psi - d_c_psi
    beta = (y_t_prediction - y_c_prediction) / (d_t_prediction - d_c_prediction)

    # Compute influence functions
    xi_t = np.zeros((len(X), output_size))
    xi_c = np.zeros((len(X), output_size))

    for i in range(len(X)):
        w_s = weights[S[i]]

        # Compute outcome indicators (different for LDTE vs LPTE)
        if use_intervals:
            bi = (Y[i] <= locations) * 1
            bi = bi[1:] - bi[:-1]  # Convert to interval probabilities
        else:
            bi = (Y[i] <= locations) * 1

        xi_t[i] = ((1 - 1 / w_s) * y_t_mu[i] - y_c_mu[i] + bi / w_s) - beta * (
            (1 - 1 / w_s) * d_t_eta[i] - d_c_eta[i] + D[i] / w_s
        )

        xi_c[i] = (
            (1 / (1 - w_s) - 1) * y_c_mu[i] - y_t_mu[i] + bi / (1 - w_s)
        ) - beta * ((1 / (1 - w_s) - 1) * d_c_eta[i] - d_t_eta[i] + D[i] / (1 - w_s))

    # Center the influence functions
    t_xi_mean = {
        s: xi_t[(S == s) & (Z == target_treatment_arm)].mean(axis=0) for s in s_list
    }
    c_xi_mean = {
        s: xi_c[(S == s) & (Z == control_treatment_arm)].mean(axis=0) for s in s_list
    }

    for i in range(len(X)):
        xi_t[i] -= t_xi_mean[S[i]]
        xi_c[i] -= c_xi_mean[S[i]]

    # Compute xi function (different for LDTE vs LPTE)
    def xi(s):
        if use_intervals:
            a = (
                Y[(S == s) & (Z == target_treatment_arm)].reshape(-1, 1)
                < locations.reshape(1, -1)
            ) * 1
            a = a[:, 1:] - a[:, :-1]  # Convert to intervals
            b = (
                Y[(S == s) & (Z == control_treatment_arm)].reshape(-1, 1)
                < locations.reshape(1, -1)
            ) * 1
            b = b[:, 1:] - b[:, :-1]  # Convert to intervals
        else:
            a = Y[(S == s) & (Z == target_treatment_arm)].reshape(
                -1, 1
            ) < locations.reshape(1, -1)
            b = Y[(S == s) & (Z == control_treatment_arm)].reshape(
                -1, 1
            ) < locations.reshape(1, -1)

        return (
            a
            - beta.reshape(1, -1)
            * D[(S == s) & (Z == target_treatment_arm)].reshape(-1, 1)
        ).mean(axis=0) - (
            b
            - beta.reshape(1, -1)
            * D[(S == s) & (Z == control_treatment_arm)].reshape(-1, 1)
        ).mean(axis=0)

    xi_2_dict = {s: xi(s) for s in s_list}
    xi_2 = np.array([xi_2_dict[s] for s in S])
    sigma = (
        Z.reshape(-1, 1) * xi_t**2 + (1 - Z).reshape(-1, 1) * xi_c**2 + xi_2**2
    ).mean(axis=0) / (psi_b.mean()) ** 2

    # Compute confidence intervals
    z_alpha = norm.ppf(1 - alpha / 2)
    se = sigma**0.5 / np.sqrt(len(X))
    upper_bound = beta + z_alpha * se
    lower_bound = beta - z_alpha * se

    return beta, lower_bound, upper_bound


def compute_ldte(
    estimator: "SimpleStratifiedDistributionEstimator | AdjustedLocalDistributionEstimator",
    target_treatment_arm: int,
    control_treatment_arm: int,
    locations: np.ndarray,
    alpha: float = 0.05,
    display_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Local Distribution Treatment Effects (LDTE) using the provided formula.

    Args:
        estimator: The fitted estimator instance with required attributes
        target_treatment_arm (int): The index of the treatment arm of the treatment group.
        control_treatment_arm (int): The index of the treatment arm of the control group.
        locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
        alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
        display_progress (bool, optional): Whether to display a progress bar. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - Expected LDTEs (beta)
            - Lower bounds
            - Upper bounds
    """
    return _compute_local_treatment_effects_core(
        estimator,
        target_treatment_arm,
        control_treatment_arm,
        locations,
        alpha,
        use_intervals=False,
        display_progress=display_progress,
    )


def compute_lpte(
    estimator: "SimpleStratifiedDistributionEstimator | AdjustedLocalDistributionEstimator",
    target_treatment_arm: int,
    control_treatment_arm: int,
    locations: np.ndarray,
    alpha: float = 0.05,
    display_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Local Probability Treatment Effects (LPTE) using the provided formula.

    Args:
        estimator: The fitted estimator instance with required attributes
        target_treatment_arm (int): The index of the treatment arm of the treatment group.
        control_treatment_arm (int): The index of the treatment arm of the control group.
        locations (np.ndarray): Scalar values to be used for computing the interval probabilities.
        alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
        display_progress (bool, optional): Whether to display a progress bar. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - Expected LPTEs (beta)
            - Lower bounds
            - Upper bounds
    """
    return _compute_local_treatment_effects_core(
        estimator,
        target_treatment_arm,
        control_treatment_arm,
        locations,
        alpha,
        use_intervals=True,
        display_progress=display_progress,
    )
