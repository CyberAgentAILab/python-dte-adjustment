import numpy as np
from scipy.stats import norm
from typing import Tuple


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
