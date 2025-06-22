import numpy as np
from typing import Tuple, Optional, Any
from scipy.stats import norm
from copy import deepcopy
from abc import ABC
from .util import compute_confidence_intervals

__all__ = [
    "SimpleDistributionEstimator",
    "AdjustedDistributionEstimator",
    "SimpleStratifiedDistributionEstimator",
    "AdjustedStratifiedDistributionEstimator",
]


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
        Compute DTE based on the estimator for the distribution function.

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
        n_bootstrap=500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute PTE based on the estimator for the distribution function.

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
        Compute QTE based on the estimator for the distribution function.

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
            self.covariates,
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
                self.covariates[bootstrap_indexes],
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
        n_bootstrap: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected PTEs."""
        treatment_cumulative_pre, treatment_cdf_mat_pre, _ = (
            self._compute_cumulative_distribution(
                target_treatment_arm,
                locations,
                self.covariates,
                self.treatment_arms,
                self.outcomes,
            )
        )
        treatment_cumulative_post, treatment_cdf_mat_post, _ = (
            self._compute_cumulative_distribution(
                target_treatment_arm,
                locations + width,
                self.covariates,
                self.treatment_arms,
                self.outcomes,
            )
        )
        treatment_pdf = treatment_cumulative_post - treatment_cumulative_pre
        control_cumulative_pre, control_cdf_mat_pre, _ = (
            self._compute_cumulative_distribution(
                control_treatment_arm,
                locations,
                self.covariates,
                self.treatment_arms,
                self.outcomes,
            )
        )
        control_cumulative_post, control_cdf_mat_post, _ = (
            self._compute_cumulative_distribution(
                control_treatment_arm,
                locations + width,
                self.covariates,
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute expected QTEs."""
        locations = np.sort(outcomes)

        def find_quantile(quantile, arm):
            low, high = 0, locations.shape[0] - 1
            result = -1
            while low <= high:
                mid = (low + high) // 2
                val, _, _ = self._compute_cumulative_distribution(
                    arm,
                    np.full((1), locations[mid]),
                    covariates,
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


class SimpleStratifiedDistributionEstimator(DistributionEstimatorBase):
    """A class is for estimating the empirical distribution function and computing the Distributional parameters for CAR."""

    def fit(
        self,
        covariates: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.ndarray,
        strata: np.ndarray,
    ) -> "DistributionEstimatorBase":
        """
        Train the DistributionEstimatorBase.

        Args:
            covariates (np.ndarray): Pre-treatment covariates.
            treatment_arms (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar-valued observed outcome.

        Returns:
            DistributionEstimatorBase: The fitted estimator.
        """
        if covariates.shape[0] != treatment_arms.shape[0]:
            raise ValueError("The shape of covariates and treatment_arm should be same")

        if covariates.shape[0] != outcomes.shape[0]:
            raise ValueError("The shape of covariates and outcome should be same")

        self.covariates = covariates
        self.treatment_arms = treatment_arms
        self.outcomes = outcomes
        self.strata = strata

        return self

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
            treatment_arm (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data

        Returns:
            Tuple of numpy arrays:
                - np.ndarray: Unconditional cumulative distribution values.
                - np.ndarray: Adjusted cumulative distribution for each observation.
                - np.ndarray: Conditional cumulative distribution for each observation.
        """
        n_records = outcomes.shape[0]
        n_loc = locations.shape[0]
        prediction = np.zeros((n_records, n_loc))
        treatment_mask = treatment_arms == target_treatment_arm

        strata = self.strata
        s_list = np.unique(strata)
        w_s = {}
        for s in s_list:
            s_mask = strata == s
            w_s[s] = (s_mask & treatment_mask).sum() / s_mask.sum()
        n_obs = outcomes.shape[0]
        n_loc = locations.shape[0]
        for i, outcome in enumerate(locations):
            for j in range(n_obs):
                s = strata[j]
                prediction[j, i] = (outcomes[j] <= outcome) / w_s[s] * treatment_mask[j]

        unconditional_pred = {s: prediction[s == strata].mean(axis=0) for s in s_list}
        conditional_prediction = np.array([unconditional_pred[s] for s in strata])
        weights = np.array([w_s[s] for s in strata])[:, np.newaxis]
        prediction = (
            (outcomes[:, np.newaxis] <= locations) - conditional_prediction
        ) / weights * treatment_mask[:, np.newaxis] + conditional_prediction

        return prediction.mean(axis=0), prediction, conditional_prediction

    def _compute_interval_probability(
        self,
        target_treatment_arm: int,
        locations: np.ndarray,
        covariates: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the interval probabilities.

        Args:
            target_treatment_arm (int): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the interval probabilities.
            covariates: (np.ndarray): An array of covariates variables in the observed data.
            treatment_arm (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data

        Returns:
            Tuple of numpy arrays:
                - np.ndarray: Estimated unconditional interval probabilities.
                - np.ndarray: Adjusted for each observation.
                - np.ndarray: Conditional for each observation.
        """
        n_records = outcomes.shape[0]
        n_loc = locations.shape[0]
        prediction = np.zeros((n_records, n_loc))
        treatment_mask = treatment_arms == target_treatment_arm

        strata = self.strata
        s_list = np.unique(strata)
        w_s = {}
        for s in s_list:
            s_mask = strata == s
            w_s[s] = (s_mask & treatment_mask).sum() / s_mask.sum()
        n_obs = outcomes.shape[0]
        n_loc = locations.shape[0]
        for i, outcome in enumerate(locations):
            for j in range(n_obs):
                s = strata[j]
                prediction[j, i] = (outcomes[j] <= outcome) / w_s[s] * treatment_mask[j]

        unconditional_pred = {s: prediction[s == strata].mean(axis=0) for s in s_list}
        conditional_prediction = np.array([unconditional_pred[s] for s in strata])
        weights = np.array([w_s[s] for s in strata])[:, np.newaxis]
        prediction = (
            (outcomes[:, np.newaxis] <= locations) - conditional_prediction
        ) / weights * treatment_mask[:, np.newaxis] + conditional_prediction

        cdf = prediction.mean(axis=0)
        return (
            cdf[1:] - cdf[:-1],
            prediction[:, 1:] - prediction[:, :-1],
            conditional_prediction[:, 1:] - conditional_prediction[:, :-1],
        )


class AdjustedStratifiedDistributionEstimator(DistributionEstimatorBase):
    """A class is for estimating the adjusted distribution function and computing the Distributional parameters for CAR."""

    def __init__(self, base_model: Any, folds=3, is_multi_task=False):
        """
        Initializes the AdjustedDistributionEstimator.

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

    def fit(
        self,
        covariates: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.ndarray,
        strata: np.ndarray,
    ) -> "DistributionEstimatorBase":
        """
        Train the DistributionEstimatorBase.

        Args:
            covariates (np.ndarray): Pre-treatment covariates.
            treatment_arms (np.ndarray): The index of the treatment arm.
            outcomes (np.ndarray): Scalar-valued observed outcome.

        Returns:
            DistributionEstimatorBase: The fitted estimator.
        """
        if covariates.shape[0] != treatment_arms.shape[0]:
            raise ValueError("The shape of covariates and treatment_arm should be same")

        if covariates.shape[0] != outcomes.shape[0]:
            raise ValueError("The shape of covariates and outcome should be same")

        self.covariates = covariates
        self.treatment_arms = treatment_arms
        self.outcomes = outcomes
        self.strata = strata

        return self

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
            treatment_arm (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data

        Returns:
            Tuple of numpy arrays:
                - np.ndarray: Unconditional cumulative distribution values.
                - np.ndarray: Adjusted cumulative distribution for each observation.
                - np.ndarray: Conditional cumulative distribution for each observation.
        """
        n_records = outcomes.shape[0]
        n_loc = locations.shape[0]
        superset_prediction = np.zeros((n_records, n_loc))
        prediction = np.zeros((n_records, n_loc))
        treatment_mask = treatment_arms == target_treatment_arm
        folds = np.random.randint(self.folds, size=n_records)
        strata = self.strata
        s_list = np.unique(strata)
        if self.is_multi_task:
            binomial = (outcomes.reshape(-1, 1) <= locations) * 1  # (n_records, n_loc)
            for fold in range(self.folds):
                fold_mask = (folds != fold) & treatment_mask
                for s in s_list:
                    s_mask = strata == s
                    weight = (s_mask & treatment_mask).sum() / s_mask.sum()
                    superset_mask = (folds == fold) & s_mask
                    subset_train_mask = (folds != fold) & s_mask & treatment_mask
                    covariates_train = covariates[subset_train_mask]
                    binomial_train = binomial[subset_train_mask]
                    if len(np.unique(binomial_train)) > 1:
                        self.model = deepcopy(self.base_model)
                        self.model.fit(covariates_train, binomial_train)

                    pred = self._compute_model_prediction(
                        self.model, covariates[superset_mask]
                    )
                    prediction[superset_mask] = (
                        pred
                        + treatment_mask[superset_mask].reshape(-1, 1)
                        * (binomial[superset_mask] - pred)
                        / weight
                    )
                    superset_prediction[superset_mask] = pred
        else:
            for i, location in enumerate(locations):
                binomial = (outcomes <= location) * 1  # (n_records)
                for fold in range(self.folds):
                    fold_mask = (folds != fold) & treatment_mask
                    covariates_train = covariates[fold_mask]
                    binomial_train = binomial[fold_mask]
                    # Pool the records across strata and train the model
                    if len(np.unique(binomial_train)) > 1:
                        self.model = deepcopy(self.base_model)
                        self.model.fit(covariates_train, binomial_train)
                    for s in s_list:
                        s_mask = strata == s
                        weight = (s_mask & treatment_mask).sum() / s_mask.sum()
                        superset_mask = (folds == fold) & s_mask
                        subset_train_mask = (folds != fold) & s_mask & treatment_mask
                        covariates_train = covariates[subset_train_mask]
                        binomial_train = binomial[subset_train_mask]
                        # TODO: revisit the logic here
                        if len(np.unique(binomial_train)) > 1:
                            # self.model = deepcopy(self.base_model)
                            # self.model.fit(covariates_train, binomial_train)
                            pass
                        else:
                            pred = binomial_train[0]
                            superset_prediction[superset_mask, i] = pred
                            prediction[superset_mask, i] = (
                                pred
                                + treatment_mask[superset_mask]
                                * (binomial[superset_mask] - pred)
                                / weight
                            )
                            continue
                        pred = self._compute_model_prediction(
                            self.model, covariates[superset_mask]
                        )
                        prediction[superset_mask, i] = (
                            pred
                            + treatment_mask[superset_mask]
                            * (binomial[superset_mask] - pred)
                            / weight
                        )
                        superset_prediction[superset_mask, i] = pred

        return prediction.mean(axis=0), prediction, superset_prediction

    def _compute_interval_probability(
        self,
        target_treatment_arm: int,
        locations: np.ndarray,
        covariates: np.ndarray,
        treatment_arms: np.ndarray,
        outcomes: np.array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the interval probabilities.

        Args:
            target_treatment_arm (int): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            covariates: (np.ndarray): An array of covariates variables in the observed data.
            treatment_arm (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data

        Returns:
            Tuple of numpy arrays:
                - np.ndarray: Unconditional interval probabilities.
                - np.ndarray: Adjusted interval probabilities for each observation.
                - np.ndarray: Conditional interval probabilities for each observation.
        """
        n_records = outcomes.shape[0]
        n_loc = locations.shape[0]
        superset_prediction = np.zeros((n_records, n_loc - 1))
        prediction = np.zeros((n_records, n_loc - 1))
        treatment_mask = treatment_arms == target_treatment_arm
        folds = np.random.randint(self.folds, size=n_records)
        strata = self.strata
        s_list = np.unique(strata)
        binominals = (outcomes[:, np.newaxis] <= locations) * 1  # (n_records, n_loc)
        for i in range(len(locations) - 1):
            binomial = binominals[:, i + 1] - binominals[:, i]
            for fold in range(self.folds):
                fold_mask = (folds != fold) & treatment_mask
                covariates_train = covariates[fold_mask]
                binomial_train = binomial[fold_mask]
                if len(np.unique(binomial_train)) > 1:
                    self.model = deepcopy(self.base_model)
                    self.model.fit(covariates_train, binomial_train)
                for s in s_list:
                    s_mask = strata == s
                    wight = (s_mask & treatment_mask).sum() / s_mask.sum()
                    superset_mask = (folds == fold) & s_mask
                    subset_train_mask = (folds != fold) & s_mask & treatment_mask
                    covariates_train = covariates[subset_train_mask]
                    binomial_train = binomial[subset_train_mask]
                    if len(np.unique(binomial_train)) == 1:
                        pred = binomial_train[0]
                        superset_prediction[superset_mask, i] = pred
                        prediction[superset_mask, i] = (
                            pred
                            + treatment_mask[superset_mask]
                            * (binomial[superset_mask] - pred)
                            / wight
                        )
                        continue
                    pred = self._compute_model_prediction(
                        self.model, covariates[superset_mask]
                    )
                    prediction[superset_mask, i] = (
                        pred
                        + treatment_mask[superset_mask]
                        * (binomial[superset_mask] - pred)
                        / wight
                    )
                    superset_prediction[superset_mask, i] = pred

        return prediction.mean(axis=0), prediction, superset_prediction

    def _compute_model_prediction(self, model, covariates: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            if self.is_multi_task:
                # suppose the shape of prediction is (n_records, n_locations)
                return model.predict_proba(covariates)
            probabilities = model.predict_proba(covariates)
            if probabilities.ndim == 1:
                # when the shape of prediction is (n_records)
                return probabilities
            # when the shape of prediction is (n_records, 2)
            return probabilities[:, 1]
        else:
            return model.predict(covariates)


class SimpleDistributionEstimator(SimpleStratifiedDistributionEstimator):
    """
    A class for computing the empirical distribution function and the distributional parameters
    based on the distribution function.
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
    """A class is for estimating the adjusted distribution function and computing the Distributional parameters based on the trained conditional estimator."""

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
