from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Any
from copy import deepcopy
from scipy.stats import norm
from tqdm.auto import tqdm
from dte_adj.base import DistributionEstimatorBase
from dte_adj.util import ArrayLike, _convert_to_ndarray


class SimpleStratifiedDistributionEstimator(DistributionEstimatorBase):
    """A class is for estimating the empirical distribution function and computing the Distributional parameters for CAR."""

    def fit(
        self,
        covariates: ArrayLike,
        treatment_arms: ArrayLike,
        outcomes: ArrayLike,
        strata: ArrayLike,
    ) -> DistributionEstimatorBase:
        """
        Train the DistributionEstimatorBase.

        Args:
            covariates: Pre-treatment covariates.
            treatment_arms: The index of the treatment arm.
            outcomes: Scalar-valued observed outcome.
            strata: Stratum indicators.

        Returns:
            DistributionEstimatorBase: The fitted estimator.
        """
        covariates = _convert_to_ndarray(covariates)
        treatment_arms = _convert_to_ndarray(treatment_arms)
        outcomes = _convert_to_ndarray(outcomes)
        strata = _convert_to_ndarray(strata)

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
        display_progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the cumulative distribution values.

        Args:
            target_treatment_arm (int): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            covariates: (np.ndarray): An array of covariates variables in the observed data.
            treatment_arm (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data
            display_progress (bool): Whether to display a progress bar.

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
        for j in range(n_records):
            s = strata[j]
            prediction[j] = (outcomes[j] <= locations) / w_s[s] * treatment_mask[j]

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
        display_progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the interval probabilities.

        Args:
            target_treatment_arm (int): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the interval probabilities.
            covariates: (np.ndarray): An array of covariates variables in the observed data.
            treatment_arm (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data
            display_progress (bool): Whether to display a progress bar.

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
        for i, outcome in enumerate(locations):
            for j in range(n_records):
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

    def predict_qte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        quantiles: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap=500,
        display_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Quantile Treatment Effects (QTE) using stratified bootstrap.

        Uses stratified bootstrap (resampling independently within each stratum) to
        correctly estimate variance under covariate adaptive randomization (CAR).

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            quantiles (np.ndarray, optional): Quantiles used for QTE. Defaults to [0.1, 0.2, ..., 0.9].
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.
            display_progress (bool, optional): Whether to display a progress bar. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected QTEs (np.ndarray): Treatment effect estimates at each quantile
                - Lower bounds (np.ndarray): Lower confidence interval bounds
                - Upper bounds (np.ndarray): Upper confidence interval bounds
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

        # Precompute stratum indices for stratified bootstrap
        unique_strata = np.unique(self.strata)
        strata_indices = {s: np.where(self.strata == s)[0] for s in unique_strata}

        qtes = np.zeros((n_bootstrap, qte.shape[0]))
        bootstrap_iter = range(n_bootstrap)
        if display_progress:
            bootstrap_iter = tqdm(bootstrap_iter, desc="Bootstrap QTE")
        for b in bootstrap_iter:
            # Stratified bootstrap: resample within each stratum independently
            bootstrap_indexes = np.concatenate([
                np.random.choice(idx, size=len(idx), replace=True)
                for idx in strata_indices.values()
            ])

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
        covariates: ArrayLike,
        treatment_arms: ArrayLike,
        outcomes: ArrayLike,
        strata: ArrayLike,
    ) -> DistributionEstimatorBase:
        """
        Train the DistributionEstimatorBase.

        Args:
            covariates: Pre-treatment covariates.
            treatment_arms: The index of the treatment arm.
            outcomes: Scalar-valued observed outcome.
            strata: Stratum indicators.

        Returns:
            DistributionEstimatorBase: The fitted estimator.
        """
        covariates = _convert_to_ndarray(covariates)
        treatment_arms = _convert_to_ndarray(treatment_arms)
        outcomes = _convert_to_ndarray(outcomes)
        strata = _convert_to_ndarray(strata)

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
        display_progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the cumulative distribution values.

        Args:
            target_treatment_arm (int): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            covariates: (np.ndarray): An array of covariates variables in the observed data.
            treatment_arm (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data
            display_progress (bool): Whether to display a progress bar.

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
            fold_iter = range(self.folds)
            if display_progress:
                fold_iter = tqdm(fold_iter, desc="Cross-fitting (multi-task)")
            for fold in fold_iter:
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
            loc_iter = enumerate(locations)
            if display_progress:
                loc_iter = tqdm(loc_iter, total=len(locations), desc="Computing CDF")
            for i, location in loc_iter:
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
        display_progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the interval probabilities.

        Args:
            target_treatment_arm (int): The index of the treatment arm.
            locations (np.ndarray): Scalar values to be used for computing the cumulative distribution.
            covariates: (np.ndarray): An array of covariates variables in the observed data.
            treatment_arm (np.ndarray): An array of treatment arms in the observed data.
            outcomes (np.ndarray): An array of outcomes in the observed data
            display_progress (bool): Whether to display a progress bar.

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
        interval_iter = range(len(locations) - 1)
        if display_progress:
            interval_iter = tqdm(interval_iter, desc="Computing interval prob.")
        for i in interval_iter:
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
                    weight = (s_mask & treatment_mask).sum() / s_mask.sum()
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

    def predict_qte(
        self,
        target_treatment_arm: int,
        control_treatment_arm: int,
        quantiles: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap=500,
        display_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Quantile Treatment Effects (QTE) using stratified bootstrap.

        Uses stratified bootstrap (resampling independently within each stratum) to
        correctly estimate variance under covariate adaptive randomization (CAR).

        Args:
            target_treatment_arm (int): The index of the treatment arm of the treatment group.
            control_treatment_arm (int): The index of the treatment arm of the control group.
            quantiles (np.ndarray, optional): Quantiles used for QTE. Defaults to [0.1, 0.2, ..., 0.9].
            alpha (float, optional): Significance level of the confidence bound. Defaults to 0.05.
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 500.
            display_progress (bool, optional): Whether to display a progress bar. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Expected QTEs (np.ndarray): Treatment effect estimates at each quantile
                - Lower bounds (np.ndarray): Lower confidence interval bounds
                - Upper bounds (np.ndarray): Upper confidence interval bounds
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

        # Precompute stratum indices for stratified bootstrap
        unique_strata = np.unique(self.strata)
        strata_indices = {s: np.where(self.strata == s)[0] for s in unique_strata}

        qtes = np.zeros((n_bootstrap, qte.shape[0]))
        bootstrap_iter = range(n_bootstrap)
        if display_progress:
            bootstrap_iter = tqdm(bootstrap_iter, desc="Bootstrap QTE")
        for b in bootstrap_iter:
            # Stratified bootstrap: resample within each stratum independently
            bootstrap_indexes = np.concatenate([
                np.random.choice(idx, size=len(idx), replace=True)
                for idx in strata_indices.values()
            ])

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
