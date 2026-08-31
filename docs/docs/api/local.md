# Local Distribution Estimators

This page documents local distribution treatment effect estimators that compute treatment effects weighted by treatment propensity within each stratum. These estimators are particularly useful for handling treatment assignment heterogeneity across strata and scenarios with imperfect compliance.

Local distribution treatment effects (LDTE) and local probability treatment effects (LPTE) provide methods for causal inference that account for treatment assignment vs. treatment receipt differences. For theoretical foundations on imperfect compliance scenarios, see:

- Byambadalai, U., Hirata, T., Oka, T., & Yasui, S. (2025). *Beyond the Average: Distributional Causal Inference under Imperfect Compliance*. In Advances in Neural Information Processing Systems 38 (NeurIPS'25). [arXiv:2509.15594](https://arxiv.org/abs/2509.15594).

## SimpleLocalDistributionEstimator

::: dte_adj.SimpleLocalDistributionEstimator

## AdjustedLocalDistributionEstimator

::: dte_adj.AdjustedLocalDistributionEstimator
