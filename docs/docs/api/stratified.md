# Covariate Adaptive Randomization Estimators

This page documents estimators that work with stratified experimental designs, particularly for covariate-adaptive randomization (CAR) within strata.

These estimators are designed to handle stratified block randomization where participants are grouped into strata based on baseline covariates before treatment assignment. The key methodological contribution is leveraging additional covariates beyond strata indicators using machine learning methods to enhance the precision of distributional treatment effect estimates.

Byambadalai et al. (2025)[^car2025] propose a flexible distribution regression framework that achieves the semiparametric efficiency bound for distributional treatment effects under CAR, demonstrating that regression-adjusted estimators can optimally utilize covariate information in stratified designs.

## SimpleStratifiedDistributionEstimator

::: dte_adj.SimpleStratifiedDistributionEstimator
    options:
      inherited_members: true

## AdjustedStratifiedDistributionEstimator

::: dte_adj.AdjustedStratifiedDistributionEstimator
    options:
      inherited_members: true

[^car2025]: Byambadalai, U., Hirata, T., Oka, T., & Yasui, S. (2025). On Efficient Estimation of Distributional Treatment Effects under Covariate-Adaptive Randomization. arXiv preprint [arXiv:2506.05945](https://arxiv.org/abs/2506.05945).
