Simple Randomization Estimators
===============================

This page documents estimators that work with simple randomized experimental designs where treatment assignment is completely randomized.

These estimators leverage pre-treatment covariates through distributional regression frameworks to improve the precision of distributional treatment effect estimates. The key methodological contribution is using machine learning techniques for variance reduction while maintaining validity as long as nuisance components are reasonably well estimated.

Byambadalai et al. (2024) [#simple2024]_ propose a regression adjustment method that incorporates covariates into distributional regression, enabling deeper insights beyond average treatment effects by estimating full distributional treatment effects in randomized experiments.

.. [#simple2024] Byambadalai, U., Oka, T., & Yasui, S. (2024). Estimating Distributional Treatment Effects in Randomized Experiments: Machine Learning for Variance Reduction. arXiv preprint `arXiv:2407.16037 <https://arxiv.org/abs/2407.16037>`_.

SimpleDistributionEstimator
---------------------------

.. autoclass:: dte_adj.SimpleDistributionEstimator
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :no-index:

AdjustedDistributionEstimator
-----------------------------

.. autoclass:: dte_adj.AdjustedDistributionEstimator
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :no-index:
