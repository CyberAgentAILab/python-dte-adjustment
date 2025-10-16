API Reference
=============

This section provides comprehensive documentation for all classes and functions in the dte_adj package. The API is organized into logical groups based on functionality and use cases.

Overview
--------

The dte_adj package provides several types of estimators for computing distribution treatment effects:

* **Simple Randomization Estimators**: For estimating distributional effects in simple randomized experiments where treatment assignment is independent of all covariates
* **Covariate Adaptive Randomization Estimators**: For estimating distributional effects under covariate-adaptive randomization (CAR) designs, including stratified block randomization and other adaptive schemes
* **Local Distribution Estimators**: For estimating local distribution treatment effects weighted by treatment propensity within strata
* **Utility Functions**: Helper functions for confidence intervals and statistical computations
* **Plotting Utilities**: Visualization tools for treatment effects and distributions

For theoretical foundations, see Byambadalai et al. (2024) [#simple2024]_ for simple randomization, Byambadalai et al. (2025) [#car2025]_ for covariate-adaptive randomization, and Byambadalai et al. (2024) [#compliance2025]_ for imperfect compliance scenarios.

For multi-task learning approaches that train models for all locations simultaneously (using ``is_multi_task=True``), see the neural network framework in [#multitask2025]_.

.. [#simple2024] Byambadalai, U., Oka, T., & Yasui, S. (2024). Estimating Distributional Treatment Effects in Randomized Experiments: Machine Learning for Variance Reduction. In Proceedings of the 41st International Conference on Machine Learning (ICML'24). `arXiv:2407.16037 <https://arxiv.org/abs/2407.16037>`_.

.. [#car2025] Byambadalai, U., Hirata, T., Oka, T., & Yasui, S. (2025). On Efficient Estimation of Distributional Treatment Effects under Covariate-Adaptive Randomization. In Proceedings of the 42nd International Conference on Machine Learning (ICML'25). `arXiv:2506.05945 <https://arxiv.org/abs/2506.05945>`_.

.. [#multitask2025] Hirata, T., Byambadalai, U., Oka, T., Yasui, S., & Uto, S. (2025). Efficient and Scalable Estimation of Distributional Treatment Effects with Multi-Task Neural Networks. arXiv preprint `arXiv:2507.07738 <https://arxiv.org/abs/2507.07738>`_.

.. [#compliance2025] Byambadalai, U., Hirata, T., Oka, T., & Yasui, S. (2025). Beyond the Average: Distributional Causal Inference under Imperfect Compliance. arXiv preprint `arXiv:2509.15594 <https://arxiv.org/abs/2509.15594>`_.

Detailed Documentation
----------------------

.. toctree::
   :maxdepth: 2

   api/simple
   api/stratified
   api/local
   api/plot
