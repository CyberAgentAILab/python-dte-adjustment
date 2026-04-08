---
title: 'dte_adj: A Python Package for Estimating Distributional Treatment Effects in Randomized Experiments'
tags:
  - Python
  - randomized experiments
  - causal inference
  - distributional treatment effects
  - machine learning
  - variance reduction
authors:
  - name: Tomu Hirata
    orcid: 0009-0006-3140-291X
    corresponding: true
    affiliation: "1, 3"
  - name: Undral Byambadalai
    affiliation: 1
  - name: Tatsushi Oka
    affiliation: "1, 2"
  - name: Shota Yasui
    affiliation: 1
affiliations:
 - name: CyberAgent, Inc., Japan
   index: 1
 - name: Keio University, Japan
   index: 2
 - name: Databricks, Inc., United States
   index: 3
date: 24 August 2025
bibliography: paper.bib
---

# Summary

`dte_adj` is a Python package for estimating distributional treatment effects (DTEs) in randomized experiments (RCTs, also known as A/B tests). Unlike traditional approaches that focus on average treatment effects, `dte_adj` enables researchers to analyze the full distributional impact of interventions across different outcome levels. The package implements machine learning-enhanced regression adjustment methods for variance reduction, supports multiple experimental designs including simple randomization, covariate-adaptive randomization, and settings with imperfect compliance, and provides a scikit-learn compatible API with comprehensive functionality for computing distribution functions, probability treatment effects, and quantile treatment effects with confidence intervals.

# Statement of Need

Randomized experiments have been fundamental to scientific inquiry since @fisher1935design, providing the gold standard for causal inference. While most experimental analyses focus on average treatment effects (ATEs), many research questions require understanding how treatments affect the entire distribution of outcomes. Distributional treatment effects (DTEs) capture these richer patterns, revealing heterogeneous impacts across different outcome levels that averages can mask. For example, a policy intervention might have no effect on average income while substantially reducing poverty rates at lower quantiles, or a medical treatment might benefit patients at the tails of the distribution differently than those near the median.

Despite the growing importance of distributional analysis in economics, medicine, and technology, the Python ecosystem lacks comprehensive tools for DTE estimation with modern variance reduction techniques. Researchers often resort to basic empirical CDFs or manual implementations that lack statistical rigor. `dte_adj` fills this gap by providing a unified framework for distributional treatment effect analysis that integrates state-of-the-art machine learning methods for improved precision, rigorous confidence interval construction, and support for complex experimental designs.

# State of the Field

Several Python packages address causal inference, but none focus on distributional treatment effects with machine learning-based variance reduction:

- **SciPy** [@2020SciPy-NMeth]: Provides basic empirical cumulative distribution functions but offers no functionality for treatment effect estimation or confidence interval construction in experimental settings.
- **DoWhy** [@dowhy]: Focuses on causal graph-based inference and average treatment effects, without distributional analysis capabilities.
- **EconML** [@econml]: Incorporates machine learning for heterogeneous treatment effect estimation (CATE) but does not address distributional effects.
- **causal-curve** [@kobrosly2020causalcurve]: Estimates dose-response curves but targets continuous treatments rather than distributional outcomes.

In the R ecosystem, packages like `qte` provide quantile treatment effect estimation but lack machine learning integration for variance reduction. `dte_adj` uniquely combines: (1) distributional treatment effect estimation across the full outcome distribution, (2) machine learning-enhanced regression adjustment for precision gains, and (3) support for multiple experimental designs including covariate-adaptive randomization and imperfect compliance settings.

# Software Design

`dte_adj` follows a class-based architecture with a template method pattern, where a base class defines the algorithm structure and subclasses implement design-specific computations:

- **`SimpleDistributionEstimator`** and **`AdjustedDistributionEstimator`**: For simple randomized experiments, implementing methods from @byambadalai2024estimatingdistributionaltreatmenteffects.
- **`SimpleStratifiedDistributionEstimator`** and **`AdjustedStratifiedDistributionEstimator`**: For covariate-adaptive randomization designs, implementing methods from @byambadalai2025efficientestimationdistributionaltreatment.
- **`SimpleLocalDistributionEstimator`** and **`AdjustedLocalDistributionEstimator`**: For settings with imperfect compliance, implementing methods from @byambadalai2025imperfectcompliance.

All estimators implement a consistent API with three primary methods: `predict_dte()` for distributional treatment effects, `predict_pte()` for probability treatment effects over intervals, and `predict_qte()` for quantile treatment effects. The adjusted estimators use K-fold cross-fitting to prevent overfitting and support both single-task and multi-task learning modes [@hirata2025efficientscalableestimationdistributional] for computational efficiency. Bootstrap methods provide confidence intervals with multiple variance estimation approaches.

![Distributional treatment effects for the Hillstrom email marketing dataset [@hillstrom2008], comparing Women's vs Men's email campaigns. The simple estimator (left, purple) and ML-adjusted estimator (right, green) show that adjustment substantially tightens confidence bands, demonstrating the variance reduction benefit of regression adjustment.](hillstrom_dte.png)

![Local distributional treatment effects for emergency department costs in the Oregon Health Insurance Experiment [@finkelstein2012], estimated using `SimpleLocalDistributionEstimator` (left) and `AdjustedLocalDistributionEstimator` (right). Health insurance coverage shifts the distribution of ED costs, with ML adjustment again yielding narrower confidence intervals.](oregon_ldte_costs_comparison.png)

# Research Impact Statement

The methods implemented in `dte_adj` have been published at top machine learning venues: ICML 2024 [@byambadalai2024estimatingdistributionaltreatmenteffects] and ICML 2025 [@byambadalai2025efficientestimationdistributionaltreatment]. The package has been used internally at CyberAgent, Inc. for analyzing A/B tests where distributional impacts are critical, such as evaluating interventions on user engagement metrics where tail behavior matters more than averages. The documentation includes tutorials demonstrating applications to the Hillstrom email marketing dataset (Figure 1) and the Oregon Health Insurance Experiment (Figure 2), facilitating adoption by researchers in economics, marketing, and healthcare.

# AI Usage Disclosure

Generative AI tools (Claude) were used to assist with documentation writing and code review during development. All AI-generated content was reviewed and validated by the human authors.

# Acknowledgements

We thank CyberAgent, Inc. for supporting this research and the open-source community for valuable feedback during development.

# References
