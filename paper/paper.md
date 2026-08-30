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
    affiliation: 3
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

`dte_adj` is a Python package for analyzing how an intervention, such as a marketing campaign, a medical treatment, or a policy change, affects the entire range of an outcome, not just its average. Practitioners running randomized experiments (RCTs, also known as A/B tests) can use it to answer questions such as "did the treatment help the users who were struggling the most?" or "did it move the whole distribution, or only shift the top end?", together with rigorous confidence bands around those answers.

More formally, `dte_adj` estimates distributional treatment effects (DTEs) in randomized experiments. Unlike traditional approaches that focus on average treatment effects, `dte_adj` enables researchers to analyze the full distributional impact of interventions across different outcome levels. The package implements machine learning-enhanced regression adjustment methods for variance reduction, supports multiple experimental designs including simple randomization, covariate-adaptive randomization, and settings with imperfect compliance, and provides a scikit-learn-like API [@scikit-learn] with comprehensive functionality for computing distribution functions, probability treatment effects, and quantile treatment effects with confidence intervals.

# Statement of Need

Randomized experiments have been fundamental to scientific inquiry since @fisher1935design, providing the gold standard for causal inference. While most experimental analyses focus on average treatment effects (ATEs), many research questions require understanding how treatments affect the entire distribution of outcomes. Distributional treatment effects (DTEs) capture these richer patterns, revealing heterogeneous impacts across different outcome levels that averages can mask. For example, a policy intervention might have no effect on average income while substantially reducing poverty rates at lower quantiles, or a medical treatment might benefit patients at the tails of the distribution differently than those near the median.

Despite the growing importance of distributional analysis in economics, medicine, and technology, the Python ecosystem lacks comprehensive tools for DTE estimation with modern variance reduction techniques. Applied researchers in economics, biostatistics, and the social sciences, along with data scientists and experimentation engineers running A/B tests in industry, are often left resorting to basic empirical CDFs or manual implementations that do not provide the inferential guarantees implemented here. `dte_adj` fills this gap for these users with a unified framework that integrates state-of-the-art machine learning methods for improved precision, rigorous confidence interval construction, and support for complex experimental designs. It complements general causal inference libraries such as `DoWhy` [@dowhy] and `EconML` [@econml], which target average or conditional average treatment effects, and the R package `qte` [@qte], which supports quantile treatment effects but without machine learning-based variance reduction.

# State of the Field

Several Python packages address causal inference, but none focus on distributional treatment effects with machine learning-based variance reduction:

- **SciPy** [@2020SciPy-NMeth]: Provides basic empirical cumulative distribution functions but offers no functionality for treatment effect estimation or confidence interval construction in experimental settings.
- **DoWhy** [@dowhy]: Focuses on causal graph-based inference and average treatment effects, without distributional analysis capabilities.
- **EconML** [@econml]: Incorporates machine learning for heterogeneous treatment effect estimation (CATE) but does not address distributional effects.
- **causal-curve** [@kobrosly2020causalcurve]: Estimates dose-response curves but targets continuous treatments rather than distributional outcomes.

In the R ecosystem, packages like `qte` [@qte] provide quantile treatment effect estimation but lack machine learning integration for variance reduction. `dte_adj` uniquely combines: (1) distributional treatment effect estimation across the full outcome distribution, (2) machine learning-enhanced regression adjustment for precision gains, and (3) support for multiple experimental designs including covariate-adaptive randomization and imperfect compliance settings.

A standalone package is warranted because existing libraries are organized around scalar estimands (ATEs in `DoWhy`, heterogeneous CATEs in `EconML`), whereas distributional estimation requires a distinct set of primitives: distribution functions evaluated over grids of locations, interval probabilities, quantile inversion, pointwise and uniform confidence bands, and cross-fitted distributional regression for variance reduction. These do not map cleanly onto point-estimate abstractions, and a focused package also allows the implementation to track a rapidly evolving methodological literature on covariate-adaptive randomization, imperfect compliance, and multi-task learning for DTEs.

# Software Design

`dte_adj` follows a class-based architecture with a template method pattern, where a base class defines the algorithm structure and subclasses implement design-specific computations:

- **`SimpleDistributionEstimator`** and **`AdjustedDistributionEstimator`**: For simple randomized experiments, implementing methods from @byambadalai2024estimatingdistributionaltreatmenteffects.
- **`SimpleStratifiedDistributionEstimator`** and **`AdjustedStratifiedDistributionEstimator`**: For covariate-adaptive randomization designs, implementing methods from @byambadalai2025efficientestimationdistributionaltreatment.
- **`SimpleLocalDistributionEstimator`** and **`AdjustedLocalDistributionEstimator`**: For settings with imperfect compliance, implementing methods from @byambadalai2025imperfectcompliance.

The stratified and non-stratified estimators share a common API with three methods: `predict_dte()` for distributional treatment effects, `predict_pte()` for probability treatment effects over intervals, and `predict_qte()` for quantile treatment effects. The local estimators expose `predict_ldte()` and `predict_lpte()` instead, for the imperfect-compliance setting. The adjusted estimators use K-fold cross-fitting to prevent overfitting and support both single-task and multi-task learning modes [@hirata2025efficientscalableestimationdistributional] for computational efficiency. Bootstrap methods provide confidence intervals with multiple variance estimation approaches.

The template method pattern is a natural fit here because every estimator shares the same outer algorithm (evaluate a distribution function on a grid, difference across treatment arms, and construct confidence bands) and differs only in how the conditional distribution is estimated, which depends on the experimental design and on whether a plug-in or a cross-fitted machine learning estimator is used. Defining the outer algorithm once in the base class keeps its statistical invariants in one place, whereas a strategy-based configuration would push that structure into runtime flags and obscure them. For the same reason, the estimators are exposed as distinct classes rather than a single configurable one: many flag combinations would be confusing (e.g., stratification without strata) or correspond to different estimands with different identifying assumptions, and separate classes make the required inputs explicit at construction time and let each estimator evolve independently.

![Distributional treatment effects for the Hillstrom email marketing dataset [@hillstrom2008], comparing Women's vs Men's email campaigns. The simple estimator (left, purple) and ML-adjusted estimator (right, green) show that adjustment substantially tightens confidence bands, demonstrating the variance reduction benefit of regression adjustment.](hillstorm_dte.png)

![Local distributional treatment effects for emergency department costs in the Oregon Health Insurance Experiment [@finkelstein2012], estimated using `SimpleLocalDistributionEstimator` (left) and `AdjustedLocalDistributionEstimator` (right). Health insurance coverage shifts the distribution of ED costs, with ML adjustment again yielding narrower confidence intervals.](oregon_ldte_costs_comparison.png)

# Research Impact Statement

The methods implemented in `dte_adj` have been published across machine learning and econometrics venues: ICML 2024 [@byambadalai2024estimatingdistributionaltreatmenteffects], Econometric Reviews [@oka2025regression], ICML 2025 [@byambadalai2025efficientestimationdistributionaltreatment], and NeurIPS 2025 [@byambadalai2025imperfectcompliance]. The package has been applied in industry settings, including analyzing the distributional impact of content promotion on user engagement at ABEMA, a major video streaming platform [@yasui2026abema]. The documentation includes tutorials demonstrating applications to the Hillstrom email marketing dataset [@hillstrom2008] (Figure 1) and the Oregon Health Insurance Experiment [@finkelstein2012] (Figure 2), facilitating adoption by researchers in economics, marketing, and healthcare. End-to-end code for reproducing both figures is available in the online tutorials at <https://cyberagentailab.github.io/python-dte-adjustment/tutorials/>.

# AI Usage Disclosure

Generative AI tools (Claude) were used to assist with documentation writing and code review during development, and to copy-edit this paper; the initial draft of the paper was written by the human authors without generative AI. All AI-generated content was reviewed and validated by the human authors.

# Acknowledgements

We thank CyberAgent, Inc. for supporting this research and the open-source community for valuable feedback during development.

# References
