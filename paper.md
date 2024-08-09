---
title: 'dte_adj: A Python package for Distributional Treatment Effects'
tags:
  - Python
  - Distributional Treatment Effects
  - Variance Reduction
authors:
  - name: Tomu Hirata
    orcid: 0009-0006-3140-291X
    equal-contrib: true
    affiliation: "1, 3"
  - name: Undral Byambadalai 
    corresponding: true
    affiliation: 1
  - name: Tatsushi Oka
    corresponding: true
    affiliation: "1, 2"
  - name: Shota Yasui 
    corresponding: true
    affiliation: 1
affiliations:
 - name: Cyber Agent, Inc, Japan
   index: 1
 - name: Keio University, Japan
   index: 2
 - name: Indeed Technologies Japan, Japan
   index: 3
date: 9 August 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx 
aas-journal: International Conference on Machine Learning
---

# Summary

`dte_adj` is a Python package for computing empirical cumulative distribution function (CDF) and distributional treatment effect (DTE) from data obtained by Randomized control tests. This package also contains a novel method to reduce variance of DTE using pre-treatment covariates introduced in `@Undral:2024`.

# Statement of need

Since the groundbreaking work by `@Fisher:1935`, randomized experiments have been essential in understanding the impact of interventions and shaping policy decisions. A widely used metric in this context is the Average Treatment Effect (ATE). However, exploring the distributional treatment effects often offers a more nuanced understanding than focusing solely on the average effects.
Python is widely used in the research community recently with its flexibility and ease-of-use in the user-interface. However, there is no popular Python library for computing Distributional Treatment Effect from data obtained from randomized experiments. While scipy provides a method for computing the empirical cumulative distribution function, it lacks convenient functions for calculating DTE or for estimating the variance of the distribution.
`dte_adj` was developed to fill the gap by offering the functionalities for 1) computing CDF from data, 2) calculating DTE and its confidence band based on CDF and 3) visualizing DTE. This library uses `numpy` as input and output of methods, which is widely used for matrix computation in Python. The main classes of this library also follows the interface of popular library `scikit-learn`, which makes it easy for the users with Machine Learning development experieneces.

# Functionalities

The high level functionalities of `dte_adj` are as follows:
1. Computing CDF and its variance based on number arrays
2. Calculating distributional parameters and their confidence bands
3. Visualiving distributional parameters and the confidence bands

It currently offers two classes to compute CDF and its variance.
- `SimpleDistributionEstimator`: this class offers a standard way to compute empirical CDF
- `AdjustedDistributionEstimator`: this class offers a way to compute CDF with smaller variance adjusted by pre-treatment covariates introduced in `@Undral:2024`

Both classes implement following methods to calculate distributional parameters.
- `predict_dte`: method for computing Distributional Treatment Effect $DTE_{w, w'}(y) := F_{Y(w)}(y) - F_{Y(w')}(y)$, where $y$ is an outcome variable, $w$ is treatment type , and $F_{Y(w)}(y)$ is cumulative likelihood for treatment type $w$ and outcome $y$.
- `predict_pte`: method for computing Probability Treatment Effect (PTE) $PTE_{w, w'}(y, h) := \left( F_{Y(w)}(y+h) - F_{Y(w)}(y) \right) - \left( F_{Y(w')}(y+h) - F_{Y(w')}(y) \right)$, where $h > 0$ is an interval of each evaluation window.
- `predict_qte`: method for computing Quantile Treatment Effect (QTE) $QTE_{w, w'}(\tau) := F_{Y(w)}^{-1}(\tau) - F_{Y(w')}^{-1}(\tau)$, where $\tau$ is quantile.

Lastly, `dte_adj.plot` module can be used for visualiting the distribution parameters. The examples of the visualization are available in the figures below.

![DTE](docs/source/_static/dte_moment.png)
![PTE](docs/source/_static/pte_simple.png)
![QTE](docs/source/_static/qte.png)

# Acknowledgements

# References