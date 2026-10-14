# Changelog

## 0.1.11 (unreleased)

This release incorporates the fixes, inference options, and documentation updates
from the [JOSS review](https://github.com/openjournals/joss-reviews/issues/11047).

### Added

- Multiplier-bootstrap confidence intervals and uniform confidence bands for
  LDTE and LPTE through `variance_type` and `n_bootstrap`. The default remains
  analytic moment inference (#151).
- Statistical baseline tests and regression coverage for input shapes,
  quantile inversion, local bootstrap inference, and small-sample cross-fitting
  (#150, #151, #152).

### Fixed

- Accept single-column outcome, treatment-arm, stratum, and treatment-indicator
  inputs by flattening them; reject unsupported shapes and missing values in
  these inputs with informative errors (#150).
- Compute QTEs using the generalized inverse of the CDF: the smallest outcome
  whose cumulative probability is at least the requested quantile (#150).
- Skip empty prediction folds, report missing training data in folds or strata,
  and use constant predictions for constant multi-task training labels (#152).
- Repair the Hillstrom and Oregon tutorial examples (#137, #139).

### Documentation and maintenance

- Migrate the documentation to MkDocs Material and convert API docstrings to
  Markdown (#130, #132).
- Clarify that regression adjustment improves precision in randomized
  experiments, and document the missing-value policy (#150).
- Update the JOSS paper, references, and author affiliation (#131, #136, #140,
  #142), and align software citation metadata with the paper for archiving.
- Update development dependencies and GitHub Actions.

### Compatibility notes

QTE results can change because of the quantile-inversion correction. Invalid
shapes and missing values in outcomes, treatment arms, strata, or treatment
indicators now raise errors. Missing covariates remain subject to the selected
model's support. Existing LDTE/LPTE calls retain moment inference by default.

[Changes since v0.1.10](https://github.com/CyberAgentAILab/python-dte-adjustment/compare/v0.1.10...v0.1.11)
