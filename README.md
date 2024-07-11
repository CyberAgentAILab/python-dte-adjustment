## Overview

This a Python package for building the regression adjusted distribution function estimator proposed in "Estimating Distributional Treatment Effects in Randomized Experiments: Machine Learning for Variance Reduction". For the details of this package, see [the documentation](https://cyberagentailab.github.io/python-dte-adjustment/).

## Installation

1. **Install from PyPI**
    ```sh
    pip install dte_adj
    ```

2. **Install from Source**

    ```sh
    git clone https://github.com/CyberAgentAILab/python-dte-adjustment
    cd python-dte-adjustment
    pip install -e .
    ```

## Development

### Install Pipenv
If you don't have `pipenv` installed, you can install it using `pip`:

```sh
pip install pipenv
```

### Linting
We use `ruff` for linting the code. To run the linter, use the following command:
```sh
pipenv run lint
```

### Auto format
We use `ruff` for formatting the code. To run the formatter, use the following command:
```sh
pipenv run format
```

### Unit test
We use `unittest` for testing the code. To run the unit tests, use the following command:
```sh
pipenv run unittest
```
