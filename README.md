## Overview

`dte_adj` is a Python package for estimating distribution treatment effects. It provides APIs for conducting regression adjustment to estimate precise distribution functions as well as convenient utils. For the details of this package, see [the documentation](https://cyberagentailab.github.io/python-dte-adjustment/).

## Installation

1. **Install from PyPI**
    ```sh
    pip install dte_adj
    ```

2. **Install from source**

    ```sh
    git clone https://github.com/CyberAgentAILab/python-dte-adjustment
    cd python-dte-adjustment
    pip install -e .
    ```

## Basic Usage
Examples of how to use this package are available in [this Get-started Guide](https://cyberagentailab.github.io/python-dte-adjustment/get_started.html).

## Theoretical Foundations

This package implements methods from the following research papers:

### Simple Randomization
- **Byambadalai, U., Oka, T., & Yasui, S.** (2024). *Estimating Distributional Treatment Effects in Randomized Experiments: Machine Learning for Variance Reduction*. [arXiv:2407.16037](https://arxiv.org/abs/2407.16037)

### Covariate-Adaptive Randomization
- **Byambadalai, U., Hirata, T., Oka, T., & Yasui, S.** (2025). *On Efficient Estimation of Distributional Treatment Effects under Covariate-Adaptive Randomization*. [arXiv:2506.05945](https://arxiv.org/abs/2506.05945)

### Multi-Task Learning
- **Hirata, T., Byambadalai, U., Oka, T., Yasui, S., & Uto, S.** (2025). *Efficient and Scalable Estimation of Distributional Treatment Effects with Multi-Task Neural Networks*. [arXiv:2507.07738](https://arxiv.org/abs/2507.07738)

## Citation

If you use this software in your research, please cite our work:

```bibtex
@article{byambadalai2024estimating,
  title={Estimating Distributional Treatment Effects in Randomized Experiments: Machine Learning for Variance Reduction},
  author={Byambadalai, Undral and Oka, Tatsushi and Yasui, Shota},
  journal={arXiv preprint arXiv:2407.16037},
  year={2024}
}
```

For other citation formats, see our [CITATION.cff](CITATION.cff) file.

## Development
We welcome contributions to the project! Please review our [Contribution Guide](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Maintainers
- [Tomu Hirata](https://github.com/TomeHirata)
