
# Contribution
Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## How can I contribute to this project?
### Reporting Bugs
If you find a bug, please report it by opening an issue in the Github issue tracker. Provide as much detail as possible to help us understand and reproduce the issue:
- A clear and descriptive title.
- A detailed description of the problem.
- Steps to reproduce the issue.
- Any error messages or screenshots.

### Suggesting Enhancements
We welcome suggestions for improvements! To suggest an enhancement:
- Check the issue tracker to see if someone else has already suggested it.
- If not, open a new issue and describe your idea clearly.
- Explain why you believe the enhancement would be beneficial.

### Pull Requests
Pull requests are welcome! If you plan to make significant changes, please open an issue first to discuss your idea. This helps us ensure that your contribution fits with the project's direction. Follow these steps for a smooth pull request process:

- Fork the repository.
- Clone your fork to your local machine.
- Create a new branch: `git checkout -b my-feature-branch`.
- Make your changes.
- Commit your changes: `git commit -m 'Add some feature'.
- Push to the branch: `git push origin my-feature-branch`.
- Open a pull request in the original repository.

## Development
Important commands and major dependencies are managed with uv. Here are the basic commands you can use to develop this package.

### Install Dependencies
Install all development dependencies using uv:

```sh
uv sync --dev
```

### Pre-commit Hooks
We use pre-commit hooks to automatically format code and check for issues. Install them with:

```sh
uv run pre-commit install
```

### Manual Commands

#### Linting
We use `ruff` for linting the code. To run the linter manually:
```sh
uv run ruff check
```

#### Auto format
We use `ruff` for formatting the code. To run the formatter manually:
```sh
uv run ruff format
```

#### Unit test
We use `unittest` for testing the code. To run the unit tests:
```sh
uv run python -m unittest
```
