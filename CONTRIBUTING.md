
# Contribution
Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## How can I contribute to this project?
### Reporting Bugs
If you find a bug, please report it by opening an issue in the issue tracker. Provide as much detail as possible to help us understand and reproduce the issue:
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
Here are the basic commands you can use to develop this package.

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