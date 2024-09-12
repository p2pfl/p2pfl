# 🎉 Welcome to p2pfl contributing guide

## 🤝 How to contribute

### 🐞 Reporting bugs

If you find a bug, please open an issue in the [issue tracker](https://github.com/pguijas/p2pfl/issues/new). Please include as much information as possible, including the version of p2pfl you are using, the operating system, and any relevant stack traces or error messages.

### 💡 Suggesting enhancements

If you have an idea for a new feature, please open an issue in the [issue tracker](https://github.com/pguijas/p2pfl/issues/new). Please include as much information as possible, including a clear and descriptive title, a description of the problem you're trying to solve, and a description of the feature you'd like to see.

### 💻 Contributing code

If you'd like to contribute code, please open a pull request. Please include as much information as possible, including a clear and descriptive title, a description of the problem you're trying to solve, and a description of the feature you'd like to see.

For more information, see "[Creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)" in the GitHub Help documentation.

#### 🖋 Code quality expectations

In order to maintain a high level of code quality, we expect all contributions to meet the following standards:

- All code should be formatted with [Ruff](https://docs.astral.sh/ruff/).
- All code should pass [mypy](https://github.com/python/mypy) type checking.
- All code should be accompanied by tests that pass (we use [Pytest](https://docs.pytest.org/) and [Pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)).
- All code should be accompanied by documentation (we use [Sphinx](https://www.sphinx-doc.org/en/master/)).

Before submitting a pull request, please run the following commands:

```bash
poetry run ruff format p2pfl
poetry run ruff check p2pfl
poetry run mypy -p p2pfl
poetry run pytest -v --cov=p2pfl
```

Please see more details about design patterns and code style in the [documentation](https://pguijas.github.io/federated_learning_p2p/).

## 📜 License

By contributing to p2pfl, you agree that your contributions will be licensed under its [GNU General Public License, Version 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## 🤝 Code of Conduct

p2pfl has adopted the [Contributer Covenant](https://www.contributor-covenant.org/) as its Code of Conduct. All community members are expected to adhere to it. Please see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.
