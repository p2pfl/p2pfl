# 👫 Contributing

## 🤝 How to contribute

We welcome contributions to p2pfl! Whether you're reporting bugs, suggesting enhancements, or contributing code, your involvement is valuable.

### 🐞 Reporting bugs

If you find a bug, please open an issue in the [issue tracker](https://github.com/pguijas/p2pfl/issues/new). Please include as much information as possible, including the version of p2pfl you are using, the operating system, and any relevant stack traces or error messages.

### 💡 Suggesting enhancements

If you have an idea for a new feature or improvement, please open an issue in the [issue tracker](https://github.com/pguijas/p2pfl/issues/new). Clearly describe the problem you're trying to solve and the proposed solution. Consider including use cases and potential benefits.

### 💻 Contributing code

We encourage code contributions!  To contribute code, please follow these steps:

1. **Fork the repository:** Create a fork of the p2pfl repository on GitHub.
2. **Create a branch:** Create a new branch for your changes.  Use a descriptive branch name (e.g., `fix-bug-123` or `feature-new-aggregator`).
3. **Make your changes:** Implement your changes, following the code style guidelines below.
4. **Test your changes:** Thoroughly test your changes to ensure they work as expected and don't introduce new issues.
5. **Commit your changes:** Commit your changes with clear and concise commit messages.
6. **Push your branch:** Push your branch to your forked repository.
7. **Open a pull request:** Open a pull request against the `develop` branch of the p2pfl repository.  Provide a detailed description of your changes, including the problem you're solving and how your changes address it.  Reference any related issues.

For more information on creating pull requests, see the GitHub documentation: [Creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)


## 🖋 Code quality expectations

To maintain a high level of code quality, we require all contributions to adhere to the following standards:

* **Formatting:** Use [Ruff](https://docs.astral.sh/ruff/) for code formatting. Run `uv run ruff format p2pfl` before submitting a pull request.  Also, check for any remaining issues with `uv run ruff check p2pfl`.

* **Type Hinting:**  Use type hints and ensure your code passes [mypy](https://github.com/python/mypy) type checking. Run `uv run mypy -p p2pfl` before submitting.

* **Testing:** Provide comprehensive tests for all new code using [Pytest](https://docs.pytest.org/) and ensure test coverage with [Pytest-cov](https://pytest-cov.readthedocs.io/en/latest/). Run `uv run pytest -v --cov=p2pfl` before submitting.

* **Documentation:**  Document your code using [Sphinx](https://www.sphinx-doc.org/en/master/).  Pay particular attention to documenting the module you are contributing to.  Refer to the [main components documentation](https://p2pfl.github.io/p2pfl/components.html) for examples and guidance.

* **Design Principles:** Adhere to SOLID principles, KISS (Keep It Simple, Stupid), DRY (Don't Repeat Yourself), and YAGNI (You Ain't Gonna Need It).

* **Design Patterns:** Utilize appropriate design patterns for modularity and flexibility.  We commonly use:
    * **Template Pattern:** For consistent interfaces (e.g., `CommunicationProtocol`, `Learner`).
    * **Command Pattern:** For defining executable commands over the `CommunicationProtocol`.
    * **Strategy Pattern:** For interchangeable algorithms and workflows (e.g., `Aggregator`, `Workflow`).
    * For more detailed information on the design patterns used, please refer to the [main components documentation](https://p2pfl.github.io/p2pfl/components.html), with emphasis on the module you are going to contribute to.

Before submitting a pull request, ensure all of the following commands run successfully:

```bash
uv run ruff format p2pfl
uv run ruff check p2pfl
uv run mypy -p p2pfl
uv run pytest -v --cov=p2pfl
```

## 📜 License

By contributing to p2pfl, you agree that your contributions will be licensed under its [GNU General Public License, Version 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## 🤝 Code of Conduct

p2pfl has adopted the [Contributer Covenant](https://www.contributor-covenant.org/) as its Code of Conduct. All community members are expected to adhere to it. Please see [CODE_OF_CONDUCT.md](/blob/main/CODE_OF_CONDUCT.md) for details.
