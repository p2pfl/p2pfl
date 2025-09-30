![GitHub Logo](https://raw.githubusercontent.com/p2pfl/p2pfl/main/other/logo.png)

# P2PFL - Federated Learning over P2P networks

[![GitHub license](https://img.shields.io/github/license/p2pfl/p2pfl)](/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/p2pfl/p2pfl)](/issues)
![GitHub contributors](https://img.shields.io/github/contributors/p2pfl/p2pfl)
![GitHub forks](https://img.shields.io/github/forks/p2pfl/p2pfl)
![GitHub stars](https://img.shields.io/github/stars/p2pfl/p2pfl)
![GitHub activity](https://img.shields.io/github/commit-activity/m/p2pfl/p2pfl)
[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fp2pfl%2Fp2pfl%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?/blob/python-coverage-comment-action-data/htmlcov/index.html)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://join.slack.com/t/p2pfl/shared_invite/zt-38xyec0k7-tLPbNx873Pm~N4aoqIyjRw)

P2PFL is a general-purpose open-source library designed for the execution (simulated and in real environments) of Decentralized Federated Learning systems, specifically making use of P2P networks and the gossip protocols.

## ✨ Key Features

P2PFL offers a range of features designed to make decentralized federated learning accessible and efficient. For detailed information, please refer to our [documentation](https://p2pfl.github.io/p2pfl/).

| Feature          | Description                                      |
|-------------------|--------------------------------------------------|
| 🚀 Easy to Use   | [Get started](https://p2pfl.github.io/p2pfl/quickstart.html) quickly with our intuitive API.       |
| 🛡️ Reliable     | Built for fault tolerance and resilience.       |
| 🌐 Scalable      | Leverages the power of peer-to-peer networks.    |
| 🧪 Versatile     | Experiment in simulated or real-world environments.|
| 🔒 Private       | Prioritizes data privacy with decentralized architecture.|
| 🧩 Flexible      | Designed to be easy to modify.|
| 📈 Real-time Monitoring | Manage and track experiment through [P2PFL Web Services platform](https://p2pfl.com). |
| 🧠 ML Frameworks | Seamlessly integrate [PyTorch](https://pytorch.org/), [TensorFlow/Keras](https://www.tensorflow.org/), and [JAX](https://github.com/google/jax) models. |
| 📡 Communication Protocol Agnostic | Choose the communication protocol that best suits your needs (e.g., [gRPC](https://grpc.io/)). |
| 🔌 Integrations  | Enhanced capabilities through integrations: [Hugging Face Datasets](https://huggingface.co/datasets), ML frameworks, communication protocols, and [Ray](https://www.ray.io/) for large-scale simulations. |

## 📥 Installation

### 👨🏼‍💻 For Users

```bash
pip install "p2pfl[torch]"
```

### 👨🏼‍🔧 For Developers

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/p2pfl/p2pfl/tree/develop?quickstart=1)

#### 🐍 Python (using UV)

```bash
git clone https://github.com/p2pfl/p2pfl.git
cd p2pfl
uv sync --all-extras
```

> **Note:** The above command installs all dependencies (PyTorch, TensorFlow, and Ray). If you only need specific frameworks, you can use:
> - `uv sync` - Install only core dependencies
> - `uv sync --extra torch` - Install with PyTorch support
> - `uv sync --extra tensorflow` - Install with TensorFlow support
> - `uv sync --extra ray` - Install with Ray support
> 
> Use `--no-dev` to exclude development dependencies.

> **⚠️ Important for Ray users:** If you're using Ray, we recommend activating the virtual environment traditionally instead of using `uv run` to avoid dependency issues with Ray workers. See the [installation guide](https://p2pfl.github.io/p2pfl/installation.html#working-with-traditional-virtual-environment-activation) for details.

#### 🐳 Docker

```bash
docker build -t p2pfl .
docker run -it --rm p2pfl bash
```

## 🎬 Quickstart

To start using P2PFL, follow our [quickstart guide](https://p2pfl.github.io/p2pfl/quickstart.html) in the documentation.

## 📚 Documentation & Resources

* **Documentation:** [https://p2pfl.github.io/p2pfl/](https://p2pfl.github.io/p2pfl)
* **Technical Report:** (first version) [other/memoria.pdf](other/memoria.pdf)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Please adhere to the project's code of conduct in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 💬 Community

Connect with us and stay updated:

* [**GitHub Issues:**](/issues) - For reporting bugs and requesting features.
* [**Google Group:**](https://groups.google.com/g/p2pfl) - For discussions and announcements.
* [**Slack:**](https://join.slack.com/t/p2pfl/shared_invite/zt-38xyec0k7-tLPbNx873Pm~N4aoqIyjRw) - For real-time conversations and support.

## ⭐ Star History

A big thank you to the community for your interest in P2PFL! We appreciate your support and contributions.

[![Star History Chart](https://api.star-history.com/svg?repos=p2pfl/p2pfl&type=Date)](https://star-history.com/#p2pfl/p2pfl&Date)

## 📜 License

[GNU General Public License, Version 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
