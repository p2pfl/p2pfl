![GitHub Logo](https://raw.githubusercontent.com/p2pfl/p2pfl/main/other/logo.png)

# P2PFL - Federated Learning over P2P networks

[![GitHub license](https://img.shields.io/github/license/p2pfl/p2pfl)](https://github.com/p2pfl/p2pfl/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/p2pfl/p2pfl)](https://github.com/p2pfl/p2pfl/issues)
![GitHub contributors](https://img.shields.io/github/contributors/p2pfl/p2pfl)
![GitHub forks](https://img.shields.io/github/forks/p2pfl/p2pfl)
![GitHub stars](https://img.shields.io/github/stars/p2pfl/p2pfl)
![GitHub activity](https://img.shields.io/github/commit-activity/m/p2pfl/p2pfl)
[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fp2pfl%2Fp2pfl%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/p2pfl/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://join.slack.com/t/p2pfl/shared_invite/zt-2lbqvfeqt-FkutD1LCZ86yK5tP3Duztw)

P2PFL is a general-purpose open-source library for the execution (simulated and in real environments) of Decentralized Federated Learning systems, specifically making use of P2P networks and the Gossip protocol.

## ✨ Key Features

P2PFL offers a range of features designed to make decentralized federated learning accessible and efficient. For detailed information, please refer to our [documentation](https://p2pfl.github.io/p2pfl/).

| Feature          | Description                                      |
|-------------------|--------------------------------------------------|
| 🚀 Easy to Use   | Get started quickly with our intuitive API.       |
| 🛡️ Reliable     | Built for fault tolerance and resilience.       |
| 🌐 Scalable      | Leverages the power of peer-to-peer networks.    |
| 🧪 Versatile     | Experiment in simulated or real-world environments.|
| 🔒 Private       | Prioritizes data privacy with decentralized architecture.|
| 🧩 Flexible      | Integrate with PyTorch and TensorFlow (coming soon!).|
| 📈 Real-time Monitoring | Manage and track experiment through [P2PFL Web Services](https://p2pfl.com). | 
| 🧠 Model Agnostic | Use any machine learning model you prefer (e.g., PyTorch models). |
| 📡 Communication Protocol Agnostic | Choose the communication protocol that best suits your needs (e.g., gRPC). |
## 📥 Installation

> **Note:** We recommend using Python 3.9 or lower. We have found some compatibility issues with Python 3.10 and PyTorch.

### 👨🏼‍💻 For Users

```bash
pip install "p2pfl[torch]"
```

### 👨🏼‍🔧 For Developers

#### 🐍 Python (using Poetry)

```bash
git clone https://github.com/p2pfl/p2pfl.git
cd p2pfl
poetry install -E torch 
```

> **Note:** Use the extras (`-E`) flag to install specific dependencies (e.g., `-E torch`). Use `--no-dev` to exclude development dependencies.

#### 🐳 Docker

```bash
docker build -t p2pfl .
docker run -it --rm p2pfl bash
```

## 🎬 Quickstart

To start using P2PFL, follow our [quickstart guide](https://p2pfl.github.io/p2pfl/quickstart.html) in the documentation.

## 📚 Documentation & Resources

* **Documentation:** [https://p2pfl.github.io/p2pfl/](https://p2pfl.github.io/p2pfl)
* **End-of-Degree Project Report:** [other/memoria.pdf](other/memoria.pdf)
* **Open Source Project Award Report:** [other/memoria-open-source.pdf](other/memoria-open-source.pdf)

## 🤝 Contributing

We welcome contributions! See `CONTRIBUTING.md` for guidelines. Please adhere to the project's code of conduct in `CODE_OF_CONDUCT.md`.

## 💬 Community

Connect with us and stay updated:

* [**GitHub Discussions:**](https://github.com/p2pfl/p2pfl/discussions) - For general discussions, questions, and ideas.
* [**GitHub Issues:**](https://github.com/p2pfl/p2pfl/issues) - For reporting bugs and requesting features.
* [**Google Group:**](https://groups.google.com/g/p2pfl) - For discussions and announcements.
* [**Slack:**](https://join.slack.com/t/p2pfl/shared_invite/zt-2lbqvfeqt-FkutD1LCZ86yK5tP3Duztw) - For real-time conversations and support.


## ⭐ Star History

A big thank you to the community for your interest in P2PFL! We appreciate your support and contributions.

[![Star History Chart](https://api.star-history.com/svg?repos=p2pfl/p2pfl&type=Date)](https://star-history.com/#p2pfl/p2pfl&Date)

## 📜 License

[GNU General Public License, Version 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
