![GitHub Logo](other/logo.png)

# P2PFL - Federated Learning over P2P networks

[![GitHub license](https://img.shields.io/github/license/pguijas/federated_learning_p2p)](https://github.com/pguijas/federated_learning_p2p/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/pguijas/federated_learning_p2p)](https://github.com/pguijas/federated_learning_p2p/issues)
![GitHub contributors](https://img.shields.io/github/contributors/pguijas/federated_learning_p2p)
![GitHub forks](https://img.shields.io/github/forks/pguijas/federated_learning_p2p)
![GitHub stars](https://img.shields.io/github/stars/pguijas/federated_learning_p2p)
![GitHub activity](https://img.shields.io/github/commit-activity/m/pguijas/federated_learning_p2p)

P2PFL is a general-purpose open-source library for the execution (simulated and in real environments) of Decentralized Federated Learning systems, specifically making use of P2P networks and the Gossisp protocol.

## ✅ Features

- Easy to use and extend
- Fault tolerant
- Decentralized and Scalable
- Simulated and real environments
- Privacy-preserving
- Framework agnostic

## 📥 Installation

> **Note**
> We recommend using Python 3.9 or lower. We have found some compatibility issues with Python 3.10 and PyTorch.

### 👨🏼‍💻 For users

To install the library, you can simply run:

```bash
pip install p2pfl
```

Or you can install it from source:

```bash
git clone https://github.com/pguijas/p2pfl.git
cd p2pfl
pip install -e .
```

### 👨🏼‍🔧 For developers

To install the library for development we recommend using a virtual environment. For example, with `pipenv`:

```bash
pipenv install --requirements requirements.txt
```

## 📚 Documentation

- [Documentation](https://pguijas.github.io/federated_learning_p2p/).

- [Report of the end-of-degree project](other/memoria.pdf).

- [Report for the award for the best open source end-of-degree project](other/memoria-open-source.pdf).

## 🚀 TO DO

> **Note**
> Don't be shy, share your ideas with us!

- Agnostic installation with variants for different frameworks (include TensorFlow)
- Add secure channels and node authentication
- Improved simulation environment
- Control panel
- add FEMNIST example
- add typing
- New aggregation methods
- Hot node inclusion
- Secure aggregation

## 👫 Contributing

Contributions are always welcome!

See `CONTRIBUTING.md` for ways to get started.

Please adhere to this project's code of conduct specified in `CODE_OF_CONDUCT.md`.

## 💬 Google Group

If you have any questions, or you to be notified of any updates, you can join the Google Group [here](https://groups.google.com/g/p2pfl).

## 📜 License

[GNU General Public License, Version 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)