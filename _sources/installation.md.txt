# 📥 Installation

## 👨🏼‍💻 For users


To install the library, you can simply run:

```bash
pip install "p2pfl[torch]"
```

Or you can install it from source. This installation method is recommended for developers (detailed in the next section).


## 👨🏼‍🔧 For developers

### 🐍 Python


To install the library for development we recommend using a virtual environment. We use [Poetry](https://python-poetry.org/) for this purpose.


```bash
git clone https://github.com/pguijas/p2pfl.git
cd p2pfl
poetry install -E torch
```
> **Note**: You can use the extras (`-E`) flag to install the library with the desired dependencies. Also, you can use the `--no-dev` flag to install the library without the development dependencies.


### 🐳 Docker

You can also use the library with Docker. We provide a Docker image with the library installed. You can use it as follows:

```bash
docker build -t p2pfl .
docker run -it --rm p2pfl bash
```