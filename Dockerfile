FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VENV_PATH="/app/.venv"
    # Use custom venv, avoid auto-creation by Poetry

# Install system tools and libraries.
# Utilize --mount flag of Docker Buildx to cache downloaded packages, avoiding repeated downloads
RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    apt-get update && \ 
    apt-get install -y software-properties-common && \
    # Add the Deadsnakes PPA for Python 3.9
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        python3.9 \
        python3-pip \
        python3.9-venv \
        python3.9-dev && \
    # Clean up to keep the image size small
    rm -rf /var/lib/apt/lists/*  && \
    # Set Python 3.11 as the default Python version
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --set python /usr/bin/python3.9

# Set PATH to include Poetry and custom venv
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION

# Create and prepare the virtual environment
RUN python -m venv $VENV_PATH && \
    python -m pip install --upgrade pip && \
    pip cache purge && rm -Rf /root/.cache/pip/http
    
WORKDIR /app

# Copy dependency files to the app directory
COPY . /app/

# Install dependencies with Poetry and Torch with pip, caching downloaded packages
RUN --mount=type=cache,target=/root/.cache \
    poetry install --without dev && \
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html 

ENTRYPOINT [ "/bin/bash" ]