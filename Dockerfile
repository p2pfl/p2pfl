FROM python:3.9-slim

WORKDIR /app

COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    HOME=/app

# Update and install dependencies

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Install dependencies
RUN poetry install --without dev,docs

# Install torch (cpu) - poetry present problem with torch-cpu installation
RUN poetry run pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Install torchmetrics and pytorch-lightning
RUN poetry run pip install torchmetrics==1.4.0.post0 pytorch-lightning==1.9.5