FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build dependencies
RUN apt-get update && apt-get install -y gcc

# Add source code to the image
ADD . /app
WORKDIR /app

# Install
RUN uv sync --all-extras --locked

# Default command - runs bash shell with environment activated
# You can override this to run p2pfl directly: docker run <image> /bin/bash -c "source .venv/bin/activate && p2pfl"
# Or run any other command: docker run <image> /bin/bash -c "source .venv/bin/activate && python script.py"
CMD ["/bin/bash", "-c", "source .venv/bin/activate && bash"]
