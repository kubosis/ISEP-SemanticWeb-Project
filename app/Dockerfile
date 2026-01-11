FROM python:3.13-slim

# Set working directory to /project inside the container
WORKDIR /project

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Python deps
# Now valid because context is root
COPY pyproject.toml uv.lock ./

FROM python:3.13-slim

WORKDIR /project

# Install system dependencies + curl/wget for frpc download
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

ENV PATH="/project/.venv/bin:$PATH"

COPY app ./app

# Create cache directory with proper permissions
RUN mkdir -p /gradio_cache && chmod -R 777 /gradio_cache

# Set Gradio cache location
ENV XDG_CACHE_HOME=/gradio_cache
ENV GRADIO_TEMP_DIR=/gradio_cache

# Expose Gradio port
EXPOSE 7860

CMD ["python", "-m", "app.main"]