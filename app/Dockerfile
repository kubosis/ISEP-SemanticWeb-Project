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

# Install dependencies into .venv
RUN uv sync --frozen --no-install-project --no-dev

# CRITICAL: Add the virtual environment to PATH so 'python' command uses it
ENV PATH="/project/.venv/bin:$PATH"

# Copy the app source code folder into /project/app
COPY app ./app

# Gradio port
EXPOSE 7860

# Run the app
# Since WORKDIR is /project, this correctly finds app/main.py
CMD ["python", "-m", "app.main"]