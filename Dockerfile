# Container image for the Kani MLX FastAPI service.
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG INSTALL_SYSTEM_DEPS=true
RUN if [ "${INSTALL_SYSTEM_DEPS}" = "true" ]; then \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ffmpeg \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*; \
    else \
    echo "Skipping system dependency installation"; \
    fi

WORKDIR /workspace
COPY requirements.txt ./

ARG INSTALL_PIP_DEPS=true
RUN if [ "${INSTALL_PIP_DEPS}" = "true" ]; then \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt; \
    else \
    echo "Skipping adapter dependency installation"; \
    fi

COPY . .

EXPOSE 8000
ENTRYPOINT ["python", "server.py"]
