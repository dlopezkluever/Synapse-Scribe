# ---- Base stage ----
FROM python:3.10-slim AS base

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.10-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libhdf5-103 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from base
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ src/
COPY app/ app/
COPY config.yaml .
COPY pyproject.toml .
COPY scripts/ scripts/

# Ensure data and output directories exist
RUN mkdir -p data outputs

# Set PYTHONPATH so imports resolve
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default: run FastAPI backend
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
