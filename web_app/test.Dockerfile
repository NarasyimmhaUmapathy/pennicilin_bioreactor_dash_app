# ---------- Stage 1: build dependencies ----------
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps only needed to BUILD wheels (not needed in runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ cmake python3-dev pkg-config libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements early for layer caching
COPY web_app/requirements.txt /app/requirements.txt

# Create venv and install deps into it
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt


# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

# Runtime libs often needed by compiled Python packages (safe minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bring the venv from build stage
COPY --from=build /opt/venv /opt/venv

# Copy app code
COPY web_app/pages ./pages
COPY web_app/reports ./reports
COPY web_app/utils ./utils
COPY web_app/app.py ./app.py

# Shared code
COPY common/ ./common

EXPOSE 8080

CMD ["sh", "-c", "gunicorn app:server --bind 0.0.0.0:${PORT}"]
