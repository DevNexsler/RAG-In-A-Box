FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
# Filter out Apple Silicon-only packages (mlx-*) that won't install on Linux
RUN grep -v '^mlx' requirements.txt > requirements.docker.txt && \
    pip install --no-cache-dir -r requirements.docker.txt && \
    rm requirements.docker.txt

COPY . .

VOLUME /data

EXPOSE 7788

# Liveness/progress probe. /health returns 503 only when the indexer is
# genuinely frozen (stale heartbeat) or FTS rebuild is failing, and 200 when
# healthy or idle — so the container's health is visible to `docker ps` and
# monitoring instead of relying solely on an external probe (#0127). The slim
# image has no curl/wget, so use urllib: it exits non-zero on any non-2xx.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7788/health', timeout=5)"]

CMD ["python", "server.py"]
