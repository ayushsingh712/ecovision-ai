# EcoVision AI — CPU inference image (API + static frontend)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image small; install it before the rest of the
# requirements so pip is satisfied by the CPU build rather than the CUDA wheel.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY ml ./ml
COPY pytest.ini ./

RUN useradd --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/data/uploads /app/ml/models /app/ml/evaluation \
    && chown -R app:app /app

USER app

# Render injects PORT (defaults to 10000 for Docker web services). Runtime state
# lives under DATA_DIR, which must be writable by the non-root user.
ENV PORT=10000 \
    ECOVISION_DEVICE=cpu \
    ECOVISION_DATA_DIR=/app/data \
    ECOVISION_SERVE_FRONTEND=true \
    ECOVISION_FRONTEND_DIR=/app/frontend \
    ECOVISION_MODEL_DIR=/app/ml/models \
    ECOVISION_EVALUATION_DIR=/app/ml/evaluation \
    ECOVISION_LOG_LEVEL=INFO \
    HF_HOME=/home/app/.cache/huggingface

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request,sys; port=os.environ.get('PORT','10000'); sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/api/health' % port, timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port \"${PORT:-10000}\" --proxy-headers --forwarded-allow-ips='*'"]
