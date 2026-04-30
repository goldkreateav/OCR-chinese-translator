FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      poppler-utils \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python3 -m pip install -U pip wheel setuptools \
    && python3 -m pip install -e . \
    && python3 -m pip install -e ".[rapidocr]" \
    && python3 -m pip install onnxruntime-gpu

ENV MASKPDF_DATA_ROOT=/data
RUN mkdir -p /data

EXPOSE 54172

CMD ["maskpdf", "web", "--host", "0.0.0.0", "--port", "54172", "--data-root", "/data", "--allow-fallback", "--ocr-device", "cuda", "--ocr-fallback-cpu-on-oom"]

