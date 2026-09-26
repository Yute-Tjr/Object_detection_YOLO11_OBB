ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

ARG PYTORCH_INDEX_URL=
ARG DEBIAN_MIRROR_URL=
ARG DEBIAN_SECURITY_MIRROR_URL=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN if [ -n "$DEBIAN_MIRROR_URL" ] && [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|http://deb.debian.org/debian-security|$DEBIAN_SECURITY_MIRROR_URL|g; s|http://deb.debian.org/debian|$DEBIAN_MIRROR_URL|g" \
            /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && apt-get install --no-install-recommends -y fonts-noto-cjk libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN if [ -n "$PYTORCH_INDEX_URL" ]; then \
        grep -E '^(torch|torchvision)==' requirements.txt > /tmp/pytorch-requirements.txt; \
        pip install --index-url "$PYTORCH_INDEX_URL" -r /tmp/pytorch-requirements.txt; \
    fi \
    && pip install -r requirements.txt \
    && rm -f /tmp/pytorch-requirements.txt

COPY terminal_web ./terminal_web
COPY scripts/run_terminal_worker.py ./scripts/
COPY obb_detection ./obb_detection
COPY yolo11_obb ./yolo11_obb

RUN useradd --create-home --uid 10001 terminal \
    && mkdir -p /data/terminal-inspection \
    && chown -R terminal:terminal /app /data/terminal-inspection

USER terminal

CMD ["python", "scripts/run_terminal_worker.py"]
