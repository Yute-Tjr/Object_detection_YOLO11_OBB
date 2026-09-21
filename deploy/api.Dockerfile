FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY alembic.ini ./
COPY migrations ./migrations
COPY terminal_web ./terminal_web
COPY scripts ./scripts
COPY obb_detection ./obb_detection
COPY yolo11_obb ./yolo11_obb

RUN useradd --create-home --uid 10001 terminal \
    && mkdir -p /data/terminal-inspection \
    && chown -R terminal:terminal /app /data/terminal-inspection

USER terminal

CMD ["python", "scripts/run_terminal_api.py", "--host", "0.0.0.0", "--port", "8000"]
