ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

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
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN sed -E 's/^([[:alnum:]_.-]+)\[[^]]+\](==.*)$/\1\2/' \
        requirements.txt > /tmp/api-constraints.txt \
    && pip install --constraint /tmp/api-constraints.txt \
    alembic \
    argon2-cffi \
    fastapi \
    numpy \
    opencv-python \
    pillow \
    'psycopg[binary]' \
    pydantic \
    pydantic-settings \
    python-multipart \
    SQLAlchemy \
    'uvicorn[standard]' \
    && rm -f /tmp/api-constraints.txt

COPY alembic.ini ./
COPY migrations ./migrations
COPY terminal_web ./terminal_web
COPY scripts/run_terminal_api.py scripts/manage_users.py ./scripts/

RUN useradd --create-home --uid 10001 terminal \
    && mkdir -p /data/terminal-inspection \
    && chown -R terminal:terminal /app /data/terminal-inspection

USER terminal

CMD ["python", "scripts/run_terminal_api.py", "--host", "0.0.0.0", "--port", "8000"]
