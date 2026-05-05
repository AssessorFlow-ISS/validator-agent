FROM rockylinux/rockylinux:10-ubi AS builder

ENV UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/app/.venv \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv

RUN dnf install -y ca-certificates tar gzip findutils poppler-utils && \
    dnf clean all

RUN uv python install 3.12 --install-dir /opt/python
RUN ln -s /opt/python/cpython-3.12* /opt/python/current

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN /usr/local/bin/uv venv /opt/app/.venv \
    --python /opt/python/current/bin/python3.12 \
    --relocatable \
    --link-mode copy

RUN /usr/local/bin/uv sync --frozen --no-dev --no-editable --active

FROM rockylinux/rockylinux:10-ubi

RUN dnf install -y poppler-utils && \
    dnf clean all

ENV VIRTUAL_ENV=/opt/app/.venv \
    PATH=/opt/app/.venv/bin:/opt/python/current/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app/src

WORKDIR /opt/app

COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/app/.venv /opt/app/.venv
COPY --from=builder /workspace/src /opt/app/src
COPY prompts/ /opt/app/prompts/

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin appuser
USER 10001:10001

EXPOSE 8080

ENTRYPOINT ["/opt/app/.venv/bin/python", "-m", "uvicorn"]
CMD ["validator_agent.main:app", "--host", "0.0.0.0", "--port", "8080"]
