FROM python:3.12-slim AS base

WORKDIR /app

# Accept build argument for private registry access (optional for local builds)
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

COPY pyproject.toml README.md ./
COPY src/ src/

# Install dependencies - uses extra index if provided, otherwise uses public PyPI only
RUN if [ -n "$PIP_EXTRA_INDEX_URL" ]; then \
        pip install --no-cache-dir --extra-index-url "$PIP_EXTRA_INDEX_URL" .; \
    else \
        pip install --no-cache-dir .; \
    fi

COPY prompts/ prompts/

EXPOSE 8080

CMD ["uvicorn", "validator_agent.main:app", "--host", "0.0.0.0", "--port", "8080"]
