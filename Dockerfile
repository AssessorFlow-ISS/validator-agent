FROM python:3.12-slim AS base

WORKDIR /app

# Install af-shared (peer dependency for all agents)
COPY --from=shared . /tmp/af-shared/
RUN pip install --no-cache-dir "/tmp/af-shared[langfuse]" && rm -rf /tmp/af-shared

# Copy everything then install
COPY pyproject.toml README.md ./
COPY src/ src/
COPY prompts/ prompts/
RUN pip install --no-cache-dir -e ".[postgres]"

EXPOSE 8080

CMD ["uvicorn", "validator_agent.main:app", "--host", "0.0.0.0", "--port", "8080"]
