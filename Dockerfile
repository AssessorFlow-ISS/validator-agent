FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY prompts/ prompts/

EXPOSE 8080

CMD ["uvicorn", "validator_agent.main:app", "--host", "0.0.0.0", "--port", "8080"]
