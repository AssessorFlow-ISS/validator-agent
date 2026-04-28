FROM python:3.12-slim AS base

WORKDIR /app

# poppler-utils provides pdftoppm — required by pdf2image.convert_from_bytes
# in pipeline/pdf_to_images.py (page-classifier feeds visual pages to GPT-4o).
# Without it, pdf_pages_to_images() raises and ocr_pipeline silently early-returns
# leaving all pages with classification=None → "0 TEXT, 0 VISUAL" in traces.
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    rm -rf /var/lib/apt/lists/*

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
