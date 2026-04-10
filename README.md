# Validator Agent (#12)

Centralized content safety gate for AssessorFlow. Validates all content entering the system in Phase 3 (uploaded materials) and Phase 5 (web research content).

## Architecture

Stateless FastAPI microservice (no graph.py, no Redis). All external dependencies are behind ABC port interfaces with stub adapters for local development.

## Validation Pipeline

For each file:
1. **Download** from Cloud Storage via `StoragePort`
2. **MRC check** via `MrcPort` — blur/readability classification
3. **OCR extraction** via `OcrPort` — text extraction from documents
4. **Content safety** via `ModelBrokerPort` — LLM-based harmful content, PII, copyright detection
5. **Forward to Knowledge Service** via `KnowledgeServicePort` (only on PROCEED)

Returns a **Terminal Signal**: `PROCEED` or `TERMINATE` with reason code and message.

## Running

```bash
pip install -e ".[dev]"
uvicorn validator_agent.main:app --reload --port 8080
```

## Testing

```bash
pytest tests/ -v
```

## Adapter Configuration

Set environment variables to swap between stub and real adapters:

| Variable | Values | Default |
|----------|--------|---------|
| `MRC_ADAPTER` | `stub`, `vertex_ai` | `stub` |
| `OCR_ADAPTER` | `stub`, `vertex_ai` | `stub` |
| `LLM_PROVIDER` | `stub`, `google_ai_studio`, `vertex_ai` | `stub` |
| `KNOWLEDGE_ADAPTER` | `stub`, `grpc` | `stub` |
| `AUDIT_ADAPTER` | `stub`, `grpc` | `stub` |
| `EVENT_ADAPTER` | `stub`, `pubsub` | `stub` |
| `STORAGE_ADAPTER` | `stub`, `gcs` | `stub` |
