# Quick Start Guide

Get the Validator Agent running locally in minutes.

## Prerequisites

- **Python 3.12+**
- **pip** package manager
- (Optional) **Google Cloud SDK** for Vertex AI features

## 1. Installation

Clone the repository and install dependencies:

```bash
# Install with dev dependencies (includes pytest, asyncpg)
pip install -e ".[dev]"

# Or install with postgres support only
pip install -e ".[postgres]"

# Or minimal install (stub adapters only)
pip install -e .
```

## 2. Quick Start with Stub Adapters (No External Dependencies)

The Validator Agent defaults to **stub adapters** that simulate external services. This is the fastest way to get started with no external dependencies.

```bash
# Run with all defaults (stub adapters)
uvicorn validator_agent.main:app --reload --port 8080
```

The API will be available at: `http://localhost:8080`

### Test the Health Endpoint

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status": "healthy"}
```

## 3. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=validator_agent --cov-report=html
```

## 4. Using Real Adapters

To use real AI services (Vertex AI, OpenAI), set the appropriate adapter configuration:

### 4.1 MRC + OCR with Vertex AI

Requires [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) authentication:

```bash
# Authenticate with GCP
gcloud auth login
gcloud auth application-default login
gcloud config set project aflow-491809

# Set environment variables
export MRC_ADAPTER=vertex_ai
export OCR_ADAPTER=vertex_ai
export LLM_PROVIDER=vertex_ai
export MODEL_BROKER_URL=http://localhost:8010

# Run
uvicorn validator_agent.main:app --reload --port 8080
```

### 4.2 With OpenAI (for Moderation + Visual Fallback)

```bash
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=google_ai_studio  # or vertex_ai

uvicorn validator_agent.main:app --reload --port 8080
```

### 4.3 With Knowledge Service

```bash
export KNOWLEDGE_ADAPTER=http
export KNOWLEDGE_SERVICE_URL=http://localhost:8020

uvicorn validator_agent.main:app --reload --port 8080
```

## 5. Docker Deployment

```bash
# Build image
docker build -t validator-agent .

# Run with stub adapters
docker run -p 8080:8080 validator-agent

# Run with environment variables
docker run -p 8080:8080 \
  -e MRC_ADAPTER=vertex_ai \
  -e OCR_ADAPTER=vertex_ai \
  -e OPENAI_API_KEY=sk-... \
  validator-agent
```

## 6. Event-Driven Mode (Pub/Sub)

To run as an event-driven agent listening for Pub/Sub triggers:

```bash
# Using Pub/Sub emulator (local testing)
export EVENT_ADAPTER=emulator
export PUBSUB_EMULATOR_HOST=localhost:8085
export PUBSUB_PROJECT_ID=test-project

# Or using real Pub/Sub
export EVENT_ADAPTER=pubsub
export PUBSUB_PROJECT_ID=aflow-491809

# Run
uvicorn validator_agent.main:app --reload --port 8080
```

## 7. Environment Configuration

Copy the example environment file and customize:

```bash
cp .env.example .env
# Edit .env with your values
source .env
uvicorn validator_agent.main:app --reload --port 8080
```

## Adapter Quick Reference

| Adapter | Purpose | Requirements |
|---------|---------|--------------|
| `MRC_ADAPTER=vertex_ai` | Blur detection | `gcloud auth login` |
| `OCR_ADAPTER=vertex_ai` | Text extraction | GCP service account |
| `LLM_PROVIDER=vertex_ai` | Content safety | Model Broker URL |
| `STORAGE_ADAPTER=local` | File downloads | `UPLOAD_DIR` path |
| `STORAGE_ADAPTER=gcs` | Cloud Storage | `GCS_BUCKET_NAME` |
| `AUDIT_ADAPTER=postgres` | Decision logging | Postgres connection |
| `EVENT_ADAPTER=pubsub` | Event messaging | GCP project ID |
| `KNOWLEDGE_ADAPTER=http` | Content forwarding | Knowledge Service URL |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe - returns 200 if process is running |
| `/ready` | GET | Readiness probe - returns 200 if agent is ready |
| `/invoke` | POST | Run full validation pipeline for submitted files |

### Example Validation Request

```bash
curl -X POST http://localhost:8080/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "test-123",
    "assessment_id": "assess-456",
    "files": [
      {
        "file_name": "document.pdf",
        "storage_path": "/path/to/document.pdf",
        "file_type": "pdf"
      }
    ]
  }'
```

## Troubleshooting

### "Failed to get GCP access token"
Run `gcloud auth login` and `gcloud auth application-default login`

### "No OPENAI_API_KEY set"
Add `export OPENAI_API_KEY=sk-...` to your `.env` file

### Module not found errors
Ensure you're using Python 3.12+ and installed with `pip install -e ".[dev]"`

## Next Steps

- Read the full [README.md](README.md) for architecture details
- Check [tests/](tests/) for usage examples
- See `.env.example` for all configuration options
