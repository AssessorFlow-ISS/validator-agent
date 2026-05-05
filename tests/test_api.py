"""Tests for FastAPI endpoints (POST /invoke, GET /health, GET /ready)."""

from __future__ import annotations

from httpx import AsyncClient


class TestHealthEndpoint:
    async def test_health_returns_200(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestReadyEndpoint:
    async def test_ready_returns_200(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}


class TestInvokeEndpoint:
    async def test_invoke_clean_files_returns_proceed(self, api_client: AsyncClient) -> None:
        payload = {
            "workflow_id": "wf-api-001",
            "assessment_id": "assess-001",
            "validation_type": "material_validation",
            "files": [
                {
                    "file_name": "chapter1.pdf",
                    "storage_path": "materials/wf-api-001/chapter1.pdf",
                    "file_type": "pdf",
                },
            ],
        }
        response = await api_client.post("/invoke", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["workflow_id"] == "wf-api-001"
        assert body["terminal_signal"]["status"] == "PROCEED"
        assert body["terminal_signal"]["reason_code"] == "VALIDATION_PASSED"
        assert len(body["file_results"]) == 1

    async def test_invoke_returns_file_results(self, api_client: AsyncClient) -> None:
        payload = {
            "workflow_id": "wf-api-002",
            "assessment_id": "assess-002",
            "validation_type": "material_validation",
            "files": [
                {
                    "file_name": "doc1.pdf",
                    "storage_path": "materials/wf-api-002/doc1.pdf",
                    "file_type": "pdf",
                },
                {
                    "file_name": "doc2.pdf",
                    "storage_path": "materials/wf-api-002/doc2.pdf",
                    "file_type": "pdf",
                },
            ],
        }
        response = await api_client.post("/invoke", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert len(body["file_results"]) == 2

    async def test_invoke_missing_fields_returns_422(self, api_client: AsyncClient) -> None:
        # Missing required "files" field
        payload = {
            "workflow_id": "wf-api-003",
            "assessment_id": "assess-003",
        }
        response = await api_client.post("/invoke", json=payload)
        assert response.status_code == 422

    async def test_invoke_empty_files_returns_422(self, api_client: AsyncClient) -> None:
        payload = {
            "workflow_id": "wf-api-004",
            "assessment_id": "assess-004",
            "validation_type": "material_validation",
            "files": [],
        }
        response = await api_client.post("/invoke", json=payload)
        assert response.status_code == 422

    async def test_invoke_missing_file_fields_returns_422(self, api_client: AsyncClient) -> None:
        payload = {
            "workflow_id": "wf-api-005",
            "assessment_id": "assess-005",
            "validation_type": "material_validation",
            "files": [
                {"file_name": "doc.pdf"},  # missing storage_path and file_type
            ],
        }
        response = await api_client.post("/invoke", json=payload)
        assert response.status_code == 422

    async def test_invoke_response_schema_shape(self, api_client: AsyncClient) -> None:
        payload = {
            "workflow_id": "wf-api-006",
            "assessment_id": "assess-006",
            "validation_type": "material_validation",
            "files": [
                {
                    "file_name": "chapter1.pdf",
                    "storage_path": "materials/wf-api-006/chapter1.pdf",
                    "file_type": "pdf",
                },
            ],
        }
        response = await api_client.post("/invoke", json=payload)
        body = response.json()

        # Validate top-level shape
        assert "workflow_id" in body
        assert "terminal_signal" in body
        assert "file_results" in body

        # Validate terminal_signal shape
        ts = body["terminal_signal"]
        assert "status" in ts
        assert "reason_code" in ts
        assert "message" in ts

        # Validate file_results shape
        fr = body["file_results"][0]
        assert "file_name" in fr
        assert "terminal_signal" in fr
