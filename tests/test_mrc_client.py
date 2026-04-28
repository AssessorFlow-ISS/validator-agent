"""Unit tests for validator_agent.pipeline.mrc_client.

Covers:
  - No-endpoint escape hatch (auto-PROCEED).
  - Connection error / general exception → TERMINATE.
  - Local vs. Vertex AI routing.
  - Blurry threshold application (PROCEED / PROCEED_WITH_EXCLUSIONS / TERMINATE).
  - _is_vertex_ai_endpoint() helper.
  - _get_gcp_token() in dev and prod modes.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from validator_agent.pipeline import mrc_client as mrc_mod


class TestIsVertexAiEndpoint:
    def test_vertex_ai_url_detected(self) -> None:
        assert mrc_mod._is_vertex_ai_endpoint(
            "https://asia-southeast1-aiplatform.googleapis.com/v1/projects/..."
        )

    def test_local_url_not_detected(self) -> None:
        assert not mrc_mod._is_vertex_ai_endpoint("http://localhost:8000/infer")


class TestGetGcpToken:
    def test_prod_uses_metadata_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "APP_MODE", "prod")
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "xyz"}
        mock_response.raise_for_status = MagicMock()
        with patch.object(mrc_mod.requests, "get", return_value=mock_response):
            assert mrc_mod._get_gcp_token() == "xyz"

    def test_prod_token_empty_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "APP_MODE", "prod")
        with patch.object(
            mrc_mod.requests, "get", side_effect=requests.exceptions.ConnectionError("x")
        ):
            assert mrc_mod._get_gcp_token() == ""

    def test_dev_uses_gcloud_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "APP_MODE", "dev")
        fake_proc = MagicMock()
        fake_proc.stdout = "token-abc\n"
        with patch.object(mrc_mod.subprocess, "run", return_value=fake_proc):
            assert mrc_mod._get_gcp_token() == "token-abc"

    def test_dev_token_empty_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "APP_MODE", "dev")
        with patch.object(mrc_mod.subprocess, "run", side_effect=FileNotFoundError("no gcloud")):
            assert mrc_mod._get_gcp_token() == ""


class TestCallVertexAi:
    def test_vertex_ai_raises_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "https://x-aiplatform.googleapis.com/v1/x")
        with patch.object(mrc_mod, "_get_gcp_token", return_value=""):
            with pytest.raises(ConnectionError, match="GCP access token"):
                mrc_mod._call_vertex_ai(b"bytes", "doc.pdf")

    def test_vertex_ai_sends_bearer_and_returns_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "https://x-aiplatform.googleapis.com/v1/x")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"k": "v"}
        mock_resp.raise_for_status = MagicMock()
        with (
            patch.object(mrc_mod, "_get_gcp_token", return_value="TOKEN"),
            patch.object(mrc_mod.requests, "post", return_value=mock_resp) as post_mock,
        ):
            data = mrc_mod._call_vertex_ai(b"bytes", "doc.pdf")
        assert data == {"k": "v"}
        headers = post_mock.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer TOKEN"


class TestCallLocal:
    def test_local_posts_and_returns_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "http://localhost:8000/infer")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(mrc_mod.requests, "post", return_value=mock_resp):
            assert mrc_mod._call_local(b"b", "doc.pdf") == {"ok": True}


class TestCheckReadability:
    def test_no_endpoint_returns_proceed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "")

        result = mrc_mod.check_readability(b"bytes", "doc.pdf")

        assert result.overall_status == "PROCEED"
        assert result.total_pages == 1
        assert result.file_name == "doc.pdf"

    def test_connection_error_returns_terminate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "http://localhost:8000/mrc")
        with patch.object(
            mrc_mod,
            "_call_local",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            result = mrc_mod.check_readability(b"bytes", "doc.pdf")
        assert result.overall_status == "TERMINATE"
        assert result.total_pages == 0

    def test_generic_exception_returns_terminate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "http://localhost:8000/mrc")
        with patch.object(mrc_mod, "_call_local", side_effect=ValueError("boom")):
            result = mrc_mod.check_readability(b"bytes", "doc.pdf")
        assert result.overall_status == "TERMINATE"

    def test_local_proceed_all_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "http://localhost:8000/mrc")
        data = {
            "overall_readiness": True,
            "overall_confidence": 0.98,
            "total_pages": 2,
            "readable_pages": 2,
            "unreadable_pages": 0,
            "page_results": [
                {"page": 1, "readiness": True, "confidence": 0.99},
                {"page": 2, "readiness": True, "confidence": 0.97},
            ],
            "file_name": "doc.pdf",
            "inference_time_ms": 42.0,
        }
        with patch.object(mrc_mod, "_call_local", return_value=data):
            result = mrc_mod.check_readability(b"bytes", "doc.pdf")

        assert result.overall_status == "PROCEED"
        assert result.readable_page_numbers == [1, 2]
        assert result.excluded_pages == []
        assert result.blurry_ratio == 0.0

    def test_proceed_with_exclusions_when_some_blurry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "http://localhost:8000/mrc")
        monkeypatch.setattr(mrc_mod, "BLUR_THRESHOLD", 0.30)
        data = {
            "overall_readiness": True,
            "overall_confidence": 0.9,
            "total_pages": 10,
            "readable_pages": 8,
            "unreadable_pages": 2,
            "page_results": (
                [{"page": i, "readiness": True, "confidence": 0.9} for i in range(1, 9)]
                + [{"page": i, "readiness": False, "confidence": 0.1} for i in (9, 10)]
            ),
            "file_name": "doc.pdf",
            "inference_time_ms": 50.0,
        }
        with patch.object(mrc_mod, "_call_local", return_value=data):
            result = mrc_mod.check_readability(b"bytes", "doc.pdf")

        assert result.overall_status == "PROCEED_WITH_EXCLUSIONS"
        assert result.excluded_pages == [9, 10]
        assert result.readable_page_numbers == list(range(1, 9))

    def test_terminate_when_blurry_above_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "http://localhost:8000/mrc")
        monkeypatch.setattr(mrc_mod, "BLUR_THRESHOLD", 0.30)
        data = {
            "overall_readiness": False,
            "overall_confidence": 0.4,
            "total_pages": 10,
            "readable_pages": 5,
            "unreadable_pages": 5,
            "page_results": [
                {"page": i, "readiness": i <= 5, "confidence": 0.9 if i <= 5 else 0.1}
                for i in range(1, 11)
            ],
            "file_name": "doc.pdf",
            "inference_time_ms": 50.0,
        }
        with patch.object(mrc_mod, "_call_local", return_value=data):
            result = mrc_mod.check_readability(b"bytes", "doc.pdf")

        assert result.overall_status == "TERMINATE"
        assert result.blurry_ratio == 0.5

    def test_zero_total_pages_terminates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mrc_mod, "MRC_ENDPOINT", "http://localhost:8000/mrc")
        data = {
            "overall_readiness": False,
            "overall_confidence": 0.0,
            "total_pages": 0,
            "readable_pages": 0,
            "unreadable_pages": 0,
            "page_results": [],
            "file_name": "doc.pdf",
            "inference_time_ms": 0.0,
        }
        with patch.object(mrc_mod, "_call_local", return_value=data):
            result = mrc_mod.check_readability(b"bytes", "doc.pdf")

        assert result.overall_status == "TERMINATE"
        assert result.blurry_ratio == 1.0

    def test_routes_to_vertex_ai_when_aiplatform_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            mrc_mod, "MRC_ENDPOINT", "https://x-aiplatform.googleapis.com/v1/predict"
        )
        data = {
            "overall_readiness": True,
            "overall_confidence": 0.9,
            "total_pages": 1,
            "readable_pages": 1,
            "unreadable_pages": 0,
            "page_results": [{"page": 1, "readiness": True, "confidence": 0.9}],
            "file_name": "doc.pdf",
            "inference_time_ms": 1.0,
        }
        with (
            patch.object(mrc_mod, "_call_vertex_ai", return_value=data) as vertex_mock,
            patch.object(mrc_mod, "_call_local") as local_mock,
        ):
            mrc_mod.check_readability(b"b", "doc.pdf")
        vertex_mock.assert_called_once()
        local_mock.assert_not_called()
