"""Unit tests for validator_agent.main helpers.

Covers `_load_files_for_validation`, `_write_material_decisions`, and the
`/trigger` HTTP endpoint. The Pub/Sub lifespan path is covered indirectly
by api_client (which creates the stub-mode app with no subscription) and
by focused unit tests that exercise the build-service matrix.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from validator_agent.api.schemas import (
    FileResult,
    ValidationResponse,
)
from validator_agent.config import ValidatorConfig
from validator_agent.domain.terminal_signal import (
    ReasonCode,
    TerminalSignal,
    TerminalSignalStatus,
)
from validator_agent.main import (
    _READINESS_MAP,
    _build_service,
    _load_files_for_validation,
    _write_material_decisions,
    create_app,
)
from validator_agent.ports.material_validation_port import Material


def _material(
    *,
    material_id: str = "m-1",
    file_name: str = "doc.pdf",
    storage_path: str = "doc.pdf",
    file_type: str = "pdf",
    readiness_status: str = "pending",
    source: str = "upload",
    source_url: str = "",
) -> Material:
    return Material(
        material_id=material_id,
        file_name=file_name,
        storage_path=storage_path,
        file_type=file_type,
        readiness_status=readiness_status,
        source=source,
        source_url=source_url,
    )


def _make_signal(status: TerminalSignalStatus, code: ReasonCode) -> TerminalSignal:
    return TerminalSignal(status=status, reason_code=code, message="ok")


class _StubMatVal:
    def __init__(self, materials: list[Material] | Exception | None = None) -> None:
        self._materials = materials if materials is not None else []
        self.updates: list[dict] = []

    async def get_materials(
        self,
        *,
        assessment_id: str,
        unvalidated_only: bool = False,
        source: str | None = None,
    ) -> list[Material]:
        if isinstance(self._materials, Exception):
            raise self._materials
        mats = list(self._materials)
        if source:
            mats = [m for m in mats if m.source == source]
        return mats

    async def update_material_validation(
        self,
        *,
        assessment_id: str,
        material_id: str,
        readiness_status: str,
        validation_reason_code: str = "",
        validation_message: str = "",
    ) -> None:
        self.updates.append(
            {
                "assessment_id": assessment_id,
                "material_id": material_id,
                "readiness_status": readiness_status,
                "validation_reason_code": validation_reason_code,
                "validation_message": validation_message,
            }
        )


# ── _load_files_for_validation ──


class TestLoadFilesForValidation:
    async def test_inline_materials_fast_path_prefers_gcs_path(self) -> None:
        stub = _StubMatVal(materials=[])
        file_infos, mapping = await _load_files_for_validation(
            material_validation=stub,
            assessment_id="a-1",
            validation_type="material_validation",
            inline_materials=[
                {
                    "gcs_path": "gs://bucket/a.pdf",
                    "file_name": "a.pdf",
                    "file_type": "pdf",
                    "material_id": "mat-a",
                },
                {
                    "storage_path": "gs://bucket/b.pdf",
                    "blob_id": "blob-b",
                },
                # Skipped — not a dict
                "not-a-dict",
                # Skipped — no path
                {"file_name": "no-path.pdf"},
            ],
        )
        assert [f.storage_path for f in file_infos] == [
            "gs://bucket/a.pdf",
            "gs://bucket/b.pdf",
        ]
        assert file_infos[1].file_name == "b.pdf"  # derived via os.path.basename
        assert file_infos[1].file_type == "pdf"  # default
        assert mapping == {"a.pdf": "mat-a", "b.pdf": "blob-b"}

    async def test_inline_materials_empty_falls_through_to_grpc(self) -> None:
        stub = _StubMatVal(materials=[_material()])
        file_infos, mapping = await _load_files_for_validation(
            material_validation=stub,
            assessment_id="a-1",
            validation_type="material_validation",
            inline_materials=[],
        )
        assert len(file_infos) == 1
        assert mapping == {"doc.pdf": "m-1"}

    async def test_grpc_path_with_gcs_storage(self) -> None:
        stub = _StubMatVal(materials=[_material(storage_path="gs://bucket/real.pdf")])
        file_infos, mapping = await _load_files_for_validation(
            material_validation=stub,
            assessment_id="a-1",
            validation_type="material_validation",
        )
        assert file_infos[0].storage_path == "gs://bucket/real.pdf"

    async def test_grpc_path_with_web_research_source_joins_upload_dir(self) -> None:
        with patch.dict(os.environ, {"UPLOAD_DIR": "/tmp/x"}):
            stub = _StubMatVal(
                materials=[
                    _material(
                        source="web_research",
                        storage_path="webpage.pdf",
                    )
                ]
            )
            file_infos, _ = await _load_files_for_validation(
                material_validation=stub,
                assessment_id="a-1",
                validation_type="web_research_validation",
            )
        assert file_infos[0].storage_path == "/tmp/x/webpage.pdf"

    async def test_grpc_path_local_fallback_builds_prefixed_path(self) -> None:
        with patch.dict(os.environ, {"UPLOAD_DIR": "/tmp/y"}):
            stub = _StubMatVal(
                materials=[
                    _material(
                        material_id="mid",
                        file_name="raw.pdf",
                        storage_path="raw.pdf",
                        source="upload",
                    )
                ]
            )
            file_infos, _ = await _load_files_for_validation(
                material_validation=stub,
                assessment_id="a-1",
                validation_type="material_validation",
            )
        assert file_infos[0].storage_path == "/tmp/y/mid_raw.pdf"

    async def test_grpc_fetch_failure_returns_empty(self) -> None:
        stub = _StubMatVal(materials=RuntimeError("boom"))
        file_infos, mapping = await _load_files_for_validation(
            material_validation=stub,
            assessment_id="a-1",
            validation_type="material_validation",
        )
        assert file_infos == []
        assert mapping == {}

    async def test_material_without_file_type_defaults_to_pdf(self) -> None:
        stub = _StubMatVal(materials=[_material(file_type="")])
        file_infos, _ = await _load_files_for_validation(
            material_validation=stub,
            assessment_id="a-1",
            validation_type="material_validation",
        )
        assert file_infos[0].file_type == "pdf"

    async def test_material_without_material_id_not_mapped(self) -> None:
        stub = _StubMatVal(materials=[_material(material_id="")])
        _, mapping = await _load_files_for_validation(
            material_validation=stub,
            assessment_id="a-1",
            validation_type="material_validation",
        )
        assert mapping == {}


# ── _write_material_decisions ──


def _response_with(results: list[tuple[str, TerminalSignalStatus, ReasonCode]]) -> ValidationResponse:
    file_results = [
        FileResult(
            file_name=name,
            terminal_signal=_make_signal(status, code),
        )
        for name, status, code in results
    ]
    return ValidationResponse(
        workflow_id="wf-1",
        terminal_signal=_make_signal(TerminalSignalStatus.PROCEED, ReasonCode.VALIDATION_PASSED),
        file_results=file_results,
    )


class TestWriteMaterialDecisions:
    async def test_writes_one_per_mapped_file(self) -> None:
        stub = _StubMatVal()
        resp = _response_with(
            [
                ("a.pdf", TerminalSignalStatus.PROCEED, ReasonCode.VALIDATION_PASSED),
                ("b.pdf", TerminalSignalStatus.TERMINATE, ReasonCode.HARMFUL_CONTENT),
                ("c.pdf", TerminalSignalStatus.PROCEED_WITH_WARNINGS, ReasonCode.PII_DETECTED),
            ]
        )
        mapping = {"a.pdf": "mid-a", "b.pdf": "mid-b", "c.pdf": "mid-c"}
        await _write_material_decisions(
            material_validation=stub,
            assessment_id="a-1",
            response=resp,
            file_to_material_id=mapping,
        )
        assert len(stub.updates) == 3
        by_id = {u["material_id"]: u["readiness_status"] for u in stub.updates}
        assert by_id == {"mid-a": "PROCEED", "mid-b": "REJECT", "mid-c": "PARTIAL"}

    async def test_skips_unmapped_files(self) -> None:
        stub = _StubMatVal()
        resp = _response_with(
            [
                ("mapped.pdf", TerminalSignalStatus.PROCEED, ReasonCode.VALIDATION_PASSED),
                ("orphan.pdf", TerminalSignalStatus.PROCEED, ReasonCode.VALIDATION_PASSED),
            ]
        )
        await _write_material_decisions(
            material_validation=stub,
            assessment_id="a-1",
            response=resp,
            file_to_material_id={"mapped.pdf": "mid-1"},
        )
        assert [u["material_id"] for u in stub.updates] == ["mid-1"]

    async def test_rpc_failure_is_swallowed(self) -> None:
        """Failures are logged but do not raise."""
        stub = _StubMatVal()

        async def boom(**kwargs):
            raise RuntimeError("rpc down")

        stub.update_material_validation = boom  # type: ignore[assignment]
        resp = _response_with(
            [
                ("a.pdf", TerminalSignalStatus.PROCEED, ReasonCode.VALIDATION_PASSED),
            ]
        )
        # Should not raise
        await _write_material_decisions(
            material_validation=stub,
            assessment_id="a-1",
            response=resp,
            file_to_material_id={"a.pdf": "mid-a"},
        )

    def test_readiness_map_covers_all_statuses(self) -> None:
        assert _READINESS_MAP[TerminalSignalStatus.PROCEED] == "PROCEED"
        assert _READINESS_MAP[TerminalSignalStatus.PROCEED_WITH_WARNINGS] == "PARTIAL"
        assert _READINESS_MAP[TerminalSignalStatus.TERMINATE] == "REJECT"


# ── /trigger HTTP endpoint ──


@pytest.fixture(autouse=False)
def _reset_workflow_env():
    prior = os.environ.get("CURRENT_WORKFLOW_ID")
    yield
    if prior is None:
        os.environ.pop("CURRENT_WORKFLOW_ID", None)
    else:
        os.environ["CURRENT_WORKFLOW_ID"] = prior


class TestTriggerEndpoint:
    def test_trigger_with_no_materials_returns_no_materials_terminate(self, _reset_workflow_env) -> None:
        app = create_app()
        client = TestClient(app)
        # Stub material validation returns empty by default
        resp = client.post(
            "/trigger",
            json={
                "workflow_id": "wf-1",
                "assessment_id": "unseeded",
                "validation_type": "material_validation",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["terminal_signal"]["reason_code"] == "NO_MATERIALS"
        assert body["file_results"] == []
        assert body["source_agent"] == "validator-agent"

    def test_trigger_skips_terminate_on_stale_retrigger(self, _reset_workflow_env, monkeypatch) -> None:
        # Force the stub material adapter (default is grpc when env unset
        # the way our prod deploy reads it).
        monkeypatch.setenv("MATERIAL_ADAPTER", "stub")
        """When all materials carry readiness_status != NULL (already
        validated by a prior run), a duplicate trigger must NOT emit
        TERMINATE NO_MATERIALS — that would kill an in-flight downstream
        phase. Production repro: WF-9F1CF1 + WF-18AF1C.

        Returns STALE_RETRIGGER status with reason_code ALREADY_VALIDATED
        instead, which the orchestrator treats as a no-op.
        """
        from validator_agent.adapters.material_validation_stub import (
            StubMaterialValidationAdapter,
        )
        from validator_agent.ports.material_validation_port import Material

        seeded_materials = [
            Material(
                material_id="m-already-validated",
                file_name="scrum.pdf",
                file_type="pdf",
                storage_path="gs://bucket/scrum.pdf",
                readiness_status="PROCEED",
                source="upload",
                source_url="",
            ),
        ]

        original_init = StubMaterialValidationAdapter.__init__

        def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            original_init(self, *args, **kwargs)
            if not self.materials:
                self.materials = list(seeded_materials)

        monkeypatch.setattr(StubMaterialValidationAdapter, "__init__", patched_init)

        app = create_app()
        client = TestClient(app)
        resp = client.post(
            "/trigger",
            json={
                "workflow_id": "wf-stale",
                "assessment_id": "assess-stale",
                "validation_type": "material_validation",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["terminal_signal"]["status"] == "STALE_RETRIGGER"
        assert body["terminal_signal"]["reason_code"] == "ALREADY_VALIDATED"
        assert body["file_results"] == []

    def test_trigger_with_inline_materials_runs_pipeline(self, _reset_workflow_env) -> None:
        app = create_app()
        client = TestClient(app)
        resp = client.post(
            "/trigger",
            json={
                "workflow_id": "wf-inline",
                "assessment_id": "a-inline",
                "validation_type": "material_validation",
                "materials": [
                    {
                        "gcs_path": "gs://bucket/inline.pdf",
                        "file_name": "inline.pdf",
                        "file_type": "pdf",
                    }
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "terminal_signal" in body
        assert body["workflow_id"] == "wf-inline"


# ── _build_service ──


class TestBuildService:
    def _config(self, **overrides) -> ValidatorConfig:
        base = dict(
            mrc_adapter="stub",
            ocr_adapter="stub",
            llm_provider="stub",
            knowledge_adapter="stub",
            audit_adapter="stub",
            event_adapter="stub",
            material_adapter="stub",
            storage_adapter="stub",
        )
        base.update(overrides)
        return ValidatorConfig(**base)

    def test_stub_config_builds_default_service(self) -> None:
        service, ev, da, ks, mv = _build_service(self._config())
        assert service is not None
        # stub adapters shouldn't need real backends
        assert type(ev).__name__ == "StubEventPublisherAdapter"
        assert type(mv).__name__ == "StubMaterialValidationAdapter"

    def test_unknown_event_adapter_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown EVENT_ADAPTER"):
            _build_service(self._config(event_adapter="unknown"))

    def test_local_storage_selected_when_configured(self) -> None:
        service, *_ = _build_service(self._config(storage_adapter="local"))
        assert service is not None

    def test_pubsub_event_branch(self) -> None:
        with patch("validator_agent.adapters.pubsub_event_publisher.PubSubEventPublisherAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            service, ev, *_ = _build_service(self._config(event_adapter="pubsub"))
        assert service is not None

    def test_grpc_material_adapter_branch(self) -> None:
        with patch("validator_agent.adapters.material_validation_grpc.GrpcMaterialValidationAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            service, *_ = _build_service(self._config(material_adapter="grpc"))
        assert service is not None

    def test_gcs_storage_branch(self) -> None:
        with patch("validator_agent.adapters.storage_gcs.GcsStorageAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            service, *_ = _build_service(self._config(storage_adapter="gcs"))
        assert service is not None
