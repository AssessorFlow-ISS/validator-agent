"""Unit tests for the GrpcMaterialValidationAdapter.

All gRPC calls go through an injected ``SubmissionClient`` mock so
these tests require no real proto/grpc wiring.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from validator_agent.adapters.material_validation_grpc import (
    GrpcMaterialValidationAdapter,
    _from_proto_material,
)


def _make_proto(
    *,
    material_id: str = "m-1",
    file_name: str = "doc.pdf",
    storage_path: str = "path/doc.pdf",
    file_type: str = "pdf",
    readiness_status: str = "pending",
    source: str = "upload",
    source_url: str = "",
) -> MagicMock:
    m = MagicMock()
    m.material_id = material_id
    m.file_name = file_name
    m.storage_path = storage_path
    m.file_type = file_type
    m.readiness_status = readiness_status
    m.source = source
    m.source_url = source_url
    return m


def _make_client(materials: list[MagicMock] | None = None) -> MagicMock:
    client = MagicMock()
    client.close = AsyncMock()

    response = MagicMock()
    response.materials = materials or []
    client.get_materials = AsyncMock(return_value=response)

    update_resp = MagicMock()
    update_resp.readiness_status = "PROCEED"
    update_resp.status = "OK"
    client.update_material_validation = AsyncMock(return_value=update_resp)
    return client


class TestFromProtoMaterial:
    def test_maps_all_fields(self) -> None:
        proto = _make_proto(
            material_id="mid",
            file_name="f.pdf",
            storage_path="p",
            file_type="pdf",
            readiness_status="pending",
            source="upload",
            source_url="https://x",
        )
        mat = _from_proto_material(proto)
        assert mat.material_id == "mid"
        assert mat.file_name == "f.pdf"
        assert mat.storage_path == "p"
        assert mat.source_url == "https://x"


class TestGrpcMaterialValidationAdapter:
    async def test_get_materials_filters_by_source(self) -> None:
        client = _make_client(
            materials=[
                _make_proto(material_id="m1", source="upload"),
                _make_proto(material_id="m2", source="web_research"),
                _make_proto(material_id="m3", source="web_research"),
            ]
        )
        adapter = GrpcMaterialValidationAdapter(client=client)
        out = await adapter.get_materials(
            assessment_id="a-1",
            unvalidated_only=True,
            source="web_research",
        )
        assert len(out) == 2
        assert {m.material_id for m in out} == {"m2", "m3"}
        client.get_materials.assert_awaited_once()

    async def test_get_materials_no_source_filter(self) -> None:
        client = _make_client(materials=[_make_proto(), _make_proto(material_id="m2")])
        adapter = GrpcMaterialValidationAdapter(client=client)
        out = await adapter.get_materials(assessment_id="a-1")
        assert len(out) == 2

    async def test_update_material_validation_passes_through(self) -> None:
        client = _make_client()
        adapter = GrpcMaterialValidationAdapter(client=client)
        await adapter.update_material_validation(
            assessment_id="a-1",
            material_id="mid",
            readiness_status="PROCEED",
            validation_reason_code="VALIDATION_PASSED",
            validation_message="ok",
        )
        client.update_material_validation.assert_awaited_once_with(
            assessment_id="a-1",
            material_id="mid",
            readiness_status="PROCEED",
            validation_reason_code="VALIDATION_PASSED",
            validation_message="ok",
        )

    async def test_close_closes_underlying_client(self) -> None:
        client = _make_client()
        adapter = GrpcMaterialValidationAdapter(client=client)
        await adapter.close()
        client.close.assert_awaited_once()

    def test_default_client_is_instantiated_when_none_provided(self) -> None:
        # Should not raise — client is lazily created from env via SubmissionClient()
        adapter = GrpcMaterialValidationAdapter()
        assert adapter._client is not None
