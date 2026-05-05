from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MaterialInfo(_message.Message):
    __slots__ = ("material_id", "file_name", "storage_path", "file_type", "readiness_status", "source", "source_url", "validation_reason_code", "validation_message")
    MATERIAL_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    STORAGE_PATH_FIELD_NUMBER: _ClassVar[int]
    FILE_TYPE_FIELD_NUMBER: _ClassVar[int]
    READINESS_STATUS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_URL_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_REASON_CODE_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    material_id: str
    file_name: str
    storage_path: str
    file_type: str
    readiness_status: str
    source: str
    source_url: str
    validation_reason_code: str
    validation_message: str
    def __init__(self, material_id: _Optional[str] = ..., file_name: _Optional[str] = ..., storage_path: _Optional[str] = ..., file_type: _Optional[str] = ..., readiness_status: _Optional[str] = ..., source: _Optional[str] = ..., source_url: _Optional[str] = ..., validation_reason_code: _Optional[str] = ..., validation_message: _Optional[str] = ...) -> None: ...

class GetMaterialsRequest(_message.Message):
    __slots__ = ("assessment_id", "unvalidated_only")
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    UNVALIDATED_ONLY_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    unvalidated_only: bool
    def __init__(self, assessment_id: _Optional[str] = ..., unvalidated_only: bool = ...) -> None: ...

class GetMaterialsResponse(_message.Message):
    __slots__ = ("materials",)
    MATERIALS_FIELD_NUMBER: _ClassVar[int]
    materials: _containers.RepeatedCompositeFieldContainer[MaterialInfo]
    def __init__(self, materials: _Optional[_Iterable[_Union[MaterialInfo, _Mapping]]] = ...) -> None: ...

class UpdateMaterialValidationRequest(_message.Message):
    __slots__ = ("assessment_id", "material_id", "readiness_status", "validation_reason_code", "validation_message")
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    MATERIAL_ID_FIELD_NUMBER: _ClassVar[int]
    READINESS_STATUS_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_REASON_CODE_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    material_id: str
    readiness_status: str
    validation_reason_code: str
    validation_message: str
    def __init__(self, assessment_id: _Optional[str] = ..., material_id: _Optional[str] = ..., readiness_status: _Optional[str] = ..., validation_reason_code: _Optional[str] = ..., validation_message: _Optional[str] = ...) -> None: ...

class UpdateMaterialValidationResponse(_message.Message):
    __slots__ = ("material_id", "readiness_status", "status")
    MATERIAL_ID_FIELD_NUMBER: _ClassVar[int]
    READINESS_STATUS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    material_id: str
    readiness_status: str
    status: str
    def __init__(self, material_id: _Optional[str] = ..., readiness_status: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...
