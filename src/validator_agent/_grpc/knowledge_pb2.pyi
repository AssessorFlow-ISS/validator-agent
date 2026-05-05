from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ProcessMaterialRequest(_message.Message):
    __slots__ = ("workflow_id", "content_text", "source_type", "source_file", "source_url", "assessor_id", "assessment_id")
    WORKFLOW_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TEXT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FILE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_URL_FIELD_NUMBER: _ClassVar[int]
    ASSESSOR_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    workflow_id: str
    content_text: str
    source_type: str
    source_file: str
    source_url: str
    assessor_id: str
    assessment_id: str
    def __init__(self, workflow_id: _Optional[str] = ..., content_text: _Optional[str] = ..., source_type: _Optional[str] = ..., source_file: _Optional[str] = ..., source_url: _Optional[str] = ..., assessor_id: _Optional[str] = ..., assessment_id: _Optional[str] = ...) -> None: ...

class ProcessMaterialResponse(_message.Message):
    __slots__ = ("chunks_created", "status", "chunk_ids")
    CHUNKS_CREATED_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_IDS_FIELD_NUMBER: _ClassVar[int]
    chunks_created: int
    status: str
    chunk_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, chunks_created: _Optional[int] = ..., status: _Optional[str] = ..., chunk_ids: _Optional[_Iterable[str]] = ...) -> None: ...
