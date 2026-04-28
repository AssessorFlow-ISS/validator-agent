from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AssessmentConfig(_message.Message):
    __slots__ = ("assessment_id", "workflow_id", "assessor_id", "assessment_title", "purpose", "duration_minutes", "difficulty_level", "structured_question_count", "non_structured_question_count", "web_research_mode", "status", "deadline")
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSOR_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_TITLE_FIELD_NUMBER: _ClassVar[int]
    PURPOSE_FIELD_NUMBER: _ClassVar[int]
    DURATION_MINUTES_FIELD_NUMBER: _ClassVar[int]
    DIFFICULTY_LEVEL_FIELD_NUMBER: _ClassVar[int]
    STRUCTURED_QUESTION_COUNT_FIELD_NUMBER: _ClassVar[int]
    NON_STRUCTURED_QUESTION_COUNT_FIELD_NUMBER: _ClassVar[int]
    WEB_RESEARCH_MODE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    workflow_id: str
    assessor_id: str
    assessment_title: str
    purpose: str
    duration_minutes: int
    difficulty_level: str
    structured_question_count: int
    non_structured_question_count: int
    web_research_mode: str
    status: str
    deadline: str
    def __init__(self, assessment_id: _Optional[str] = ..., workflow_id: _Optional[str] = ..., assessor_id: _Optional[str] = ..., assessment_title: _Optional[str] = ..., purpose: _Optional[str] = ..., duration_minutes: _Optional[int] = ..., difficulty_level: _Optional[str] = ..., structured_question_count: _Optional[int] = ..., non_structured_question_count: _Optional[int] = ..., web_research_mode: _Optional[str] = ..., status: _Optional[str] = ..., deadline: _Optional[str] = ...) -> None: ...

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

class Question(_message.Message):
    __slots__ = ("question_id", "question_type", "content", "structured_answer", "non_structured_model_answer", "metadata_json", "topic_id", "iteration", "sort_order")
    QUESTION_ID_FIELD_NUMBER: _ClassVar[int]
    QUESTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    STRUCTURED_ANSWER_FIELD_NUMBER: _ClassVar[int]
    NON_STRUCTURED_MODEL_ANSWER_FIELD_NUMBER: _ClassVar[int]
    METADATA_JSON_FIELD_NUMBER: _ClassVar[int]
    TOPIC_ID_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    SORT_ORDER_FIELD_NUMBER: _ClassVar[int]
    question_id: str
    question_type: str
    content: str
    structured_answer: str
    non_structured_model_answer: str
    metadata_json: str
    topic_id: str
    iteration: int
    sort_order: int
    def __init__(self, question_id: _Optional[str] = ..., question_type: _Optional[str] = ..., content: _Optional[str] = ..., structured_answer: _Optional[str] = ..., non_structured_model_answer: _Optional[str] = ..., metadata_json: _Optional[str] = ..., topic_id: _Optional[str] = ..., iteration: _Optional[int] = ..., sort_order: _Optional[int] = ...) -> None: ...

class EvaluationDetail(_message.Message):
    __slots__ = ("question_id", "group_evaluation_id", "score", "max_score", "reasoning", "evaluation_method")
    QUESTION_ID_FIELD_NUMBER: _ClassVar[int]
    GROUP_EVALUATION_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    MAX_SCORE_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_METHOD_FIELD_NUMBER: _ClassVar[int]
    question_id: str
    group_evaluation_id: str
    score: float
    max_score: float
    reasoning: str
    evaluation_method: str
    def __init__(self, question_id: _Optional[str] = ..., group_evaluation_id: _Optional[str] = ..., score: _Optional[float] = ..., max_score: _Optional[float] = ..., reasoning: _Optional[str] = ..., evaluation_method: _Optional[str] = ...) -> None: ...

class WebResearchFile(_message.Message):
    __slots__ = ("file_name", "storage_path", "file_type", "source_url")
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    STORAGE_PATH_FIELD_NUMBER: _ClassVar[int]
    FILE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_URL_FIELD_NUMBER: _ClassVar[int]
    file_name: str
    storage_path: str
    file_type: str
    source_url: str
    def __init__(self, file_name: _Optional[str] = ..., storage_path: _Optional[str] = ..., file_type: _Optional[str] = ..., source_url: _Optional[str] = ...) -> None: ...

class GetAssessmentConfigRequest(_message.Message):
    __slots__ = ("assessment_id", "workflow_id")
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_ID_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    workflow_id: str
    def __init__(self, assessment_id: _Optional[str] = ..., workflow_id: _Optional[str] = ...) -> None: ...

class GetAssessmentConfigResponse(_message.Message):
    __slots__ = ("config",)
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    config: AssessmentConfig
    def __init__(self, config: _Optional[_Union[AssessmentConfig, _Mapping]] = ...) -> None: ...

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

class CreateQuestionSetRequest(_message.Message):
    __slots__ = ("workflow_id",)
    WORKFLOW_ID_FIELD_NUMBER: _ClassVar[int]
    workflow_id: str
    def __init__(self, workflow_id: _Optional[str] = ...) -> None: ...

class CreateQuestionSetResponse(_message.Message):
    __slots__ = ("question_set_id", "status")
    QUESTION_SET_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    question_set_id: str
    status: str
    def __init__(self, question_set_id: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class WriteGeneratedQuestionsRequest(_message.Message):
    __slots__ = ("question_set_id", "questions")
    QUESTION_SET_ID_FIELD_NUMBER: _ClassVar[int]
    QUESTIONS_FIELD_NUMBER: _ClassVar[int]
    question_set_id: str
    questions: _containers.RepeatedCompositeFieldContainer[Question]
    def __init__(self, question_set_id: _Optional[str] = ..., questions: _Optional[_Iterable[_Union[Question, _Mapping]]] = ...) -> None: ...

class WriteGeneratedQuestionsResponse(_message.Message):
    __slots__ = ("questions_written", "status")
    QUESTIONS_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    questions_written: int
    status: str
    def __init__(self, questions_written: _Optional[int] = ..., status: _Optional[str] = ...) -> None: ...

class GetGeneratedQuestionsRequest_(_message.Message):
    __slots__ = ("question_set_id",)
    QUESTION_SET_ID_FIELD_NUMBER: _ClassVar[int]
    question_set_id: str
    def __init__(self, question_set_id: _Optional[str] = ...) -> None: ...

class GetGeneratedQuestionsResponse(_message.Message):
    __slots__ = ("questions",)
    QUESTIONS_FIELD_NUMBER: _ClassVar[int]
    questions: _containers.RepeatedCompositeFieldContainer[Question]
    def __init__(self, questions: _Optional[_Iterable[_Union[Question, _Mapping]]] = ...) -> None: ...

class IncrementIterationRequest(_message.Message):
    __slots__ = ("question_set_id",)
    QUESTION_SET_ID_FIELD_NUMBER: _ClassVar[int]
    question_set_id: str
    def __init__(self, question_set_id: _Optional[str] = ...) -> None: ...

class IncrementIterationResponse(_message.Message):
    __slots__ = ("question_set_id", "iteration_count", "status")
    QUESTION_SET_ID_FIELD_NUMBER: _ClassVar[int]
    ITERATION_COUNT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    question_set_id: str
    iteration_count: int
    status: str
    def __init__(self, question_set_id: _Optional[str] = ..., iteration_count: _Optional[int] = ..., status: _Optional[str] = ...) -> None: ...

class GetApprovedQuestionsRequest(_message.Message):
    __slots__ = ("assessment_id",)
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    def __init__(self, assessment_id: _Optional[str] = ...) -> None: ...

class GetApprovedQuestionsResponse(_message.Message):
    __slots__ = ("questions",)
    QUESTIONS_FIELD_NUMBER: _ClassVar[int]
    questions: _containers.RepeatedCompositeFieldContainer[Question]
    def __init__(self, questions: _Optional[_Iterable[_Union[Question, _Mapping]]] = ...) -> None: ...

class CreateEvaluationRequest(_message.Message):
    __slots__ = ("workflow_id", "participant_id", "submission_id", "total_score", "max_score", "details")
    WORKFLOW_ID_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_ID_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SCORE_FIELD_NUMBER: _ClassVar[int]
    MAX_SCORE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    workflow_id: str
    participant_id: str
    submission_id: str
    total_score: float
    max_score: float
    details: _containers.RepeatedCompositeFieldContainer[EvaluationDetail]
    def __init__(self, workflow_id: _Optional[str] = ..., participant_id: _Optional[str] = ..., submission_id: _Optional[str] = ..., total_score: _Optional[float] = ..., max_score: _Optional[float] = ..., details: _Optional[_Iterable[_Union[EvaluationDetail, _Mapping]]] = ...) -> None: ...

class CreateEvaluationResponse(_message.Message):
    __slots__ = ("evaluation_id", "status")
    EVALUATION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    evaluation_id: str
    status: str
    def __init__(self, evaluation_id: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class CreateGroupEvaluationRequest(_message.Message):
    __slots__ = ("workflow_id", "group_id", "question_id", "group_score", "max_score", "reasoning")
    WORKFLOW_ID_FIELD_NUMBER: _ClassVar[int]
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    QUESTION_ID_FIELD_NUMBER: _ClassVar[int]
    GROUP_SCORE_FIELD_NUMBER: _ClassVar[int]
    MAX_SCORE_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    workflow_id: str
    group_id: str
    question_id: str
    group_score: float
    max_score: float
    reasoning: str
    def __init__(self, workflow_id: _Optional[str] = ..., group_id: _Optional[str] = ..., question_id: _Optional[str] = ..., group_score: _Optional[float] = ..., max_score: _Optional[float] = ..., reasoning: _Optional[str] = ...) -> None: ...

class CreateGroupEvaluationResponse(_message.Message):
    __slots__ = ("group_evaluation_id", "status")
    GROUP_EVALUATION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    group_evaluation_id: str
    status: str
    def __init__(self, group_evaluation_id: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class GetEvaluationRequest(_message.Message):
    __slots__ = ("workflow_id", "participant_id")
    WORKFLOW_ID_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_ID_FIELD_NUMBER: _ClassVar[int]
    workflow_id: str
    participant_id: str
    def __init__(self, workflow_id: _Optional[str] = ..., participant_id: _Optional[str] = ...) -> None: ...

class GetEvaluationResponse(_message.Message):
    __slots__ = ("evaluation_id", "total_score", "max_score", "status", "details")
    EVALUATION_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SCORE_FIELD_NUMBER: _ClassVar[int]
    MAX_SCORE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    evaluation_id: str
    total_score: float
    max_score: float
    status: str
    details: _containers.RepeatedCompositeFieldContainer[EvaluationDetail]
    def __init__(self, evaluation_id: _Optional[str] = ..., total_score: _Optional[float] = ..., max_score: _Optional[float] = ..., status: _Optional[str] = ..., details: _Optional[_Iterable[_Union[EvaluationDetail, _Mapping]]] = ...) -> None: ...

class CreateReportRequest(_message.Message):
    __slots__ = ("workflow_id", "participant_id", "evaluation_id", "report_content_json")
    WORKFLOW_ID_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_ID_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_ID_FIELD_NUMBER: _ClassVar[int]
    REPORT_CONTENT_JSON_FIELD_NUMBER: _ClassVar[int]
    workflow_id: str
    participant_id: str
    evaluation_id: str
    report_content_json: str
    def __init__(self, workflow_id: _Optional[str] = ..., participant_id: _Optional[str] = ..., evaluation_id: _Optional[str] = ..., report_content_json: _Optional[str] = ...) -> None: ...

class CreateReportResponse(_message.Message):
    __slots__ = ("report_id", "status")
    REPORT_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    report_id: str
    status: str
    def __init__(self, report_id: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class UploadWebResearchRequest(_message.Message):
    __slots__ = ("assessment_id", "files")
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    files: _containers.RepeatedCompositeFieldContainer[WebResearchFile]
    def __init__(self, assessment_id: _Optional[str] = ..., files: _Optional[_Iterable[_Union[WebResearchFile, _Mapping]]] = ...) -> None: ...

class UploadWebResearchResponse(_message.Message):
    __slots__ = ("materials_registered", "status", "material_ids")
    MATERIALS_REGISTERED_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MATERIAL_IDS_FIELD_NUMBER: _ClassVar[int]
    materials_registered: int
    status: str
    material_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, materials_registered: _Optional[int] = ..., status: _Optional[str] = ..., material_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class StartWorkflowRequest(_message.Message):
    __slots__ = ("assessment_id",)
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    def __init__(self, assessment_id: _Optional[str] = ...) -> None: ...

class StartWorkflowResponse(_message.Message):
    __slots__ = ("workflow_id", "correlation_id", "status")
    WORKFLOW_ID_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    workflow_id: str
    correlation_id: str
    status: str
    def __init__(self, workflow_id: _Optional[str] = ..., correlation_id: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

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
