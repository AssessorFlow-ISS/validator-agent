"""Unit tests for validator_agent.pipeline.llm_client.

Covers:
  - set_workflow_context + get_stats side-effects.
  - generate() routing to Model Broker when MODEL_BROKER_URL is set.
  - generate() fallback to OpenAI when MODEL_BROKER_URL is unset.
  - generate_structured() schema + parse.
  - _call_model_broker happy path with usage accumulation.
  - _call_openai_direct without API key returns empty content.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from validator_agent.pipeline import llm_client as llm_mod


class _Payload(BaseModel):
    answer: str


class TestContextAndStats:
    def test_set_and_get_workflow_context(self) -> None:
        llm_mod.set_workflow_context("wf-123")
        # ContextVar value is read inside generate(); accessible via getter
        assert llm_mod._workflow_id_var.get() == "wf-123"

    def test_get_stats_returns_same_accumulator(self) -> None:
        s1 = llm_mod.get_stats()
        s2 = llm_mod.get_stats()
        assert s1 is s2


class TestCallModelBroker:
    def test_routes_via_httpx_with_body_fields(
        self, monkeypatch: pytest.MonkeyPatch  # noqa: F821
    ) -> None:
        monkeypatch.setattr(llm_mod, "MODEL_BROKER_URL", "http://broker")
        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json.return_value = {
            "content": '{"answer": "ok"}',
            "model_used": "gemini-flash",
            "model_tier": "CHEAP",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cost_usd": 0.001,
            "latency_ms": 42,
        }
        fake_client_cm = MagicMock()
        fake_client = MagicMock()
        fake_client.post.return_value = fake_response
        fake_client_cm.__enter__.return_value = fake_client
        fake_client_cm.__exit__.return_value = False

        with patch.object(llm_mod.httpx, "Client", return_value=fake_client_cm):
            # Reset stats so accumulator assertions are deterministic
            llm_mod._stats.request_count = 0
            llm_mod._stats.total_tokens = 0

            resp = llm_mod._call_model_broker(
                task_key="validator.test",
                prompt="Hello",
                system_prompt="sys",
                max_tokens=100,
                temperature=0.0,
                response_format="json",
                response_schema={"type": "object"},
                prompt_version="v@1",
                session_id="wf-brok",
            )

        assert resp.content == '{"answer": "ok"}'
        assert resp.model_used == "gemini-flash"
        assert resp.model_tier == "CHEAP"
        assert resp.tokens_input == 10
        assert resp.tokens_output == 5
        assert resp.cost_usd == 0.001

        # Body contained optional keys
        body = fake_client.post.call_args[1]["json"]
        assert body["system_prompt"] == "sys"
        assert body["response_format"] == "json"
        assert body["response_schema"] == {"type": "object"}
        assert body["session_id"] == "wf-brok"

        # Stats were accumulated
        assert llm_mod._stats.request_count >= 1
        assert llm_mod._stats.total_tokens >= 15


class TestGenerateRouting:
    def test_generate_prefers_model_broker_when_url_set(
        self, monkeypatch: pytest.MonkeyPatch  # noqa: F821
    ) -> None:
        monkeypatch.setattr(llm_mod, "MODEL_BROKER_URL", "http://broker")
        stub = MagicMock(
            return_value=llm_mod.LlmResponse(
                content="OK", model_used="x", model_tier="HIGH"
            )
        )
        with patch.object(llm_mod, "_call_model_broker", stub):
            resp = llm_mod.generate(task_key="t", prompt="p")
        stub.assert_called_once()
        assert resp.content == "OK"

    def test_generate_falls_back_to_openai_when_no_url(
        self, monkeypatch: pytest.MonkeyPatch  # noqa: F821
    ) -> None:
        monkeypatch.setattr(llm_mod, "MODEL_BROKER_URL", "")
        fallback = MagicMock(return_value=llm_mod.LlmResponse(content="local"))
        with patch.object(llm_mod, "_call_openai_direct", fallback):
            resp = llm_mod.generate(task_key="t", prompt="p")
        fallback.assert_called_once()
        assert resp.content == "local"


class TestOpenAIDirect:
    def test_missing_api_key_returns_empty_json(
        self, monkeypatch: pytest.MonkeyPatch  # noqa: F821
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        resp = llm_mod._call_openai_direct(
            prompt="p",
            system_prompt=None,
            max_tokens=16,
            temperature=0.0,
            response_format=None,
            response_schema=None,
            task_key="validator.classifier",
        )
        assert resp.content == "{}"
        assert resp.model_used == "no-key"

    def test_with_api_key_uses_client(self, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F821
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        fake_choice = MagicMock()
        fake_choice.message.content = "hello"
        fake_usage = SimpleNamespace(prompt_tokens=3, completion_tokens=7)
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = fake_usage
        fake_response.model = "gpt-4o-mini"

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch("openai.OpenAI", return_value=fake_client):
            resp = llm_mod._call_openai_direct(
                prompt="p",
                system_prompt="sys",
                max_tokens=16,
                temperature=0.1,
                response_format="json",
                response_schema={"type": "object"},
                task_key="validator.classifier",  # routes to gpt-4o-mini
            )

        assert resp.content == "hello"
        assert resp.model_used == "gpt-4o-mini"
        assert resp.tokens_input == 3
        assert resp.tokens_output == 7
        # Verify response_format={"type":"json_object"} was passed to OpenAI
        kwargs = fake_client.chat.completions.create.call_args[1]
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_openai_none_content_returns_empty_object_literal(
        self, monkeypatch: pytest.MonkeyPatch  # noqa: F821
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fake_choice = MagicMock()
        fake_choice.message.content = None
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = None
        fake_response.model = "gpt-4o"

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response
        with patch("openai.OpenAI", return_value=fake_client):
            resp = llm_mod._call_openai_direct(
                prompt="p",
                system_prompt=None,
                max_tokens=8,
                temperature=0,
                response_format=None,
                response_schema=None,
                task_key="validator.reasoning",  # routes to gpt-4o
            )
        assert resp.content == "{}"
        # usage=None → zero-token counts
        assert resp.tokens_input == 0
        assert resp.tokens_output == 0


class TestGenerateStructured:
    def test_parses_pydantic_response(
        self, monkeypatch: pytest.MonkeyPatch  # noqa: F821
    ) -> None:
        # The schema cleaner is pulled in-line — patch generate() directly
        fake_resp = llm_mod.LlmResponse(content='{"answer": "yes"}')
        with patch.object(llm_mod, "generate", return_value=fake_resp):
            parsed, raw = llm_mod.generate_structured(
                task_key="t",
                system_prompt="sys",
                user_prompt="u",
                response_model=_Payload,
            )
        assert isinstance(parsed, _Payload)
        assert parsed.answer == "yes"
        assert raw is fake_resp
