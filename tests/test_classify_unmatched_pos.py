from __future__ import annotations

import importlib.util
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "classify_unmatched_pos",
    "scripts/classify_unmatched_pos.py",
)
assert SPEC and SPEC.loader
classify_unmatched_pos = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classify_unmatched_pos)


def test_classify_batch_uses_openai_client(monkeypatch) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"classifications":[{"parts_of_speech":["noun"]}]}'
                )
            )
        ]
    )
    create_calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **payload: create_calls.append(payload) or response
                )
            )

    monkeypatch.setattr(classify_unmatched_pos, "OpenAI", FakeOpenAI)

    assert classify_unmatched_pos.classify_batch(
        ["crane"],
        api_key="secret",
        gateway_url="http://localhost/v1/chat/completions",
    ) == [{"word": "crane", "parts_of_speech": ["noun"]}]
    assert create_calls[0]["extra_body"]["thinking_budget"] == 0
