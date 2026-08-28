from __future__ import annotations

from types import SimpleNamespace

from tiny_wordle_lab_v2.game import Observation, Turn
from tiny_wordle_lab_v2.litellm_policy import (
    DEFAULT_PROMPT,
    OpenAIWordlePolicy,
    render_observation,
)


def observation() -> Observation:
    return Observation(
        history=(Turn("crane", "BBYBB"),),
        previous_actions=("crane",),
        remaining_opportunities=5,
        candidates=("light", "pilot"),
    )


def test_render_observation_contains_only_public_game_state() -> None:
    prompt = render_observation(observation())
    assert "CRANE -> BBYBB" in prompt
    assert "Opportunities remaining: 5" in prompt
    assert "light" not in prompt
    assert "pilot" not in prompt


def test_openai_policy_returns_raw_content_and_records_usage(monkeypatch) -> None:
    policy = OpenAIWordlePolicy(api_key="secret")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="CRANE"),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )
    monkeypatch.setattr(
        policy._client.chat.completions,
        "create",
        lambda **_kwargs: response,
    )

    assert policy.choose(observation()) == "CRANE"
    assert policy.usage.as_dict() == {
        "requests": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "truncated_responses": 0,
    }


def test_descriptor_does_not_expose_api_key() -> None:
    policy = OpenAIWordlePolicy(api_key="secret")
    assert "secret" not in repr(policy.descriptor)
    assert policy.descriptor.parameters["prompt_sha256"] == DEFAULT_PROMPT.sha256
