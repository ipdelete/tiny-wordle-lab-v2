from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .game import Observation
from .policy import PolicyDescriptor
from .prompt import WordlePrompt


DEFAULT_API_BASE = "http://127.0.0.1:4000/v1"
DEFAULT_ENV_FILE = Path("~/src/wmd-router/.env").expanduser()
DEFAULT_MODEL = "gpt-oss-20b"
DEFAULT_PROMPT = WordlePrompt.from_path()

@dataclass
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    truncated_responses: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "truncated_responses": self.truncated_responses,
        }


def read_api_key(path: Path = DEFAULT_ENV_FILE) -> str:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "LITELLM_MASTER_KEY":
            key = value.strip().strip("\"'")
            if key:
                return key
    raise ValueError(f"LITELLM_MASTER_KEY is not set in {path}")


def render_observation(observation: Observation) -> str:
    if not observation.history:
        history = "No guesses have been made."
    else:
        history = "\n".join(
            f"{turn.guess.upper()} -> {turn.feedback}"
            for turn in observation.history
        )
    return (
        f"Game history:\n{history}\n\n"
        f"Opportunities remaining: {observation.remaining_opportunities}\n"
        "Choose the next guess."
    )


class OpenAIWordlePolicy:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        api_base: str = DEFAULT_API_BASE,
        temperature: float = 0,
        seed: int = 0,
        reasoning_effort: str = "low",
        max_tokens: int = 2_048,
        timeout_seconds: float = 180,
        prompt: WordlePrompt = DEFAULT_PROMPT,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout_seconds,
        )
        self._model = model
        self._api_base = api_base
        self._temperature = temperature
        self._seed = seed
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._prompt = prompt
        self.usage = UsageTotals()

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            "openai-snapshot",
            {
                "model": self._model,
                "api_base": self._api_base,
                "temperature": self._temperature,
                "seed": self._seed,
                "reasoning_effort": self._reasoning_effort,
                "max_tokens": self._max_tokens,
                "prompt_source": self._prompt.source,
                "prompt_sha256": self._prompt.sha256,
                "candidate_list_provided": False,
                "structured_output": False,
            },
        )

    def messages(self, observation: Observation) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._prompt.content},
            {"role": "user", "content": render_observation(observation)},
        ]

    def choose(self, observation: Observation) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self.messages(observation),
            temperature=self._temperature,
            seed=self._seed,
            reasoning_effort=self._reasoning_effort,
            max_tokens=self._max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage
        self.usage.requests += 1
        if usage is not None:
            self.usage.input_tokens += usage.prompt_tokens
            self.usage.output_tokens += usage.completion_tokens
            self.usage.total_tokens += usage.total_tokens
        if choice.finish_reason == "length":
            self.usage.truncated_responses += 1
        return choice.message.content or ""


LiteLLMPolicy = OpenAIWordlePolicy
