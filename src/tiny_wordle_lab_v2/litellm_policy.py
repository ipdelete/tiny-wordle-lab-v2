from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx2 as httpx
from agents import (
    Agent,
    ModelSettings,
    Runner,
    SQLiteSession,
    set_tracing_disabled,
)
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI, OpenAI
from openai.types.shared import Reasoning

from .game import Observation, parse_word
from .policy import PolicyDescriptor


DEFAULT_API_BASE = "http://127.0.0.1:4000/v1"
DEFAULT_ENV_FILE = Path("~/src/wmd-router/.env").expanduser()
DEFAULT_MODEL = "gpt-oss-20b"
PROMPT_VERSION = 2

SYSTEM_PROMPT = """Play Wordle.

Reply with exactly one lowercase five-letter English word and nothing else.
Do not use punctuation or explain your choice.
Never repeat a previous guess.

Feedback marks:
G = correct letter in the correct position
Y = correct letter in the wrong position
B = this occurrence of the letter is not matched

Use all previous guesses and feedback to choose the next guess."""

INITIAL_USER_MESSAGE = "No guesses have been made. Choose the first guess."


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


def transcript_messages(observation: Observation) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    next_history = 0
    total_actions = len(observation.previous_actions)

    if total_actions == 0:
        messages.append({"role": "user", "content": INITIAL_USER_MESSAGE})
        return messages

    messages.append({"role": "user", "content": INITIAL_USER_MESSAGE})
    for index, raw_output in enumerate(observation.previous_actions):
        messages.append({"role": "assistant", "content": raw_output})
        parsed = parse_word(raw_output)
        accepted = (
            next_history < len(observation.history)
            and parsed == observation.history[next_history].guess
        )
        remaining = observation.remaining_opportunities + total_actions - index - 1
        if accepted:
            turn = observation.history[next_history]
            next_history += 1
            content = (
                f"{turn.guess.upper()} produced {turn.feedback}. "
                f"Opportunities remaining: {remaining}. Choose the next guess."
            )
        else:
            content = (
                f"Your response {raw_output!r} was not a legal Wordle guess. "
                f"Opportunities remaining: {remaining}. Choose the next guess."
            )
        messages.append({"role": "user", "content": content})

    if next_history != len(observation.history):
        raise ValueError("observation history could not be aligned with policy outputs")
    return messages


class OpenAIWordlePolicy:
    def __init__(
        self,
        *,
        api_key: str,
        context_mode: Literal["snapshot", "transcript"],
        model: str = DEFAULT_MODEL,
        api_base: str = DEFAULT_API_BASE,
        temperature: float = 0,
        seed: int = 0,
        reasoning_effort: str = "low",
        max_tokens: int = 2_048,
        timeout_seconds: float = 180,
        capture_requests: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if context_mode not in ("snapshot", "transcript"):
            raise ValueError(f"unsupported context_mode: {context_mode}")
        self.request_bodies: list[dict] = []

        def capture_request(request: httpx.Request) -> None:
            if capture_requests and request.content:
                self.request_bodies.append(json.loads(request.content))

        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout_seconds,
            http_client=httpx.Client(
                event_hooks={"request": [capture_request]},
            ),
        )
        self._context_mode = context_mode
        self._model = model
        self._api_base = api_base
        self._temperature = temperature
        self._seed = seed
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self.usage = UsageTotals()

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            f"openai-{self._context_mode}",
            {
                "model": self._model,
                "api_base": self._api_base,
                "temperature": self._temperature,
                "seed": self._seed,
                "reasoning_effort": self._reasoning_effort,
                "max_tokens": self._max_tokens,
                "prompt_version": PROMPT_VERSION,
                "candidate_list_provided": False,
                "structured_output": False,
            },
        )

    def messages(self, observation: Observation) -> list[dict[str, str]]:
        if self._context_mode == "snapshot":
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_observation(observation)},
            ]
        return transcript_messages(observation)

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


class AgentsSessionWordlePolicy:
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
        capture_requests: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        set_tracing_disabled(True)
        self.request_bodies: list[dict] = []
        self.usage = UsageTotals()

        async def capture_request(request: httpx.Request) -> None:
            if capture_requests and request.content:
                self.request_bodies.append(json.loads(request.content))

        async def capture_response(response: httpx.Response) -> None:
            await response.aread()
            if response.content:
                body = json.loads(response.content)
                if body.get("choices", [{}])[0].get("finish_reason") == "length":
                    self.usage.truncated_responses += 1

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout_seconds,
            http_client=httpx.AsyncClient(
                event_hooks={
                    "request": [capture_request],
                    "response": [capture_response],
                },
            ),
        )
        chat_model = OpenAIChatCompletionsModel(
            model=model,
            openai_client=client,
            should_replay_reasoning_content=lambda _context: False,
        )
        self._agent = Agent(
            name="Wordle player",
            instructions=SYSTEM_PROMPT,
            model=chat_model,
            model_settings=ModelSettings(
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=Reasoning(effort=reasoning_effort),
                extra_body={"seed": seed},
            ),
        )
        self._model = model
        self._api_base = api_base
        self._temperature = temperature
        self._seed = seed
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._session: SQLiteSession | None = None
        self._game_number = 0
        self._prior_action_count = 0

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            "agents-session",
            {
                "model": self._model,
                "api_base": self._api_base,
                "temperature": self._temperature,
                "seed": self._seed,
                "reasoning_effort": self._reasoning_effort,
                "max_tokens": self._max_tokens,
                "prompt_version": PROMPT_VERSION,
                "candidate_list_provided": False,
                "structured_output": False,
                "reasoning_replay": False,
            },
        )

    def _new_game(self) -> None:
        self._game_number += 1
        self._session = SQLiteSession(f"wordle-{self._game_number}")
        self._prior_action_count = 0

    def choose(self, observation: Observation) -> str:
        if not observation.previous_actions:
            self._new_game()
        elif len(observation.previous_actions) != self._prior_action_count:
            raise ValueError("Agents session is out of sync with the evaluator")
        if self._session is None:
            raise RuntimeError("Agents session was not initialized")

        messages = transcript_messages(observation)
        user_input = messages[-1]["content"]
        result = Runner.run_sync(
            self._agent,
            user_input,
            session=self._session,
            max_turns=1,
        )
        run_usage = result.context_wrapper.usage
        self.usage.requests += run_usage.requests
        self.usage.input_tokens += run_usage.input_tokens
        self.usage.output_tokens += run_usage.output_tokens
        self.usage.total_tokens += run_usage.total_tokens
        output = result.final_output
        if not isinstance(output, str):
            raise ValueError("Agents SDK final output was not a string")
        self._prior_action_count = len(observation.previous_actions) + 1
        return output


LiteLLMPolicy = OpenAIWordlePolicy
