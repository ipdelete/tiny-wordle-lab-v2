from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from .rollout import PolicyDecision

MASK_VERSION = "answer-token-trie-v1"


def digest_action_vocabulary(answers: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(answers).encode()).hexdigest()


def digest_tokenizer(tokenizer: Any) -> str:
    vocabulary = tokenizer.get_vocab()
    payload = {
        "vocab": sorted(vocabulary.items()),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class _TrieNode:
    children: dict[int, _TrieNode]
    word: str | None = None


class TokenTrie:
    def __init__(
        self,
        sequences: Mapping[str, Sequence[int]],
        *,
        eos_token_id: int | None = None,
    ) -> None:
        if not sequences:
            raise ValueError("token trie requires at least one word")
        self.root = _TrieNode(children={})
        self._sequences: dict[str, tuple[int, ...]] = {}
        self._words_by_sequence: dict[tuple[int, ...], str] = {}
        self.eos_token_id = eos_token_id
        for word, sequence in sequences.items():
            self._add(word, tuple(int(token) for token in sequence))

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        answers: Sequence[str],
        *,
        rendered_prompt: str | None = None,
    ) -> TokenTrie:
        sequences = build_word_token_sequences(
            tokenizer,
            answers,
            rendered_prompt=rendered_prompt,
        )
        return cls(
            sequences,
            eos_token_id=int(tokenizer.eos_token_id),
        )

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(self._sequences)

    @property
    def sequences(self) -> Mapping[str, tuple[int, ...]]:
        return dict(self._sequences)

    def sequence_for_word(self, word: str) -> tuple[int, ...]:
        try:
            return self._sequences[word]
        except KeyError as error:
            raise ValueError(f"word is not in token trie: {word}") from error

    def word_for_sequence(self, sequence: Sequence[int]) -> str:
        key = tuple(int(token) for token in sequence)
        try:
            return self._words_by_sequence[key]
        except KeyError as error:
            raise ValueError("token sequence is not a trie leaf") from error

    def node_for_prefix(self, prefix: Sequence[int]) -> _TrieNode:
        node = self.root
        for token in prefix:
            try:
                node = node.children[int(token)]
            except KeyError as error:
                raise ValueError("prefix is not in token trie") from error
        return node

    def allowed_tokens(self, prefix: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted(self.node_for_prefix(prefix).children))

    def branching_prefixes_by_depth(self) -> dict[int, tuple[tuple[int, ...], ...]]:
        by_depth: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        stack = [((), self.root)]
        while stack:
            prefix, node = stack.pop()
            if len(node.children) > 1:
                by_depth[len(prefix)].append(prefix)
            for token, child in node.children.items():
                stack.append((prefix + (token,), child))
        return {
            depth: tuple(sorted(prefixes))
            for depth, prefixes in sorted(by_depth.items())
        }

    def shape(self) -> dict[str, Any]:
        total_nodes = 0
        internal_nodes = 0
        leaf_count = 0
        maximum_depth = 0
        branching_by_depth: dict[int, int] = defaultdict(int)
        stack = [((), self.root)]
        while stack:
            prefix, node = stack.pop()
            total_nodes += 1
            maximum_depth = max(maximum_depth, len(prefix))
            if node.children:
                internal_nodes += 1
            if len(node.children) > 1:
                branching_by_depth[len(prefix)] += 1
            if node.word is not None:
                leaf_count += 1
            for token, child in node.children.items():
                stack.append((prefix + (token,), child))
        return {
            "total_nodes": total_nodes,
            "internal_nodes": internal_nodes,
            "leaf_count": leaf_count,
            "branching_nodes": sum(branching_by_depth.values()),
            "branching_by_depth": dict(sorted(branching_by_depth.items())),
            "maximum_depth": maximum_depth,
            "edge_count": total_nodes - 1,
            "token_positions": sum(
                len(sequence) for sequence in self._sequences.values()
            ),
        }

    def _add(self, word: str, sequence: tuple[int, ...]) -> None:
        if not sequence:
            raise ValueError("trie sequences must not be empty")
        if self.eos_token_id is not None and sequence[-1] != self.eos_token_id:
            raise ValueError(f"sequence for {word} does not end in EOS")
        if word in self._sequences:
            raise ValueError(f"duplicate trie word: {word}")
        if sequence in self._words_by_sequence:
            raise ValueError(f"duplicate trie sequence for {word}")
        self._sequences[word] = sequence
        self._words_by_sequence[sequence] = word
        node = self.root
        for token in sequence:
            node = node.children.setdefault(token, _TrieNode(children={}))
        if node.word is not None:
            raise ValueError("duplicate trie leaf")
        node.word = word


def _encoded_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        values = encoded["input_ids"]
    else:
        values = encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def build_word_token_sequences(
    tokenizer: Any,
    answers: Sequence[str],
    *,
    rendered_prompt: str | None = None,
) -> dict[str, tuple[int, ...]]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    eos_token_id = int(eos_token_id)
    eos_text = tokenizer.eos_token
    if eos_text is None:
        raise ValueError("tokenizer must define eos_token")

    prompt_ids = (
        _encoded_ids(tokenizer, rendered_prompt)
        if rendered_prompt is not None
        else None
    )
    sequences: dict[str, tuple[int, ...]] = {}
    for word in answers:
        standalone = tuple(
            _encoded_ids(tokenizer, word) + [eos_token_id]
        )
        if rendered_prompt is not None:
            contextual = tuple(
                _encoded_ids(tokenizer, rendered_prompt + word + eos_text)
            )
            suffix = contextual[len(prompt_ids):]
            if suffix != standalone:
                raise ValueError(
                    f"contextual tokenizer suffix differs for {word}"
                )
            decoded = tokenizer.decode(
                standalone[:-1],
                skip_special_tokens=True,
            )
            if decoded != word:
                raise ValueError(
                    f"tokenizer does not decode {word!r} exactly: {decoded!r}"
                )
        if eos_token_id in standalone[:-1]:
            raise ValueError(f"word contains EOS internally: {word}")
        sequences[word] = standalone

    if len(sequences) != len(set(sequences.values())):
        raise ValueError("answer words do not have unique token sequences")
    sequence_values = tuple(sequences.values())
    for sequence in sequence_values:
        if any(
            other != sequence and other[: len(sequence)] == sequence
            for other in sequence_values
        ):
            raise ValueError("answer token sequence is a prefix of another")
    return sequences


class TriePolicy:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        trie: TokenTrie,
        *,
        device: torch.device | str | None = None,
        prompt_renderer: Callable[[str], str] | None = None,
        checkpoint_digest: str = "",
        mask_version: str = MASK_VERSION,
        tokenizer_digest: str = "",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.trie = trie
        self.device = torch.device(device) if device is not None else None
        self.prompt_renderer = prompt_renderer
        self.checkpoint_digest = checkpoint_digest
        self.mask_version = mask_version
        self.tokenizer_digest = tokenizer_digest
        self.action_vocabulary_digest = digest_action_vocabulary(trie.words)

    @classmethod
    def from_tokenizer(
        cls,
        model: Any,
        tokenizer: Any,
        answers: Sequence[str],
        *,
        rendered_prompt: str | None = None,
        **kwargs: Any,
    ) -> TriePolicy:
        trie = TokenTrie.from_tokenizer(
            tokenizer,
            answers,
            rendered_prompt=rendered_prompt,
        )
        return cls(model, tokenizer, trie, **kwargs)

    def sample(
        self,
        observation: Any,
        *,
        temperature: float,
        seed: int,
    ) -> PolicyDecision:
        self._validate_temperature(temperature)
        prompt_ids = self._prompt_ids(observation.prompt)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        prefix: tuple[int, ...] = ()
        token_ids: list[int] = []
        log_probabilities: list[float] = []
        model_calls = 0
        node = self.trie.root
        was_training = getattr(self.model, "training", False)
        if hasattr(self.model, "eval"):
            self.model.eval()
        try:
            while node.children:
                allowed = tuple(sorted(node.children))
                if len(allowed) == 1:
                    token = allowed[0]
                    log_probability = 0.0
                else:
                    logits = self._next_logits(
                        prompt_ids,
                        [prefix],
                        requires_grad=False,
                    )[0]
                    model_calls += 1
                    token, log_probability = self._sample_token(
                        logits,
                        allowed,
                        temperature,
                        generator,
                    )
                token_ids.append(token)
                log_probabilities.append(log_probability)
                prefix += (token,)
                node = node.children[token]
                if node.word is not None:
                    break
        finally:
            if hasattr(self.model, "train"):
                self.model.train(was_training)

        if node.word is None:
            raise RuntimeError("trie sampling stopped without a word")
        return PolicyDecision(
            word=node.word,
            token_ids=tuple(token_ids),
            per_token_log_probabilities=tuple(log_probabilities),
            action_log_probability=math.fsum(log_probabilities),
            checkpoint_digest=self.checkpoint_digest,
            temperature=temperature,
            mask_version=self.mask_version,
            sampling_seed=int(seed),
            tokenizer_digest=self.tokenizer_digest,
            model_calls=model_calls,
        )

    def log_probability(
        self,
        observation: Any,
        word: str,
        *,
        temperature: float,
    ) -> tuple[float, tuple[float, ...]]:
        _, values = self.log_probability_tensor(
            observation,
            word,
            temperature=temperature,
            requires_grad=False,
        )
        per_token = tuple(float(value.item()) for value in values)
        return math.fsum(per_token), per_token

    def log_probability_tensor(
        self,
        observation: Any,
        word: str,
        *,
        temperature: float,
        requires_grad: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        self._validate_temperature(temperature)
        sequence = self.trie.sequence_for_word(word)
        prompt_ids = self._prompt_ids(observation.prompt)
        prefix: tuple[int, ...] = ()
        values: list[torch.Tensor] = []
        for token in sequence:
            allowed = self.trie.allowed_tokens(prefix)
            if token not in allowed:
                raise ValueError(f"word leaves trie at token {token}")
            if len(allowed) == 1:
                value = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self._device(),
                )
            else:
                logits = self._next_logits(
                    prompt_ids,
                    [prefix],
                    requires_grad=requires_grad,
                )[0]
                allowed_logits = logits[list(allowed)]
                scaled = allowed_logits / temperature
                value = scaled[allowed.index(token)] - torch.logsumexp(
                    scaled,
                    dim=0,
                )
            values.append(value)
            prefix += (token,)
        return torch.stack(values).sum(), tuple(values)

    def exact_log_distribution(
        self,
        observation: Any,
        *,
        temperature: float,
        max_model_calls: int = 3,
        max_batch_size: int = 256,
    ) -> dict[str, float]:
        self._validate_temperature(temperature)
        prompt_ids = self._prompt_ids(observation.prompt)
        conditional: dict[tuple[tuple[int, ...], int], float] = {}
        prefixes_by_depth = self.trie.branching_prefixes_by_depth()
        was_training = getattr(self.model, "training", False)
        if hasattr(self.model, "eval"):
            self.model.eval()
        model_calls = 0
        try:
            for prefixes in prefixes_by_depth.values():
                if len(prefixes) > max_batch_size:
                    raise RuntimeError(
                        f"exact trie batch {len(prefixes)} exceeds "
                        f"cap {max_batch_size}"
                    )
                if model_calls >= max_model_calls:
                    raise RuntimeError(
                        f"exact trie walk exceeds {max_model_calls} model calls"
                    )
                logits_batch = self._next_logits(
                    prompt_ids,
                    list(prefixes),
                    requires_grad=False,
                )
                model_calls += 1
                for prefix, logits in zip(prefixes, logits_batch):
                    allowed = self.trie.allowed_tokens(prefix)
                    allowed_logits = logits[list(allowed)] / temperature
                    normalizer = torch.logsumexp(allowed_logits, dim=0)
                    for token, value in zip(
                        allowed,
                        allowed_logits - normalizer,
                    ):
                        conditional[(prefix, token)] = float(value.item())
        finally:
            if hasattr(self.model, "train"):
                self.model.train(was_training)

        result = {}
        for word, sequence in self.trie.sequences.items():
            prefix: tuple[int, ...] = ()
            total = 0.0
            for token in sequence:
                allowed = self.trie.allowed_tokens(prefix)
                if len(allowed) > 1:
                    total += conditional[(prefix, token)]
                prefix += (token,)
            result[word] = total
        normalizer = torch.logsumexp(
            torch.tensor(tuple(result.values()), dtype=torch.float64),
            dim=0,
        ).item()
        return {
            word: value - normalizer
            for word, value in result.items()
        }

    def exact_distribution(
        self,
        observation: Any,
        *,
        temperature: float,
        max_model_calls: int = 3,
        max_batch_size: int = 256,
    ) -> dict[str, float]:
        return {
            word: float(torch.exp(torch.tensor(log_probability)).item())
            for word, log_probability in self.exact_log_distribution(
                observation,
                temperature=temperature,
                max_model_calls=max_model_calls,
                max_batch_size=max_batch_size,
            ).items()
        }

    def greedy_word(
        self,
        observation: Any,
    ) -> str:
        prompt_ids = self._prompt_ids(observation.prompt)
        prefix: tuple[int, ...] = ()
        node = self.trie.root
        was_training = getattr(self.model, "training", False)
        if hasattr(self.model, "eval"):
            self.model.eval()
        try:
            while node.children:
                allowed = tuple(sorted(node.children))
                if len(allowed) == 1:
                    token = allowed[0]
                else:
                    logits = self._next_logits(
                        prompt_ids,
                        [prefix],
                        requires_grad=False,
                    )[0]
                    token = max(
                        allowed,
                        key=lambda value: (float(logits[value]), -value),
                    )
                prefix += (token,)
                node = node.children[token]
                if node.word is not None:
                    return node.word
        finally:
            if hasattr(self.model, "train"):
                self.model.train(was_training)
        raise RuntimeError("greedy trie decoding stopped without a word")

    def verify_prompt_contract(self, rendered_prompt: str) -> None:
        expected = build_word_token_sequences(
            self.tokenizer,
            self.trie.words,
            rendered_prompt=rendered_prompt,
        )
        if expected != dict(self.trie.sequences):
            raise ValueError("prompt tokenization differs from frozen trie")

    def _prompt_ids(self, prompt: str) -> tuple[int, ...]:
        if self.prompt_renderer is not None:
            rendered = self.prompt_renderer(prompt)
        elif hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            rendered = prompt
        return tuple(_encoded_ids(self.tokenizer, rendered))

    def _next_logits(
        self,
        prompt_ids: Sequence[int],
        prefixes: Sequence[Sequence[int]],
        *,
        requires_grad: bool,
    ) -> torch.Tensor:
        if not prefixes:
            raise ValueError("at least one prefix is required")
        sequences = [
            tuple(prompt_ids) + tuple(int(token) for token in prefix)
            for prefix in prefixes
        ]
        lengths = [len(sequence) for sequence in sequences]
        if len(set(lengths)) != 1:
            raise ValueError("prefix batch must have one sequence length")
        max_length = max(lengths)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = torch.full(
            (len(sequences), max_length),
            int(pad_id),
            dtype=torch.long,
            device=self._device(),
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=self._device(),
            )
            attention_mask[row, :length] = 1
        context = nullcontext() if requires_grad else torch.no_grad()
        with context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
                use_cache=False,
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if logits.shape[1] != 1:
                raise RuntimeError("model ignored logits_to_keep=1")
            return logits[:, -1, :]

    def _sample_token(
        self,
        logits: torch.Tensor,
        allowed: Sequence[int],
        temperature: float,
        generator: torch.Generator,
    ) -> tuple[int, float]:
        allowed_logits = logits[list(allowed)] / temperature
        log_probabilities = (
            allowed_logits
            - torch.logsumexp(allowed_logits, dim=0)
        )
        probabilities = torch.exp(log_probabilities).detach().cpu()
        choice = int(
            torch.multinomial(
                probabilities,
                1,
                generator=generator,
            ).item()
        )
        return allowed[choice], float(log_probabilities[choice].item())

    def _device(self) -> torch.device:
        if self.device is not None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    @staticmethod
    def _validate_temperature(temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
