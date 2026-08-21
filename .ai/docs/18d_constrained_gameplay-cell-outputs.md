# Lab 18d - Does constrained policy improve full games?

Labs 18b and 18c established a replicated state-level capability. Across three
B-structured seeds, free generation produced usable actions on 14.5% to 16.3%
of held-out states, while ranking the 2,315 answer words produced 30.0% to
31.0%. Candidate-rank percentile stayed near 0.029.

That does not establish gameplay. Errors change the next state, repeated actions
waste turns, and the largest replicated weakness is strategic choice under broad
Turn 2 uncertainty. This lab changes only the action decoder and plays the same
19 reserved answers with every B seed.

## 18d.1 Pre-registered experiment

**Question.** If malformed-word generation is removed, does the replicated
constrained policy materially improve full-game solve rate?

**Models.** B-structured seeds 42, 45, and 47 from Lab 18c.

**Paired decoder intervention.**

| decoder | action |
| --- | --- |
| free | greedy token generation, Lab 18 behavior |
| free-continue | same generation, but invalid output consumes a turn |
| answer-constrained | argmax string likelihood over all 2,315 answers |

Everything else stays fixed: the 19 reserved answers, `RAISE` as Turn 1, the
Lab 17 structured prompt, six total turns, greedy inference, and termination
when the free decoder emits an out-of-lexicon action. `free-continue` is a
diagnostic counterfactual that keeps the state unchanged and spends the turn,
matching the older benchmark behavior. The constrained decoder cannot terminate
for malformed output, but it can still choose an inconsistent or repeated
answer. No symbolic candidate filter is applied to its action space.

**Primary output.** Solve rate for each seed and decoder, paired by answer.

**Secondary outputs.**

1. candidate count before and after every action;
2. history consistency, repetition, and usable-action rate by turn;
3. teacher action, model rank of that action, and teacher match;
4. chosen-action entropy, teacher entropy, and entropy gap;
5. realized `log2(candidates_before / candidates_after)`.
6. the candidate-restricted Tier 2 action implied by the same score vector;
7. the full 2,315-answer score vector for every constrained game state.

The canonical teacher chooses the maximum-entropy word among current answer
candidates. The model may choose any answer-list word. Therefore
`teacher_entropy - chosen_entropy` can be negative for an exploratory
out-of-candidate action; that is not automatically an error. Teacher-relative
rank is defined only when the chosen action is itself a current candidate. A
second open teacher maximizes entropy over the same 2,315-answer action space as
the model; its regret is always nonnegative and separates exploration quality
from candidate-only teacher imitation.

**Read before seeing results.**

- Free-continue beats free -> invalid termination, rather than improved actions,
  explains part of the gameplay gain.
- Answer-constrained beats free-continue -> the learned answer ranking improves
  action quality beyond merely keeping games alive.
- Validity rises but solves do not -> grounding was real but strategic ranking
  remains the binding problem.
- Teacher match is low but entropy gap is near zero -> exact teacher match is too
  strict; the model often chooses strategically equivalent alternatives.
- Entropy gap is large and candidate reduction is weak on Turn 2 -> Lab 19
  should distill relative action values, especially in broad states.
- Constrained actions have low regret but games still fail -> the remaining
  failure lies in trajectory dynamics rather than one-step ranking.

Nineteen fixed answers provide a paired diagnostic, not a precise population
solve rate. Seed-level replication is stronger evidence than a state-level
p-value.

## 18d.2 Run controls and memory guard

Run only through the system watchdog:

```
scripts/memguard.py --min-free 64 -- uv run jupyter nbconvert \
    --to notebook --execute --inplace notebooks/18d_constrained_gameplay.ipynb
```

The scoring kernel is Lab 18b's verified answer ranker. Before gameplay, seed 42
must reproduce one persisted Lab 18b score vector, then pass a 40-iteration
fixed-shape memory soak on the longest prompt in the 620-state battery.
Gameplay artifacts are written after every answer, so interruption loses at most
one game.


```python
RUN_EVALUATION = True
MEMORY_CAP_GIB = 128.0

import torch

if torch.backends.mps.is_available():
    total_gib = torch.mps.recommended_max_memory() / 1024**3
    torch.mps.set_per_process_memory_fraction(MEMORY_CAP_GIB / total_gib)
    print(f"MPS cap: {MEMORY_CAP_GIB:.0f} GiB of {total_gib:.0f} GiB")

print("RUN_EVALUATION:", RUN_EVALUATION)
```

    MPS cap: 128 GiB of 464 GiB
    RUN_EVALUATION: True



```python
from collections import defaultdict
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_wordle.benchmark import DEFAULT_EVAL_ANSWERS, parse_guess
from tiny_wordle.expert import EntropyExpert
from tiny_wordle.game import Turn, filter_candidates, is_consistent, score_string
from tiny_wordle.hardware import preferred_device

MODEL_ID = "Qwen/Qwen3-0.6B"
SEEDS = [42, 45, 47]
DECODERS = ["free", "free-continue", "answer-constrained"]
MAX_TURNS = 6
OPENING = "RAISE"
CHUNK_SIZE = 256
MEMORY_ABORT_GIB = MEMORY_CAP_GIB * 0.75

DATA_DIR = Path("../data")
CHECKPOINT_ROOT = Path("../checkpoints")
RESULTS_DIR = Path("../results/lab18d")
LAB18_RESULTS = Path("../results/lab18")
LAB18B_RESULTS = Path("../results/lab18b")
CHECKPOINTS = {
    42: CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured",
    45: CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed45",
    47: CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed47",
}
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = preferred_device()
torch.set_float32_matmul_precision("high")
print("device:", device)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


checkpoint_hashes = {}
for seed, path in CHECKPOINTS.items():
    model_file = path / "adapter_model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"missing seed {seed} adapter: {model_file}")
    checkpoint_hashes[seed] = sha256_file(model_file)
print("checkpoint fingerprints:", checkpoint_hashes)


def driver_memory_gib() -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1024**3
    return float("nan")
```

    device: mps
    checkpoint fingerprints: {42: '8f08ba4787ccaa729adb39507edb9eb389e0cf81bdf1811aa4f9077fef157f55', 45: 'a3b849ac3cbc57c085ec4f1f7697d113f13e87168377420662baaba3b75d614c', 47: '52dd5812478f9b412e875151c39a7d703144605136c57ebd330535ba8a8e9ec9'}


## 18d.3 Structured gameplay prompts

These functions reproduce the Lab 17 representation and Lab 18 gameplay prompt.
Every model starts after the fixed `RAISE` opening, so the first model decision
is Turn 2.


```python
ANSWERS = [
    line.strip().upper()
    for line in (DATA_DIR / "wordle-answers-original.txt").read_text().splitlines()
    if line.strip()
]
ANSWER_SET = set(ANSWERS)
ANSWER_ARRAY = np.array(ANSWERS)
PATTERNS = np.load(DATA_DIR / "wordle-patterns-original-2315.npy")
expert = EntropyExpert(ANSWERS, PATTERNS)
WORD_TO_INDEX = expert.word_to_index
ALL_INDICES = expert.all_indices
assert len(ANSWERS) == 2315
assert PATTERNS.shape == (2315, 2315)
assert list(DEFAULT_EVAL_ANSWERS) == [
    "SHORE", "MIGHT", "BRICK", "GHOST", "KNIFE", "DOUBT", "FLING",
    "ROUND", "CHAMP", "WASTE", "BLIND", "POINT", "SLATE", "CRANE",
    "APPLE", "SHEEP", "BANAL", "ALLEY", "AUDIO",
]


def derive_constraints(history: list[Turn]) -> dict:
    greens = [None] * 5
    minimum = defaultdict(int)
    maximum = defaultdict(lambda: 5)
    excluded = defaultdict(set)
    for turn in history:
        marks_by_letter = defaultdict(list)
        for position, (letter, mark) in enumerate(
            zip(turn.guess, turn.feedback), 1
        ):
            marks_by_letter[letter].append(mark)
            if mark == "G":
                if greens[position - 1] not in (None, letter):
                    raise ValueError("conflicting green constraints")
                greens[position - 1] = letter
            else:
                excluded[letter].add(position)
        for letter, marks in marks_by_letter.items():
            matched = sum(mark in {"Y", "G"} for mark in marks)
            minimum[letter] = max(minimum[letter], matched)
            if matched < len(marks):
                maximum[letter] = min(maximum[letter], matched)
    for letter in minimum:
        if minimum[letter] > maximum.get(letter, 5):
            raise ValueError(f"impossible count constraint for {letter}")
    return {
        "greens": greens,
        "minimum": dict(minimum),
        "maximum": dict(maximum),
        "excluded": {
            letter: sorted(positions)
            for letter, positions in excluded.items()
        },
        "previous_guesses": [turn.guess for turn in history],
    }


def render_structured_state(
    history: list[Turn], candidate_count: int
) -> str:
    state = derive_constraints(history)
    greens = " ".join(letter or "_" for letter in state["greens"])
    present = sorted(
        letter for letter, count in state["minimum"].items() if count > 0
    )
    counts = []
    for letter in present:
        low = state["minimum"][letter]
        high = state["maximum"].get(letter, 5)
        counts.append(
            f"{letter}={low}..{high}" if high < 5 else f"{letter}>={low}"
        )
    absent = sorted(
        letter for letter, count in state["maximum"].items() if count == 0
    )
    excluded = []
    for letter in sorted(state["excluded"]):
        positions = ",".join(map(str, state["excluded"][letter]))
        excluded.append(f"{letter}@{positions}")
    return "\n".join([
        f"GREENS: {greens}",
        f"LETTER_COUNTS: {', '.join(counts) or 'NONE'}",
        f"EXCLUDED_POSITIONS: {', '.join(excluded) or 'NONE'}",
        f"ABSENT_LETTERS: {' '.join(absent) or 'NONE'}",
        f"PREVIOUS_GUESSES: {', '.join(state['previous_guesses']) or 'NONE'}",
        f"CANDIDATE_COUNT: {candidate_count}",
    ])


def format_training_history(history: list[Turn]) -> str:
    return "\n".join(
        f"{' '.join(turn.guess)} -> {' '.join(turn.feedback)}"
        for turn in history
    )


def raw_next_guess_prompt(history: list[Turn]) -> str:
    return (
        "Task: NEXT_GUESS\n"
        "You are playing Wordle.\n"
        "Use the game history to choose the next guess.\n"
        "Return exactly one uppercase five-letter word.\n\n"
        f"History:\n{format_training_history(history)}"
    )


def structured_next_guess_prompt(history: list[Turn]) -> str:
    candidate_count = len(filter_candidates(ANSWERS, history))
    prefix = raw_next_guess_prompt(history).split("\n\nHistory:\n", 1)[0]
    return (
        prefix
        + "\n\nDerived state:\n"
        + render_structured_state(history, candidate_count)
    )
```

## 18d.4 Verified answer-list scoring kernel

The constrained action maximizes summed
`log P(word tokens + EOS | structured prompt)`. Words are bucketed by token
length, prompt KV state is reused across chunks, and only needed logits are
materialized.


```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def render_prompt(prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


WORD_TOKENS = [
    tokenizer.encode(word, add_special_tokens=False)
    + [tokenizer.eos_token_id]
    for word in ANSWERS
]
LENGTH_BUCKETS = {}
for length in sorted({len(tokens) for tokens in WORD_TOKENS}):
    indices = [
        index for index, tokens in enumerate(WORD_TOKENS)
        if len(tokens) == length
    ]
    padding = (-len(indices)) % CHUNK_SIZE
    padded = indices + [indices[-1]] * padding
    LENGTH_BUCKETS[length] = (
        torch.tensor(padded),
        torch.tensor(
            [WORD_TOKENS[index] for index in padded], device=device
        ),
    )


def load_adapter(seed: int):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)
    return PeftModel.from_pretrained(
        base, CHECKPOINTS[seed]
    ).to(device).eval()


def release_model(model) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


LAST_STATE_PEAK_GIB = 0.0


@torch.no_grad()
def score_all_words(model, prompt_text: str) -> np.ndarray:
    global LAST_STATE_PEAK_GIB
    input_ids = tokenizer(
        render_prompt(prompt_text),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    prefill = model(
        input_ids=input_ids, use_cache=True, logits_to_keep=1
    )
    final_logits = prefill.logits[0, -1].float()
    first_logprobs = final_logits - final_logits.logsumexp(-1)
    cache = prefill.past_key_values
    cache.batch_repeat_interleave(CHUNK_SIZE)
    peak = 0.0
    scores = torch.zeros(len(ANSWERS), dtype=torch.float32)

    for length, (indices, tokens) in LENGTH_BUCKETS.items():
        for start in range(0, len(indices), CHUNK_SIZE):
            chunk = tokens[start:start + CHUNK_SIZE]
            total = first_logprobs[chunk[:, 0]].clone()
            if length > 1:
                step = length - 1
                output = model(
                    input_ids=chunk[:, :step],
                    past_key_values=cache,
                    use_cache=True,
                )
                logits = output.logits.float()
                targets = logits.gather(
                    2, chunk[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                total = total + (
                    targets - logits.logsumexp(-1)
                ).sum(dim=1)
                peak = max(peak, driver_memory_gib())
                cache.crop(-step)
                del output, logits, targets
            scores[indices[start:start + CHUNK_SIZE]] = total.cpu()

    LAST_STATE_PEAK_GIB = peak
    del cache, prefill, final_logits, first_logprobs
    if device.type == "mps":
        torch.mps.empty_cache()
    return scores.numpy()


@torch.no_grad()
def generate_free(model, prompt_text: str) -> str:
    batch = tokenizer(
        render_prompt(prompt_text), return_tensors="pt"
    ).to(device)
    output = model.generate(
        **batch, max_new_tokens=16, do_sample=False
    )
    new_tokens = output[0, batch["input_ids"].shape[1]:]
    return tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()
```

## 18d.5 Correctness and fixed-shape memory gate

Seed 42 must reproduce Lab 18b's first persisted score vector within float32
tolerance. The same model then scores one fixed longest battery prompt 40 times.
The final third must be flat before any game runs.


```python
if RUN_EVALUATION:
    checker = load_adapter(42)
    lab18b_battery = pd.read_csv(
        LAB18B_RESULTS / "battery-states.csv"
    )
    battery_histories = lab18b_battery["state_key"].map(
        lambda key: [
            Turn(
                guess=line.split(" -> ")[0].replace(" ", ""),
                feedback=line.split(" -> ")[1].replace(" ", ""),
            )
            for line in key.splitlines()
        ]
    )
    battery_prompts = [
        structured_next_guess_prompt(history)
        for history in battery_histories
    ]
    first_scores = score_all_words(checker, battery_prompts[0])
    reference_scores = np.load(
        LAB18B_RESULTS / "scores-B-structured.npy",
        mmap_mode="r",
    )[0]
    max_abs_diff = float(np.max(np.abs(first_scores - reference_scores)))
    print("Lab 18b score-vector max abs diff:", max_abs_diff)
    assert max_abs_diff < 1e-3

    prompt_lengths = [
        len(tokenizer(render_prompt(prompt)).input_ids)
        for prompt in battery_prompts
    ]
    soak_prompt = battery_prompts[int(np.argmax(prompt_lengths))]
    soak_peaks = []
    for _ in range(40):
        score_all_words(checker, soak_prompt)
        soak_peaks.append(LAST_STATE_PEAK_GIB)
    third = len(soak_peaks) // 3
    creep = (
        np.mean(soak_peaks[-third:])
        - np.mean(soak_peaks[third:2 * third])
    )
    late_range = np.ptp(soak_peaks[-third:])
    print(
        f"scoring soak peak {max(soak_peaks):.2f} GiB, "
        f"creep {creep:+.2f} GiB, "
        f"final range {late_range:.2f} GiB"
    )
    assert creep < 0.5
    assert late_range < 0.5
    assert max(soak_peaks) < MEMORY_ABORT_GIB
    release_model(checker)
    del checker
    print("kernel verified and memory plateaued")
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    Lab 18b score-vector max abs diff: 0.0


    scoring soak peak 54.77 GiB, creep +0.00 GiB, final range 0.00 GiB


    kernel verified and memory plateaued


## 18d.6 Gameplay engine and strategic diagnostics

The teacher is evaluated on the exact state the model visits. Entropy is the
expected information gain of an action over the current candidate set. Realized
candidate reduction is answer-specific and therefore complements expected
entropy.


```python
def candidate_indices(history: list[Turn]) -> np.ndarray:
    words = filter_candidates(ANSWERS, history)
    indices = np.array(
        [WORD_TO_INDEX[word] for word in words], dtype=np.int32
    )
    if len(indices) == 0:
        raise ValueError("game history produced an empty candidate set")
    return indices


def strategic_metrics(
    guess: str | None,
    candidates: np.ndarray,
    model_scores: np.ndarray | None,
) -> dict:
    candidate_entropies = np.array([
        expert.entropy(int(index), candidates)
        for index in candidates
    ])
    best_candidate_entropy = float(candidate_entropies.max())
    tied_candidate_indices = candidates[
        np.abs(candidate_entropies - best_candidate_entropy) <= 1e-12
    ]
    teacher_index = int(min(
        tied_candidate_indices,
        key=lambda index: ANSWERS[int(index)],
    ))
    teacher_word = ANSWERS[teacher_index]
    teacher_entropy = best_candidate_entropy
    result = {
        "teacher_guess": teacher_word,
        "teacher_entropy_bits": teacher_entropy,
        "teacher_match": guess == teacher_word,
        "open_teacher_guess": None,
        "open_teacher_entropy_bits": float("nan"),
        "chosen_entropy_bits": float("nan"),
        "entropy_gap_bits": float("nan"),
        "open_entropy_regret_bits": float("nan"),
        "chosen_solve_probability": 0.0,
        "chosen_is_candidate": False,
        "chosen_candidate_entropy_rank": float("nan"),
        "chosen_candidate_entropy_percentile": float("nan"),
        "model_teacher_rank": float("nan"),
        "model_best_candidate_rank": float("nan"),
        "candidate_rank_percentile": float("nan"),
        "candidate_mass": float("nan"),
        "candidate_mass_lift": float("nan"),
        "tier2_guess": None,
        "tier2_teacher_match": float("nan"),
        "tier2_entropy_gap_bits": float("nan"),
        "chosen_token_length": float("nan"),
    }
    if guess is None or guess not in ANSWER_SET:
        return result

    guess_index = WORD_TO_INDEX[guess]
    if guess_index in set(map(int, candidates)):
        chosen_entropy = float(
            candidate_entropies[
                np.where(candidates == guess_index)[0][0]
            ]
        )
    else:
        chosen_entropy = expert.entropy(guess_index, candidates)
    result["chosen_entropy_bits"] = chosen_entropy
    result["entropy_gap_bits"] = teacher_entropy - chosen_entropy
    result["chosen_token_length"] = len(WORD_TOKENS[guess_index])
    candidate_positions = {
        int(index): position
        for position, index in enumerate(candidates)
    }
    result["chosen_is_candidate"] = guess_index in candidate_positions
    if result["chosen_is_candidate"]:
        result["chosen_solve_probability"] = 1.0 / len(candidates)
        ranked_candidates = sorted(
            (
                -float(entropy),
                ANSWERS[int(index)],
                int(index),
            )
            for index, entropy in zip(candidates, candidate_entropies)
        )
        rank_by_index = {
            index: rank
            for rank, (_, _, index) in enumerate(
                ranked_candidates, 1
            )
        }
        result["chosen_candidate_entropy_rank"] = rank_by_index[
            guess_index
        ]
        result["chosen_candidate_entropy_percentile"] = (
            rank_by_index[guess_index] / len(candidates)
        )

    if model_scores is not None:
        order = np.argsort(-model_scores, kind="stable")
        ranks = np.empty(len(model_scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(model_scores) + 1)
        result["model_teacher_rank"] = int(ranks[teacher_index])
        result["model_best_candidate_rank"] = int(
            ranks[candidates].min()
        )
        result["candidate_rank_percentile"] = float(
            ranks[candidates].mean() / len(model_scores)
        )
        shifted = model_scores - model_scores.max()
        weights = np.exp(shifted)
        candidate_mass = float(
            weights[candidates].sum() / weights.sum()
        )
        uniform_mass = len(candidates) / len(ANSWERS)
        result["candidate_mass"] = candidate_mass
        result["candidate_mass_lift"] = (
            candidate_mass / uniform_mass
        )

        tier2_index = int(
            candidates[np.argmax(model_scores[candidates])]
        )
        tier2_entropy = float(
            candidate_entropies[
                np.where(candidates == tier2_index)[0][0]
            ]
        )
        result["tier2_guess"] = ANSWERS[tier2_index]
        result["tier2_teacher_match"] = tier2_index == teacher_index
        result["tier2_entropy_gap_bits"] = (
            teacher_entropy - tier2_entropy
        )

        all_entropies = np.array([
            expert.entropy(int(index), candidates)
            for index in ALL_INDICES
        ])
        best_open_entropy = float(all_entropies.max())
        tied_open_indices = ALL_INDICES[
            np.abs(all_entropies - best_open_entropy) <= 1e-12
        ]
        open_teacher_index = int(min(
            tied_open_indices,
            key=lambda index: ANSWERS[int(index)],
        ))
        result["open_teacher_guess"] = ANSWERS[open_teacher_index]
        result["open_teacher_entropy_bits"] = best_open_entropy
        result["open_entropy_regret_bits"] = (
            best_open_entropy - chosen_entropy
        )
    return result


def play_game(
    model, seed: int, decoder: str, answer: str
) -> tuple[list[dict], dict, list[np.ndarray]]:
    history = [
        Turn(OPENING, score_string(answer, OPENING))
    ]
    seen = {OPENING}
    call_rows = []
    score_vectors = []
    solved_turn = None
    terminated_invalid = False
    started = time.perf_counter()

    for turn_number in range(2, MAX_TURNS + 1):
        before = candidate_indices(history)
        prompt = structured_next_guess_prompt(history)
        history_has_duplicate = (
            len(history) != len({turn.guess for turn in history})
        )
        model_scores = None
        if decoder in {"free", "free-continue"}:
            raw = generate_free(model, prompt)
            guess = parse_guess(raw)
        elif decoder == "answer-constrained":
            model_scores = score_all_words(model, prompt)
            assert LAST_STATE_PEAK_GIB < MEMORY_ABORT_GIB, (
                f"memory regression at seed {seed} {answer} "
                f"turn {turn_number}: {LAST_STATE_PEAK_GIB:.1f} GiB"
            )
            guess = ANSWERS[int(model_scores.argmax())]
            raw = guess
            score_vectors.append(model_scores)
        else:
            raise ValueError(f"unknown decoder {decoder}")

        format_valid = guess is not None
        in_lexicon = bool(guess and guess in ANSWER_SET)
        repeated = bool(guess and guess in seen)
        consistent = bool(
            in_lexicon and is_consistent(guess, history)
        )
        usable = bool(
            in_lexicon and consistent and not repeated
        )
        strategy = strategic_metrics(
            guess, before, model_scores
        )

        if in_lexicon:
            feedback = score_string(answer, guess)
            after = expert.update(
                before, WORD_TO_INDEX[guess], feedback
            )
            if len(after) == 0:
                raise AssertionError("answer disappeared from candidates")
            seen.add(guess)
            history.append(Turn(guess, feedback))
        else:
            feedback = None
            after = before

        reduction_fraction = 1.0 - len(after) / len(before)
        realized_log2_reduction = math.log2(
            len(before) / len(after)
        )
        call_rows.append({
            "seed": seed,
            "decoder": decoder,
            "answer": answer,
            "turn": turn_number,
            "raw": raw,
            "guess": guess,
            "format_valid": format_valid,
            "in_answer_lexicon": in_lexicon,
            "repeated": repeated,
            "history_has_duplicate_before": history_has_duplicate,
            "history_consistent": consistent,
            "usable": usable,
            "feedback": feedback,
            "candidate_count_before": len(before),
            "candidate_count_after": len(after),
            "candidate_reduction_fraction": reduction_fraction,
            "realized_log2_reduction": realized_log2_reduction,
            "driver_peak_gib": (
                LAST_STATE_PEAK_GIB
                if decoder == "answer-constrained"
                else float("nan")
            ),
            **strategy,
        })

        if not in_lexicon and decoder == "free":
            terminated_invalid = True
            break
        if not in_lexicon:
            continue
        if feedback == "GGGGG":
            solved_turn = turn_number
            break

    return call_rows, {
        "seed": seed,
        "decoder": decoder,
        "answer": answer,
        "solved": solved_turn is not None,
        "solved_turn": solved_turn,
        "terminated_invalid": terminated_invalid,
        "model_calls": len(call_rows),
        "final_candidate_count": (
            call_rows[-1]["candidate_count_after"]
            if call_rows else len(ANSWERS)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }, score_vectors
```

## 18d.7 Restartable paired gameplay

Each seed uses one loaded adapter for all decoders. Free runs first because it
is the frozen Lab 18 behavior; free-continue and answer-constrained follow.
Artifacts are atomically rewritten after every completed answer and validated
before reuse. Constrained score vectors and their `(answer, turn)` keys are
checkpointed at the same cadence.


```python
def evaluation_paths(seed: int, decoder: str) -> dict[str, Path]:
    stem = f"seed{seed}-{decoder}"
    return {
        "calls": RESULTS_DIR / f"{stem}-calls.csv",
        "games": RESULTS_DIR / f"{stem}-games.csv",
        "scores": RESULTS_DIR / f"{stem}-scores.npy",
        "score_keys": RESULTS_DIR / f"{stem}-score-keys.csv",
        "progress": RESULTS_DIR / f"{stem}-progress.json",
    }


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npy(values: np.ndarray, path: Path) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


lab18_calls = pd.read_csv(
    LAB18_RESULTS / "b-gameplay-calls.csv"
)
lab18_games = pd.read_csv(
    LAB18_RESULTS / "b-gameplay-games.csv"
)


def validate_seed42_free(
    calls: pd.DataFrame, games: pd.DataFrame
) -> None:
    paired_games = games[
        ["answer", "solved", "terminated_invalid"]
    ].merge(
        lab18_games[
            ["answer", "solved", "terminated_invalid"]
        ],
        on="answer",
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    assert (
        paired_games["solved_new"]
        == paired_games["solved_old"]
    ).all()
    assert (
        paired_games["terminated_invalid_new"]
        == paired_games["terminated_invalid_old"]
    ).all()
    paired_calls = calls[
        ["answer", "turn", "guess"]
    ].merge(
        lab18_calls[["answer", "turn", "guess"]],
        on=["answer", "turn"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    assert len(paired_calls) == len(calls) == len(lab18_calls)
    assert (
        paired_calls["guess_new"].fillna("<INVALID>")
        == paired_calls["guess_old"].fillna("<INVALID>")
    ).all()


def evaluate_decoder(
    model, seed: int, decoder: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = evaluation_paths(seed, decoder)
    calls_path = paths["calls"]
    games_path = paths["games"]
    progress = {
        "seed": seed,
        "decoder": decoder,
        "checkpoint_sha256": checkpoint_hashes[seed],
        "answers": list(DEFAULT_EVAL_ANSWERS),
    }
    if paths["progress"].exists():
        assert json.loads(paths["progress"].read_text()) == progress
    else:
        atomic_json(progress, paths["progress"])

    if games_path.exists() and not calls_path.exists():
        raise FileNotFoundError(
            f"games exist without calls for seed {seed} {decoder}"
        )
    if calls_path.exists() and not games_path.exists():
        calls = pd.DataFrame()
        games = pd.DataFrame()
    elif games_path.exists():
        calls = pd.read_csv(calls_path)
        games = pd.read_csv(games_path)
        assert set(games["answer"]).issubset(DEFAULT_EVAL_ANSWERS)
        assert not games["answer"].duplicated().any()
        # Calls are written before the matching game row. If the process dies
        # between those writes, discard that one incomplete game's calls.
        calls = calls.loc[
            calls["answer"].isin(set(games["answer"]))
        ].copy()
    else:
        calls = pd.DataFrame()
        games = pd.DataFrame()

    score_matrix = np.empty((0, len(ANSWERS)), dtype=np.float32)
    score_keys = pd.DataFrame(columns=["seed", "answer", "turn"])
    if decoder == "answer-constrained":
        score_exists = paths["scores"].exists()
        keys_exist = paths["score_keys"].exists()
        if score_exists != keys_exist:
            raise FileNotFoundError(
                f"incomplete score artifact pair for seed {seed}"
            )
        if score_exists:
            score_matrix = np.load(paths["scores"])
            score_keys = pd.read_csv(paths["score_keys"])
            assert len(score_matrix) == len(score_keys)

    completed = set(games["answer"]) if len(games) else set()
    if len(calls):
        calls = calls.loc[calls["answer"].isin(completed)].copy()
    if decoder == "answer-constrained" and len(score_keys):
        keep = score_keys["answer"].isin(completed).to_numpy()
        score_matrix = score_matrix[keep]
        score_keys = score_keys.loc[keep].reset_index(drop=True)

    for answer in DEFAULT_EVAL_ANSWERS:
        if answer in completed:
            continue
        new_calls, new_game, new_scores = play_game(
            model, seed, decoder, answer
        )
        calls = pd.concat(
            [calls, pd.DataFrame(new_calls)], ignore_index=True
        )
        games = pd.concat(
            [games, pd.DataFrame([new_game])], ignore_index=True
        )
        if decoder == "answer-constrained":
            new_score_matrix = np.stack(new_scores).astype(
                np.float32, copy=False
            )
            new_keys = pd.DataFrame([
                {
                    "seed": row["seed"],
                    "answer": row["answer"],
                    "turn": row["turn"],
                }
                for row in new_calls
            ])
            score_matrix = np.concatenate(
                [score_matrix, new_score_matrix], axis=0
            )
            score_keys = pd.concat(
                [score_keys, new_keys], ignore_index=True
            )
            atomic_npy(score_matrix, paths["scores"])
            atomic_csv(score_keys, paths["score_keys"])
        atomic_csv(calls, calls_path)
        atomic_csv(games, games_path)
        print(
            f"seed {seed} {decoder} {answer}: "
            f"{'SOLVED' if new_game['solved'] else 'FAILED'} "
            f"turn={new_game['solved_turn']} "
            f"calls={new_game['model_calls']}",
            flush=True,
        )

    assert games["answer"].tolist() == list(DEFAULT_EVAL_ANSWERS)
    assert set(calls["answer"]) == set(DEFAULT_EVAL_ANSWERS)
    assert not calls.duplicated(
        ["seed", "decoder", "answer", "turn"]
    ).any()
    if decoder == "answer-constrained":
        expected_keys = calls[
            ["seed", "answer", "turn"]
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            score_keys.reset_index(drop=True),
            expected_keys,
            check_dtype=False,
        )
        assert score_matrix.shape == (
            len(calls), len(ANSWERS)
        )
    return calls, games


all_calls = []
all_games = []
if RUN_EVALUATION:
    for seed in SEEDS:
        model = load_adapter(seed)
        for decoder in DECODERS:
            calls, games = evaluate_decoder(
                model, seed, decoder
            )
            if seed == 42 and decoder == "free":
                validate_seed42_free(calls, games)
                print("seed 42 free gameplay reproduces Lab 18")
            all_calls.append(calls)
            all_games.append(games)
        release_model(model)
        del model
else:
    for seed in SEEDS:
        for decoder in DECODERS:
            paths = evaluation_paths(seed, decoder)
            all_calls.append(pd.read_csv(paths["calls"]))
            all_games.append(pd.read_csv(paths["games"]))

gameplay_calls = pd.concat(all_calls, ignore_index=True)
gameplay_games = pd.concat(all_games, ignore_index=True)
print("calls:", len(gameplay_calls), "games:", len(gameplay_games))
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 42 free SHORE: FAILED turn=None calls=1


    seed 42 free MIGHT: FAILED turn=None calls=1


    seed 42 free BRICK: SOLVED turn=6 calls=5


    seed 42 free GHOST: FAILED turn=None calls=1


    seed 42 free KNIFE: FAILED turn=None calls=4


    seed 42 free DOUBT: FAILED turn=None calls=1


    seed 42 free FLING: FAILED turn=None calls=1


    seed 42 free ROUND: SOLVED turn=2 calls=1


    seed 42 free CHAMP: FAILED turn=None calls=1


    seed 42 free WASTE: FAILED turn=None calls=2


    seed 42 free BLIND: FAILED turn=None calls=1


    seed 42 free POINT: FAILED turn=None calls=1


    seed 42 free SLATE: SOLVED turn=4 calls=3


    seed 42 free CRANE: SOLVED turn=4 calls=3


    seed 42 free APPLE: FAILED turn=None calls=1


    seed 42 free SHEEP: SOLVED turn=3 calls=2


    seed 42 free BANAL: FAILED turn=None calls=2


    seed 42 free ALLEY: FAILED turn=None calls=1


    seed 42 free AUDIO: FAILED turn=None calls=1


    seed 42 free gameplay reproduces Lab 18


    seed 42 free-continue SHORE: FAILED turn=None calls=5


    seed 42 free-continue MIGHT: FAILED turn=None calls=5


    seed 42 free-continue BRICK: SOLVED turn=6 calls=5


    seed 42 free-continue GHOST: FAILED turn=None calls=5


    seed 42 free-continue KNIFE: FAILED turn=None calls=5


    seed 42 free-continue DOUBT: FAILED turn=None calls=5


    seed 42 free-continue FLING: FAILED turn=None calls=5


    seed 42 free-continue ROUND: SOLVED turn=2 calls=1


    seed 42 free-continue CHAMP: FAILED turn=None calls=5


    seed 42 free-continue WASTE: FAILED turn=None calls=5


    seed 42 free-continue BLIND: FAILED turn=None calls=5


    seed 42 free-continue POINT: FAILED turn=None calls=5


    seed 42 free-continue SLATE: SOLVED turn=4 calls=3


    seed 42 free-continue CRANE: SOLVED turn=4 calls=3


    seed 42 free-continue APPLE: FAILED turn=None calls=5


    seed 42 free-continue SHEEP: SOLVED turn=3 calls=2


    seed 42 free-continue BANAL: FAILED turn=None calls=5


    seed 42 free-continue ALLEY: FAILED turn=None calls=5


    seed 42 free-continue AUDIO: FAILED turn=None calls=5


    seed 42 answer-constrained SHORE: SOLVED turn=2 calls=1


    seed 42 answer-constrained MIGHT: FAILED turn=None calls=5


    seed 42 answer-constrained BRICK: SOLVED turn=6 calls=5


    seed 42 answer-constrained GHOST: FAILED turn=None calls=5


    seed 42 answer-constrained KNIFE: FAILED turn=None calls=5


    seed 42 answer-constrained DOUBT: FAILED turn=None calls=5


    seed 42 answer-constrained FLING: SOLVED turn=3 calls=2


    seed 42 answer-constrained ROUND: FAILED turn=None calls=5


    seed 42 answer-constrained CHAMP: SOLVED turn=4 calls=3


    seed 42 answer-constrained WASTE: FAILED turn=None calls=5


    seed 42 answer-constrained BLIND: SOLVED turn=4 calls=3


    seed 42 answer-constrained POINT: FAILED turn=None calls=5


    seed 42 answer-constrained SLATE: SOLVED turn=4 calls=3


    seed 42 answer-constrained CRANE: SOLVED turn=5 calls=4


    seed 42 answer-constrained APPLE: SOLVED turn=6 calls=5


    seed 42 answer-constrained SHEEP: SOLVED turn=5 calls=4


    seed 42 answer-constrained BANAL: SOLVED turn=4 calls=3


    seed 42 answer-constrained ALLEY: FAILED turn=None calls=5


    seed 42 answer-constrained AUDIO: FAILED turn=None calls=5



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45 free SHORE: FAILED turn=None calls=1


    seed 45 free MIGHT: FAILED turn=None calls=2


    seed 45 free BRICK: SOLVED turn=4 calls=3


    seed 45 free GHOST: FAILED turn=None calls=4


    seed 45 free KNIFE: FAILED turn=None calls=1


    seed 45 free DOUBT: SOLVED turn=5 calls=4


    seed 45 free FLING: FAILED turn=None calls=1


    seed 45 free ROUND: FAILED turn=None calls=1


    seed 45 free CHAMP: FAILED turn=None calls=2


    seed 45 free WASTE: FAILED turn=None calls=2


    seed 45 free BLIND: FAILED turn=None calls=1


    seed 45 free POINT: FAILED turn=None calls=1


    seed 45 free SLATE: FAILED turn=None calls=1


    seed 45 free CRANE: FAILED turn=None calls=2


    seed 45 free APPLE: FAILED turn=None calls=1


    seed 45 free SHEEP: SOLVED turn=3 calls=2


    seed 45 free BANAL: FAILED turn=None calls=3


    seed 45 free ALLEY: FAILED turn=None calls=1


    seed 45 free AUDIO: FAILED turn=None calls=2


    seed 45 free-continue SHORE: FAILED turn=None calls=5


    seed 45 free-continue MIGHT: FAILED turn=None calls=5


    seed 45 free-continue BRICK: SOLVED turn=4 calls=3


    seed 45 free-continue GHOST: FAILED turn=None calls=5


    seed 45 free-continue KNIFE: FAILED turn=None calls=5


    seed 45 free-continue DOUBT: SOLVED turn=5 calls=4


    seed 45 free-continue FLING: FAILED turn=None calls=5


    seed 45 free-continue ROUND: FAILED turn=None calls=5


    seed 45 free-continue CHAMP: FAILED turn=None calls=5


    seed 45 free-continue WASTE: FAILED turn=None calls=5


    seed 45 free-continue BLIND: FAILED turn=None calls=5


    seed 45 free-continue POINT: FAILED turn=None calls=5


    seed 45 free-continue SLATE: FAILED turn=None calls=5


    seed 45 free-continue CRANE: FAILED turn=None calls=5


    seed 45 free-continue APPLE: FAILED turn=None calls=5


    seed 45 free-continue SHEEP: SOLVED turn=3 calls=2


    seed 45 free-continue BANAL: FAILED turn=None calls=5


    seed 45 free-continue ALLEY: FAILED turn=None calls=5


    seed 45 free-continue AUDIO: FAILED turn=None calls=5


    seed 45 answer-constrained SHORE: SOLVED turn=3 calls=2


    seed 45 answer-constrained MIGHT: FAILED turn=None calls=5


    seed 45 answer-constrained BRICK: SOLVED turn=4 calls=3


    seed 45 answer-constrained GHOST: FAILED turn=None calls=5


    seed 45 answer-constrained KNIFE: FAILED turn=None calls=5


    seed 45 answer-constrained DOUBT: SOLVED turn=6 calls=5


    seed 45 answer-constrained FLING: SOLVED turn=3 calls=2


    seed 45 answer-constrained ROUND: FAILED turn=None calls=5


    seed 45 answer-constrained CHAMP: SOLVED turn=4 calls=3


    seed 45 answer-constrained WASTE: FAILED turn=None calls=5


    seed 45 answer-constrained BLIND: SOLVED turn=5 calls=4


    seed 45 answer-constrained POINT: FAILED turn=None calls=5


    seed 45 answer-constrained SLATE: SOLVED turn=3 calls=2


    seed 45 answer-constrained CRANE: SOLVED turn=5 calls=4


    seed 45 answer-constrained APPLE: SOLVED turn=5 calls=4


    seed 45 answer-constrained SHEEP: FAILED turn=None calls=5


    seed 45 answer-constrained BANAL: SOLVED turn=4 calls=3


    seed 45 answer-constrained ALLEY: FAILED turn=None calls=5


    seed 45 answer-constrained AUDIO: FAILED turn=None calls=5



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 47 free SHORE: SOLVED turn=2 calls=1


    seed 47 free MIGHT: FAILED turn=None calls=1


    seed 47 free BRICK: SOLVED turn=6 calls=5


    seed 47 free GHOST: FAILED turn=None calls=2


    seed 47 free KNIFE: FAILED turn=None calls=1


    seed 47 free DOUBT: FAILED turn=None calls=1


    seed 47 free FLING: FAILED turn=None calls=1


    seed 47 free ROUND: FAILED turn=None calls=1


    seed 47 free CHAMP: FAILED turn=None calls=2


    seed 47 free WASTE: FAILED turn=None calls=2


    seed 47 free BLIND: FAILED turn=None calls=1


    seed 47 free POINT: FAILED turn=None calls=1


    seed 47 free SLATE: SOLVED turn=4 calls=3


    seed 47 free CRANE: FAILED turn=None calls=2


    seed 47 free APPLE: FAILED turn=None calls=1


    seed 47 free SHEEP: FAILED turn=None calls=2


    seed 47 free BANAL: FAILED turn=None calls=2


    seed 47 free ALLEY: FAILED turn=None calls=1


    seed 47 free AUDIO: FAILED turn=None calls=1


    seed 47 free-continue SHORE: SOLVED turn=2 calls=1


    seed 47 free-continue MIGHT: FAILED turn=None calls=5


    seed 47 free-continue BRICK: SOLVED turn=6 calls=5


    seed 47 free-continue GHOST: FAILED turn=None calls=5


    seed 47 free-continue KNIFE: FAILED turn=None calls=5


    seed 47 free-continue DOUBT: FAILED turn=None calls=5


    seed 47 free-continue FLING: FAILED turn=None calls=5


    seed 47 free-continue ROUND: FAILED turn=None calls=5


    seed 47 free-continue CHAMP: FAILED turn=None calls=5


    seed 47 free-continue WASTE: FAILED turn=None calls=5


    seed 47 free-continue BLIND: FAILED turn=None calls=5


    seed 47 free-continue POINT: FAILED turn=None calls=5


    seed 47 free-continue SLATE: SOLVED turn=4 calls=3


    seed 47 free-continue CRANE: FAILED turn=None calls=5


    seed 47 free-continue APPLE: FAILED turn=None calls=5


    seed 47 free-continue SHEEP: FAILED turn=None calls=5


    seed 47 free-continue BANAL: FAILED turn=None calls=5


    seed 47 free-continue ALLEY: FAILED turn=None calls=5


    seed 47 free-continue AUDIO: FAILED turn=None calls=5


    seed 47 answer-constrained SHORE: SOLVED turn=2 calls=1


    seed 47 answer-constrained MIGHT: SOLVED turn=5 calls=4


    seed 47 answer-constrained BRICK: SOLVED turn=5 calls=4


    seed 47 answer-constrained GHOST: SOLVED turn=5 calls=4


    seed 47 answer-constrained KNIFE: FAILED turn=None calls=5


    seed 47 answer-constrained DOUBT: FAILED turn=None calls=5


    seed 47 answer-constrained FLING: SOLVED turn=5 calls=4


    seed 47 answer-constrained ROUND: SOLVED turn=2 calls=1


    seed 47 answer-constrained CHAMP: FAILED turn=None calls=5


    seed 47 answer-constrained WASTE: FAILED turn=None calls=5


    seed 47 answer-constrained BLIND: SOLVED turn=6 calls=5


    seed 47 answer-constrained POINT: FAILED turn=None calls=5


    seed 47 answer-constrained SLATE: SOLVED turn=4 calls=3


    seed 47 answer-constrained CRANE: FAILED turn=None calls=5


    seed 47 answer-constrained APPLE: FAILED turn=None calls=5


    seed 47 answer-constrained SHEEP: SOLVED turn=5 calls=4


    seed 47 answer-constrained BANAL: SOLVED turn=4 calls=3


    seed 47 answer-constrained ALLEY: FAILED turn=None calls=5


    seed 47 answer-constrained AUDIO: FAILED turn=None calls=5


    calls: 594 games: 171


## 18d.8 Reproduce the Lab 18 seed-42 free baseline

The seed-42 free trajectory must match Lab 18 answer by answer and turn by turn.
This is the regression test that only the constrained decoder changed.


```python
seed42_free_calls = gameplay_calls.query(
    "seed == 42 and decoder == 'free'"
)
seed42_free_games = gameplay_games.query(
    "seed == 42 and decoder == 'free'"
)
validate_seed42_free(seed42_free_calls, seed42_free_games)
print("seed 42 free gameplay reproduces Lab 18 exactly")
```

    seed 42 free gameplay reproduces Lab 18 exactly


## 18d.9 Solve rate and paired decoder effect

Each row below is one independently trained adapter. Per-answer flips show which
reserved games change outcome, but the seed remains the replication unit.


```python
game_summary = gameplay_games.groupby(
    ["seed", "decoder"], sort=False
).agg(
    games=("answer", "size"),
    solved=("solved", "sum"),
    solve_rate=("solved", "mean"),
    invalid_termination_rate=("terminated_invalid", "mean"),
    mean_model_calls=("model_calls", "mean"),
    mean_final_candidates=("final_candidate_count", "mean"),
)
turns_on_wins = gameplay_games.loc[
    gameplay_games["solved"]
].groupby(["seed", "decoder"])["solved_turn"].mean()
game_summary["mean_turns_on_wins"] = turns_on_wins
game_summary = game_summary.reset_index()
display(game_summary)


paired_solve_rows = []
for seed in SEEDS:
    for left_decoder, right_decoder in [
        ("free", "free-continue"),
        ("free-continue", "answer-constrained"),
        ("free", "answer-constrained"),
    ]:
        left = gameplay_games.query(
            "seed == @seed and decoder == @left_decoder"
        )[["answer", "solved"]].rename(columns={"solved": "left"})
        right = gameplay_games.query(
            "seed == @seed and decoder == @right_decoder"
        )[["answer", "solved"]].rename(columns={"solved": "right"})
        paired = left.merge(
            right, on="answer", validate="one_to_one"
        )
        paired_solve_rows.append({
            "seed": seed,
            "left_decoder": left_decoder,
            "right_decoder": right_decoder,
            "left_solved": int(paired["left"].sum()),
            "right_solved": int(paired["right"].sum()),
            "solve_delta": (
                paired["right"].mean()
                - paired["left"].mean()
            ),
            "left_only": int(
                (paired["left"] & ~paired["right"]).sum()
            ),
            "right_only": int(
                (~paired["left"] & paired["right"]).sum()
            ),
            "both": int(
                (paired["left"] & paired["right"]).sum()
            ),
            "neither": int(
                (~paired["left"] & ~paired["right"]).sum()
            ),
        })
paired_solves = pd.DataFrame(paired_solve_rows)
display(paired_solves)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seed</th>
      <th>decoder</th>
      <th>games</th>
      <th>solved</th>
      <th>solve_rate</th>
      <th>invalid_termination_rate</th>
      <th>mean_model_calls</th>
      <th>mean_final_candidates</th>
      <th>mean_turns_on_wins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>free</td>
      <td>19</td>
      <td>5</td>
      <td>0.263158</td>
      <td>0.736842</td>
      <td>1.736842</td>
      <td>37.263158</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>free-continue</td>
      <td>19</td>
      <td>5</td>
      <td>0.263158</td>
      <td>0.000000</td>
      <td>4.421053</td>
      <td>37.263158</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>10</td>
      <td>0.526316</td>
      <td>0.000000</td>
      <td>4.105263</td>
      <td>1.210526</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>free</td>
      <td>19</td>
      <td>3</td>
      <td>0.157895</td>
      <td>0.842105</td>
      <td>1.842105</td>
      <td>19.421053</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>free-continue</td>
      <td>19</td>
      <td>3</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>4.684211</td>
      <td>19.421053</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>10</td>
      <td>0.526316</td>
      <td>0.000000</td>
      <td>4.052632</td>
      <td>1.315789</td>
      <td>4.2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>47</td>
      <td>free</td>
      <td>19</td>
      <td>3</td>
      <td>0.157895</td>
      <td>0.842105</td>
      <td>1.631579</td>
      <td>34.578947</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>47</td>
      <td>free-continue</td>
      <td>19</td>
      <td>3</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>4.684211</td>
      <td>34.578947</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>10</td>
      <td>0.526316</td>
      <td>0.000000</td>
      <td>4.105263</td>
      <td>1.210526</td>
      <td>4.3</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seed</th>
      <th>left_decoder</th>
      <th>right_decoder</th>
      <th>left_solved</th>
      <th>right_solved</th>
      <th>solve_delta</th>
      <th>left_only</th>
      <th>right_only</th>
      <th>both</th>
      <th>neither</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>free</td>
      <td>free-continue</td>
      <td>5</td>
      <td>5</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>free-continue</td>
      <td>answer-constrained</td>
      <td>5</td>
      <td>10</td>
      <td>0.263158</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>free</td>
      <td>answer-constrained</td>
      <td>5</td>
      <td>10</td>
      <td>0.263158</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>free</td>
      <td>free-continue</td>
      <td>3</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>free-continue</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>10</td>
      <td>0.368421</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>free</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>10</td>
      <td>0.368421</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>47</td>
      <td>free</td>
      <td>free-continue</td>
      <td>3</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>47</td>
      <td>free-continue</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>10</td>
      <td>0.368421</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>free</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>10</td>
      <td>0.368421</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>


## 18d.10 Action quality by decoder and turn

Constrained validity is guaranteed, but consistency is not. Turn 2 is displayed
separately because Lab 18c found replicated below-chance teacher match there.
Turn 2 is the only fully paired state comparison. Later turns are
conditional-on-survival and follow decoder-specific trajectories, so differences
there describe deployed behavior rather than a controlled one-step contrast.


```python
action_summary = gameplay_calls.groupby(
    ["seed", "decoder"], sort=False
).agg(
    calls=("answer", "size"),
    format_valid_rate=("format_valid", "mean"),
    in_lexicon_rate=("in_answer_lexicon", "mean"),
    history_consistency_rate=("history_consistent", "mean"),
    usable_rate=("usable", "mean"),
    repeat_rate=("repeated", "mean"),
    teacher_match_rate=("teacher_match", "mean"),
    mean_candidates_before=("candidate_count_before", "mean"),
    mean_candidates_after=("candidate_count_after", "mean"),
    mean_realized_log2_reduction=(
        "realized_log2_reduction", "mean"
    ),
)
display(action_summary.reset_index())

by_turn = gameplay_calls.groupby(
    ["seed", "decoder", "turn"], sort=False
).agg(
    calls=("answer", "size"),
    usable_rate=("usable", "mean"),
    consistency_rate=("history_consistent", "mean"),
    repeat_rate=("repeated", "mean"),
    teacher_match_rate=("teacher_match", "mean"),
    mean_candidates_before=("candidate_count_before", "mean"),
    mean_candidates_after=("candidate_count_after", "mean"),
    mean_entropy_gap_bits=("entropy_gap_bits", "mean"),
    mean_realized_log2_reduction=(
        "realized_log2_reduction", "mean"
    ),
)
display(by_turn.reset_index())

turn2 = gameplay_calls.query("turn == 2").groupby(
    ["seed", "decoder"], sort=False
).agg(
    calls=("answer", "size"),
    usable_rate=("usable", "mean"),
    teacher_match_rate=("teacher_match", "mean"),
    chosen_candidate_rate=("chosen_is_candidate", "mean"),
    mean_teacher_rank=("model_teacher_rank", "mean"),
    median_teacher_rank=("model_teacher_rank", "median"),
    mean_entropy_gap_bits=("entropy_gap_bits", "mean"),
    median_entropy_gap_bits=("entropy_gap_bits", "median"),
    mean_open_entropy_regret_bits=(
        "open_entropy_regret_bits", "mean"
    ),
    tier2_teacher_match_rate=("tier2_teacher_match", "mean"),
    mean_candidate_rank_percentile=(
        "candidate_rank_percentile", "mean"
    ),
    mean_realized_log2_reduction=(
        "realized_log2_reduction", "mean"
    ),
    mean_candidates_after=("candidate_count_after", "mean"),
)
display(turn2.reset_index())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seed</th>
      <th>decoder</th>
      <th>calls</th>
      <th>format_valid_rate</th>
      <th>in_lexicon_rate</th>
      <th>history_consistency_rate</th>
      <th>usable_rate</th>
      <th>repeat_rate</th>
      <th>teacher_match_rate</th>
      <th>mean_candidates_before</th>
      <th>mean_candidates_after</th>
      <th>mean_realized_log2_reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>free</td>
      <td>33</td>
      <td>0.787879</td>
      <td>0.575758</td>
      <td>0.303030</td>
      <td>0.303030</td>
      <td>0.000000</td>
      <td>0.151515</td>
      <td>30.696970</td>
      <td>23.696970</td>
      <td>0.963745</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>free-continue</td>
      <td>84</td>
      <td>0.583333</td>
      <td>0.226190</td>
      <td>0.119048</td>
      <td>0.119048</td>
      <td>0.000000</td>
      <td>0.059524</td>
      <td>45.369048</td>
      <td>42.619048</td>
      <td>0.378614</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>78</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.307692</td>
      <td>0.307692</td>
      <td>0.153846</td>
      <td>0.141026</td>
      <td>14.961538</td>
      <td>3.217949</td>
      <td>1.222411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>free</td>
      <td>35</td>
      <td>0.800000</td>
      <td>0.542857</td>
      <td>0.228571</td>
      <td>0.228571</td>
      <td>0.000000</td>
      <td>0.085714</td>
      <td>29.142857</td>
      <td>12.857143</td>
      <td>1.192860</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>free-continue</td>
      <td>89</td>
      <td>0.629213</td>
      <td>0.213483</td>
      <td>0.089888</td>
      <td>0.089888</td>
      <td>0.000000</td>
      <td>0.033708</td>
      <td>27.426966</td>
      <td>21.022472</td>
      <td>0.469102</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>77</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.337662</td>
      <td>0.337662</td>
      <td>0.142857</td>
      <td>0.116883</td>
      <td>14.623377</td>
      <td>2.753247</td>
      <td>1.212312</td>
    </tr>
    <tr>
      <th>6</th>
      <td>47</td>
      <td>free</td>
      <td>31</td>
      <td>0.741935</td>
      <td>0.483871</td>
      <td>0.225806</td>
      <td>0.225806</td>
      <td>0.000000</td>
      <td>0.064516</td>
      <td>32.677419</td>
      <td>23.580645</td>
      <td>0.862459</td>
    </tr>
    <tr>
      <th>7</th>
      <td>47</td>
      <td>free-continue</td>
      <td>89</td>
      <td>0.573034</td>
      <td>0.168539</td>
      <td>0.078652</td>
      <td>0.078652</td>
      <td>0.000000</td>
      <td>0.022472</td>
      <td>40.224719</td>
      <td>37.056180</td>
      <td>0.300407</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>78</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.294872</td>
      <td>0.294872</td>
      <td>0.115385</td>
      <td>0.128205</td>
      <td>15.576923</td>
      <td>3.833333</td>
      <td>1.214911</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seed</th>
      <th>decoder</th>
      <th>turn</th>
      <th>calls</th>
      <th>usable_rate</th>
      <th>consistency_rate</th>
      <th>repeat_rate</th>
      <th>teacher_match_rate</th>
      <th>mean_candidates_before</th>
      <th>mean_candidates_after</th>
      <th>mean_entropy_gap_bits</th>
      <th>mean_realized_log2_reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>free</td>
      <td>2</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>39.736842</td>
      <td>0.952102</td>
      <td>0.923485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>free</td>
      <td>3</td>
      <td>7</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.000000</td>
      <td>0.285714</td>
      <td>9.000000</td>
      <td>2.714286</td>
      <td>0.174060</td>
      <td>1.667489</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>free</td>
      <td>4</td>
      <td>4</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>1.750000</td>
      <td>1.250000</td>
      <td>0.416667</td>
      <td>0.396241</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>free</td>
      <td>5</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>free</td>
      <td>6</td>
      <td>1</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>42</td>
      <td>free-continue</td>
      <td>2</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>39.736842</td>
      <td>0.952102</td>
      <td>0.923485</td>
    </tr>
    <tr>
      <th>6</th>
      <td>42</td>
      <td>free-continue</td>
      <td>3</td>
      <td>18</td>
      <td>0.166667</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>41.888889</td>
      <td>39.444444</td>
      <td>0.174060</td>
      <td>0.648468</td>
    </tr>
    <tr>
      <th>7</th>
      <td>42</td>
      <td>free-continue</td>
      <td>4</td>
      <td>17</td>
      <td>0.117647</td>
      <td>0.117647</td>
      <td>0.000000</td>
      <td>0.117647</td>
      <td>41.705882</td>
      <td>41.588235</td>
      <td>0.416667</td>
      <td>0.093233</td>
    </tr>
    <tr>
      <th>8</th>
      <td>42</td>
      <td>free-continue</td>
      <td>5</td>
      <td>15</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>47.000000</td>
      <td>46.933333</td>
      <td>0.000000</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>42</td>
      <td>free-continue</td>
      <td>6</td>
      <td>15</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>46.933333</td>
      <td>46.933333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>2</td>
      <td>19</td>
      <td>0.210526</td>
      <td>0.210526</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>7.578947</td>
      <td>0.884257</td>
      <td>2.807769</td>
    </tr>
    <tr>
      <th>11</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>18</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>7.944444</td>
      <td>2.611111</td>
      <td>0.521136</td>
      <td>1.531774</td>
    </tr>
    <tr>
      <th>12</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>4</td>
      <td>17</td>
      <td>0.352941</td>
      <td>0.352941</td>
      <td>0.176471</td>
      <td>0.235294</td>
      <td>2.705882</td>
      <td>1.588235</td>
      <td>0.233894</td>
      <td>0.554617</td>
    </tr>
    <tr>
      <th>13</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>5</td>
      <td>13</td>
      <td>0.384615</td>
      <td>0.384615</td>
      <td>0.307692</td>
      <td>0.230769</td>
      <td>1.769231</td>
      <td>1.384615</td>
      <td>0.153846</td>
      <td>0.339618</td>
    </tr>
    <tr>
      <th>14</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>6</td>
      <td>11</td>
      <td>0.272727</td>
      <td>0.272727</td>
      <td>0.454545</td>
      <td>0.272727</td>
      <td>1.454545</td>
      <td>1.363636</td>
      <td>0.181818</td>
      <td>0.053178</td>
    </tr>
    <tr>
      <th>15</th>
      <td>45</td>
      <td>free</td>
      <td>2</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>21.000000</td>
      <td>0.779973</td>
      <td>1.555673</td>
    </tr>
    <tr>
      <th>16</th>
      <td>45</td>
      <td>free</td>
      <td>3</td>
      <td>10</td>
      <td>0.300000</td>
      <td>0.300000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>7.300000</td>
      <td>4.400000</td>
      <td>0.273335</td>
      <td>1.119229</td>
    </tr>
    <tr>
      <th>17</th>
      <td>45</td>
      <td>free</td>
      <td>4</td>
      <td>4</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>1.250000</td>
      <td>1.250000</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>45</td>
      <td>free</td>
      <td>5</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>1.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>45</td>
      <td>free-continue</td>
      <td>2</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>21.000000</td>
      <td>0.779973</td>
      <td>1.555673</td>
    </tr>
    <tr>
      <th>20</th>
      <td>45</td>
      <td>free-continue</td>
      <td>3</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>21.000000</td>
      <td>19.473684</td>
      <td>0.273335</td>
      <td>0.589068</td>
    </tr>
    <tr>
      <th>21</th>
      <td>45</td>
      <td>free-continue</td>
      <td>4</td>
      <td>18</td>
      <td>0.055556</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>20.500000</td>
      <td>20.500000</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>45</td>
      <td>free-continue</td>
      <td>5</td>
      <td>17</td>
      <td>0.058824</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>21.647059</td>
      <td>21.588235</td>
      <td>0.000000</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>23</th>
      <td>45</td>
      <td>free-continue</td>
      <td>6</td>
      <td>16</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>22.875000</td>
      <td>22.875000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>2</td>
      <td>19</td>
      <td>0.368421</td>
      <td>0.368421</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>5.157895</td>
      <td>0.777557</td>
      <td>3.317823</td>
    </tr>
    <tr>
      <th>25</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>19</td>
      <td>0.368421</td>
      <td>0.368421</td>
      <td>0.000000</td>
      <td>0.105263</td>
      <td>5.157895</td>
      <td>2.473684</td>
      <td>0.597538</td>
      <td>0.868190</td>
    </tr>
    <tr>
      <th>26</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>4</td>
      <td>16</td>
      <td>0.312500</td>
      <td>0.312500</td>
      <td>0.250000</td>
      <td>0.125000</td>
      <td>2.750000</td>
      <td>1.875000</td>
      <td>0.408529</td>
      <td>0.494181</td>
    </tr>
    <tr>
      <th>27</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>5</td>
      <td>13</td>
      <td>0.384615</td>
      <td>0.384615</td>
      <td>0.230769</td>
      <td>0.230769</td>
      <td>2.076923</td>
      <td>1.615385</td>
      <td>0.230769</td>
      <td>0.332456</td>
    </tr>
    <tr>
      <th>28</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>6</td>
      <td>10</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.400000</td>
      <td>0.200000</td>
      <td>1.800000</td>
      <td>1.600000</td>
      <td>0.281128</td>
      <td>0.158496</td>
    </tr>
    <tr>
      <th>29</th>
      <td>47</td>
      <td>free</td>
      <td>2</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>35.473684</td>
      <td>1.024697</td>
      <td>1.113225</td>
    </tr>
    <tr>
      <th>30</th>
      <td>47</td>
      <td>free</td>
      <td>3</td>
      <td>8</td>
      <td>0.125000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.500000</td>
      <td>6.500000</td>
      <td>0.038910</td>
      <td>0.573120</td>
    </tr>
    <tr>
      <th>31</th>
      <td>47</td>
      <td>free</td>
      <td>4</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>1.500000</td>
      <td>1.500000</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>47</td>
      <td>free</td>
      <td>5</td>
      <td>1</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>47</td>
      <td>free</td>
      <td>6</td>
      <td>1</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>47</td>
      <td>free-continue</td>
      <td>2</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>35.473684</td>
      <td>1.024697</td>
      <td>1.113225</td>
    </tr>
    <tr>
      <th>35</th>
      <td>47</td>
      <td>free-continue</td>
      <td>3</td>
      <td>18</td>
      <td>0.055556</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>37.388889</td>
      <td>36.500000</td>
      <td>0.038910</td>
      <td>0.254720</td>
    </tr>
    <tr>
      <th>36</th>
      <td>47</td>
      <td>free-continue</td>
      <td>4</td>
      <td>18</td>
      <td>0.055556</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>36.500000</td>
      <td>36.500000</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>47</td>
      <td>free-continue</td>
      <td>5</td>
      <td>17</td>
      <td>0.058824</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>38.588235</td>
      <td>38.529412</td>
      <td>0.000000</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>38</th>
      <td>47</td>
      <td>free-continue</td>
      <td>6</td>
      <td>17</td>
      <td>0.058824</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>38.529412</td>
      <td>38.529412</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>2</td>
      <td>19</td>
      <td>0.210526</td>
      <td>0.210526</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>10.052632</td>
      <td>1.172647</td>
      <td>2.366765</td>
    </tr>
    <tr>
      <th>40</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>17</td>
      <td>0.352941</td>
      <td>0.352941</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>11.117647</td>
      <td>2.647059</td>
      <td>0.551260</td>
      <td>2.052215</td>
    </tr>
    <tr>
      <th>41</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>4</td>
      <td>17</td>
      <td>0.176471</td>
      <td>0.176471</td>
      <td>0.117647</td>
      <td>0.117647</td>
      <td>2.647059</td>
      <td>1.705882</td>
      <td>0.517790</td>
      <td>0.489525</td>
    </tr>
    <tr>
      <th>42</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>5</td>
      <td>15</td>
      <td>0.533333</td>
      <td>0.533333</td>
      <td>0.200000</td>
      <td>0.333333</td>
      <td>1.800000</td>
      <td>1.333333</td>
      <td>0.172331</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>6</td>
      <td>10</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.400000</td>
      <td>0.200000</td>
      <td>1.500000</td>
      <td>1.400000</td>
      <td>0.258496</td>
      <td>0.058496</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seed</th>
      <th>decoder</th>
      <th>calls</th>
      <th>usable_rate</th>
      <th>teacher_match_rate</th>
      <th>chosen_candidate_rate</th>
      <th>mean_teacher_rank</th>
      <th>median_teacher_rank</th>
      <th>mean_entropy_gap_bits</th>
      <th>median_entropy_gap_bits</th>
      <th>mean_open_entropy_regret_bits</th>
      <th>tier2_teacher_match_rate</th>
      <th>mean_candidate_rank_percentile</th>
      <th>mean_realized_log2_reduction</th>
      <th>mean_candidates_after</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>free</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.952102</td>
      <td>1.058764</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.923485</td>
      <td>39.736842</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>free-continue</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.952102</td>
      <td>1.058764</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.923485</td>
      <td>39.736842</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.210526</td>
      <td>0.0</td>
      <td>0.210526</td>
      <td>281.263158</td>
      <td>244.0</td>
      <td>0.884257</td>
      <td>0.800000</td>
      <td>1.158016</td>
      <td>0.000000</td>
      <td>0.083417</td>
      <td>2.807769</td>
      <td>7.578947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>free</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.779973</td>
      <td>0.800929</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.555673</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>free-continue</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.779973</td>
      <td>0.800929</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.555673</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.368421</td>
      <td>0.0</td>
      <td>0.368421</td>
      <td>243.578947</td>
      <td>210.0</td>
      <td>0.777557</td>
      <td>0.794693</td>
      <td>1.051316</td>
      <td>0.000000</td>
      <td>0.086622</td>
      <td>3.317823</td>
      <td>5.157895</td>
    </tr>
    <tr>
      <th>6</th>
      <td>47</td>
      <td>free</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.024697</td>
      <td>1.184322</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.113225</td>
      <td>35.473684</td>
    </tr>
    <tr>
      <th>7</th>
      <td>47</td>
      <td>free-continue</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.024697</td>
      <td>1.184322</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.113225</td>
      <td>35.473684</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.210526</td>
      <td>0.0</td>
      <td>0.210526</td>
      <td>285.947368</td>
      <td>254.0</td>
      <td>1.172647</td>
      <td>1.224417</td>
      <td>1.446406</td>
      <td>0.052632</td>
      <td>0.086815</td>
      <td>2.366765</td>
      <td>10.052632</td>
    </tr>
  </tbody>
</table>
</div>


## 18d.11 Constrained strategic ranking on visited states

These rows exclude the free decoder because it does not compute a full answer
ranking. Negative entropy gaps identify exploratory out-of-candidate actions
that split the candidate set more than the canonical candidate-only teacher.


```python
constrained_calls = gameplay_calls.query(
    "decoder == 'answer-constrained'"
).copy()
constrained_strategy = constrained_calls.groupby(
    ["seed", "turn"], sort=False
).agg(
    calls=("answer", "size"),
    candidate_action_rate=("chosen_is_candidate", "mean"),
    teacher_match_rate=("teacher_match", "mean"),
    median_model_teacher_rank=("model_teacher_rank", "median"),
    median_best_candidate_rank=("model_best_candidate_rank", "median"),
    mean_candidate_rank_percentile=(
        "candidate_rank_percentile", "mean"
    ),
    mean_candidate_mass=("candidate_mass", "mean"),
    mean_candidate_mass_lift=("candidate_mass_lift", "mean"),
    tier2_teacher_match_rate=("tier2_teacher_match", "mean"),
    mean_tier2_entropy_gap_bits=(
        "tier2_entropy_gap_bits", "mean"
    ),
    mean_chosen_entropy_bits=("chosen_entropy_bits", "mean"),
    mean_teacher_entropy_bits=("teacher_entropy_bits", "mean"),
    mean_entropy_gap_bits=("entropy_gap_bits", "mean"),
    mean_open_entropy_regret_bits=(
        "open_entropy_regret_bits", "mean"
    ),
    mean_chosen_solve_probability=(
        "chosen_solve_probability", "mean"
    ),
    negative_entropy_gap_rate=(
        "entropy_gap_bits", lambda values: (values < 0).mean()
    ),
    mean_realized_log2_reduction=(
        "realized_log2_reduction", "mean"
    ),
)
display(constrained_strategy.reset_index())

candidate_policy_calls = constrained_calls.loc[
    constrained_calls["chosen_is_candidate"]
]
candidate_policy_summary = candidate_policy_calls.groupby(
    ["seed", "turn"], sort=False
).agg(
    calls=("answer", "size"),
    teacher_match_rate=("teacher_match", "mean"),
    median_teacher_entropy_rank=(
        "chosen_candidate_entropy_rank", "median"
    ),
    mean_entropy_regret_bits=("entropy_gap_bits", "mean"),
)
display(candidate_policy_summary.reset_index())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seed</th>
      <th>turn</th>
      <th>calls</th>
      <th>candidate_action_rate</th>
      <th>teacher_match_rate</th>
      <th>median_model_teacher_rank</th>
      <th>median_best_candidate_rank</th>
      <th>mean_candidate_rank_percentile</th>
      <th>mean_candidate_mass</th>
      <th>mean_candidate_mass_lift</th>
      <th>tier2_teacher_match_rate</th>
      <th>mean_tier2_entropy_gap_bits</th>
      <th>mean_chosen_entropy_bits</th>
      <th>mean_teacher_entropy_bits</th>
      <th>mean_entropy_gap_bits</th>
      <th>mean_open_entropy_regret_bits</th>
      <th>mean_chosen_solve_probability</th>
      <th>negative_entropy_gap_rate</th>
      <th>mean_realized_log2_reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>2</td>
      <td>19</td>
      <td>0.210526</td>
      <td>0.000000</td>
      <td>244.0</td>
      <td>2.0</td>
      <td>0.083417</td>
      <td>0.290236</td>
      <td>33.036849</td>
      <td>0.000000</td>
      <td>0.584024</td>
      <td>2.792719</td>
      <td>3.676975</td>
      <td>0.884257</td>
      <td>1.158016</td>
      <td>0.016953</td>
      <td>0.0</td>
      <td>2.807769</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>3</td>
      <td>18</td>
      <td>0.333333</td>
      <td>0.055556</td>
      <td>16.0</td>
      <td>4.5</td>
      <td>0.020695</td>
      <td>0.287967</td>
      <td>134.044274</td>
      <td>0.111111</td>
      <td>0.222108</td>
      <td>1.460677</td>
      <td>1.981812</td>
      <td>0.521136</td>
      <td>0.789360</td>
      <td>0.063685</td>
      <td>0.0</td>
      <td>1.531774</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>4</td>
      <td>17</td>
      <td>0.352941</td>
      <td>0.235294</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0.007368</td>
      <td>0.266722</td>
      <td>428.486085</td>
      <td>0.705882</td>
      <td>0.050420</td>
      <td>0.656858</td>
      <td>0.890752</td>
      <td>0.233894</td>
      <td>0.344818</td>
      <td>0.208403</td>
      <td>0.0</td>
      <td>0.554617</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>5</td>
      <td>13</td>
      <td>0.384615</td>
      <td>0.230769</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.002816</td>
      <td>0.277600</td>
      <td>364.958711</td>
      <td>0.769231</td>
      <td>0.000000</td>
      <td>0.370098</td>
      <td>0.523944</td>
      <td>0.153846</td>
      <td>0.206825</td>
      <td>0.211538</td>
      <td>0.0</td>
      <td>0.339618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>6</td>
      <td>11</td>
      <td>0.272727</td>
      <td>0.272727</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.002785</td>
      <td>0.192849</td>
      <td>361.556198</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.083481</td>
      <td>0.265300</td>
      <td>0.181818</td>
      <td>0.242424</td>
      <td>0.212121</td>
      <td>0.0</td>
      <td>0.053178</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>2</td>
      <td>19</td>
      <td>0.368421</td>
      <td>0.000000</td>
      <td>210.0</td>
      <td>2.0</td>
      <td>0.086622</td>
      <td>0.288871</td>
      <td>28.419676</td>
      <td>0.000000</td>
      <td>0.536420</td>
      <td>2.899418</td>
      <td>3.676975</td>
      <td>0.777557</td>
      <td>1.051316</td>
      <td>0.012467</td>
      <td>0.0</td>
      <td>3.317823</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>3</td>
      <td>19</td>
      <td>0.368421</td>
      <td>0.105263</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>0.023529</td>
      <td>0.257663</td>
      <td>284.590905</td>
      <td>0.263158</td>
      <td>0.223171</td>
      <td>0.947457</td>
      <td>1.544995</td>
      <td>0.597538</td>
      <td>0.780330</td>
      <td>0.168084</td>
      <td>0.0</td>
      <td>0.868190</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>4</td>
      <td>16</td>
      <td>0.312500</td>
      <td>0.125000</td>
      <td>8.0</td>
      <td>2.5</td>
      <td>0.007906</td>
      <td>0.242149</td>
      <td>357.909909</td>
      <td>0.500000</td>
      <td>0.028697</td>
      <td>0.515249</td>
      <td>0.923778</td>
      <td>0.408529</td>
      <td>0.610612</td>
      <td>0.179167</td>
      <td>0.0</td>
      <td>0.494181</td>
    </tr>
    <tr>
      <th>8</th>
      <td>45</td>
      <td>5</td>
      <td>13</td>
      <td>0.384615</td>
      <td>0.230769</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.002816</td>
      <td>0.201379</td>
      <td>250.838448</td>
      <td>0.769231</td>
      <td>0.000000</td>
      <td>0.342423</td>
      <td>0.573192</td>
      <td>0.230769</td>
      <td>0.427338</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.332456</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>6</td>
      <td>10</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.002891</td>
      <td>0.168610</td>
      <td>191.071652</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.191830</td>
      <td>0.472957</td>
      <td>0.281128</td>
      <td>0.416667</td>
      <td>0.083333</td>
      <td>0.0</td>
      <td>0.158496</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47</td>
      <td>2</td>
      <td>19</td>
      <td>0.210526</td>
      <td>0.000000</td>
      <td>254.0</td>
      <td>4.0</td>
      <td>0.086815</td>
      <td>0.269467</td>
      <td>34.816526</td>
      <td>0.052632</td>
      <td>0.520393</td>
      <td>2.504328</td>
      <td>3.676975</td>
      <td>1.172647</td>
      <td>1.446406</td>
      <td>0.016953</td>
      <td>0.0</td>
      <td>2.366765</td>
    </tr>
    <tr>
      <th>11</th>
      <td>47</td>
      <td>3</td>
      <td>17</td>
      <td>0.352941</td>
      <td>0.058824</td>
      <td>20.0</td>
      <td>2.0</td>
      <td>0.024446</td>
      <td>0.305863</td>
      <td>65.834929</td>
      <td>0.176471</td>
      <td>0.373059</td>
      <td>1.838635</td>
      <td>2.389895</td>
      <td>0.551260</td>
      <td>0.834003</td>
      <td>0.031708</td>
      <td>0.0</td>
      <td>2.052215</td>
    </tr>
    <tr>
      <th>12</th>
      <td>47</td>
      <td>4</td>
      <td>17</td>
      <td>0.176471</td>
      <td>0.117647</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>0.007836</td>
      <td>0.212607</td>
      <td>302.474587</td>
      <td>0.588235</td>
      <td>0.034962</td>
      <td>0.411474</td>
      <td>0.929264</td>
      <td>0.517790</td>
      <td>0.638046</td>
      <td>0.129412</td>
      <td>0.0</td>
      <td>0.489525</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>5</td>
      <td>15</td>
      <td>0.533333</td>
      <td>0.333333</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.004015</td>
      <td>0.212895</td>
      <td>264.472122</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.426416</td>
      <td>0.598747</td>
      <td>0.172331</td>
      <td>0.218246</td>
      <td>0.338889</td>
      <td>0.0</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>47</td>
      <td>6</td>
      <td>10</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>4.5</td>
      <td>3.5</td>
      <td>0.003326</td>
      <td>0.174102</td>
      <td>331.070865</td>
      <td>0.900000</td>
      <td>0.000000</td>
      <td>0.091830</td>
      <td>0.350326</td>
      <td>0.258496</td>
      <td>0.325163</td>
      <td>0.133333</td>
      <td>0.0</td>
      <td>0.058496</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seed</th>
      <th>turn</th>
      <th>calls</th>
      <th>teacher_match_rate</th>
      <th>median_teacher_entropy_rank</th>
      <th>mean_entropy_regret_bits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>2</td>
      <td>4</td>
      <td>0.000000</td>
      <td>7.5</td>
      <td>0.594161</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>3</td>
      <td>6</td>
      <td>0.166667</td>
      <td>2.0</td>
      <td>0.246303</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>5</td>
      <td>5</td>
      <td>0.600000</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>6</td>
      <td>3</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>4</td>
      <td>6</td>
      <td>0.666667</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>3</td>
      <td>7</td>
      <td>0.285714</td>
      <td>3.0</td>
      <td>0.333450</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>4</td>
      <td>5</td>
      <td>0.400000</td>
      <td>2.0</td>
      <td>0.091830</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>2</td>
      <td>7</td>
      <td>0.000000</td>
      <td>15.0</td>
      <td>0.447285</td>
    </tr>
    <tr>
      <th>8</th>
      <td>45</td>
      <td>6</td>
      <td>2</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>5</td>
      <td>5</td>
      <td>0.600000</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47</td>
      <td>2</td>
      <td>4</td>
      <td>0.000000</td>
      <td>6.5</td>
      <td>0.494161</td>
    </tr>
    <tr>
      <th>11</th>
      <td>47</td>
      <td>5</td>
      <td>8</td>
      <td>0.625000</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>47</td>
      <td>3</td>
      <td>6</td>
      <td>0.166667</td>
      <td>3.5</td>
      <td>0.430755</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>4</td>
      <td>3</td>
      <td>0.666667</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>47</td>
      <td>6</td>
      <td>2</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


## 18d.12 Persist the diagnostic

The per-call table is the durable artifact. It contains every visited state,
chosen action, candidate trajectory, and teacher diagnostic needed to redesign
Lab 19 without rerunning the models.


```python
gameplay_calls.to_csv(
    RESULTS_DIR / "gameplay-calls.csv", index=False
)
gameplay_games.to_csv(
    RESULTS_DIR / "gameplay-games.csv", index=False
)
game_summary.to_csv(
    RESULTS_DIR / "game-summary.csv", index=False
)
paired_solves.to_csv(
    RESULTS_DIR / "paired-solves.csv", index=False
)
action_summary.reset_index().to_csv(
    RESULTS_DIR / "action-summary.csv", index=False
)
by_turn.reset_index().to_csv(
    RESULTS_DIR / "by-turn.csv", index=False
)
turn2.reset_index().to_csv(
    RESULTS_DIR / "turn2-summary.csv", index=False
)
constrained_strategy.reset_index().to_csv(
    RESULTS_DIR / "constrained-strategy.csv", index=False
)
candidate_policy_summary.reset_index().to_csv(
    RESULTS_DIR / "candidate-policy-summary.csv", index=False
)

run_manifest = {
    "experiment": "Lab 18d constrained full-game evaluation",
    "model_id": MODEL_ID,
    "seeds": SEEDS,
    "decoders": DECODERS,
    "answers": list(DEFAULT_EVAL_ANSWERS),
    "opening": OPENING,
    "max_turns": MAX_TURNS,
    "action_space": "2,315 answer words",
    "scoring_rule": "summed log P(word tokens + EOS | prompt)",
    "checkpoint_sha256": checkpoint_hashes,
}
(RESULTS_DIR / "lab18d-run.json").write_text(
    json.dumps(run_manifest, indent=2)
)
print("written to", RESULTS_DIR)
```

    written to ../results/lab18d


## Lab 18d checkpoint

Read the result in this order:

1. Did answer-constrained solve rate exceed free solve rate for every seed?
2. Did the gain come from avoiding invalid termination, or from better actions?
3. On Turn 2, how often was the constrained action a current candidate?
4. Was exact teacher mismatch accompanied by large entropy regret, or by a
   near-tied alternative?
5. Did constrained Turn 2 actions reduce the realized candidate set more than
   free actions?
6. Which failures remain after lexical grounding is removed?

If constrained games still collapse with large Turn 2 entropy gaps, Lab 19
should distill relative teacher scores. If gaps are small, exact teacher match is
the wrong target and the next diagnosis should follow trajectory-level failure.
