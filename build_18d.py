"""Generate notebooks/18d_constrained_gameplay.ipynb."""

import json
from pathlib import Path


cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md("""
# Lab 18d - Does constrained policy improve full games?

Labs 18b and 18c established a replicated state-level capability. Across three
B-structured seeds, free generation produced usable actions on 14.5% to 16.3%
of held-out states, while ranking the 2,315 answer words produced 30.0% to
31.0%. Candidate-rank percentile stayed near 0.029.

That does not establish gameplay. Errors change the next state, repeated actions
waste turns, and the largest replicated weakness is strategic choice under broad
Turn 2 uncertainty. This lab changes only the action decoder and plays the same
19 reserved answers with every B seed.
""")

md("""
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
""")

md("""
## 18d.2 Run controls and memory guard

Run only through the system watchdog:

```
scripts/memguard.py --min-free 64 -- uv run jupyter nbconvert \\
    --to notebook --execute --inplace notebooks/18d_constrained_gameplay.ipynb
```

The scoring kernel is Lab 18b's verified answer ranker. Before gameplay, seed 42
must reproduce one persisted Lab 18b score vector, then pass a 40-iteration
fixed-shape memory soak on the longest prompt in the 620-state battery.
Gameplay artifacts are written after every answer, so interruption loses at most
one game.
""")

code("""
RUN_EVALUATION = True
MEMORY_CAP_GIB = 128.0

import torch

if torch.backends.mps.is_available():
    total_gib = torch.mps.recommended_max_memory() / 1024**3
    torch.mps.set_per_process_memory_fraction(MEMORY_CAP_GIB / total_gib)
    print(f"MPS cap: {MEMORY_CAP_GIB:.0f} GiB of {total_gib:.0f} GiB")

print("RUN_EVALUATION:", RUN_EVALUATION)
""")

code("""
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
""")

md("""
## 18d.3 Structured gameplay prompts

These functions reproduce the Lab 17 representation and Lab 18 gameplay prompt.
Every model starts after the fixed `RAISE` opening, so the first model decision
is Turn 2.
""")

code("""
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
    return "\\n".join([
        f"GREENS: {greens}",
        f"LETTER_COUNTS: {', '.join(counts) or 'NONE'}",
        f"EXCLUDED_POSITIONS: {', '.join(excluded) or 'NONE'}",
        f"ABSENT_LETTERS: {' '.join(absent) or 'NONE'}",
        f"PREVIOUS_GUESSES: {', '.join(state['previous_guesses']) or 'NONE'}",
        f"CANDIDATE_COUNT: {candidate_count}",
    ])


def format_training_history(history: list[Turn]) -> str:
    return "\\n".join(
        f"{' '.join(turn.guess)} -> {' '.join(turn.feedback)}"
        for turn in history
    )


def raw_next_guess_prompt(history: list[Turn]) -> str:
    return (
        "Task: NEXT_GUESS\\n"
        "You are playing Wordle.\\n"
        "Use the game history to choose the next guess.\\n"
        "Return exactly one uppercase five-letter word.\\n\\n"
        f"History:\\n{format_training_history(history)}"
    )


def structured_next_guess_prompt(history: list[Turn]) -> str:
    candidate_count = len(filter_candidates(ANSWERS, history))
    prefix = raw_next_guess_prompt(history).split("\\n\\nHistory:\\n", 1)[0]
    return (
        prefix
        + "\\n\\nDerived state:\\n"
        + render_structured_state(history, candidate_count)
    )
""")

md("""
## 18d.4 Verified answer-list scoring kernel

The constrained action maximizes summed
`log P(word tokens + EOS | structured prompt)`. Words are bucketed by token
length, prompt KV state is reused across chunks, and only needed logits are
materialized.
""")

code("""
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
""")

md("""
## 18d.5 Correctness and fixed-shape memory gate

Seed 42 must reproduce Lab 18b's first persisted score vector within float32
tolerance. The same model then scores one fixed longest battery prompt 40 times.
The final third must be flat before any game runs.
""")

code("""
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
""")

md("""
## 18d.6 Gameplay engine and strategic diagnostics

The teacher is evaluated on the exact state the model visits. Entropy is the
expected information gain of an action over the current candidate set. Realized
candidate reduction is answer-specific and therefore complements expected
entropy.
""")

code("""
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
""")

md("""
## 18d.7 Restartable paired gameplay

Each seed uses one loaded adapter for all decoders. Free runs first because it
is the frozen Lab 18 behavior; free-continue and answer-constrained follow.
Artifacts are atomically rewritten after every completed answer and validated
before reuse. Constrained score vectors and their `(answer, turn)` keys are
checkpointed at the same cadence.
""")

code("""
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
""")

md("""
## 18d.8 Reproduce the Lab 18 seed-42 free baseline

The seed-42 free trajectory must match Lab 18 answer by answer and turn by turn.
This is the regression test that only the constrained decoder changed.
""")

code("""
seed42_free_calls = gameplay_calls.query(
    "seed == 42 and decoder == 'free'"
)
seed42_free_games = gameplay_games.query(
    "seed == 42 and decoder == 'free'"
)
validate_seed42_free(seed42_free_calls, seed42_free_games)
print("seed 42 free gameplay reproduces Lab 18 exactly")
""")

md("""
## 18d.9 Solve rate and paired decoder effect

Each row below is one independently trained adapter. Per-answer flips show which
reserved games change outcome, but the seed remains the replication unit.
""")

code("""
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
""")

md("""
## 18d.10 Action quality by decoder and turn

Constrained validity is guaranteed, but consistency is not. Turn 2 is displayed
separately because Lab 18c found replicated below-chance teacher match there.
Turn 2 is the only fully paired state comparison. Later turns are
conditional-on-survival and follow decoder-specific trajectories, so differences
there describe deployed behavior rather than a controlled one-step contrast.
""")

code("""
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
""")

md("""
## 18d.11 Constrained strategic ranking on visited states

These rows exclude the free decoder because it does not compute a full answer
ranking. Negative entropy gaps identify exploratory out-of-candidate actions
that split the candidate set more than the canonical candidate-only teacher.
""")

code("""
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
""")

md("""
## 18d.12 Persist the diagnostic

The per-call table is the durable artifact. It contains every visited state,
chosen action, candidate trajectory, and teacher diagnostic needed to redesign
Lab 19 without rerunning the models.
""")

code("""
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
""")

md("""
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
""")


for index, cell in enumerate(cells):
    cell["id"] = f"lab18d-{index:02d}-{cell['cell_type']}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = Path("notebooks/18d_constrained_gameplay.ipynb")
path.write_text(json.dumps(notebook, indent=1))
print(f"wrote {path} with {len(cells)} cells")
