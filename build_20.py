"""Generate notebooks/20_policy_state_correction.ipynb."""

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
# Lab 20 - Correct policy-created states

Lab 18d gave every B-structured seed a working incumbent: 10 of 19 reserved
answer-constrained games solved, with a replicated strategic weakness at broad
Turn 2 states and at singleton closure. Lab 19 tried to fix that weakness by
continuing each incumbent against a 12-action distillation objective, mining
four hard negatives per state from the incumbent's own full-list ranking.

Lab 19 failed on the deployed action space, not on its training objective.
Both arms learned their twelve-action target, but constrained solve rate fell
from 10/19 for every incumbent to 4/19, 4/19, and 1/19 under `hard`, and 0/19
under `value`. Lab 19d then read the persisted 2,315-word score vectors and
found the same shape of failure on all three seeds:

* **Global rank collapse.** On paired Turn 2 states the new winner's rank under
  the *frozen* incumbent had a median between 130 and 657 out of 2,315. These
  were not marginal improvements to words the incumbent already favored; they
  were words the incumbent had barely considered.
* **Generic attractors.** Across the 19 paired Turn 2 states, `hard` and
  `value` each collapsed onto only 3 to 5 distinct winning words per seed, and
  the single most common winner captured 7 to 15 of the 19 states (37% to
  79%). A twelve-action objective that never compares two states directly
  still produced one word winning almost everywhere.
* **Candidate-mass loss.** Mean candidate probability mass fell by 0.20 to
  0.25 (on a 0-1 scale) at the same paired states, so the collapse was not only
  a change of winner, it was a loss of usable signal across the whole
  candidate set.
* **Singleton answer rank collapse.** On states the deployed trajectory itself
  reached with exactly one candidate left, the incumbent's median full-list
  rank for that sole candidate was 1.5 to 3. After training it was 41 to 168.5
  under `hard` and 381 to 996 under `value`, out of 2,315 words - the state
  that should be a certain win became a near-random one.

Lab 19d's diagnostic could not separate two explanations: frozen hard
negatives becoming stale as the policy moved during training, or a broader
preservation failure with no cause specific to negative mining. Lab 20 does
not resolve that mechanism. It tests a different lever entirely: whether the
*source* of new correction states, not the training objective, explains
whether full-game behavior improves.
""")

md("""
## 20.1 Pre-registered experiment

**Question.** Does expert correction on states reached by the policy improve
held-out full-game behavior more than the same amount of static expert data,
when every arm receives the same incumbent-preservation mechanism?

This is supervised learning, not RL. Every training presentation is a
`(prompt, response)` pair scored with the same response-only token
cross-entropy Lab 18c used. Nothing samples an action during training, nothing
optimizes a reward, and no arm ever trains against its own prior checkpoint's
output. The only intended changed variable across arms is where the new
correction states came from.

**Incumbent.** All three arms in a seed start from that seed's exact Lab 18d
incumbent - the same `qwen3-0.6b-wordle-lora-dataset-b-structured[-seed45/47]`
checkpoint Lab 18d played 19 reserved games against, verified by adapter file
hash below.

**Arms.**

| arm | new labeled states |
| --- | --- |
| `rollout_correction` | states reached during fixed answer-constrained games the incumbent itself plays on structured dev answers, relabeled by the symbolic teacher |
| `static_random` | expert-labeled dev states sampled without reference to the policy's rollouts |
| `static_matched` | expert-labeled dev states sampled to match each rollout state's answer branch, turn, and candidate-count stratum |

**Held fixed across every arm and every seed:** the incumbent checkpoint, the
`derived_state_v1` prompt representation, the fixed `RAISE` opening, the
2,315-word answer vocabulary and pattern matrix, the answer-constrained
decoder and scoring kernel, the number of formatted training presentations,
the padded-token budget, the optimizer, the learning-rate schedule, the
shared-replay incumbent-preservation mechanism, the drift-checkpoint
schedule, and the held-out evaluation protocol. Arms differ in exactly one
thing: which states are labeled and trained on.

**Shared incumbent preservation.** Lab 19d could not rule out a general
preservation failure, so Lab 20 does not test correction without a
preservation control. Every optimizer update trains on a batch of exactly two
presentations: one arm-specific correction presentation, and one identical
replay presentation sampled from the original structured `TRAIN` corpus. The
replay sequence and its order are identical across all three arms of a seed,
so replay cannot itself explain a between-arm difference.

**Primary estimate.** The equal-weight mean of the three seed-paired held-out
solve-rate differences, `rollout_correction - static_random`, on the same 19
reserved answers Lab 18d used.

**Correction gate.** Passes only when every one of the following holds:

1. no seed triplet tripped a drift stop rule during training;
2. the mean paired solve-rate gain is at least 5 percentage points;
3. `rollout_correction` beats `static_random` in all three seed pairs;
4. the pooled answer-level paired bootstrap 95% interval excludes zero.

`static_matched` is diagnostic, not part of the gate. If `rollout_correction`
beats `static_matched`, the gain is specific to the policy's own states beyond
branch, turn, and candidate count. If `static_matched` alone already recovers
most of the gain, coarse state difficulty was the active ingredient and the
policy did not need to generate its own states.

**Read before seeing results.**

| observation | pre-registered interpretation |
| --- | --- |
| `rollout_correction` beats `static_random`, and `static_matched` does not | the policy's own states carry information beyond branch, turn, and candidate-count difficulty |
| `static_matched` recovers most of the `rollout_correction` gain | the gain is explained by state difficulty stratification, not by policy-specific rollouts |
| all three arms improve about equally over the incumbent | the shared replay and additional labeled data explain the gain; rollout source does not matter at this scale |
| any arm trips a drift stop rule | Lab 19's collapse mode reproduces under a fixed-teacher correction target and the triplet's gate fails regardless of its raw solve rate |
| no arm improves on the incumbent | 27 to 57 additional correction presentations at this learning rate and replay ratio were not enough signal, not evidence that policy-created correction data cannot work |

Nineteen reserved answers are a paired diagnostic, not a precise population
solve rate. The bootstrap below resamples answer identity, and the seed is
still the unit of replication.

**Cost warning.** This notebook plays up to 414 structured-dev answer-constrained
games per seed to collect rollout states (roughly 3.4-3.8 seconds per
2,315-word ranking observed in Lab 18d, so a few hours per seed before any
optimizer step), trains three arms per seed to a shared, seed-specific
number of optimizer updates with a full 2,315-word anchor evaluation at six
checkpoints per arm, and then replays the 19-answer Lab 18d gameplay protocol
for every arm that reaches or is stopped at a checkpoint. `RUN_COLLECTION`,
`RUN_TRAINING`, and `RUN_EVALUATION` gate each phase independently so a
restart never repeats completed, hash-verified work.
""")

md("""
## 20.2 Run controls and memory guard

Run only through the total-system watchdog:

```
scripts/memguard.py --min-free 64 -- uv run jupyter nbconvert \\
    --to notebook --execute --inplace notebooks/20_policy_state_correction.ipynb
```

Three independent flags gate the three expensive phases. Each phase writes its
artifacts atomically and validates them by hash before reuse, so turning a flag
off after a partial run resumes rather than repeats. Before any of the three
phases runs, this notebook reproduces a persisted Lab 18b/18d score vector and
holds a flat driver-memory trace across fixed-shape 40-iteration scoring and
training soaks - the same mandatory gate Lab 18d and Lab 19 used.
""")

code("""
RUN_COLLECTION = True
RUN_TRAINING = True
RUN_EVALUATION = True

MEMORY_CAP_GIB = 128.0
MEM_ABORT = 0.75

import torch

if torch.backends.mps.is_available():
    total_gib = torch.mps.recommended_max_memory() / 1024**3
    torch.mps.set_per_process_memory_fraction(MEMORY_CAP_GIB / total_gib)
    print(f"MPS cap: {MEMORY_CAP_GIB:.0f} GiB of {total_gib:.0f} GiB")

print("RUN_COLLECTION:", RUN_COLLECTION)
print("RUN_TRAINING:", RUN_TRAINING)
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
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display
from peft import PeftModel
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_wordle.benchmark import DEFAULT_EVAL_ANSWERS, parse_guess
from tiny_wordle.expert import EntropyExpert
from tiny_wordle.game import Turn, filter_candidates, is_consistent, score_string
from tiny_wordle.hardware import preferred_device, trainable_parameter_count

MODEL_ID = "Qwen/Qwen3-0.6B"
SEEDS = [42, 45, 47]
ARMS = ["rollout_correction", "static_random", "static_matched"]
REGIMES = ["1", "2", "3-10", "11+"]
ANCHOR_PER_REGIME = 6
ANCHOR_STATES = ANCHOR_PER_REGIME * len(REGIMES)

OPENING = "RAISE"
MAX_TURNS = 6
CHUNK_SIZE = 256
MEMORY_ABORT_GIB = MEMORY_CAP_GIB * MEM_ABORT

MAX_LENGTH = 256
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.05
GRAD_CLIP = 1.0
VISIT_CAP = 3
CHECKPOINT_FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

# Shared seed-triplet drift stop rules. These are conservative relative rules
# aimed at reproducing the Lab 19-scale collapse Lab 19d measured, not at
# tuning performance. Any tripped rule stops every arm of that seed at the
# current checkpoint, so update counts stay matched across the triplet.
DRIFT_CANDIDATE_MASS_RATIO = 0.70
DRIFT_RANK_MULTIPLIER = 4
DRIFT_RANK_FLOOR = 10
DRIFT_WINNER_SHARE_FLOOR = 0.50
DRIFT_WINNER_SHARE_MARGIN = 0.25

GATE_MIN_MEAN_SOLVE_GAIN = 0.05
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 4200

DATA_DIR = Path("../data")
GENERATED_DIR = DATA_DIR / "generated"
CHECKPOINT_ROOT = Path("../checkpoints")
RESULTS_DIR = Path("../results/lab20")
LAB18B_RESULTS = Path("../results/lab18b")
LAB18D_RESULTS = Path("../results/lab18d")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURED_FILES = {
    "train": GENERATED_DIR / "wordle-part2-structured-train.jsonl",
    "validation": GENERATED_DIR / "wordle-part2-structured-dev.jsonl",
    "test": GENERATED_DIR / "wordle-part2-structured-test.jsonl",
}
INCUMBENTS = {
    42: CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured",
    45: CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed45",
    47: CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed47",
}

device = preferred_device()
torch.set_float32_matmul_precision("high")
print("device:", device)


def driver_memory_gib() -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1024**3
    return float("nan")


def clear_device_cache() -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_write(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npy(values: np.ndarray, path: Path) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    atomic_write(json.dumps(value, indent=2), path)
""")

md("""
## 20.3 Freeze the rollout contract

Every piece that could silently change the game this notebook plays is pinned
and checked before anything is collected. The incumbent checkpoint hash must
equal the one Lab 18d loaded and evaluated; the structured corpus hashes must
equal the ones Lab 17 through Lab 19 used; the answer vocabulary, pattern
matrix, opening, and reserved 19-answer split are asserted against their known
values.
""")

code("""
structured_manifest = json.loads(
    (GENERATED_DIR / "wordle-part2-structured-manifest.json").read_text()
)
structured_hashes = {
    split: sha256_file(path) for split, path in STRUCTURED_FILES.items()
}
assert structured_hashes == structured_manifest["structured_sha256"]

lab18d_manifest = json.loads((LAB18D_RESULTS / "lab18d-run.json").read_text())
assert lab18d_manifest["opening"] == OPENING
assert lab18d_manifest["seeds"] == SEEDS

INCUMBENT_MANIFEST_NAMES = ["lab17-run.json", "lab18c-run.json"]
incumbent_manifests = {}
checkpoint_hashes = {}
for seed, path in INCUMBENTS.items():
    model_file = path / "adapter_model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"missing seed {seed} adapter: {model_file}")
    checkpoint_hashes[seed] = sha256_file(model_file)
    assert checkpoint_hashes[seed] == lab18d_manifest["checkpoint_sha256"][str(seed)], (
        f"seed {seed} checkpoint no longer matches the Lab 18d incumbent"
    )
    manifest_path = next(
        (path / name for name in INCUMBENT_MANIFEST_NAMES if (path / name).exists()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(f"seed {seed} incumbent has no run manifest: {path}")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == seed
    assert manifest["base_model"] == MODEL_ID
    assert manifest["representation"] == "derived_state_v1"
    assert manifest["structured_data_sha256"] == structured_hashes
    adapter_config = json.loads((path / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == MODEL_ID
    assert adapter_config["r"] == 8
    assert adapter_config["lora_alpha"] == 16
    assert adapter_config["lora_dropout"] == 0.05
    assert sorted(adapter_config["target_modules"]) == [
        "k_proj", "o_proj", "q_proj", "v_proj"
    ]
    incumbent_manifests[seed] = manifest

print("checkpoint fingerprints tie to Lab 18d incumbent:", checkpoint_hashes)

RESERVED_ANSWERS = list(DEFAULT_EVAL_ANSWERS)
assert RESERVED_ANSWERS == [
    "SHORE", "MIGHT", "BRICK", "GHOST", "KNIFE", "DOUBT", "FLING",
    "ROUND", "CHAMP", "WASTE", "BLIND", "POINT", "SLATE", "CRANE",
    "APPLE", "SHEEP", "BANAL", "ALLEY", "AUDIO",
]
RESERVED_SET = set(RESERVED_ANSWERS)

ANSWERS = [
    line.strip().upper()
    for line in (DATA_DIR / "wordle-answers-original.txt").read_text().splitlines()
    if line.strip()
]
ANSWER_SET = set(ANSWERS)
WORD_TO_INDEX = {word: index for index, word in enumerate(ANSWERS)}
PATTERNS = np.load(DATA_DIR / "wordle-patterns-original-2315.npy")
expert = EntropyExpert(ANSWERS, PATTERNS)
ALL_INDICES = expert.all_indices
assert len(ANSWERS) == 2315
assert PATTERNS.shape == (2315, 2315)
assert expert.word_to_index == WORD_TO_INDEX

structured_rows = {
    split: [json.loads(line) for line in path.read_text().splitlines()]
    for split, path in STRUCTURED_FILES.items()
}
for split, rows in structured_rows.items():
    assert not any(row["answer"] in RESERVED_SET for row in rows), (
        f"reserved gameplay answer leaked into {split}"
    )
    assert all(row["representation"] == "derived_state_v1" for row in rows)
next_guess_rows = {
    split: [row for row in rows if row["task"] == "NEXT_GUESS"]
    for split, rows in structured_rows.items()
}
print(
    "structured rows:",
    {split: len(rows) for split, rows in structured_rows.items()},
)
print(
    "NEXT_GUESS rows:",
    {split: len(rows) for split, rows in next_guess_rows.items()},
)
""")

md("""
## 20.4 The frozen `derived_state_v1` representation

These functions are copied unchanged from Lab 18d and Lab 19. They rebuild the
structured prompt from a state key alone, through the pattern matrix, so
rollout collection, static-pool construction, training, and held-out gameplay
all share one code path with no divergence between "training representation"
and "deployed representation."
""")

code("""
def parse_state_key(state_key: str) -> list[Turn]:
    if not state_key:
        return []
    history = []
    for line in state_key.splitlines():
        guess_text, feedback_text = line.split(" -> ")
        history.append(Turn(
            guess=guess_text.replace(" ", ""),
            feedback=feedback_text.replace(" ", ""),
        ))
    return history


def render_state_key(history: list[Turn]) -> str:
    # Exact inverse of parse_state_key: reproduces the stored space-separated
    # "R A I S E -> B G B B Y" convention so a freshly played rollout state's
    # key string is byte-identical to the same logical state's key in the
    # structured corpus - required for duplicate aggregation, anchor
    # exclusion, and train-corpus overlap to compare correctly.
    return "\\n".join(
        f"{' '.join(turn.guess)} -> {' '.join(turn.feedback)}" for turn in history
    )


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


def render_structured_state(history: list[Turn], candidate_count: int) -> str:
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


def structured_next_guess_prompt(history: list[Turn], candidate_count: int) -> str:
    return (
        "Task: NEXT_GUESS\\n"
        "You are playing Wordle.\\n"
        "Use the game history to choose the next guess.\\n"
        "Return exactly one uppercase five-letter word.\\n\\n"
        "Derived state:\\n"
        + render_structured_state(history, candidate_count)
    )


def candidate_indices_from_history(history: list[Turn]) -> np.ndarray:
    indices = ALL_INDICES
    for turn in history:
        if turn.guess not in WORD_TO_INDEX:
            raise ValueError(f"guess outside the answer lexicon: {turn.guess}")
        indices = expert.update(indices, WORD_TO_INDEX[turn.guess], turn.feedback)
    if len(indices) == 0:
        raise ValueError("state key produced an empty candidate set")
    return indices


def candidate_stratum(candidate_count: int) -> str:
    if candidate_count == 1:
        return "1"
    if candidate_count == 2:
        return "2"
    if candidate_count <= 10:
        return "3-10"
    return "11+"


def answer_branch_of(history: list[Turn]) -> str:
    if not history or history[0].guess != OPENING:
        raise ValueError("state does not start from the fixed RAISE opening")
    return history[0].feedback
""")

md("""
## 20.5 Verify representation fidelity on every stored state

Before any new state is manufactured, every unique `NEXT_GUESS` state stored by
Lab 14/17 in `train` and `validation` is rebuilt from its state key alone and
checked two ways: the pattern-matrix candidate count must equal Lab 14's
stored count, and the freshly rendered prompt must equal the stored prompt
character for character. This is the same check Lab 19 ran before mining.
""")

code("""
unique_states = {}
for split in ["train", "validation"]:
    for row in next_guess_rows[split]:
        unique_states.setdefault((split, row["state_key"]), row)

state_records = []
for (split, state_key), row in unique_states.items():
    history = parse_state_key(state_key)
    candidates = candidate_indices_from_history(history)
    assert len(candidates) == row["candidate_count"], (
        f"candidate reconstruction disagrees with Lab 14 for {state_key!r}"
    )
    prompt = structured_next_guess_prompt(history, len(candidates))
    assert prompt == row["prompt"], "representation drift in derived_state_v1"
    state_records.append({
        "split": split,
        "state_key": state_key,
        "turn": row["turn"],
        "answer": row["answer"],
        "response": row["response"],
        "candidate_count": len(candidates),
        "candidate_stratum": candidate_stratum(len(candidates)),
        "answer_branch": answer_branch_of(history),
        "prompt": prompt,
    })

all_states = pd.DataFrame(state_records).sort_values(
    ["split", "state_key"], kind="stable"
).reset_index(drop=True)

verification_sample = all_states.iloc[::75]
for row in verification_sample.itertuples():
    history = parse_state_key(row.state_key)
    rebuilt = sorted(ANSWERS[int(index)] for index in candidate_indices_from_history(history))
    assert rebuilt == sorted(filter_candidates(ANSWERS, history))
    teacher_index = expert.choose(candidate_indices_from_history(history))
    assert ANSWERS[teacher_index] == row.response, (
        f"canonical teacher disagrees with stored NEXT_GUESS response for {row.state_key!r}"
    )
print(
    f"pattern-matrix candidates, prompts, and canonical teacher labels verified "
    f"on {len(verification_sample)} of {len(all_states)} unique states"
)

dev_states = all_states.query("split == 'validation'").reset_index(drop=True)
train_states = all_states.query("split == 'train'").reset_index(drop=True)
print(
    "unique states:",
    {"train": len(train_states), "validation": len(dev_states)},
)
display(dev_states.groupby("candidate_stratum").size().rename("dev_states").reset_index())
""")

md("""
## 20.6 Verified answer-list scoring kernel

Identical to the Lab 18d/19 kernel: a `logits_to_keep=1` prefill, a KV cache
repeated `CHUNK_SIZE=256` ways, gather-minus-logsumexp token log-probabilities,
and an MPS cache clear after every state. This is the scorer used for rollout
collection, anchor drift evaluation, and held-out gameplay.
""")

code("""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
PAD_ID = tokenizer.pad_token_id or tokenizer.eos_token_id


def render_prompt(prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


WORD_TOKENS = [
    tokenizer.encode(word, add_special_tokens=False) + [tokenizer.eos_token_id]
    for word in ANSWERS
]
LENGTH_BUCKETS = {}
for length in sorted({len(tokens) for tokens in WORD_TOKENS}):
    indices = [
        index for index, tokens in enumerate(WORD_TOKENS) if len(tokens) == length
    ]
    padding = (-len(indices)) % CHUNK_SIZE
    padded = indices + [indices[-1]] * padding
    LENGTH_BUCKETS[length] = (
        torch.tensor(padded),
        torch.tensor([WORD_TOKENS[index] for index in padded], device=device),
    )


def load_frozen_adapter(seed: int):
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    return PeftModel.from_pretrained(base, INCUMBENTS[seed]).to(device).eval()


def load_arm_adapter(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing adapter {path}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    return PeftModel.from_pretrained(base, path).to(device).eval()


def release_model(model) -> None:
    model.to("cpu")
    del model
    gc.collect()
    clear_device_cache()


LAST_STATE_PEAK_GIB = 0.0


@torch.no_grad()
def score_all_words(model, prompt_text: str) -> np.ndarray:
    global LAST_STATE_PEAK_GIB
    input_ids = tokenizer(
        render_prompt(prompt_text), return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    prefill = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
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
                    input_ids=chunk[:, :step], past_key_values=cache, use_cache=True
                )
                logits = output.logits.float()
                targets = logits.gather(2, chunk[:, 1:].unsqueeze(-1)).squeeze(-1)
                total = total + (targets - logits.logsumexp(-1)).sum(dim=1)
                peak = max(peak, driver_memory_gib())
                cache.crop(-step)
                del output, logits, targets
            scores[indices[start:start + CHUNK_SIZE]] = total.cpu()

    LAST_STATE_PEAK_GIB = peak
    del cache, prefill, final_logits, first_logprobs
    clear_device_cache()
    return scores.numpy()
""")

md("""
## 20.7 Mandatory scorer regression and scoring memory gate

Before any of the three expensive phases runs, seed 42's incumbent must
reproduce Lab 18b's first persisted `B-structured` score vector (the same
regression Lab 18d ran), and every seed's incumbent must reproduce its own
first persisted Lab 18d constrained-gameplay score vector. Then one model
scores the longest known battery prompt 40 times with the driver-memory trace
asserted flat. Nothing else runs until both pass.
""")

code("""
lab18b_battery = pd.read_csv(LAB18B_RESULTS / "battery-states.csv")
battery_histories = lab18b_battery["state_key"].map(parse_state_key)
battery_candidate_counts = lab18b_battery["candidate_count"].tolist()
battery_prompts = [
    structured_next_guess_prompt(history, candidate_count)
    for history, candidate_count in zip(battery_histories, battery_candidate_counts)
]

checker = load_frozen_adapter(42)
first_scores = score_all_words(checker, battery_prompts[0])
reference_scores = np.load(
    LAB18B_RESULTS / "scores-B-structured.npy", mmap_mode="r"
)[0]
max_abs_diff = float(np.max(np.abs(first_scores - reference_scores)))
print("Lab 18b score-vector max abs diff:", max_abs_diff)
assert max_abs_diff < 1e-3

for seed in SEEDS:
    keys = pd.read_csv(LAB18D_RESULTS / f"seed{seed}-answer-constrained-score-keys.csv")
    first_key = keys.iloc[0]
    reserved_history = [
        Turn(OPENING, score_string(first_key["answer"], OPENING))
    ]
    assert int(first_key["turn"]) == 2
    reserved_prompt = structured_next_guess_prompt(
        reserved_history, len(candidate_indices_from_history(reserved_history))
    )
    model = checker if seed == 42 else load_frozen_adapter(seed)
    reproduced = score_all_words(model, reserved_prompt)
    persisted = np.load(LAB18D_RESULTS / f"seed{seed}-answer-constrained-scores.npy")[0]
    diff = float(np.max(np.abs(reproduced - persisted)))
    print(f"seed {seed} Lab 18d score-vector max abs diff: {diff:.2e}")
    assert diff < 1e-3
    if seed != 42:
        release_model(model)
        del model

prompt_lengths = [
    len(tokenizer(render_prompt(prompt)).input_ids) for prompt in battery_prompts
]
soak_prompt = battery_prompts[int(np.argmax(prompt_lengths))]
soak_peaks = []
for _ in range(40):
    score_all_words(checker, soak_prompt)
    soak_peaks.append(LAST_STATE_PEAK_GIB)
third = len(soak_peaks) // 3
creep = np.mean(soak_peaks[-third:]) - np.mean(soak_peaks[third:2 * third])
late_range = np.ptp(soak_peaks[-third:])
print(
    f"scoring soak peak {max(soak_peaks):.2f} GiB, creep {creep:+.2f} GiB, "
    f"final range {late_range:.2f} GiB"
)
assert creep < 0.5
assert late_range < 0.5
assert max(soak_peaks) < MEMORY_ABORT_GIB
release_model(checker)
del checker
print("scoring kernel verified against Lab 18b and Lab 18d, memory plateaued")
""")

md("""
## 20.8 Training kernel: response-only CE with fixed padding

Every arm uses the same loss: response-only token cross-entropy, computed
exactly as Lab 18c computed it. The only deliberate departure from Lab 18c is
padding. Lab 18c padded each batch to its own longest row; Lab 20 pads every
presentation to a fixed `MAX_LENGTH=256` regardless of its real length, so a
batch's padded-token budget (`BATCH_SIZE * MAX_LENGTH = 512` tokens) is
identical for every arm, every seed, and every optimizer update - a stronger,
content-independent equal-budget guarantee than matching mean lengths after
the fact would give.
""")

code("""
def encode_presentation(prompt: str, response: str) -> dict:
    prompt_text = render_prompt(prompt)
    full_text = prompt_text + response + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) >= MAX_LENGTH:
        raise ValueError(f"sequence length {len(full_ids)} reached {MAX_LENGTH}")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    pad = MAX_LENGTH - len(full_ids)
    return {
        "input_ids": [PAD_ID] * pad + full_ids,
        "labels": [-100] * pad + labels,
        "attention_mask": [0] * pad + [1] * len(full_ids),
    }


def collate_presentations(rows: list[dict]) -> dict[str, torch.Tensor]:
    encoded = [encode_presentation(row["prompt"], row["response"]) for row in rows]
    return {
        "input_ids": torch.tensor([e["input_ids"] for e in encoded], dtype=torch.long),
        "labels": torch.tensor([e["labels"] for e in encoded], dtype=torch.long),
        "attention_mask": torch.tensor(
            [e["attention_mask"] for e in encoded], dtype=torch.long
        ),
    }


def response_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    supervised = batch["labels"].ne(-100)
    supervised_tokens = int(supervised.sum())
    first_target = int(supervised.nonzero(as_tuple=False)[:, 1].min())
    logit_positions = torch.arange(
        first_target - 1, batch["input_ids"].shape[1] - 1, device=device
    )
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        logits_to_keep=logit_positions,
        use_cache=False,
    ).logits
    targets = batch["labels"][:, logit_positions + 1]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
    )
    return loss, supervised_tokens


def training_step(model, optimizer, scheduler, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss, _ = response_loss(model, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), max_norm=GRAD_CLIP
    )
    optimizer.step()
    scheduler.step()
    peak = driver_memory_gib()
    clear_device_cache()
    return float(loss.detach().cpu()), peak


def lr_multiplier_factory(total_updates: int):
    warmup_steps = max(1, int(total_updates * WARMUP_FRACTION))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_multiplier, warmup_steps


def reset_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_trainable_incumbent(seed: int):
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, INCUMBENTS[seed], is_trainable=True).to(device)
    model.train()
    trainable, total = trainable_parameter_count(model)
    assert trainable > 0, "incumbent adapter loaded without trainable parameters"
    return model, trainable, total


def load_trainable_checkpoint(path: Path):
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, path, is_trainable=True).to(device)
    model.train()
    return model
""")

md("""
## 20.9 Training memory soak

The longest known state-response pair repeats into a fixed batch of
`BATCH_SIZE=2`, run for 40 optimizer steps on a disposable LoRA-continued copy
of the seed-42 incumbent. This copy is discarded afterward; it never touches a
real checkpoint path and it cannot alter any real arm's optimizer state or
dropout stream.
""")

code("""
if (RUN_TRAINING or RUN_COLLECTION):
    all_lengths = pd.concat([train_states, dev_states])["prompt"].str.len() + (
        pd.concat([train_states, dev_states])["response"].str.len()
    )
    longest_row = pd.concat([train_states, dev_states]).loc[all_lengths.idxmax()]
    soak_batch = collate_presentations([
        {"prompt": longest_row["prompt"], "response": longest_row["response"]}
    ] * BATCH_SIZE)
    soak_batch = {key: value.to(device) for key, value in soak_batch.items()}

    reset_seeds(999_999)
    soak_model, _, _ = load_trainable_incumbent(42)
    soak_optimizer = AdamW(
        (p for p in soak_model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    soak_lr_multiplier, _ = lr_multiplier_factory(40)
    soak_scheduler = torch.optim.lr_scheduler.LambdaLR(
        soak_optimizer, lr_lambda=soak_lr_multiplier
    )
    soak_peaks = []
    for _ in range(40):
        _, peak = training_step(soak_model, soak_optimizer, soak_scheduler, soak_batch)
        soak_peaks.append(peak)
    third = len(soak_peaks) // 3
    creep = np.mean(soak_peaks[-third:]) - np.mean(soak_peaks[third:2 * third])
    late_range = np.ptp(soak_peaks[-third:])
    print(
        f"training soak peak {max(soak_peaks):.2f} GiB, creep {creep:+.2f} GiB, "
        f"final range {late_range:.2f} GiB"
    )
    assert creep < 0.5
    assert late_range < 0.5
    assert max(soak_peaks) < MEMORY_ABORT_GIB
    del soak_optimizer, soak_scheduler, soak_batch
    release_model(soak_model)
    del soak_model
    print("training memory plateaued")
else:
    print("training soak skipped: RUN_COLLECTION and RUN_TRAINING are both False")
""")

md("""
## 20.10 Frozen full-list anchor suite

Twenty-four states, six per candidate regime (`1`, `2`, `3-10`, `11+`), drawn
from Lab 18b's 620-state battery and frozen before any correction data is
built. States outside the structured `dev` split are preferred, because dev
states are also the source pool for `static_random` and `static_matched`.
Where a regime cannot fill six states from outside dev, the remaining slots
come from dev and those exact state keys are excluded from every intervention
and static pool below - the anchor suite must stay untouched by anything an
arm ever trains on.
""")

code("""
battery_frame = lab18b_battery.copy()
battery_frame["candidate_stratum"] = battery_frame["candidate_count"].map(candidate_stratum)
dev_state_keys = set(dev_states["state_key"])

anchor_records = []
anchor_dev_exclusions = set()
for regime in REGIMES:
    regime_keys = sorted(
        battery_frame.loc[battery_frame["candidate_stratum"] == regime, "state_key"]
    )
    outside_dev = [key for key in regime_keys if key not in dev_state_keys]
    inside_dev = [key for key in regime_keys if key in dev_state_keys]
    chosen = outside_dev[:ANCHOR_PER_REGIME]
    if len(chosen) < ANCHOR_PER_REGIME:
        needed = ANCHOR_PER_REGIME - len(chosen)
        fill = inside_dev[:needed]
        assert len(fill) == needed, (
            f"regime {regime} cannot fill {ANCHOR_PER_REGIME} anchor states "
            f"({len(outside_dev)} outside dev, {len(inside_dev)} inside dev)"
        )
        anchor_dev_exclusions.update(fill)
        chosen = chosen + fill
    assert len(chosen) == ANCHOR_PER_REGIME
    for state_key in chosen:
        row = battery_frame.loc[battery_frame["state_key"] == state_key].iloc[0]
        history = parse_state_key(state_key)
        candidates = candidate_indices_from_history(history)
        assert len(candidates) == row["candidate_count"]
        teacher_index = expert.choose(candidates)
        anchor_records.append({
            "state_key": state_key,
            "turn": int(row["turn"]),
            "regime": regime,
            "candidate_count": int(row["candidate_count"]),
            "candidates": [int(index) for index in candidates],
            "teacher_index": int(teacher_index),
            "teacher_word": ANSWERS[int(teacher_index)],
            "from_dev": state_key in dev_state_keys,
            "prompt": structured_next_guess_prompt(history, len(candidates)),
        })

anchor_states = pd.DataFrame(anchor_records)
assert len(anchor_states) == ANCHOR_STATES
assert (anchor_states.groupby("regime").size() == ANCHOR_PER_REGIME).all()
assert not anchor_states["state_key"].duplicated().any()
ANCHOR_EXCLUDED_KEYS = set(anchor_dev_exclusions)
assert ANCHOR_EXCLUDED_KEYS <= dev_state_keys
train_state_key_set = set(train_states["state_key"])
assert not (set(anchor_states["state_key"]) & train_state_key_set), (
    "anchor states must not enter the shared replay pool"
)

anchor_manifest = {
    "experiment": "Lab 20 frozen full-list anchor suite",
    "anchor_per_regime": ANCHOR_PER_REGIME,
    "regimes": REGIMES,
    "states": ANCHOR_STATES,
    "dev_exclusions": sorted(ANCHOR_EXCLUDED_KEYS),
    "state_keys_sha256": sha256_text(
        json.dumps(sorted(anchor_states["state_key"]))
    ),
}
atomic_json(anchor_manifest, RESULTS_DIR / "anchor-manifest.json")
atomic_csv(
    anchor_states.drop(columns=["candidates", "prompt"]),
    RESULTS_DIR / "anchor-states.csv",
)
print(
    f"anchor suite: {ANCHOR_STATES} states, "
    f"{len(ANCHOR_EXCLUDED_KEYS)} drawn from dev and reserved out of every pool"
)
display(anchor_states[["state_key", "turn", "regime", "candidate_count", "from_dev"]])
""")

md("""
## 20.11 Static pool from structured dev

The static pool is every unique canonical `NEXT_GUESS` state in the structured
`dev` split, after removing the seven anchor state keys reserved above. Each
row already carries its Lab 14 canonical teacher response, verified against
`expert.choose` in 20.5. `static_random` draws from this whole pool.
`static_matched` draws from the subset sharing a rollout state's exact
`(answer_branch, turn, candidate_stratum)` key.
""")

code("""
static_pool = dev_states.loc[
    ~dev_states["state_key"].isin(ANCHOR_EXCLUDED_KEYS)
].reset_index(drop=True)
assert static_pool["state_key"].is_unique
assert not set(static_pool["state_key"]) & ANCHOR_EXCLUDED_KEYS
assert not set(static_pool["state_key"]) & set(anchor_states["state_key"])

static_pool["match_key"] = list(zip(
    static_pool["answer_branch"], static_pool["turn"], static_pool["candidate_stratum"]
))
static_pool_by_key = defaultdict(list)
for row in static_pool.itertuples():
    static_pool_by_key[row.match_key].append(row.Index)

static_pool_manifest = {
    "experiment": "Lab 20 static pool",
    "source_states_sha256": sha256_file(STRUCTURED_FILES["validation"]),
    "anchor_exclusions": sorted(ANCHOR_EXCLUDED_KEYS),
    "unique_states": len(static_pool),
    "match_keys": len(static_pool_by_key),
}
atomic_json(static_pool_manifest, RESULTS_DIR / "static-pool-manifest.json")
atomic_csv(
    static_pool.drop(columns=["match_key"]),
    RESULTS_DIR / "static-pool-states.csv",
)
print(
    f"static pool: {len(static_pool)} unique dev states "
    f"({len(dev_states) - len(static_pool)} excluded as anchors), "
    f"{len(static_pool_by_key)} distinct (answer_branch, turn, stratum) keys"
)
display(static_pool.groupby("candidate_stratum").size().rename("states").reset_index())
""")

md("""
## 20.12 Restartable rollout collection engine

Each incumbent plays one fixed answer-constrained game (greedy over the
2,315-word action space, exactly Lab 18d's decoder) against every unique
answer represented by a structured-dev `NEXT_GUESS` row. All 19
`DEFAULT_EVAL_ANSWERS` stay excluded, matching Lab 19's reserved-answer
invariant. Every visited state is written to a per-answer raw trace before any
teacher label exists, atomically after each answer, so an interruption loses
at most one game's worth of rollout.
""")

code("""
DEV_ANSWERS = sorted({
    row["answer"] for row in next_guess_rows["validation"] if row["answer"] not in RESERVED_SET
})
assert not (set(DEV_ANSWERS) & RESERVED_SET)
print(f"structured-dev intervention answers: {len(DEV_ANSWERS)}")


def rollout_paths(seed: int) -> dict[str, Path]:
    stem = f"rollout-seed{seed}"
    return {
        "raw": RESULTS_DIR / f"{stem}-raw.csv",
        "games": RESULTS_DIR / f"{stem}-games.csv",
        "progress": RESULTS_DIR / f"{stem}-progress.json",
    }


def play_rollout_game(model, seed: int, answer: str) -> tuple[list[dict], dict]:
    history = [Turn(OPENING, score_string(answer, OPENING))]
    call_rows = []
    solved_turn = None
    started = time.perf_counter()

    for turn_number in range(2, MAX_TURNS + 1):
        before = candidate_indices_from_history(history)
        prompt = structured_next_guess_prompt(history, len(before))
        step_started = time.perf_counter()
        scores = score_all_words(model, prompt)
        assert LAST_STATE_PEAK_GIB < MEMORY_ABORT_GIB, (
            f"memory regression at seed {seed} {answer} turn {turn_number}: "
            f"{LAST_STATE_PEAK_GIB:.1f} GiB"
        )
        guess = ANSWERS[int(scores.argmax())]
        call_rows.append({
            "seed": seed,
            "answer": answer,
            "turn": turn_number,
            "state_key": render_state_key(history),
            "prompt": prompt,
            "candidate_count": len(before),
            "answer_branch": answer_branch_of(history),
            "learner_guess": guess,
            "elapsed_seconds": time.perf_counter() - step_started,
        })
        feedback = score_string(answer, guess)
        history.append(Turn(guess, feedback))
        if feedback == "GGGGG":
            solved_turn = turn_number
            break

    return call_rows, {
        "seed": seed,
        "answer": answer,
        "solved": solved_turn is not None,
        "solved_turn": solved_turn,
        "model_calls": len(call_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }


def collect_rollout(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = rollout_paths(seed)
    progress = {
        "seed": seed,
        "checkpoint_sha256": checkpoint_hashes[seed],
        "answers": DEV_ANSWERS,
    }
    if paths["progress"].exists():
        assert json.loads(paths["progress"].read_text()) == progress, (
            f"seed {seed} rollout progress marker does not match the current contract"
        )
    else:
        atomic_json(progress, paths["progress"])

    if paths["games"].exists():
        if not paths["raw"].exists():
            raise FileNotFoundError(
                f"seed {seed} rollout games exist without raw state traces"
            )
        raw = pd.read_csv(paths["raw"])
        games = pd.read_csv(paths["games"])
    else:
        raw = pd.DataFrame()
        games = pd.DataFrame()
    completed = set(games["answer"]) if len(games) else set()
    if len(games):
        assert not games["answer"].duplicated().any()
        raw = raw.loc[raw["answer"].isin(completed)].reset_index(drop=True)

    if not RUN_COLLECTION and completed != set(DEV_ANSWERS):
        raise FileNotFoundError(
            f"seed {seed} rollout collection incomplete and RUN_COLLECTION=False"
        )

    model = None
    if RUN_COLLECTION and completed != set(DEV_ANSWERS):
        model = load_frozen_adapter(seed)
    for answer in DEV_ANSWERS:
        if answer in completed:
            continue
        new_rows, new_game = play_rollout_game(model, seed, answer)
        raw = pd.concat([raw, pd.DataFrame(new_rows)], ignore_index=True)
        games = pd.concat([games, pd.DataFrame([new_game])], ignore_index=True)
        atomic_csv(raw, paths["raw"])
        atomic_csv(games, paths["games"])
        print(
            f"seed {seed} rollout {answer}: "
            f"{'SOLVED' if new_game['solved'] else 'FAILED'} "
            f"turn={new_game['solved_turn']} calls={new_game['model_calls']}",
            flush=True,
        )
    if model is not None:
        release_model(model)
        del model

    assert set(games["answer"]) == set(DEV_ANSWERS)
    assert not games["answer"].duplicated().any()
    return raw, games
""")

md("""
## 20.13 Collect rollout states for every seed
""")

code("""
rollout_raw = {}
rollout_games = {}
for seed in SEEDS:
    raw, games = collect_rollout(seed)
    rollout_raw[seed] = raw
    rollout_games[seed] = games
    print(
        f"seed {seed}: {len(raw)} raw visited states across {len(games)} games, "
        f"{int(games['solved'].sum())} solved"
    )
""")

md("""
## 20.14 Aggregate to unique states and apply the symbolic teacher

Exact state-key duplicates are aggregated only now that every raw trace
exists. A state's `visit_count` is how many of the 414 games reached it;
`source_answers` names every answer that did. The canonical teacher - the same
candidate-only maximum-entropy rule with lexicographic tie-break every earlier
lab used, via `expert.choose` - labels each unique state after aggregation.
Anchor-reserved states are excluded before anything else, even though the
7 dev-drawn anchors were themselves picked from the same broad regimes a
rollout is likely to revisit.
""")

code("""
def aggregate_rollout(seed: int, raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby("state_key", sort=False)
    records = []
    for state_key, group in grouped:
        assert group["turn"].nunique() == 1, f"turn disagreement for {state_key!r}"
        assert group["candidate_count"].nunique() == 1, (
            f"candidate_count disagreement for {state_key!r}"
        )
        assert group["answer_branch"].nunique() == 1, (
            f"answer_branch disagreement for {state_key!r}"
        )
        assert group["learner_guess"].nunique() == 1, (
            f"non-deterministic learner guess for {state_key!r}: "
            f"a fixed argmax decoder must return the same action on the same state"
        )
        first = group.iloc[0]
        history = parse_state_key(state_key)
        candidates = candidate_indices_from_history(history)
        assert len(candidates) == int(first["candidate_count"])
        teacher_index = expert.choose(candidates)
        teacher_word = ANSWERS[int(teacher_index)]
        records.append({
            "seed": seed,
            "state_key": state_key,
            "turn": int(first["turn"]),
            "candidate_count": int(first["candidate_count"]),
            "candidate_stratum": candidate_stratum(int(first["candidate_count"])),
            "answer_branch": first["answer_branch"],
            "prompt": first["prompt"],
            "learner_guess": first["learner_guess"],
            "teacher_guess": teacher_word,
            "teacher_disagreement": first["learner_guess"] != teacher_word,
            "visit_count": len(group),
            "source_answers": json.dumps(sorted(group["answer"].unique().tolist())),
            "collection_seconds": float(group["elapsed_seconds"].sum()),
        })
    frame = pd.DataFrame(records).sort_values("state_key", kind="stable").reset_index(drop=True)
    assert frame["state_key"].is_unique
    assert frame["visit_count"].sum() == len(raw)
    return frame


rollout_unique = {seed: aggregate_rollout(seed, rollout_raw[seed]) for seed in SEEDS}
for seed in SEEDS:
    frame = rollout_unique[seed]
    print(
        f"seed {seed}: {len(frame)} unique rollout states, "
        f"teacher disagreement rate {frame['teacher_disagreement'].mean():.3f}"
    )
""")

md("""
## 20.15 Eligibility, visit cap, and presentation expansion

A rollout state is eligible for `rollout_correction` only if the anchor suite
never claimed it, and only if the static pool contains at least one state
sharing its exact `(answer_branch, turn, candidate_stratum)` key - otherwise
`static_matched` could not build a matched control for it. Every excluded
state is preserved with its reason rather than silently dropped.
`visit_count` is capped at `VISIT_CAP=3` and expands each eligible unique
state into that many correction presentations. `static_matched` draws one
stratum-matched static row per correction presentation, inheriting the same
capped weight; `static_random` draws the same total count from the whole
static pool, ignoring stratum.
""")

code("""
def classify_eligibility(seed: int, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["match_key"] = list(zip(
        frame["answer_branch"], frame["turn"], frame["candidate_stratum"]
    ))
    reasons = []
    for row in frame.itertuples():
        if row.state_key in ANCHOR_EXCLUDED_KEYS or row.state_key in set(anchor_states["state_key"]):
            reasons.append("anchor_reserved")
        elif row.match_key not in static_pool_by_key:
            reasons.append("no_matching_static_stratum")
        else:
            reasons.append("")
    frame["exclusion_reason"] = reasons
    frame["eligible"] = frame["exclusion_reason"] == ""
    frame["visit_count_capped"] = frame["visit_count"].clip(upper=VISIT_CAP)
    return frame


rollout_classified = {
    seed: classify_eligibility(seed, rollout_unique[seed]) for seed in SEEDS
}
for seed in SEEDS:
    frame = rollout_classified[seed]
    eligible = frame.loc[frame["eligible"]]
    assert not (set(eligible["state_key"]) & ANCHOR_EXCLUDED_KEYS)
    assert not (set(eligible["state_key"]) & set(anchor_states["state_key"]))
    print(
        f"seed {seed}: {frame['eligible'].sum()}/{len(frame)} unique states eligible, "
        f"presentations after capped expansion: {int(eligible['visit_count_capped'].sum())}"
    )
    display(frame.groupby("exclusion_reason").size().rename("states").reset_index())
""")

md("""
## 20.16 Build the three arm corpora and the shared replay sequence

`rollout_correction` expands each eligible unique state into
`visit_count_capped` identical presentations, in deterministic `state_key`
order. `static_matched` walks that exact same expansion and draws one
stratum-matched static row per presentation. `static_random` draws the same
total count from the whole static pool. A single permutation, one per seed,
shuffles all three arms' presentation lists into the same position stream; the
shared replay sequence is sampled once per seed from the original structured
`TRAIN` `NEXT_GUESS` corpus (not deduplicated - a state's natural repetition
rate in that corpus is preserved) and is reused, unshuffled and in the same
order, by every arm.
""")

code("""
BASE_SEED = 2000
STATIC_MATCHED_STREAM = 10
STATIC_RANDOM_STREAM = 11
REPLAY_STREAM = 12
ORDER_STREAM = 13


def seeded_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([BASE_SEED, seed, stream])


def build_seed_corpora(seed: int) -> dict:
    frame = rollout_classified[seed]
    eligible = frame.loc[frame["eligible"]].sort_values("state_key", kind="stable")
    assert len(eligible) > 0, f"seed {seed} has no eligible rollout states"

    rollout_presentations = []
    static_matched_presentations = []
    matched_rng = seeded_rng(seed, STATIC_MATCHED_STREAM)
    for row in eligible.itertuples():
        for visit_index in range(int(row.visit_count_capped)):
            rollout_presentations.append({
                "state_key": row.state_key,
                "prompt": row.prompt,
                "response": row.teacher_guess,
                "visit_index": visit_index,
            })
            group = static_pool_by_key[row.match_key]
            static_row = static_pool.loc[group[int(matched_rng.integers(len(group)))]]
            static_matched_presentations.append({
                "state_key": static_row["state_key"],
                "prompt": static_row["prompt"],
                "response": static_row["response"],
                "matched_to": row.state_key,
            })
    total_presentations = len(rollout_presentations)

    random_rng = seeded_rng(seed, STATIC_RANDOM_STREAM)
    random_indices = random_rng.integers(len(static_pool), size=total_presentations)
    static_random_presentations = [
        {
            "state_key": static_pool.loc[index, "state_key"],
            "prompt": static_pool.loc[index, "prompt"],
            "response": static_pool.loc[index, "response"],
        }
        for index in random_indices
    ]

    replay_rng = seeded_rng(seed, REPLAY_STREAM)
    train_pool = next_guess_rows["train"]
    replay_indices = replay_rng.integers(len(train_pool), size=total_presentations)
    replay_sequence = [
        {
            "state_key": train_pool[index]["state_key"],
            "prompt": train_pool[index]["prompt"],
            "response": train_pool[index]["response"],
        }
        for index in replay_indices
    ]

    order_rng = seeded_rng(seed, ORDER_STREAM)
    permutation = order_rng.permutation(total_presentations)

    arm_presentations = {
        "rollout_correction": rollout_presentations,
        "static_matched": static_matched_presentations,
        "static_random": static_random_presentations,
    }
    ordered = {
        arm: [presentations[int(index)] for index in permutation]
        for arm, presentations in arm_presentations.items()
    }
    return {
        "total_presentations": total_presentations,
        "ordered": ordered,
        "replay_sequence": replay_sequence,
        "permutation": permutation,
        "unordered": arm_presentations,
    }


seed_corpora = {seed: build_seed_corpora(seed) for seed in SEEDS}
for seed in SEEDS:
    print(f"seed {seed}: {seed_corpora[seed]['total_presentations']} presentations per arm")
""")

md("""
## 20.17 Fail fast on presentation length

Collection can create deeper histories than the original static corpus. Every
correction and replay presentation is tokenized now, before any optimizer
state is created, so an over-length rollout cannot fail halfway through a
training block.
""")

code("""
presentation_lengths = []
for seed in SEEDS:
    bundle = seed_corpora[seed]
    streams = {
        **bundle["ordered"],
        "shared_replay": bundle["replay_sequence"],
    }
    for stream_name, rows in streams.items():
        lengths = [
            len(encode_presentation(row["prompt"], row["response"])["input_ids"])
            for row in rows
        ]
        presentation_lengths.append({
            "seed": seed,
            "stream": stream_name,
            "presentations": len(lengths),
            "max_tokens": max(lengths),
        })
length_frame = pd.DataFrame(presentation_lengths)
assert (length_frame["max_tokens"] < MAX_LENGTH).all()
display(length_frame)
""")

md("""
## 20.18 Persist corpora and the per-seed manifest

Every ordered presentation stream and the shared replay sequence are written
as JSONL. The manifest records everything needed to audit the contract without
rerunning collection: hashes proving the replay sequence and the correction
order are identical objects across the three arms of a seed, unique-state and
presentation counts, the visit-count distribution, overlap with the base
`TRAIN` corpus (states the rollout revisited that the incumbent already saw
in Lab 17/18c training), teacher disagreement, matching/exclusion counts, and
collection cost.
""")

code("""
def jsonl_of(records: list[dict]) -> str:
    return "\\n".join(json.dumps(record, sort_keys=True) for record in records) + "\\n"


seed_manifests = {}
for seed in SEEDS:
    bundle = seed_corpora[seed]
    for arm in ARMS:
        payload = jsonl_of(bundle["ordered"][arm])
        atomic_write(payload, GENERATED_DIR / f"lab20-{arm}-seed{seed}.jsonl")
    replay_payload = jsonl_of(bundle["replay_sequence"])
    atomic_write(replay_payload, GENERATED_DIR / f"lab20-replay-seed{seed}.jsonl")

    frame = rollout_classified[seed]
    eligible = frame.loc[frame["eligible"]]
    overlap_with_train = len(set(eligible["state_key"]) & train_state_key_set)
    visit_histogram = (
        eligible["visit_count"].clip(upper=VISIT_CAP + 2).value_counts().sort_index()
    )
    manifest = {
        "experiment": "Lab 20 seed-specific correction corpora",
        "seed": seed,
        "checkpoint_sha256": checkpoint_hashes[seed],
        "visit_cap": VISIT_CAP,
        "unique_rollout_states": int(len(frame)),
        "eligible_rollout_states": int(len(eligible)),
        "exclusion_counts": {
            reason: int(count)
            for reason, count in frame["exclusion_reason"].value_counts().items()
        },
        "total_presentations": bundle["total_presentations"],
        "visit_count_histogram": {
            str(count): int(occurrences)
            for count, occurrences in visit_histogram.items()
        },
        "overlap_with_train_corpus": overlap_with_train,
        "teacher_disagreement_rate": float(eligible["teacher_disagreement"].mean()),
        "collection_seconds": float(frame["collection_seconds"].sum()),
        "replay_sequence_sha256": sha256_text(replay_payload),
        "correction_order_sha256": sha256_text(
            json.dumps([int(index) for index in bundle["permutation"]])
        ),
        "arm_corpus_sha256": {
            arm: sha256_text(jsonl_of(bundle["ordered"][arm])) for arm in ARMS
        },
        "padded_token_budget_per_update": BATCH_SIZE * MAX_LENGTH,
    }
    atomic_json(manifest, RESULTS_DIR / f"seed{seed}-corpus-manifest.json")
    seed_manifests[seed] = manifest

display(pd.DataFrame([
    {
        "seed": seed,
        "presentations": manifest["total_presentations"],
        "eligible_states": manifest["eligible_rollout_states"],
        "overlap_with_train": manifest["overlap_with_train_corpus"],
        "teacher_disagreement_rate": manifest["teacher_disagreement_rate"],
        "collection_minutes": manifest["collection_seconds"] / 60,
    }
    for seed, manifest in seed_manifests.items()
]))
""")

md("""
## 20.19 Cross-arm equality assertions

The training contract requires exact equality across the three arms of a
seed: the same presentation count, the same padded-token budget, and the same
replay content and order. The manifest stores one shared value per seed
because the code above builds the replay sequence and the shuffle permutation
once and reuses them for every arm; this cell asserts that sharing actually
held, rather than only documenting the intent.
""")

code("""
for seed in SEEDS:
    bundle = seed_corpora[seed]
    presentation_counts = {arm: len(bundle["ordered"][arm]) for arm in ARMS}
    assert len(set(presentation_counts.values())) == 1, presentation_counts
    assert presentation_counts[ARMS[0]] == len(bundle["replay_sequence"])
    assert presentation_counts[ARMS[0]] == bundle["total_presentations"]
    padded_budget = BATCH_SIZE * MAX_LENGTH
    assert padded_budget == 512
    print(
        f"seed {seed}: {presentation_counts[ARMS[0]]} presentations per arm, "
        f"{presentation_counts[ARMS[0]]} replay rows, "
        f"{padded_budget} padded tokens per update x "
        f"{presentation_counts[ARMS[0]]} updates = "
        f"{padded_budget * presentation_counts[ARMS[0]]:,} total padded tokens per arm"
    )
print("presentation count, replay identity, and padded token budget match across every arm")
""")

md("""
## 20.20 Drift-checkpoint schedule

Each seed's checkpoint steps are `0%, 10%, 25%, 50%, 75%, 100%` of that seed's
own total presentation count, rounded to integers, deduplicated, and sorted -
always including step 0 (the shared incumbent, before any update) and the
final step. All three arms of a seed train to the same steps, so this
schedule is one list per seed, not per arm.
""")

code("""
CHECKPOINT_FRACTION_LABELS = [f"{int(round(f * 100))}%" for f in CHECKPOINT_FRACTIONS]


def checkpoint_steps(total_presentations: int) -> list[int]:
    steps = sorted({
        int(round(total_presentations * fraction)) for fraction in CHECKPOINT_FRACTIONS
    })
    assert steps[0] == 0
    assert steps[-1] == total_presentations
    return steps


seed_checkpoint_steps = {
    seed: checkpoint_steps(seed_corpora[seed]["total_presentations"]) for seed in SEEDS
}
for seed in SEEDS:
    print(f"seed {seed} checkpoint steps: {seed_checkpoint_steps[seed]}")
""")

md("""
## 20.21 Anchor evaluation and shared drift stop rules

At every checkpoint, every arm's adapter scores all 2,315 words at all 24
frozen anchor states with the exact scorer verified in 20.7. Four rules,
fixed before training and aimed at reproducing Lab 19-scale collapse rather
than at tuning performance, compare that arm's anchor summary against the
seed's own incumbent baseline (step 0, identical for all three arms):

| rule | condition to stop |
| --- | --- |
| candidate-mass collapse | median candidate mass < `0.70 x` incumbent median candidate mass |
| best-candidate rank collapse | median best-candidate rank > `max(10, 4x` incumbent median best-candidate rank`)` |
| singleton rank collapse | median singleton-candidate rank > `max(10, 4x` incumbent singleton median rank`)` |
| winner concentration | largest winner share > `max(0.50,` incumbent winner share `+ 0.25)` |

If any arm trips any rule at a checkpoint, every arm of that seed stops at
that checkpoint - the whole triplet, so update counts stay matched - and the
triplet cannot pass the correction gate in 20.26 regardless of its raw solve
rate.
""")

code("""
ANCHOR_CANDIDATES = [np.array(row, dtype=np.int64) for row in anchor_states["candidates"]]
ANCHOR_TEACHER_INDEX = anchor_states["teacher_index"].to_numpy()
ANCHOR_REGIME = anchor_states["regime"].to_numpy()
ANCHOR_PROMPTS = anchor_states["prompt"].tolist()


def rank_vector(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


@torch.no_grad()
def score_anchor_states(model) -> np.ndarray:
    matrix = np.zeros((ANCHOR_STATES, len(ANSWERS)), dtype=np.float32)
    for position, prompt in enumerate(ANCHOR_PROMPTS):
        matrix[position] = score_all_words(model, prompt)
        assert LAST_STATE_PEAK_GIB < MEMORY_ABORT_GIB, (
            f"memory regression scoring anchor state {position}: "
            f"{LAST_STATE_PEAK_GIB:.1f} GiB"
        )
    return matrix


def anchor_metrics(score_matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for position in range(ANCHOR_STATES):
        scores = score_matrix[position]
        candidates = ANCHOR_CANDIDATES[position]
        ranks = rank_vector(scores)
        winner_index = int(scores.argmax())
        shifted = scores - scores.max()
        weights = np.exp(shifted)
        candidate_mass = float(weights[candidates].sum() / weights.sum())
        best_candidate_rank = int(ranks[candidates].min())
        candidate_teacher_rank = int(ranks[ANCHOR_TEACHER_INDEX[position]])
        regime = ANCHOR_REGIME[position]
        singleton_rank = best_candidate_rank if regime == "1" else np.nan
        rows.append({
            "state_key": anchor_states["state_key"].iloc[position],
            "regime": regime,
            "winner_index": winner_index,
            "winner_word": ANSWERS[winner_index],
            "candidate_mass": candidate_mass,
            "best_candidate_rank": best_candidate_rank,
            "candidate_teacher_rank": candidate_teacher_rank,
            "singleton_candidate_rank": singleton_rank,
        })
    return pd.DataFrame(rows)


def anchor_summary(metrics: pd.DataFrame) -> dict:
    winner_counts = metrics["winner_word"].value_counts()
    singleton = metrics.loc[metrics["regime"] == "1", "singleton_candidate_rank"]
    return {
        "median_candidate_mass": float(metrics["candidate_mass"].median()),
        "median_best_candidate_rank": float(metrics["best_candidate_rank"].median()),
        "singleton_median_rank": float(singleton.median()),
        "unique_winners": int(winner_counts.size),
        "largest_winner_share": float(winner_counts.iloc[0] / len(metrics)),
    }


def drift_check(current: dict, baseline: dict) -> dict:
    candidate_mass_floor = DRIFT_CANDIDATE_MASS_RATIO * baseline["median_candidate_mass"]
    best_rank_ceiling = max(
        DRIFT_RANK_FLOOR, DRIFT_RANK_MULTIPLIER * baseline["median_best_candidate_rank"]
    )
    singleton_rank_ceiling = max(
        DRIFT_RANK_FLOOR, DRIFT_RANK_MULTIPLIER * baseline["singleton_median_rank"]
    )
    winner_share_ceiling = max(
        DRIFT_WINNER_SHARE_FLOOR,
        baseline["largest_winner_share"] + DRIFT_WINNER_SHARE_MARGIN,
    )
    rules = {
        "candidate_mass_collapse": {
            "value": current["median_candidate_mass"],
            "threshold": candidate_mass_floor,
            "tripped": current["median_candidate_mass"] < candidate_mass_floor,
        },
        "best_candidate_rank_collapse": {
            "value": current["median_best_candidate_rank"],
            "threshold": best_rank_ceiling,
            "tripped": current["median_best_candidate_rank"] > best_rank_ceiling,
        },
        "singleton_rank_collapse": {
            "value": current["singleton_median_rank"],
            "threshold": singleton_rank_ceiling,
            "tripped": current["singleton_median_rank"] > singleton_rank_ceiling,
        },
        "winner_concentration": {
            "value": current["largest_winner_share"],
            "threshold": winner_share_ceiling,
            "tripped": current["largest_winner_share"] > winner_share_ceiling,
        },
    }
    rules["any_tripped"] = any(rule["tripped"] for rule in rules.values())
    return rules
""")

md("""
## 20.22 Restartable, triplet-synchronous training blocks

A checkpoint directory name encodes seed, arm, and step; a finalized
directory always carries a manifest naming the exact correction and replay
content it was trained on, so a restart can verify a checkpoint before
reusing it instead of trusting a bare adapter file. An `-in-progress` sibling
means a previous run was interrupted mid-write; reruns stop there for
inspection rather than silently resuming from an unknown optimizer state,
matching every earlier lab's convention.

All three arms of a seed always train to the *same* next checkpoint step
before either any arm advances further or any anchor evaluation happens, so
update counts never drift apart mid-triplet. A block replays presentations
`start_step .. end_step-1` of that arm's own shuffled correction stream
together with the identical positions of the shared replay sequence - one
`(correction, replay)` pair, one optimizer update, exactly the incumbent
preservation contract from 20.16-20.18.
""")

code("""
def checkpoint_dir(seed: int, arm: str, step: int) -> Path:
    return CHECKPOINT_ROOT / f"lab20-{arm}-seed{seed}-step{step:05d}"


def resolve_checkpoint(seed: int, arm: str, step: int) -> Path:
    return INCUMBENTS[seed] if step == 0 else checkpoint_dir(seed, arm, step)


def block_manifest_contract(seed: int, arm: str, start_step: int, end_step: int) -> dict:
    bundle = seed_corpora[seed]
    corrections = bundle["ordered"][arm][start_step:end_step]
    replay = bundle["replay_sequence"][start_step:end_step]
    return {
        "experiment": "Lab 20 restartable training block",
        "seed": seed,
        "arm": arm,
        "start_step": start_step,
        "end_step": end_step,
        "updates_in_block": end_step - start_step,
        "correction_block_sha256": sha256_text(jsonl_of(corrections)),
        "replay_block_sha256": sha256_text(jsonl_of(replay)),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "grad_clip": GRAD_CLIP,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "total_presentations": bundle["total_presentations"],
        "checkpoint_sha256": checkpoint_hashes[seed],
    }


def validate_block_manifest(manifest: dict, contract: dict) -> None:
    for key, value in contract.items():
        assert manifest[key] == value, (
            f"checkpoint manifest mismatch on {key!r}: {manifest[key]!r} != {value!r}"
        )


def train_block(seed: int, arm: str, start_step: int, end_step: int) -> dict:
    end_dir = checkpoint_dir(seed, arm, end_step)
    contract = block_manifest_contract(seed, arm, start_step, end_step)
    manifest_path = end_dir / "lab20-run.json"
    if end_dir.exists():
        if not manifest_path.exists():
            raise FileNotFoundError(f"checkpoint exists without a manifest: {end_dir}")
        manifest = json.loads(manifest_path.read_text())
        validate_block_manifest(manifest, contract)
        return manifest

    in_progress = end_dir.with_name(end_dir.name + "-in-progress")
    if in_progress.exists():
        raise FileExistsError(
            f"incomplete checkpoint needs inspection before reuse: {in_progress}"
        )
    if not RUN_TRAINING:
        raise FileNotFoundError(
            f"seed {seed} {arm} step {end_step} checkpoint missing and RUN_TRAINING=False"
        )

    start_path = resolve_checkpoint(seed, arm, start_step)
    assert start_path.exists(), f"missing start checkpoint {start_path}"
    if start_step == 0:
        reset_seeds(seed)
        model, _, _ = load_trainable_incumbent(seed)
    else:
        model = load_trainable_checkpoint(start_path)
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_updates = seed_corpora[seed]["total_presentations"]
    lr_multiplier, _ = lr_multiplier_factory(total_updates)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_multiplier, last_epoch=-1
    )
    for _ in range(start_step):
        scheduler.step()
    optimizer_path = start_path / "lab20-optimizer.pt"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    bundle = seed_corpora[seed]
    corrections = bundle["ordered"][arm]
    replay = bundle["replay_sequence"]
    peak_memory = 0.0
    for position in range(start_step, end_step):
        batch = collate_presentations([corrections[position], replay[position]])
        batch = {key: value.to(device) for key, value in batch.items()}
        _, peak = training_step(model, optimizer, scheduler, batch)
        peak_memory = max(peak_memory, peak)
        assert peak < MEMORY_ABORT_GIB, (
            f"memory regression training seed {seed} {arm} at update {position + 1}: "
            f"{peak:.1f} GiB"
        )

    in_progress.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(in_progress)
    torch.save(optimizer.state_dict(), in_progress / "lab20-optimizer.pt")
    manifest = dict(contract, peak_driver_memory_gib=peak_memory)
    atomic_json(manifest, in_progress / "lab20-run.json")
    release_model(model)
    del model, optimizer, scheduler
    os.replace(in_progress, end_dir)
    print(f"seed {seed} {arm}: trained step {start_step} -> {end_step}")
    return json.loads((end_dir / "lab20-run.json").read_text())
""")

md("""
## 20.23 Anchor evaluation and the shared stop record

Every arm's adapter is scored against all 24 anchors at every checkpoint it
reaches. Step 0 is identical for all three arms - the untouched incumbent -
so its anchor summary is computed once per seed and reused as the baseline
`drift_check` compares every later checkpoint against.
""")

code("""
def anchor_eval_path(seed: int, arm: str, step: int) -> Path:
    return RESULTS_DIR / f"anchor-eval-{arm}-seed{seed}-step{step:05d}.json"


def evaluate_anchor_checkpoint(seed: int, arm: str, step: int) -> dict:
    result_path = anchor_eval_path(seed, arm, step)
    if result_path.exists():
        return json.loads(result_path.read_text())
    if not (RUN_TRAINING or RUN_COLLECTION):
        raise FileNotFoundError(
            f"anchor evaluation missing for seed {seed} {arm} step {step} "
            "and RUN_TRAINING=False"
        )
    checkpoint_path = resolve_checkpoint(seed, arm, step)
    assert checkpoint_path.exists(), f"missing checkpoint {checkpoint_path}"
    model = load_arm_adapter(checkpoint_path)
    score_matrix = score_anchor_states(model)
    release_model(model)
    del model
    metrics = anchor_metrics(score_matrix)
    summary = anchor_summary(metrics)
    atomic_csv(metrics, RESULTS_DIR / f"anchor-metrics-{arm}-seed{seed}-step{step:05d}.csv")
    payload = dict(summary, seed=seed, arm=arm, step=step)
    atomic_json(payload, result_path)
    return payload


def seed_baseline(seed: int) -> dict:
    return evaluate_anchor_checkpoint(seed, "incumbent", 0)
""")

md("""
## 20.24 Train every seed triplet to its shared drift checkpoints

For a seed: compute the incumbent baseline once, then walk the checkpoint
schedule. At every step, all three arms train their block first; only once
all three have reached that step does anchor evaluation and the drift check
run. If any arm trips any rule, the stop record is written and the *entire*
triplet halts at that step - no arm trains further, regardless of whether its
own anchors looked fine. A previously written stop record short-circuits the
loop on rerun instead of re-deriving it.
""")

code("""
def stop_record_path(seed: int) -> Path:
    return RESULTS_DIR / f"seed{seed}-stop.json"


def train_seed_triplet(seed: int) -> dict:
    stop_path = stop_record_path(seed)
    if stop_path.exists():
        record = json.loads(stop_path.read_text())
        print(
            f"seed {seed}: triplet already resolved at step {record['final_step']} "
            f"(stopped={record['stopped']})"
        )
        return record

    baseline = seed_baseline(seed)
    steps = seed_checkpoint_steps[seed]
    final_step = 0
    stopped = False
    stop_details = None

    for start_step, end_step in zip(steps[:-1], steps[1:]):
        for arm in ARMS:
            train_block(seed, arm, start_step, end_step)
        arm_summaries = {
            arm: evaluate_anchor_checkpoint(seed, arm, end_step) for arm in ARMS
        }
        arm_checks = {
            arm: drift_check(summary, baseline) for arm, summary in arm_summaries.items()
        }
        final_step = end_step
        tripped_arms = [arm for arm, check in arm_checks.items() if check["any_tripped"]]
        if tripped_arms:
            stopped = True
            stop_details = {arm: arm_checks[arm] for arm in ARMS}
            print(
                f"seed {seed}: drift stop at step {end_step}, tripped arms: {tripped_arms}"
            )
            break
        print(f"seed {seed}: triplet cleared drift checks through step {end_step}")

    record = {
        "seed": seed,
        "checkpoint_steps": steps,
        "final_step": final_step,
        "stopped": stopped,
        "drift_checks_at_stop": stop_details,
        "baseline": baseline,
    }
    atomic_json(record, stop_path)
    return record
""")

md("""
## 20.25 Run the triplet-synchronous training loop for every seed
""")

code("""
seed_training_records = {seed: train_seed_triplet(seed) for seed in SEEDS}
display(pd.DataFrame([
    {
        "seed": seed,
        "final_step": record["final_step"],
        "of_total": seed_corpora[seed]["total_presentations"],
        "stopped": record["stopped"],
    }
    for seed, record in seed_training_records.items()
]))
""")

md("""
## 20.26 Held-out evaluation on the 19 reserved answers

Every arm that reached a checkpoint - by finishing all steps or by tripping a
drift rule - is replayed under the exact Lab 18d answer-constrained rules:
the same fixed `RAISE` opening, the same greedy full-list decoder, the same
19 `DEFAULT_EVAL_ANSWERS`. The incumbent's own Lab 18d outcomes are loaded
rather than replayed, since Lab 18d already produced them under an identical
protocol and re-running them would only spend compute to reproduce numbers
already on disk.
""")

code("""
def held_out_paths(seed: int, arm: str) -> dict[str, Path]:
    stem = f"seed{seed}-{arm}"
    return {
        "calls": RESULTS_DIR / f"{stem}-calls.csv",
        "games": RESULTS_DIR / f"{stem}-games.csv",
        "scores": RESULTS_DIR / f"{stem}-scores.npy",
        "score_keys": RESULTS_DIR / f"{stem}-score-keys.csv",
        "progress": RESULTS_DIR / f"{stem}-progress.json",
    }


def play_held_out_game(model, seed: int, arm: str, answer: str) -> tuple[list[dict], dict, list[np.ndarray]]:
    history = [Turn(OPENING, score_string(answer, OPENING))]
    seen = {OPENING}
    call_rows = []
    score_vectors = []
    solved_turn = None
    started = time.perf_counter()

    for turn_number in range(2, MAX_TURNS + 1):
        before = candidate_indices_from_history(history)
        prompt = structured_next_guess_prompt(history, len(before))
        model_scores = score_all_words(model, prompt)
        assert LAST_STATE_PEAK_GIB < MEMORY_ABORT_GIB, (
            f"memory regression at seed {seed} {arm} {answer} turn {turn_number}: "
            f"{LAST_STATE_PEAK_GIB:.1f} GiB"
        )
        score_vectors.append(model_scores)
        winner_index = int(model_scores.argmax())
        guess = ANSWERS[winner_index]
        repeated = guess in seen
        seen.add(guess)
        feedback = score_string(answer, guess)
        history.append(Turn(guess, feedback))

        ranks = rank_vector(model_scores)
        teacher_index = expert.choose(before)
        candidate_mass = float(
            np.exp(model_scores[before] - model_scores.max()).sum()
            / np.exp(model_scores - model_scores.max()).sum()
        )
        open_entropy_regret_bits = float("nan")
        if turn_number == 2:
            # "Open" regret compares the chosen guess against the best
            # possible information gain over the *full* lexicon, not just the
            # candidate set - the fixed RAISE opening leaves every word still
            # legal, so this is the one turn where "open" and "candidate-only"
            # differ. Matches Lab 18d's `open_entropy_regret_bits`.
            chosen_entropy = expert.entropy(winner_index, before)
            open_entropies = np.array([
                expert.entropy(int(index), before) for index in ALL_INDICES
            ])
            open_entropy_regret_bits = float(open_entropies.max() - chosen_entropy)
        call_rows.append({
            "seed": seed,
            "arm": arm,
            "answer": answer,
            "turn": turn_number,
            "guess": guess,
            "repeated": repeated,
            "candidate_count_before": len(before),
            "candidate_mass": candidate_mass,
            "best_candidate_rank": int(ranks[before].min()),
            "candidate_teacher_rank": int(ranks[teacher_index]),
            "singleton_candidate_rank": (
                int(ranks[before[0]]) if len(before) == 1 else float("nan")
            ),
            "open_entropy_regret_bits": open_entropy_regret_bits,
            "driver_peak_gib": LAST_STATE_PEAK_GIB,
        })
        if feedback == "GGGGG":
            solved_turn = turn_number
            break

    return call_rows, {
        "seed": seed,
        "arm": arm,
        "answer": answer,
        "solved": solved_turn is not None,
        "solved_turn": solved_turn,
        "model_calls": len(call_rows),
        "turn2_guess": call_rows[0]["guess"] if call_rows else None,
        "elapsed_seconds": time.perf_counter() - started,
    }, score_vectors
""")

md("""
## 20.27 Run the restartable held-out evaluation for every completed arm
""")

code("""
def evaluate_arm(seed: int, arm: str, step: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = held_out_paths(seed, arm)
    progress = {
        "seed": seed,
        "arm": arm,
        "step": step,
        "answers": list(DEFAULT_EVAL_ANSWERS),
    }
    if paths["progress"].exists():
        assert json.loads(paths["progress"].read_text()) == progress
    else:
        atomic_json(progress, paths["progress"])

    if paths["games"].exists():
        required = [
            paths["calls"], paths["scores"], paths["score_keys"],
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"seed {seed} {arm} games exist without artifacts: {missing}"
            )
        calls = pd.read_csv(paths["calls"])
        games = pd.read_csv(paths["games"])
        score_matrix = np.load(paths["scores"])
        score_keys = pd.read_csv(paths["score_keys"])
    else:
        calls = pd.DataFrame()
        games = pd.DataFrame()
        score_matrix = np.empty((0, len(ANSWERS)), dtype=np.float32)
        score_keys = pd.DataFrame(columns=["seed", "answer", "turn"])
    completed = set(games["answer"]) if len(games) else set()
    if len(games):
        assert not games["answer"].duplicated().any()
        retained = min(len(calls), len(score_keys), len(score_matrix))
        calls = calls.iloc[:retained].reset_index(drop=True)
        score_keys = score_keys.iloc[:retained].reset_index(drop=True)
        score_matrix = score_matrix[:retained]
        assert calls["answer"].tolist() == score_keys["answer"].tolist()
        assert calls["turn"].tolist() == score_keys["turn"].tolist()
        keep = score_keys["answer"].isin(completed).to_numpy()
        calls = calls.loc[keep].reset_index(drop=True)
        score_keys = score_keys.loc[keep].reset_index(drop=True)
        score_matrix = score_matrix[keep]

    if not RUN_EVALUATION and completed != set(DEFAULT_EVAL_ANSWERS):
        raise FileNotFoundError(
            f"seed {seed} {arm} held-out evaluation incomplete and RUN_EVALUATION=False"
        )

    model = None
    if completed != set(DEFAULT_EVAL_ANSWERS):
        checkpoint_path = resolve_checkpoint(seed, arm, step)
        model = load_arm_adapter(checkpoint_path)
    for answer in DEFAULT_EVAL_ANSWERS:
        if answer in completed:
            continue
        new_calls, new_game, new_scores = play_held_out_game(model, seed, arm, answer)
        calls = pd.concat([calls, pd.DataFrame(new_calls)], ignore_index=True)
        games = pd.concat([games, pd.DataFrame([new_game])], ignore_index=True)
        new_score_matrix = np.stack(new_scores).astype(np.float32, copy=False)
        new_keys = pd.DataFrame([
            {"seed": row["seed"], "answer": row["answer"], "turn": row["turn"]}
            for row in new_calls
        ])
        score_matrix = np.concatenate([score_matrix, new_score_matrix], axis=0)
        score_keys = pd.concat([score_keys, new_keys], ignore_index=True)
        atomic_npy(score_matrix, paths["scores"])
        atomic_csv(score_keys, paths["score_keys"])
        atomic_csv(calls, paths["calls"])
        atomic_csv(games, paths["games"])
        print(
            f"seed {seed} {arm} {answer}: "
            f"{'SOLVED' if new_game['solved'] else 'FAILED'} "
            f"turn={new_game['solved_turn']} calls={new_game['model_calls']}",
            flush=True,
        )
    if model is not None:
        release_model(model)
        del model

    assert set(games["answer"]) == set(DEFAULT_EVAL_ANSWERS)
    assert score_matrix.shape == (len(calls), len(ANSWERS))
    return calls, games


held_out_calls = {}
held_out_games = {}
for seed in SEEDS:
    step = seed_training_records[seed]["final_step"]
    for arm in ARMS:
        calls, games = evaluate_arm(seed, arm, step)
        held_out_calls[(seed, arm)] = calls
        held_out_games[(seed, arm)] = games
        print(
            f"seed {seed} {arm} (step {step}): "
            f"{int(games['solved'].sum())}/{len(games)} solved"
        )
""")

md("""
## 20.28 Load the Lab 18d incumbent outcomes and report held-out results

The incumbent's own 19-answer outcomes come straight from Lab 18d's persisted
`answer-constrained` games rather than a rerun, since both used the identical
decoder and answer set. Reporting covers solve rate, turns on wins, Turn 2
open regret and realized candidate reduction, singleton closure, repeats,
candidate mass/rank statistics, and winner concentration at Turn 2.
""")

code("""
incumbent_games = {}
incumbent_calls = {}
for seed in SEEDS:
    incumbent_games[seed] = pd.read_csv(
        LAB18D_RESULTS / f"seed{seed}-answer-constrained-games.csv"
    )
    incumbent_calls[seed] = pd.read_csv(
        LAB18D_RESULTS / f"seed{seed}-answer-constrained-calls.csv"
    )
    assert set(incumbent_games[seed]["answer"]) == set(DEFAULT_EVAL_ANSWERS)


def normalize_calls(calls: pd.DataFrame) -> pd.DataFrame:
    # Lab 18d's calls frame names the same two quantities
    # `model_best_candidate_rank` / `model_teacher_rank`; every other column
    # this report needs already shares its name with the Lab 20 held-out
    # calls frame built in 20.25 (`candidate_count_before`, `candidate_mass`,
    # `repeated`, `open_entropy_regret_bits`, `turn`, `guess`).
    if "model_best_candidate_rank" in calls.columns:
        calls = calls.rename(columns={
            "model_best_candidate_rank": "best_candidate_rank",
            "model_teacher_rank": "candidate_teacher_rank",
        })
    return calls


def turn2_open_regret(calls: pd.DataFrame) -> float:
    turn2 = calls.loc[calls["turn"] == 2]
    return float(turn2["open_entropy_regret_bits"].mean())


def winner_concentration(calls: pd.DataFrame, turn: int = 2) -> dict:
    guesses = calls.loc[calls["turn"] == turn, "guess"]
    counts = guesses.value_counts()
    return {
        "unique_winners": int(counts.size),
        "largest_winner_share": float(counts.iloc[0] / len(guesses)) if len(guesses) else float("nan"),
    }


def summarize_arm(seed: int, arm: str, games: pd.DataFrame, calls: pd.DataFrame) -> dict:
    calls = normalize_calls(calls)
    wins = games.loc[games["solved"]]
    singleton_calls = calls.loc[calls["candidate_count_before"] == 1]
    concentration = winner_concentration(calls, turn=2)
    return {
        "seed": seed,
        "arm": arm,
        "solve_rate": float(games["solved"].mean()),
        "mean_turns_on_win": float(wins["solved_turn"].mean()) if len(wins) else float("nan"),
        "turn2_open_regret": turn2_open_regret(calls),
        "mean_candidate_reduction_turn2": float(
            1.0 - (
                calls.loc[calls["turn"] == 3, "candidate_count_before"].mean()
                / calls.loc[calls["turn"] == 2, "candidate_count_before"].mean()
            )
        ) if (calls["turn"] == 3).any() else float("nan"),
        "singleton_closure_rate": float(
            (singleton_calls["best_candidate_rank"] == 1).mean()
        ) if len(singleton_calls) else float("nan"),
        "repeat_rate": float(calls["repeated"].mean()),
        "mean_candidate_mass": float(calls["candidate_mass"].mean()),
        "median_best_candidate_rank": float(calls["best_candidate_rank"].median()),
        "unique_turn2_winners": concentration["unique_winners"],
        "largest_turn2_winner_share": concentration["largest_winner_share"],
    }


arm_summaries_frame = pd.DataFrame(
    [
        summarize_arm(seed, arm, held_out_games[(seed, arm)], held_out_calls[(seed, arm)])
        for seed in SEEDS
        for arm in ARMS
    ]
    + [
        summarize_arm(seed, "incumbent", incumbent_games[seed], incumbent_calls[seed])
        for seed in SEEDS
    ]
)
atomic_csv(arm_summaries_frame, RESULTS_DIR / "held-out-summary.csv")
display(arm_summaries_frame)
""")

md("""
## 20.29 Primary estimate: pooled answer-level bootstrap and the gate

The primary contrast is `rollout_correction - static_random`, equal-weighted
across the three seeds. The pooled bootstrap resamples the 19 answer IDs with
replacement; for each sampled answer it keeps every seed's paired outcome for
both arms, averages the per-seed solve-rate difference over the resampled
multiset of answers, and then averages the three seeds - one statistic per
resample, 10,000 resamples, a fixed seed. The gate requires that no seed
triplet stopped, the mean gain is at least 0.05, every one of the three
seed-paired gains is strictly positive, and the 95% percentile CI lower bound
is above zero. `static_matched` is diagnostic only and never enters the gate.
""")

code("""
solved_wide = {
    (seed, arm): held_out_games[(seed, arm)].set_index("answer")["solved"].astype(float)
    for seed in SEEDS for arm in ARMS
}

PRIMARY_ARM = "rollout_correction"
CONTROL_ARM = "static_random"

seed_paired_diff = {
    seed: (solved_wide[(seed, PRIMARY_ARM)] - solved_wide[(seed, CONTROL_ARM)])
    .loc[list(DEFAULT_EVAL_ANSWERS)]
    for seed in SEEDS
}
seed_paired_gain = {seed: float(diff.mean()) for seed, diff in seed_paired_diff.items()}
mean_gain = float(np.mean(list(seed_paired_gain.values())))

diff_matrix = np.stack([
    seed_paired_diff[seed].to_numpy() for seed in SEEDS
])  # shape (3 seeds, 19 answers), answer order == DEFAULT_EVAL_ANSWERS

bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
num_answers = len(DEFAULT_EVAL_ANSWERS)
bootstrap_stats = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
for sample in range(BOOTSTRAP_SAMPLES):
    resample = bootstrap_rng.integers(num_answers, size=num_answers)
    per_seed_means = diff_matrix[:, resample].mean(axis=1)
    bootstrap_stats[sample] = per_seed_means.mean()

ci_low, ci_high = np.percentile(bootstrap_stats, [2.5, 97.5])
any_triplet_stopped = any(
    seed_training_records[seed]["stopped"] for seed in SEEDS
)
gate = {
    "primary_arm": PRIMARY_ARM,
    "control_arm": CONTROL_ARM,
    "seed_paired_gain": seed_paired_gain,
    "mean_gain": mean_gain,
    "bootstrap_ci_low": float(ci_low),
    "bootstrap_ci_high": float(ci_high),
    "any_triplet_stopped": any_triplet_stopped,
    "checks": {
        "no_triplet_stopped": not any_triplet_stopped,
        "mean_gain_at_least_0.05": mean_gain >= GATE_MIN_MEAN_SOLVE_GAIN,
        "every_seed_gain_positive": all(
            gain > 0 for gain in seed_paired_gain.values()
        ),
        "ci_lower_bound_above_zero": float(ci_low) > 0.0,
    },
}
gate["passed"] = all(gate["checks"].values())
atomic_json(gate, RESULTS_DIR / "correction-gate.json")

print(f"per-seed gain (rollout_correction - static_random): {seed_paired_gain}")
print(f"mean gain: {mean_gain:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
for check, value in gate["checks"].items():
    print(f"  {check}: {value}")
print(f"GATE PASSED: {gate['passed']}")
""")

md("""
## 20.30 `static_matched` diagnostic

`static_matched` is not part of the gate. It exists to separate two possible
explanations for any `rollout_correction` advantage over `static_random`:
matching the exact `(answer_branch, turn, candidate_stratum)` distribution
the rollout visited, versus the rollout states themselves carrying
information a stratum-matched-but-otherwise-arbitrary dev state cannot
capture. If `static_matched` closes most of the gap between `static_random`
and `rollout_correction`, distributional matching - not the specific states
the policy reached - would be the more likely explanation.
""")

code("""
static_matched_gain = {
    seed: float(
        (solved_wide[(seed, PRIMARY_ARM)] - solved_wide[(seed, "static_matched")])
        .loc[list(DEFAULT_EVAL_ANSWERS)]
        .mean()
    )
    for seed in SEEDS
}
matched_vs_random_gain = {
    seed: float(
        (solved_wide[(seed, "static_matched")] - solved_wide[(seed, CONTROL_ARM)])
        .loc[list(DEFAULT_EVAL_ANSWERS)]
        .mean()
    )
    for seed in SEEDS
}
diagnostic = {
    "rollout_correction_minus_static_matched": static_matched_gain,
    "static_matched_minus_static_random": matched_vs_random_gain,
    "mean_rollout_minus_matched": float(np.mean(list(static_matched_gain.values()))),
    "mean_matched_minus_random": float(np.mean(list(matched_vs_random_gain.values()))),
}
atomic_json(diagnostic, RESULTS_DIR / "static-matched-diagnostic.json")
print("static_matched diagnostic (not gated):")
for key, value in diagnostic.items():
    print(f"  {key}: {value}")
""")

md("""
## 20.31 Final run manifest

One manifest ties every artifact this notebook produced back to the frozen
contract in 20.3: incumbent checkpoint hashes, the anchor suite, each seed's
corpus and training outcome, and the gate verdict.
""")

code("""
final_manifest = {
    "experiment": "Lab 20 - correct policy-created states",
    "incumbent_checkpoint_sha256": checkpoint_hashes,
    "anchor_manifest_sha256": sha256_file(RESULTS_DIR / "anchor-manifest.json"),
    "static_pool_manifest_sha256": sha256_file(RESULTS_DIR / "static-pool-manifest.json"),
    "seed_corpus_manifest_sha256": {
        seed: sha256_file(RESULTS_DIR / f"seed{seed}-corpus-manifest.json") for seed in SEEDS
    },
    "seed_training_records": seed_training_records,
    "held_out_summary_sha256": sha256_file(RESULTS_DIR / "held-out-summary.csv"),
    "gate": gate,
    "static_matched_diagnostic": diagnostic,
}
atomic_json(final_manifest, RESULTS_DIR / "lab20-run.json")
print("Lab 20 run manifest written to results/lab20/lab20-run.json")
""")

md("""
## Lab 20 checkpoint

**What this notebook establishes, if the gate passes.** Correcting states the
policy itself reached beats an equal amount of static expert data at the same
presentation count, optimizer-update count, padded-token budget, and replay
protection - holding the incumbent-preservation mechanism fixed across arms.
That isolates the *source* of new correction states as the active variable,
separate from Lab 19's training-objective manipulations.

**Read the results in this order.** (1) Did any seed triplet stop on a drift
rule? A stopped triplet cannot pass the gate, regardless of its raw solve
rate - the rule exists to keep update counts matched, not to reward whichever
arm happened to drift least. (2) Does the primary bootstrap gate pass? (3) Only
then, does `static_matched` closing most of the `rollout_correction` -
`static_random` gap suggest the effect is really about matching the
rollout's state distribution rather than the specific states it reached?

**Limitations.** Nineteen held-out answers is a small evaluation set; the
bootstrap CI reflects that. The static pools are drawn from the same
structured `dev` split the rollout draws its answers from, so `static_random`
and `static_matched` are already fairly strong controls - a null result here
says less about whether *any* new data helps than about whether policy-reached
states specifically help beyond static dev data. Optimizer-state continuity
across a resumed training block relies on saved Adam moments and step-indexed
scheduler state; per-layer dropout RNG state is not separately checkpointed,
so a resumed run's dropout stream can diverge slightly from an uninterrupted
one - noted here rather than silently assumed away.

**Deliberately out of scope.** This experiment does not iterate: it collects
one rollout corpus, trains once per arm to a stop point, and evaluates once.
A dynamic negative-refresh design - repeatedly re-mining policy-created
states as training proceeds, DAgger-style - is a different, harder-to-control
experiment and is deferred rather than folded in here.
""")


for index, cell in enumerate(cells):
    cell["id"] = f"lab20-{index:02d}-{cell['cell_type']}"

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

path = Path("notebooks/20_policy_state_correction.ipynb")
path.write_text(json.dumps(notebook, indent=1))
print(f"wrote {path} with {len(cells)} cells")