"""Generate notebooks/19_value_aware_distillation.ipynb."""

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
# Lab 19 - Distillation revisited with value-aware targets

Lab 18d removed lexical grounding as an excuse. With an answer-constrained
decoder every B-structured seed solved 10 of 19 reserved games, against 5, 3,
and 3 under free generation. The remaining failures are strategic, and they
split into two very different shapes.

**Broad-state action value.** On Turn 2 the constrained policy matched the
candidate teacher on 0 of 57 calls and left 1.05 to 1.45 bits of open-action
entropy regret on the table. It ranks plausible words, but it does not rank them
by how much they would divide the candidate set.

**Late closure.** Eighteen of the 27 constrained failures ended with exactly one
candidate remaining, and across all constrained play the sole surviving
candidate was chosen on only 18 of 65 singleton calls. The model reaches a state
where the answer is fully determined and then names a different word.

One objective cannot be read off a single number, because these two failures do
not share a scale. Entropy is the right currency in broad states. In a singleton
state every action has zero entropy, including the correct one, so entropy is
silent exactly where the game is decided. Lab 19 builds a teacher target that is
explicit about both regimes and tests it against a matched one-hot control.
""")

md("""
## 19.1 Pre-registered experiment

**Question.** Does distilling a *relative action-value* distribution improve the
constrained policy more than distilling the same teacher's single best action?

**Arms.** Each of the three B-structured incumbents is continued twice.

| arm | teacher target on 12 scored actions |
| --- | --- |
| `hard` | one-hot on the teacher's chosen action |
| `value` | soft distribution over all 12 actions |

Everything else is identical inside a seed pair: the same starting adapter, the
same 1,029 states in the same order, the same 12 actions per state, the same
update count, optimizer, schedule, and dropout seed. The two arms differ only in
the probability vector on the right-hand side of one cross-entropy.

| seed | incumbent | source |
| ---: | --- | --- |
| 42 | `qwen3-0.6b-wordle-lora-dataset-b-structured` | Lab 17 |
| 45 | `...-b-structured-seed45` | Lab 18c |
| 47 | `...-b-structured-seed47` | Lab 18c |

**Objective.** For a state with actions `a_1..a_12`, the student score is the
summed `log P(word tokens + EOS | structured prompt)` used since Lab 18b, and
the loss is

```text
-(target_probs * log_softmax(student_action_scores)).sum()
```

for both arms. `hard` merely supplies a one-hot vector, so the arms cannot
differ through the loss formula, the action support, or the optimization path.

**Teacher targets by regime.**

| regime | states | value target | hard target |
| --- | --- | --- | --- |
| broad, `candidate_count >= 3` | 686 | `softmax(entropy_bits / 0.5)` over the 12 actions | argmax entropy, preferring a tied candidate, then lexicographic |
| sharp, `candidate_count <= 2` | 343 | equal mass on the 1-2 current candidates, zero elsewhere | lexicographically first candidate |

The sharp rule is the whole point of calling this *value-aware* rather than
*entropy-aware*. At a singleton the entropy of every action is 0.0 bits, so an
entropy target is uniform noise precisely where naming the candidate wins the
game. Value says: this word ends it, the others do not.

**Primary outputs.**

1. Held-out dev shortlist metrics for incumbent, `hard`, and `value` on all
   three seeds: cross-entropy and KL to the value target, `hard` top-1 rate,
   mean broad open regret in bits, and sharp candidate-selection rate.
2. Frozen 19-answer gameplay under exact Lab 18d rules with the free and
   answer-constrained decoders, reported by seed and arm.

**Secondary outputs.** Stratification by regime, candidate bucket, and turn;
Turn 2 open-teacher regret and realized `log2` candidate reduction; repeat and
consistency rates; singleton closure rate; and the number of failures that end
at a singleton. Closure is evaluated per game at the first singleton
opportunity; pooled singleton calls are descriptive because failures create
additional calls.

**Read before seeing results.**

| observation | pre-registered interpretation |
| --- | --- |
| `hard` improves closure while `value` improves broad regret | the *relative* target matters, and the two regimes need different information |
| both arms move together | explicit regime coverage plus 1,029 more updates explains the gain; the soft distribution adds nothing |
| `value` harms sharp closure | the soft broad objective interferes with committing to a determined answer |
| neither arm beats its incumbent | the objective, the 12-action support, or the optimization budget failed; not evidence that distillation cannot work |

Three seeds and 19 answers give a paired diagnostic, not a population solve
rate. The replication unit is the seed. State rows are not independent training
runs, and 1,029 training states are not 1,029 replications of anything.

The source pool is still thinner than deployment at the broadest states. Lab
19 includes every available broad Turn 2 and 11-plus-candidate training state,
but a null Turn 2 result can still mean that the existing curriculum does not
contain enough comparable broad states, not only that the target shape failed.
""")

md("""
## 19.2 Why Lab 09 distillation was not enough

Lab 09 already distilled a soft policy from this same symbolic teacher, and Lab
12 measured the result: 0.927 top-1 agreement with the teacher under a
candidate-restricted decoder and a mean policy KL of 0.186. The same model
scored 0.0 solve rate with a 100% invalid free-output rate.

Near-perfect agreement with a teacher, and no gameplay. Three design choices
explain it, and Lab 19 changes all three.

**The action space was the candidate set.** Lab 09 defined the shared action
space as the surviving candidates only. A model trained that way is never told
anything about the 2,300-odd words it will actually be ranked against at
deployment. Lab 18b showed the deployed decision is a ranking over all 2,315
answers; the teacher distribution has to say something about words outside the
candidate set, or it constrains nothing there. Lab 19 puts 6 open actions, 3
candidates, and deterministic filler in every support, so non-candidates receive
explicit low probability instead of no signal.

**Entropy is degenerate where games end.** Lab 09's targets were entropy-shaped
everywhere. With one candidate left, every action has entropy 0.0 bits and the
softmax of a constant vector is uniform. Lab 18d's 18-of-27 singleton failures
are what that looks like in a game. Lab 19 switches the utility to value in
sharp states.

**Agreement was measured inside the teacher's own restriction.** A 0.927
candidate-restricted top-1 rate says the student ranks candidates like the
teacher. It cannot detect that the student would rather emit something else
entirely. Lab 19 reports the unrestricted argmax over the support, the regret in
bits against the *open* optimum, and full games.

There is one more reason to run `hard` as a control. Lab 19 changes the action
support, the regime coverage, and adds 1,029 optimizer updates. Any of those
could produce a gain on its own. Only the matched `hard` arm isolates the part
of the gain attributable to the soft value distribution.
""")

md("""
## 19.3 Run controls and memory guard

Run this notebook only through the total-system watchdog:

```
scripts/memguard.py --min-free 64 -- uv run jupyter nbconvert \\
    --to notebook --execute --inplace notebooks/19_value_aware_distillation.ipynb
```

The in-process MPS cap turns a runaway allocation into an ordinary exception.
Four gates run before anything expensive: a numerical regression of the batched
12-action scorer against plain single-action forwards, a fixed-shape 40-step
training soak on the longest state, a reproduction of one persisted Lab 18b
score vector, and a fixed-shape 40-repeat full-list scoring soak.

Completed training arms, dev shortlist scores, and gameplay artifacts are all
written atomically. An interruption can lose the current training arm, one
model's dev pass, or one game; already completed arms and games are reused.
""")

code("""
RUN_TRAINING = True
RUN_EVALUATION = True

MEMORY_CAP_GIB = 128.0

import torch

if torch.backends.mps.is_available():
    total_gib = torch.mps.recommended_max_memory() / 1024**3
    torch.mps.set_per_process_memory_fraction(MEMORY_CAP_GIB / total_gib)
    print(f"MPS cap: {MEMORY_CAP_GIB:.0f} GiB of {total_gib:.0f} GiB")

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
ARMS = ["hard", "value"]

UPDATES = 1029
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.05
GRAD_CLIP = 1.0
LOG_EVERY = 50
SUPPORT_SIZE = 12
BROAD_OPEN_SLOTS = 6
BROAD_CANDIDATE_SLOTS = 3
SHARP_RANDOM_SLOTS = 2
TEACHER_TEMPERATURE = 0.5
BROAD_THRESHOLD = 3
PRIORITY_CANDIDATE_COUNT = 11
BROAD_STATES = 686
SHARP_STATES = 343
SINGLETON_STATES = 229
TWO_CANDIDATE_STATES = 114
SELECTION_SEED = 1900
ORDER_SEED = 1907

MAX_TURNS = 6
OPENING = "RAISE"
DECODERS = ["free", "answer-constrained"]
CHUNK_SIZE = 256
MEMORY_ABORT_GIB = MEMORY_CAP_GIB * 0.75

DATA_DIR = Path("../data")
GENERATED_DIR = DATA_DIR / "generated"
CHECKPOINT_ROOT = Path("../checkpoints")
RESULTS_DIR = Path("../results/lab19")
LAB18B_RESULTS = Path("../results/lab18b")
LAB18D_RESULTS = Path("../results/lab18d")

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
ARM_CHECKPOINTS = {
    (seed, arm): CHECKPOINT_ROOT / f"qwen3-0.6b-wordle-lab19-{arm}-seed{seed}"
    for seed in SEEDS
    for arm in ARMS
}
TARGET_FILES = {
    "train": GENERATED_DIR / "lab19-value-targets-train.jsonl",
    "dev": GENERATED_DIR / "lab19-value-targets-dev.jsonl",
}
TARGET_MANIFEST = GENERATED_DIR / "lab19-value-targets-manifest.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = preferred_device()
torch.set_float32_matmul_precision("high")
print("device:", device)


def driver_memory_gib() -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1024**3
    return float("nan")


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
## 19.4 Freeze every source artifact

Lab 19 reads Lab 17's structured JSONL and three frozen adapters. Nothing is
regenerated. The file hashes must match the Dataset B manifest and the manifest
stored inside each incumbent checkpoint, and every incumbent must carry the same
LoRA geometry, or the two arms would not be continuing from a common ancestor.
""")

code("""
structured_manifest = json.loads(
    (GENERATED_DIR / "wordle-part2-structured-manifest.json").read_text()
)
structured_hashes = {
    split: sha256_file(path) for split, path in STRUCTURED_FILES.items()
}
assert structured_hashes == structured_manifest["structured_sha256"]

INCUMBENT_MANIFEST_NAMES = ["lab17-run.json", "lab18c-run.json"]
incumbent_manifests = {}
incumbent_hashes = {}
for seed, path in INCUMBENTS.items():
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
    incumbent_hashes[seed] = sha256_file(path / "adapter_model.safetensors")

display(pd.DataFrame([
    {
        "seed": seed,
        "checkpoint": INCUMBENTS[seed].name,
        "prior_steps": incumbent_manifests[seed]["optimizer_steps"],
        "prior_val_loss": incumbent_manifests[seed]["final_val_loss"],
        "adapter_sha256": incumbent_hashes[seed][:16],
    }
    for seed in SEEDS
]))
""")

code("""
structured_rows = {
    split: [json.loads(line) for line in path.read_text().splitlines()]
    for split, path in STRUCTURED_FILES.items()
}
RESERVED_ANSWERS = list(DEFAULT_EVAL_ANSWERS)
assert RESERVED_ANSWERS == [
    "SHORE", "MIGHT", "BRICK", "GHOST", "KNIFE", "DOUBT", "FLING",
    "ROUND", "CHAMP", "WASTE", "BLIND", "POINT", "SLATE", "CRANE",
    "APPLE", "SHEEP", "BANAL", "ALLEY", "AUDIO",
]
RESERVED_SET = set(RESERVED_ANSWERS)

next_guess_rows = {
    split: [row for row in rows if row["task"] == "NEXT_GUESS"]
    for split, rows in structured_rows.items()
}
for split, rows in structured_rows.items():
    assert not any(row["answer"] in RESERVED_SET for row in rows), (
        f"reserved gameplay answer leaked into {split}"
    )
    assert all(
        row["representation"] == "derived_state_v1" for row in rows
    )

split_state_keys = {
    split: {row["state_key"] for row in rows}
    for split, rows in next_guess_rows.items()
}
assert not (split_state_keys["train"] & split_state_keys["validation"])
assert not (split_state_keys["train"] & split_state_keys["test"])
assert not (split_state_keys["validation"] & split_state_keys["test"])

display(pd.DataFrame([
    {
        "split": split,
        "rows": len(structured_rows[split]),
        "next_guess_rows": len(next_guess_rows[split]),
        "unique_states": len(split_state_keys[split]),
    }
    for split in structured_rows
]))
""")

md("""
## 19.5 The frozen `derived_state_v1` representation

The input representation does not change in Lab 19. These functions rebuild it
from a state key, and the check below is the proof: every stored `NEXT_GUESS`
prompt in train and dev must be reproduced character for character. The same
builder is used later for gameplay, so training states and deployed states go
through one code path.

Candidates are reconstructed from the state key alone through the pattern
matrix. The hidden `answer` field is never consulted when building actions or
targets, and the reconstructed count must equal the count recorded by Lab 14.
""")

code("""
ANSWERS = [
    line.strip().upper()
    for line in (DATA_DIR / "wordle-answers-original.txt").read_text().splitlines()
    if line.strip()
]
ANSWER_SET = set(ANSWERS)
ANSWER_ARRAY = np.array(ANSWERS)
WORD_TO_INDEX = {word: index for index, word in enumerate(ANSWERS)}
PATTERNS = np.load(DATA_DIR / "wordle-patterns-original-2315.npy")
expert = EntropyExpert(ANSWERS, PATTERNS)
ALL_INDICES = expert.all_indices
assert len(ANSWERS) == 2315
assert PATTERNS.shape == (2315, 2315)
assert expert.word_to_index == WORD_TO_INDEX


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


def structured_next_guess_prompt(
    history: list[Turn], candidate_count: int
) -> str:
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
        indices = expert.update(
            indices, WORD_TO_INDEX[turn.guess], turn.feedback
        )
    if len(indices) == 0:
        raise ValueError("state key produced an empty candidate set")
    return indices
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
        "source": row["source"],
        "candidate_count": len(candidates),
        "regime": "broad" if len(candidates) >= BROAD_THRESHOLD else "sharp",
        "prompt": prompt,
    })

states = pd.DataFrame(state_records).sort_values(
    ["split", "state_key"], kind="stable"
).reset_index(drop=True)

verification_sample = states.iloc[::150]
for row in verification_sample.itertuples():
    history = parse_state_key(row.state_key)
    rebuilt = sorted(
        ANSWERS[int(index)]
        for index in candidate_indices_from_history(history)
    )
    assert rebuilt == sorted(filter_candidates(ANSWERS, history))
print(
    f"pattern-matrix candidates match filter_candidates on "
    f"{len(verification_sample)} states"
)
print("every stored NEXT_GUESS prompt reproduced:", len(states))
display(
    states.groupby(["split", "regime"]).size().rename("states").reset_index()
)
""")

md("""
## 19.6 Action-value utilities

`action_entropies` scores all 2,315 answer-list actions against the current
candidate set at once. It must agree with `EntropyExpert.entropy`, which is the
function Labs 05 through 18d used, because the teacher definition is supposed to
be unchanged; only the target *shape* is new.

Two teachers are named for every broad state, exactly as in Lab 18d. The
**open teacher** maximizes entropy over all 2,315 answer words. The **canonical
candidate teacher** maximizes entropy among current candidates. Ties break
lexicographically in both cases.

Ties are common and they are not an edge case: many words split a candidate set
identically. Entropies that are equal in exact arithmetic can differ by about
`1e-16` in floating point depending on summation order, so utilities are
quantized to nine decimals before the lexicographic tie-break. Without that,
the argmax over the whole answer list and the argmax over a twelve-action
support could disagree even when both contain the same winning word, which
would silently desynchronize the hard target from the open teacher.
""")

code("""
def action_entropies(candidates: np.ndarray) -> np.ndarray:
    patterns = PATTERNS[:, candidates].astype(np.int64)
    offsets = np.arange(len(ANSWERS), dtype=np.int64)[:, None] * 243
    counts = np.bincount(
        (patterns + offsets).ravel(), minlength=len(ANSWERS) * 243
    ).reshape(len(ANSWERS), 243).astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        counts, totals, out=np.zeros_like(counts), where=counts > 0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            counts > 0, probabilities * np.log2(probabilities), 0.0
        )
    return -terms.sum(axis=1)


UTILITY_DECIMALS = 9


def utility_key(index: int, utilities: np.ndarray) -> tuple[float, str]:
    # Mathematically tied entropies can differ by about 1e-16 depending on
    # summation order, so utilities are quantized before the lexicographic
    # tie-break. Quantizing first makes the ranking subset-consistent: the
    # winner over the full answer list is still the winner over any support
    # that contains it.
    return (
        -round(float(utilities[int(index)]), UTILITY_DECIMALS),
        ANSWERS[int(index)],
    )


def utility_order(indices: np.ndarray, utilities: np.ndarray) -> list[int]:
    return sorted(
        (int(index) for index in indices),
        key=lambda index: utility_key(index, utilities),
    )


def lexicographic_argmax(indices: np.ndarray, utilities: np.ndarray) -> int:
    return utility_order(indices, utilities)[0]


entropy_check_rows = []
for row in states.iloc[::400].itertuples():
    candidates = candidate_indices_from_history(parse_state_key(row.state_key))
    vectorized = action_entropies(candidates)
    probe = np.linspace(0, len(ANSWERS) - 1, 40).astype(int)
    reference = np.array([
        expert.entropy(int(index), candidates) for index in probe
    ])
    entropy_check_rows.append(
        float(np.max(np.abs(vectorized[probe] - reference)))
    )
max_entropy_difference = max(entropy_check_rows)
print(
    f"vectorized entropy agrees with EntropyExpert on "
    f"{len(entropy_check_rows)} states, max abs diff "
    f"{max_entropy_difference:.2e}"
)
assert max_entropy_difference < 1e-9
""")

md("""
## 19.7 Deterministic training-state selection

The 1,029 training states are chosen once and reused by every seed and every
arm. The split is 686 broad and 343 sharp, and inside each regime the selection
rule is fixed before any model is loaded:

1. every broad Turn 2 state and every state with 11 or more candidates enters
   first, because those are exactly the states Lab 18d found weakest;
2. the broad remainder is filled from the 3-to-10 candidate pool by a seeded
   permutation, without replacement;
3. sharp states take 229 singletons and 114 two-candidate states, roughly the
   2:1 ratio in which they occur, again by seeded permutation.

A state reachable in a frozen 19-answer evaluation can never enter the pool.
For every `RAISE`-opened state, the notebook reconstructs the candidates and
rejects it if any reserved answer remains possible. The same invariant is
required of the dev target audit rather than inferred from the row's hidden
answer field.

The presentation order is one seeded permutation of the 1,029 selected states,
identical for all six runs. Regime is therefore interleaved rather than
curricular, and the two arms of a seed see byte-identical inputs in the same
order.
""")

code("""
train_states = states.query("split == 'train'").reset_index(drop=True)
dev_states = states.query("split == 'validation'").reset_index(drop=True)


def reachable_by_reserved_answer(state_key: str) -> bool:
    history = parse_state_key(state_key)
    if not history or history[0].guess != OPENING:
        return False
    candidates = set(map(
        int, candidate_indices_from_history(history)
    ))
    return any(
        WORD_TO_INDEX[answer] in candidates
        for answer in RESERVED_ANSWERS
    )


train_reserved_reachable = train_states["state_key"].map(
    reachable_by_reserved_answer
)
dev_reserved_reachable = dev_states["state_key"].map(
    reachable_by_reserved_answer
)
print(
    "training states removed as reachable in a reserved game:",
    int(train_reserved_reachable.sum()),
)
assert not dev_reserved_reachable.any(), (
    "a dev target is reachable in a reserved RAISE-opened game"
)
pool = train_states.loc[~train_reserved_reachable].reset_index(drop=True)


def seeded_pick(keys: list[str], count: int, stream: int) -> list[str]:
    ordered = sorted(keys)
    assert count <= len(ordered), (
        f"stream {stream} needs {count} states but only {len(ordered)} exist"
    )
    rng = np.random.default_rng([SELECTION_SEED, stream])
    order = rng.permutation(len(ordered))
    return [ordered[int(position)] for position in order[:count]]


broad_pool = pool.query("candidate_count >= @BROAD_THRESHOLD")
priority_mask = (broad_pool["turn"] == 2) | (
    broad_pool["candidate_count"] >= PRIORITY_CANDIDATE_COUNT
)
priority_keys = sorted(broad_pool.loc[priority_mask, "state_key"])
broad_remainder = sorted(
    set(broad_pool["state_key"]) - set(priority_keys)
)
broad_fill = seeded_pick(broad_remainder, BROAD_STATES - len(priority_keys), 1)
broad_keys = priority_keys + broad_fill

singleton_keys = seeded_pick(
    list(pool.query("candidate_count == 1")["state_key"]),
    SINGLETON_STATES,
    2,
)
two_candidate_keys = seeded_pick(
    list(pool.query("candidate_count == 2")["state_key"]),
    TWO_CANDIDATE_STATES,
    3,
)
sharp_keys = singleton_keys + two_candidate_keys

selected_keys = broad_keys + sharp_keys
assert len(broad_keys) == BROAD_STATES
assert len(sharp_keys) == SHARP_STATES == SINGLETON_STATES + TWO_CANDIDATE_STATES
assert len(selected_keys) == UPDATES == BROAD_STATES + SHARP_STATES
assert len(set(selected_keys)) == len(selected_keys)
assert not set(selected_keys) & split_state_keys["validation"]
assert not set(selected_keys) & split_state_keys["test"]
assert not any(reachable_by_reserved_answer(key) for key in selected_keys)
assert set(priority_keys) <= set(broad_keys)
assert all(
    key in set(broad_pool["state_key"])
    for key in priority_keys
)

order_rng = np.random.default_rng([SELECTION_SEED, ORDER_SEED])
order = order_rng.permutation(len(selected_keys))
training_order = [selected_keys[int(position)] for position in order]
assert sorted(training_order) == sorted(selected_keys)

selection = train_states.set_index("state_key").loc[training_order].reset_index()
covered_priority = set(priority_keys)
available_priority = set(
    pool.loc[
        (pool["candidate_count"] >= BROAD_THRESHOLD)
        & (
            (pool["turn"] == 2)
            | (pool["candidate_count"] >= PRIORITY_CANDIDATE_COUNT)
        ),
        "state_key",
    ]
)
assert covered_priority == available_priority
print(
    f"selected {len(selection)} training states: "
    f"{len(broad_keys)} broad ({len(priority_keys)} priority-covered, "
    f"{len(broad_fill)} filled) and {len(sharp_keys)} sharp "
    f"({SINGLETON_STATES} singleton, {TWO_CANDIDATE_STATES} two-candidate)"
)
display(
    selection.assign(
        bucket=pd.cut(
            selection["candidate_count"],
            [0, 1, 2, 10, 10**6],
            labels=["1", "2", "3-10", "11+"],
        )
    ).groupby(["regime", "bucket"], observed=True).agg(
        states=("state_key", "size"),
        turn2=("turn", lambda values: int((values == 2).sum())),
        mean_turn=("turn", "mean"),
    ).reset_index()
)
""")

md("""
## 19.8 Twelve actions per state

Every state is scored on exactly 12 answer-list actions. The support is a
deterministic function of the state key: same 12 words for every seed, every
arm, and every restart, and no dependence on the model or on the hidden answer.

**Broad states** take the 6 highest-entropy open actions, the 3 highest-entropy
current candidates, and a deterministic random fill. The global open optimum and
the canonical candidate teacher are therefore always present, which is what
makes open regret measurable inside the support.

**Sharp states** take all 1-2 current candidates, any previous guess that is an
answer word, high lexical-overlap non-candidate distractors, and two
deterministic random fills. The distractors are the near misses the model
actually emits at the end of a game, and including previous guesses lets the
target assign explicit zero probability to a repeat.

The per-state random stream is seeded from a SHA-256 digest of the state key,
not from Python's salted `hash`, so the support is stable across processes.
""")

code("""
def support_rng(state_key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"lab19-support|{state_key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def overlap_key(word: str, candidate_words: list[str]) -> tuple:
    best = (0, 0)
    for candidate in candidate_words:
        positional = sum(left == right for left, right in zip(word, candidate))
        shared = len(set(word) & set(candidate))
        best = max(best, (positional, shared))
    return (-best[0], -best[1], word)


def build_action_support(state_key: str) -> dict:
    history = parse_state_key(state_key)
    candidates = candidate_indices_from_history(history)
    candidate_set = {int(index) for index in candidates}
    regime = "broad" if len(candidates) >= BROAD_THRESHOLD else "sharp"
    support: list[int] = []

    def add(index) -> None:
        index = int(index)
        if index not in support:
            support.append(index)

    if regime == "broad":
        entropies = action_entropies(candidates)
        open_order = utility_order(ALL_INDICES, entropies)
        candidate_order = utility_order(candidates, entropies)
        open_teacher = open_order[0]
        candidate_teacher = candidate_order[0]
        for index in open_order[:BROAD_OPEN_SLOTS]:
            add(index)
        for index in candidate_order[:BROAD_CANDIDATE_SLOTS]:
            add(index)
        required = {open_teacher, candidate_teacher}
    else:
        entropies = None
        open_teacher = None
        candidate_teacher = int(min(candidate_set, key=lambda i: ANSWERS[i]))
        for index in sorted(candidate_set, key=lambda i: ANSWERS[i]):
            add(index)
        for turn in history:
            if turn.guess in ANSWER_SET:
                add(WORD_TO_INDEX[turn.guess])
        candidate_words = [ANSWERS[index] for index in sorted(candidate_set)]
        distractor_slots = SUPPORT_SIZE - SHARP_RANDOM_SLOTS - len(support)
        if distractor_slots > 0:
            distractors = sorted(
                (
                    int(index)
                    for index in ALL_INDICES
                    if int(index) not in candidate_set
                    and int(index) not in support
                ),
                key=lambda index: overlap_key(ANSWERS[index], candidate_words),
            )
            for index in distractors[:distractor_slots]:
                add(index)
        required = set(candidate_set)

    rng = support_rng(state_key)
    for index in rng.permutation(len(ANSWERS)):
        if len(support) >= SUPPORT_SIZE:
            break
        add(index)

    assert len(support) == SUPPORT_SIZE
    assert len(set(support)) == SUPPORT_SIZE
    assert required <= set(support), "required teacher action missing from support"
    return {
        "state_key": state_key,
        "regime": regime,
        "candidate_count": len(candidates),
        "candidate_set": candidate_set,
        "support": support,
        "entropies": entropies,
        "open_teacher": open_teacher,
        "candidate_teacher": candidate_teacher,
        "previous_guesses": [turn.guess for turn in history],
    }
""")

md("""
## 19.9 The value target and its matched one-hot control

Both arms score the same 12 actions with the same student and use the same loss.
They differ only in `target_probs`.

```text
broad utility  u(a) = entropy of a's feedback partition over current candidates
sharp utility  u(a) = 1 if a is a current candidate else 0

value target   softmax(u / 0.5)              broad
               uniform over candidates       sharp
hard target    one-hot at argmax u; prefer a tied candidate, then lexical
```

Because the hard action is an argmax of the same utility, the two arms agree
about which utility level is best. Entropy often ties several actions,
especially with 3-10 candidates. The hard teacher prefers a current candidate
inside that tie because it preserves the same information gain while adding a
chance to solve immediately; only then does it break ties lexicographically.
The audit reports tie multiplicity so a hard-versus-value difference is not
mistaken for evidence about a unique teacher action.

One consequence is worth stating before results: at a singleton state the value
target is already one-hot on the sole candidate, so `hard` and `value` are
identical there and contribute identical gradients. The arms differ on 800 of
1,029 states, and the audit below reports the exact divergence rather than
assuming it.
""")

code("""
def build_target_record(state_key: str, row: dict) -> dict:
    support_record = build_action_support(state_key)
    support = support_record["support"]
    candidate_set = support_record["candidate_set"]
    if support_record["regime"] == "broad":
        utilities = np.array(
            [float(support_record["entropies"][index]) for index in support]
        )
        scaled = utilities / TEACHER_TEMPERATURE
        weights = np.exp(scaled - scaled.max())
        value_probs = weights / weights.sum()
    else:
        utilities = np.array([
            1.0 if index in candidate_set else 0.0 for index in support
        ])
        value_probs = utilities / utilities.sum()

    best_utility = round(float(utilities.max()), UTILITY_DECIMALS)
    tied_positions = [
        position
        for position in range(SUPPORT_SIZE)
        if round(float(utilities[position]), UTILITY_DECIMALS) == best_utility
    ]
    hard_position = min(
        tied_positions,
        key=lambda position: (
            support[position] not in candidate_set,
            ANSWERS[support[position]],
        ),
    )
    assert utilities[hard_position] >= best_utility - 10.0**-UTILITY_DECIMALS
    hard_probs = np.zeros(SUPPORT_SIZE)
    hard_probs[hard_position] = 1.0
    positive = value_probs[value_probs > 0]
    target_entropy = float(-(positive * np.log2(positive)).sum())
    return {
        "split": row["split"],
        "state_key": state_key,
        "turn": int(row["turn"]),
        "source": row["source"],
        "candidate_count": int(support_record["candidate_count"]),
        "regime": support_record["regime"],
        "prompt": row["prompt"],
        "actions": [ANSWERS[index] for index in support],
        "action_indices": [int(index) for index in support],
        "utilities": [float(value) for value in utilities],
        "value_target": [float(value) for value in value_probs],
        "hard_index": int(hard_position),
        "hard_action": ANSWERS[support[hard_position]],
        "utility_tie_count": len(tied_positions),
        "hard_is_candidate": support[hard_position] in candidate_set,
        "candidate_positions": [
            position
            for position, index in enumerate(support)
            if index in candidate_set
        ],
        "repeat_positions": [
            position
            for position, index in enumerate(support)
            if ANSWERS[index] in set(support_record["previous_guesses"])
        ],
        "open_teacher": (
            ANSWERS[support_record["open_teacher"]]
            if support_record["open_teacher"] is not None
            else None
        ),
        "candidate_teacher": ANSWERS[support_record["candidate_teacher"]],
        "target_entropy_bits": target_entropy,
        "effective_support": float(2.0**target_entropy),
        "max_target_probability": float(value_probs.max()),
        "arm_total_variation": float(
            0.5 * np.abs(hard_probs - value_probs).sum()
        ),
    }


started = time.perf_counter()
train_targets = [
    build_target_record(row.state_key, row._asdict())
    for row in selection.itertuples()
]
dev_targets = [
    build_target_record(row.state_key, row._asdict())
    for row in dev_states.sort_values("state_key", kind="stable").itertuples()
]
print(
    f"built {len(train_targets)} train and {len(dev_targets)} dev targets "
    f"in {time.perf_counter() - started:.1f}s"
)
""")

md("""
## 19.10 Persist the target data

The generated targets are an artifact, not a side effect of a run. They are
written under `data/generated` with the source hashes that produced them and a
fingerprint of the ordered training stream. If a manifest already exists it must
match exactly, which is what makes a resumed or repeated run provably the same
experiment.
""")

code("""
def jsonl_payload(records: list[dict]) -> str:
    return "".join(
        json.dumps(record, sort_keys=True) + "\\n" for record in records
    )


train_payload = jsonl_payload(train_targets)
dev_payload = jsonl_payload(dev_targets)
stream_fingerprint = sha256_text("".join(
    record["state_key"] + "|" + ",".join(record["actions"]) + "\\n"
    for record in train_targets
))
target_manifest = {
    "experiment": "Lab 19 value-aware distillation targets",
    "representation": "derived_state_v1",
    "source_sha256": structured_hashes,
    "counts": {
        "train_states": len(train_targets),
        "dev_states": len(dev_targets),
        "broad": sum(1 for r in train_targets if r["regime"] == "broad"),
        "sharp": sum(1 for r in train_targets if r["regime"] == "sharp"),
        "singleton": sum(
            1 for r in train_targets if r["candidate_count"] == 1
        ),
        "two_candidate": sum(
            1 for r in train_targets if r["candidate_count"] == 2
        ),
    },
    "config": {
        "support_size": SUPPORT_SIZE,
        "broad_open_slots": BROAD_OPEN_SLOTS,
        "broad_candidate_slots": BROAD_CANDIDATE_SLOTS,
        "sharp_random_slots": SHARP_RANDOM_SLOTS,
        "temperature": TEACHER_TEMPERATURE,
        "broad_threshold": BROAD_THRESHOLD,
        "priority_candidate_count": PRIORITY_CANDIDATE_COUNT,
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "action_space": "2,315 answer words",
    },
    "train_sha256": sha256_text(train_payload),
    "dev_sha256": sha256_text(dev_payload),
    "training_stream_sha256": stream_fingerprint,
}

if TARGET_MANIFEST.exists():
    existing = json.loads(TARGET_MANIFEST.read_text())
    assert existing == target_manifest, (
        "existing Lab 19 target manifest disagrees with regenerated targets"
    )
atomic_write(train_payload, TARGET_FILES["train"])
atomic_write(dev_payload, TARGET_FILES["dev"])
atomic_json(target_manifest, TARGET_MANIFEST)
assert sha256_file(TARGET_FILES["train"]) == target_manifest["train_sha256"]
assert sha256_file(TARGET_FILES["dev"]) == target_manifest["dev_sha256"]
print(json.dumps(target_manifest["counts"], indent=2))
print("training stream fingerprint:", stream_fingerprint[:16])
""")

md("""
## 19.11 Pre-training target audit

No model has been loaded yet. Everything below is a property of the targets, and
every one of these checks can fail in a way that would silently invalidate the
experiment: a duplicated action, a target that does not sum to one, a hard label
that is not the argmax of its own utility, a value target so peaked that the two
arms are the same experiment, or a held-out state that leaked into training.
""")

code("""
target_frame = pd.DataFrame([
    {
        key: value
        for key, value in record.items()
        if key not in {"prompt", "actions", "action_indices", "utilities",
                       "value_target", "candidate_positions", "repeat_positions"}
    }
    | {
        "actions_in_support": len(set(record["actions"])),
        "candidates_in_support": len(record["candidate_positions"]),
        "repeats_in_support": len(record["repeat_positions"]),
        "target_sum": float(np.sum(record["value_target"])),
        "effective_actions_1pct": int(
            np.sum(np.array(record["value_target"]) >= 0.01)
        ),
    }
    for record in train_targets + dev_targets
])

for record in train_targets + dev_targets:
    probabilities = np.array(record["value_target"])
    utilities = np.array(record["utilities"])
    assert len(record["actions"]) == SUPPORT_SIZE
    assert len(set(record["actions"])) == SUPPORT_SIZE
    assert all(word in ANSWER_SET for word in record["actions"])
    assert abs(probabilities.sum() - 1.0) < 1e-9
    assert (probabilities >= 0.0).all()
    assert record["actions"][record["hard_index"]] == record["hard_action"]
    best = round(float(utilities.max()), UTILITY_DECIMALS)
    tied_positions = [
        position
        for position in range(SUPPORT_SIZE)
        if round(float(utilities[position]), UTILITY_DECIMALS) == best
    ]
    expected_hard_position = min(
        tied_positions,
        key=lambda position: (
            position not in set(record["candidate_positions"]),
            record["actions"][position],
        ),
    )
    assert record["hard_index"] == expected_hard_position
    assert record["utility_tie_count"] == len(tied_positions)
    assert record["hard_is_candidate"] == (
        record["hard_index"] in set(record["candidate_positions"])
    )
    if record["regime"] == "broad":
        assert record["open_teacher"] in record["actions"]
        assert record["candidate_teacher"] in record["actions"]
        assert round(
            float(utilities[record["hard_index"]]), UTILITY_DECIMALS
        ) == best
        assert (probabilities > 0).all()
    else:
        assert record["open_teacher"] is None
        assert len(record["candidate_positions"]) == record["candidate_count"]
        assert np.allclose(
            probabilities[record["candidate_positions"]],
            1.0 / record["candidate_count"],
        )
        assert np.isclose(
            probabilities[record["candidate_positions"]].sum(), 1.0
        )
        assert record["hard_action"] == min(
            record["actions"][position]
            for position in record["candidate_positions"]
        )
    assert not set(record["candidate_positions"]) & set(
        record["repeat_positions"]
    )

assert target_frame["target_sum"].sub(1.0).abs().max() < 1e-9
assert (target_frame["actions_in_support"] == SUPPORT_SIZE).all()
print("structural target checks passed for", len(target_frame), "states")

regime_audit = target_frame.groupby(["split", "regime"], sort=True).agg(
    states=("state_key", "size"),
    mean_candidates=("candidate_count", "mean"),
    median_candidates=("candidate_count", "median"),
    max_candidates=("candidate_count", "max"),
    turn2_states=("turn", lambda values: int((values == 2).sum())),
    mean_turn=("turn", "mean"),
    mean_candidates_in_support=("candidates_in_support", "mean"),
    mean_repeats_in_support=("repeats_in_support", "mean"),
).reset_index()
display(regime_audit)
""")

code("""
BUCKETS = [0, 1, 2, 10, 10**6]
BUCKET_LABELS = ["1", "2", "3-10", "11+"]


def bucket_of(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(
        frame["candidate_count"], BUCKETS, labels=BUCKET_LABELS
    )


available = train_states.assign(bucket=bucket_of(train_states))
chosen = selection.assign(bucket=bucket_of(selection))
coverage_audit = pd.DataFrame({
    "available": available.groupby("bucket", observed=True).size(),
    "selected": chosen.groupby("bucket", observed=True).size(),
}).fillna(0).astype(int)
coverage_audit["selected_share_of_available"] = (
    coverage_audit["selected"] / coverage_audit["available"]
)
coverage_audit["selected_share_of_training"] = (
    coverage_audit["selected"] / len(selection)
)
display(coverage_audit.reset_index())

turn_audit = pd.DataFrame({
    "available": available.groupby("turn").size(),
    "selected": chosen.groupby("turn").size(),
}).fillna(0).astype(int)
turn_audit["selected_share_of_available"] = (
    turn_audit["selected"] / turn_audit["available"]
)
display(turn_audit.reset_index())
print(
    "all available broad Turn 2 states covered:",
    bool(
        set(
            available.query(
                "turn == 2 and candidate_count >= @BROAD_THRESHOLD"
            )["state_key"]
        )
        <= set(selection["state_key"])
    ),
)
print(
    "all available 11+ candidate states covered:",
    bool(
        set(
            available.query("candidate_count >= @PRIORITY_CANDIDATE_COUNT")[
                "state_key"
            ]
        )
        <= set(selection["state_key"])
    ),
)
""")

code("""
shape_audit = target_frame.groupby(["split", "regime"], sort=True).agg(
    states=("state_key", "size"),
    mean_max_probability=("max_target_probability", "mean"),
    median_max_probability=("max_target_probability", "median"),
    mean_target_entropy_bits=("target_entropy_bits", "mean"),
    mean_effective_support=("effective_support", "mean"),
    mean_actions_above_1pct=("effective_actions_1pct", "mean"),
    mean_hard_value_total_variation=("arm_total_variation", "mean"),
    min_hard_value_total_variation=("arm_total_variation", "min"),
    median_utility_tie_count=("utility_tie_count", "median"),
    states_with_five_plus_ties=(
        "utility_tie_count", lambda values: int((values >= 5).sum())
    ),
    hard_candidate_rate=("hard_is_candidate", "mean"),
    identical_arm_states=(
        "arm_total_variation", lambda values: int((values < 1e-12).sum())
    ),
).reset_index()
display(shape_audit)

train_frame = target_frame.query("split == 'train'")
identical_states = int((train_frame["arm_total_variation"] < 1e-12).sum())
print(
    f"hard and value supply identical targets on {identical_states} of "
    f"{len(train_frame)} training states "
    f"({identical_states / len(train_frame):.1%}); every one is a singleton: "
    f"{bool((train_frame.loc[train_frame['arm_total_variation'] < 1e-12, 'candidate_count'] == 1).all())}"
)
print(
    "mean total-variation distance between arms on the remaining states:",
    round(
        float(
            train_frame.loc[
                train_frame["arm_total_variation"] >= 1e-12,
                "arm_total_variation",
            ].mean()
        ),
        4,
    ),
)
""")

code("""
def preview(record: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "action": record["actions"],
        "utility": np.round(record["utilities"], 4),
        "value_target": np.round(record["value_target"], 4),
        "hard_target": [
            1.0 if position == record["hard_index"] else 0.0
            for position in range(SUPPORT_SIZE)
        ],
        "is_candidate": [
            position in set(record["candidate_positions"])
            for position in range(SUPPORT_SIZE)
        ],
        "is_previous_guess": [
            position in set(record["repeat_positions"])
            for position in range(SUPPORT_SIZE)
        ],
    }).sort_values("value_target", ascending=False)


by_key = {record["state_key"]: record for record in train_targets}
examples = [
    ("broadest Turn 2 state", max(
        (r for r in train_targets if r["turn"] == 2),
        key=lambda r: r["candidate_count"],
    )),
    ("mid-size broad state", min(
        (r for r in train_targets if 3 <= r["candidate_count"] <= 6),
        key=lambda r: (abs(r["candidate_count"] - 5), r["state_key"]),
    )),
    ("two-candidate state", min(
        (r for r in train_targets if r["candidate_count"] == 2),
        key=lambda r: r["state_key"],
    )),
    ("singleton state", min(
        (r for r in train_targets if r["candidate_count"] == 1),
        key=lambda r: r["state_key"],
    )),
]
for label, record in examples:
    print(
        f"--- {label}: turn {record['turn']}, "
        f"{record['candidate_count']} candidates, {record['regime']} ---"
    )
    print(record["state_key"].replace(chr(10), "  |  "))
    display(preview(record))
""")

code("""
selected_keys_set = set(selection["state_key"])
dev_keys_set = set(dev_states["state_key"])
assert not selected_keys_set & dev_keys_set
assert not selected_keys_set & split_state_keys["test"]
assert selected_keys_set <= split_state_keys["train"]
assert dev_keys_set == split_state_keys["validation"]
assert not any(reachable_by_reserved_answer(key) for key in selected_keys_set)
assert not any(reachable_by_reserved_answer(key) for key in dev_keys_set)

hidden_answers = {
    row["state_key"]: row["answer"]
    for split in ["train", "validation"]
    for row in next_guess_rows[split]
}
answer_visibility_rows = []
for record in train_targets + dev_targets:
    answer = hidden_answers[record["state_key"]]
    in_support = answer in set(record["actions"])
    answer_visibility_rows.append({
        "split": record["split"],
        "regime": record["regime"],
        "answer_in_support": in_support,
        # The hidden answer may only reach a support by being a state-derived
        # candidate. Any other route would mean the answer field leaked in.
        "only_via_candidacy": (
            not in_support
            or record["actions"].index(answer)
            in set(record["candidate_positions"])
        ),
    })
answer_visibility = pd.DataFrame(answer_visibility_rows).groupby(
    ["split", "regime"]
).agg(
    states=("answer_in_support", "size"),
    answer_in_support_rate=("answer_in_support", "mean"),
    only_via_candidacy=("only_via_candidacy", "all"),
).reset_index()
display(answer_visibility)
assert answer_visibility["only_via_candidacy"].all()
print(
    "The hidden answer is not read when building actions or targets. The "
    "table is a visibility diagnostic; the held-out guarantee comes from the "
    "reserved-answer reachability assertions above."
)
print(
    "held-out isolation verified: train, dev, test, and every state reachable "
    "by a reserved RAISE-opened answer are disjoint"
)
""")

md("""
## 19.12 The student action-score kernel

The student score for an action is the same quantity Lab 18b verified and Lab
18d deployed: summed `log P(word tokens + EOS | structured prompt)`. Training
needs that score for 12 actions at once, with gradients, without ever
materializing full-vocabulary logits across prompt positions.

The 12 rows share a prompt and differ only in their action suffix, so one
right-padded batch of shape `(12, prompt_len + width - 1)` covers the state.
`logits_to_keep` is passed the exact response-predicting positions, so the logit
tensor is `(12, width, vocab)` with `width` at most 4 rather than
`(12, prompt_len, vocab)`. Token log-probabilities come from a target gather
minus `logsumexp`, never a full `log_softmax` over the vocabulary, and
`use_cache=False` keeps no keys or values alive across the backward pass.

One state and its 12 actions form one optimizer update.
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


ACTION_TOKENS = [
    tokenizer.encode(word, add_special_tokens=False) + [tokenizer.eos_token_id]
    for word in ANSWERS
]
ACTION_WIDTH = max(len(tokens) for tokens in ACTION_TOKENS)
probe_prompt = render_prompt(train_targets[0]["prompt"])
probe_ids = tokenizer(probe_prompt, add_special_tokens=False)["input_ids"]
for word in ANSWERS[:200] + ANSWERS[-200:]:
    joint = tokenizer(
        probe_prompt + word + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    assert joint[:len(probe_ids)] == probe_ids
    assert joint[len(probe_ids):] == ACTION_TOKENS[WORD_TO_INDEX[word]]
print("action token widths:", sorted({len(t) for t in ACTION_TOKENS}))


def encode_state_actions(prompt_text: str, action_indices: list[int]) -> dict:
    prompt_ids = tokenizer(
        render_prompt(prompt_text), add_special_tokens=False
    )["input_ids"]
    rows = [ACTION_TOKENS[index] for index in action_indices]
    width = max(len(tokens) for tokens in rows)
    input_ids, attention, targets, mask = [], [], [], []
    for tokens in rows:
        body = prompt_ids + tokens[:-1]
        padding = width - len(tokens)
        input_ids.append(body + [PAD_ID] * padding)
        attention.append([1] * len(body) + [0] * padding)
        targets.append(tokens + [PAD_ID] * padding)
        mask.append([1.0] * len(tokens) + [0.0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.float32),
        "positions": torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + width),
        "prompt_length": len(prompt_ids),
    }


def to_device(encoded: dict) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in encoded.items()
    }


def action_scores(model, encoded: dict) -> torch.Tensor:
    logits = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        logits_to_keep=encoded["positions"],
        use_cache=False,
    ).logits.float()
    gathered = logits.gather(
        2, encoded["targets"].unsqueeze(-1)
    ).squeeze(-1)
    token_logprobs = gathered - logits.logsumexp(-1)
    return (token_logprobs * encoded["mask"]).sum(dim=1)


def distillation_loss(
    scores: torch.Tensor, target_probs: torch.Tensor
) -> torch.Tensor:
    return -(target_probs * torch.log_softmax(scores, dim=-1)).sum()


@torch.no_grad()
def reference_action_score(model, prompt_text: str, word: str) -> float:
    prompt_ids = tokenizer(
        render_prompt(prompt_text), add_special_tokens=False
    )["input_ids"]
    tokens = ACTION_TOKENS[WORD_TO_INDEX[word]]
    input_ids = torch.tensor(
        [prompt_ids + tokens[:-1]], dtype=torch.long, device=device
    )
    logits = model(input_ids=input_ids, use_cache=False).logits[0].float()
    logprobs = torch.log_softmax(logits[len(prompt_ids) - 1:], dim=-1)
    total = float(sum(
        float(logprobs[step, token]) for step, token in enumerate(tokens)
    ))
    del logits, logprobs, input_ids
    return total


started = time.perf_counter()
train_encodings = [
    encode_state_actions(record["prompt"], record["action_indices"])
    for record in train_targets
]
dev_encodings = [
    encode_state_actions(record["prompt"], record["action_indices"])
    for record in dev_targets
]
train_value_targets = [
    torch.tensor(record["value_target"], dtype=torch.float32)
    for record in train_targets
]
train_hard_targets = []
for record in train_targets:
    one_hot = torch.zeros(SUPPORT_SIZE, dtype=torch.float32)
    one_hot[record["hard_index"]] = 1.0
    train_hard_targets.append(one_hot)
ARM_TARGETS = {"hard": train_hard_targets, "value": train_value_targets}
prompt_token_counts = [
    encoded["prompt_length"] for encoded in train_encodings
]
print(
    f"encoded {len(train_encodings)} train and {len(dev_encodings)} dev states "
    f"in {time.perf_counter() - started:.1f}s"
)
print(
    "prompt tokens: min", min(prompt_token_counts),
    "median", int(np.median(prompt_token_counts)),
    "max", max(prompt_token_counts),
)
print(
    "kept logit positions per update:",
    sorted({int(len(e["positions"])) for e in train_encodings}),
)
""")

code("""
def reset_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_trainable_incumbent(seed: int):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, INCUMBENTS[seed], is_trainable=True
    ).to(device)
    model.train()
    trainable, total = trainable_parameter_count(model)
    assert trainable > 0, "incumbent adapter loaded without trainable parameters"
    return model, trainable, total


def load_eval_adapter(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing adapter {path}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)
    return PeftModel.from_pretrained(base, path).to(device).eval()


def release_model(model) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
""")

md("""
## 19.13 Numerical regression of the batched scorer

The batched kernel pads, batches, and uses a gather-minus-`logsumexp`
formulation with a position index. The reference below runs one action at a
time, unpadded and unbatched, and takes a full `log_softmax`. If those two paths
disagree, every score, target cross-entropy, and regret in this lab is wrong.

This runs before training, on the seed-42 incumbent, over the widest, the
median, and the narrowest training state, covering supports whose actions
tokenize to different widths.
""")

code("""
if RUN_TRAINING or RUN_EVALUATION:
    regression_model = load_eval_adapter(INCUMBENTS[42])
    widest = int(np.argmax(prompt_token_counts))
    narrowest = int(np.argmin(prompt_token_counts))
    middle = int(np.argsort(prompt_token_counts)[len(prompt_token_counts) // 2])
    regression_rows = []
    for position in [widest, middle, narrowest]:
        record = train_targets[position]
        with torch.no_grad():
            batched = action_scores(
                regression_model, to_device(train_encodings[position])
            ).cpu().numpy()
        for slot, word in enumerate(record["actions"]):
            regression_rows.append({
                "state_position": position,
                "prompt_tokens": prompt_token_counts[position],
                "action": word,
                "batched_score": float(batched[slot]),
                "reference_score": reference_action_score(
                    regression_model, record["prompt"], word
                ),
            })
    regression = pd.DataFrame(regression_rows)
    regression["abs_diff"] = (
        regression["batched_score"] - regression["reference_score"]
    ).abs()
    max_score_difference = float(regression["abs_diff"].max())
    print(
        f"batched vs plain single-action scores on {len(regression)} actions: "
        f"max abs diff {max_score_difference:.3e}"
    )
    display(regression.head(12))
    assert max_score_difference < 1e-3
    release_model(regression_model)
    del regression_model
    print("scoring kernel verified")
""")

md("""
## 19.14 Fixed-shape training soak

The longest training state is repeated for 40 optimizer steps on a disposable
model. Driver memory is sampled while the forward activations are still live and
again after the backward pass, so the trace measures the real peak rather than
the quiescent value between steps.

The soak model is thrown away and every real run rebuilds its model after
resetting its own seed, so the soak cannot perturb initialization, dropout, or
state order.
""")

code("""
def training_step(model, optimizer, encoded, target_probs):
    optimizer.zero_grad(set_to_none=True)
    scores = action_scores(model, encoded)
    loss = distillation_loss(scores, target_probs)
    peak = driver_memory_gib()
    loss.backward()
    peak = max(peak, driver_memory_gib())
    torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), GRAD_CLIP
    )
    optimizer.step()
    peak = max(peak, driver_memory_gib())
    cpu_scores = scores.detach().cpu()
    loss_value = float(loss.detach().cpu())
    del scores, loss
    if device.type == "mps":
        torch.mps.empty_cache()
    return cpu_scores, loss_value, peak


missing_arms = [
    key for key, path in ARM_CHECKPOINTS.items() if not path.exists()
]
if RUN_TRAINING and missing_arms:
    soak_position = int(np.argmax(prompt_token_counts))
    soak_encoded = to_device(train_encodings[soak_position])
    soak_target = train_value_targets[soak_position].to(device)
    reset_seeds(SEEDS[0])
    soak_model, trainable, total = load_trainable_incumbent(SEEDS[0])
    print(
        f"soak model: {trainable:,} trainable parameters "
        f"({trainable / total:.3%})"
    )
    soak_optimizer = AdamW(
        (p for p in soak_model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    soak_peaks = []
    for _ in range(40):
        _, _, peak = training_step(
            soak_model, soak_optimizer, soak_encoded, soak_target
        )
        soak_peaks.append(peak)
    third = len(soak_peaks) // 3
    creep = np.mean(soak_peaks[-third:]) - np.mean(soak_peaks[third:2 * third])
    late_range = np.ptp(soak_peaks[-third:])
    print(
        f"training soak peak {max(soak_peaks):.2f} GiB, "
        f"creep {creep:+.2f} GiB, final range {late_range:.2f} GiB"
    )
    pd.DataFrame({
        "step": range(1, len(soak_peaks) + 1),
        "driver_peak_gib": soak_peaks,
    }).to_csv(RESULTS_DIR / "training-soak-trace.csv", index=False)
    assert creep < 0.5
    assert late_range < 0.5
    assert max(soak_peaks) < MEMORY_ABORT_GIB
    del soak_optimizer, soak_encoded, soak_target
    release_model(soak_model)
    del soak_model
    print("training memory plateaued")
else:
    print("training soak skipped: training disabled or all six arms exist")
""")

md("""
## 19.15 Train the six arms

Each arm continues from its frozen incumbent through
`PeftModel.from_pretrained(base, incumbent, is_trainable=True)`, so the LoRA
weights start exactly where Lab 17 or Lab 18c left them. A completed checkpoint
is validated and reused; an `-in-progress` directory stops the notebook for
inspection rather than resuming from an unknown optimizer state.

Both cross-entropies are recorded at every step for both arms. The `hard` arm
optimizes `hard_ce` and merely observes `value_ce`; the `value` arm does the
reverse. That makes the two training curves directly comparable instead of two
different quantities plotted on one axis.
""")

code("""
WARMUP_STEPS = max(1, int(UPDATES * WARMUP_FRACTION))


def lr_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, UPDATES - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def target_entropy_nats(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * positive.log()).sum())


def validate_arm_manifest(manifest: dict, seed: int, arm: str) -> None:
    assert manifest["seed"] == seed
    assert manifest["arm"] == arm
    assert manifest["base_model"] == MODEL_ID
    assert manifest["updates"] == UPDATES
    assert manifest["learning_rate"] == LEARNING_RATE
    assert manifest["weight_decay"] == WEIGHT_DECAY
    assert manifest["warmup_fraction"] == WARMUP_FRACTION
    assert manifest["temperature"] == TEACHER_TEMPERATURE
    assert manifest["support_size"] == SUPPORT_SIZE
    assert manifest["incumbent_sha256"] == incumbent_hashes[seed]
    assert manifest["training_stream_sha256"] == stream_fingerprint
    assert manifest["target_train_sha256"] == target_manifest["train_sha256"]


def train_arm(seed: int, arm: str) -> dict:
    checkpoint = ARM_CHECKPOINTS[(seed, arm)]
    in_progress = checkpoint.with_name(checkpoint.name + "-in-progress")
    if checkpoint.exists():
        manifest_path = checkpoint / "lab19-run.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"checkpoint exists without a Lab 19 manifest: {checkpoint}"
            )
        manifest = json.loads(manifest_path.read_text())
        validate_arm_manifest(manifest, seed, arm)
        print(f"seed {seed} {arm}: verified existing checkpoint")
        return manifest
    if in_progress.exists():
        raise FileExistsError(
            f"incomplete seed {seed} {arm} checkpoint needs inspection: "
            f"{in_progress}"
        )
    if not RUN_TRAINING:
        raise FileNotFoundError(
            f"seed {seed} {arm} checkpoint missing and RUN_TRAINING=False"
        )

    targets = ARM_TARGETS[arm]
    reset_seeds(seed)
    model, trainable, total = load_trainable_incumbent(seed)
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_multiplier
    )
    records = []
    first_step_scores = None
    peak_memory = 0.0
    started = time.perf_counter()

    for step in range(1, UPDATES + 1):
        position = step - 1
        record = train_targets[position]
        encoded = to_device(train_encodings[position])
        target = targets[position].to(device)
        scores, loss_value, step_peak = training_step(
            model, optimizer, encoded, target
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        peak_memory = max(peak_memory, step_peak)

        cpu_scores = scores.cpu()
        log_q = torch.log_softmax(cpu_scores, dim=-1)
        value_probs = train_value_targets[position]
        value_ce = float(-(value_probs * log_q).sum())
        hard_ce = float(-log_q[record["hard_index"]])
        if step == 1:
            first_step_scores = [round(float(value), 6) for value in cpu_scores]
        records.append({
            "step": step,
            "state_key": record["state_key"],
            "regime": record["regime"],
            "candidate_count": record["candidate_count"],
            "turn": record["turn"],
            "lr": learning_rate,
            "loss": loss_value,
            "value_ce_nats": value_ce,
            "hard_ce_nats": hard_ce,
            "kl_to_value_nats": value_ce - target_entropy_nats(value_probs),
            "hard_top1": bool(int(cpu_scores.argmax()) == record["hard_index"]),
            "driver_peak_gib": step_peak,
        })
        assert peak_memory < MEMORY_ABORT_GIB, (
            f"seed {seed} {arm} exceeded the memory threshold at step {step}: "
            f"{peak_memory:.1f} GiB"
        )
        if step == 1 or step % LOG_EVERY == 0:
            window = records[-LOG_EVERY:]
            print(
                f"seed {seed} {arm} step {step:4d}/{UPDATES} "
                f"loss={np.mean([r['loss'] for r in window]):.4f} "
                f"value_ce={np.mean([r['value_ce_nats'] for r in window]):.4f} "
                f"hard_ce={np.mean([r['hard_ce_nats'] for r in window]):.4f} "
                f"lr={learning_rate:.2e} peak={peak_memory:.2f} GiB",
                flush=True,
            )

    model.save_pretrained(in_progress)
    tokenizer.save_pretrained(in_progress)
    history = pd.DataFrame(records)
    history.to_csv(in_progress / "training-history.csv", index=False)
    tail = history.tail(100)
    manifest = {
        "experiment": "Lab 19 value-aware distillation",
        "representation": "derived_state_v1",
        "base_model": MODEL_ID,
        "arm": arm,
        "seed": seed,
        "incumbent": INCUMBENTS[seed].name,
        "incumbent_sha256": incumbent_hashes[seed],
        "updates": UPDATES,
        "states_per_update": 1,
        "actions_per_state": SUPPORT_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "warmup_steps": WARMUP_STEPS,
        "temperature": TEACHER_TEMPERATURE,
        "support_size": SUPPORT_SIZE,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_stream_sha256": stream_fingerprint,
        "target_train_sha256": target_manifest["train_sha256"],
        "source_sha256": structured_hashes,
        "first_step_scores": first_step_scores,
        "mean_loss": float(history["loss"].mean()),
        "final_100_loss": float(tail["loss"].mean()),
        "final_100_value_ce_nats": float(tail["value_ce_nats"].mean()),
        "final_100_hard_ce_nats": float(tail["hard_ce_nats"].mean()),
        "final_100_hard_top1": float(tail["hard_top1"].mean()),
        "peak_driver_gib": peak_memory,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (in_progress / "lab19-run.json").write_text(
        json.dumps(manifest, indent=2)
    )
    in_progress.rename(checkpoint)
    del optimizer, scheduler
    release_model(model)
    del model
    print(f"seed {seed} {arm}: complete in {manifest['elapsed_seconds'] / 60:.1f} min")
    return manifest


arm_manifests = {
    (seed, arm): train_arm(seed, arm)
    for seed in SEEDS
    for arm in ARMS
}
""")

code("""
for seed in SEEDS:
    hard_scores = np.array(arm_manifests[(seed, "hard")]["first_step_scores"])
    value_scores = np.array(arm_manifests[(seed, "value")]["first_step_scores"])
    assert np.allclose(hard_scores, value_scores, atol=1e-5), (
        f"seed {seed} arms did not start from an identical forward pass"
    )
print(
    "matched control verified: within each seed both arms produced identical "
    "step-1 action scores, so they differ only in target probabilities"
)

training_history = pd.concat([
    pd.read_csv(
        ARM_CHECKPOINTS[(seed, arm)] / "training-history.csv"
    ).assign(seed=seed, arm=arm)
    for seed in SEEDS
    for arm in ARMS
], ignore_index=True)
atomic_csv(training_history, RESULTS_DIR / "training-history.csv")

for seed in SEEDS:
    for column in ["state_key", "candidate_count", "regime"]:
        hard_column = training_history.query(
            "seed == @seed and arm == 'hard'"
        )[column].tolist()
        value_column = training_history.query(
            "seed == @seed and arm == 'value'"
        )[column].tolist()
        assert hard_column == value_column, (
            f"seed {seed} arms disagree on the {column} stream"
        )

training_summary = pd.DataFrame([
    {
        "seed": seed,
        "arm": arm,
        "updates": manifest["updates"],
        "mean_loss": manifest["mean_loss"],
        "final_100_loss": manifest["final_100_loss"],
        "final_100_value_ce_nats": manifest["final_100_value_ce_nats"],
        "final_100_hard_ce_nats": manifest["final_100_hard_ce_nats"],
        "final_100_hard_top1": manifest["final_100_hard_top1"],
        "peak_driver_gib": manifest["peak_driver_gib"],
        "elapsed_minutes": manifest["elapsed_seconds"] / 60,
    }
    for (seed, arm), manifest in arm_manifests.items()
])
display(training_summary)

curve = training_history.assign(
    block=(training_history["step"] - 1) // 100
).groupby(["seed", "arm", "block"], sort=True).agg(
    value_ce_nats=("value_ce_nats", "mean"),
    hard_ce_nats=("hard_ce_nats", "mean"),
    hard_top1=("hard_top1", "mean"),
).reset_index()
display(
    curve.pivot_table(
        index="block", columns=["arm"], values=["value_ce_nats", "hard_ce_nats"]
    ).round(3)
)
""")

md("""
## 19.16 Held-out dev shortlist evaluation

All 466 unique dev `NEXT_GUESS` states are scored on their own 12-action
support by all nine models: three incumbents and six trained arms. These states
were never trained on by any arm, and their supports were built by the same
deterministic rule as the training supports.

This is a shortlist metric, not a full-lexicon metric. It answers "given twelve
named actions, does the model prefer the valuable one", which is exactly what
the objective optimized. Section 19.20 asks the harder deployment question over
all 2,315 answers.
""")

code("""
EVAL_MODELS = (
    [("incumbent", seed, INCUMBENTS[seed]) for seed in SEEDS]
    + [
        (arm, seed, ARM_CHECKPOINTS[(seed, arm)])
        for seed in SEEDS
        for arm in ARMS
    ]
)


def model_label(arm: str, seed: int) -> str:
    return f"{arm}-seed{seed}"


@torch.no_grad()
def score_dev_states(model) -> np.ndarray:
    matrix = np.zeros((len(dev_targets), SUPPORT_SIZE), dtype=np.float32)
    peak = 0.0
    for position, encoded in enumerate(dev_encodings):
        matrix[position] = (
            action_scores(model, to_device(encoded)).cpu().numpy()
        )
        peak = max(peak, driver_memory_gib())
        if device.type == "mps":
            torch.mps.empty_cache()
    assert peak < MEMORY_ABORT_GIB, (
        f"dev scoring exceeded the memory threshold: {peak:.1f} GiB"
    )
    return matrix


def dev_metric_rows(arm: str, seed: int, matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for position, record in enumerate(dev_targets):
        scores = matrix[position].astype(np.float64)
        shifted = scores - scores.max()
        log_q = shifted - np.log(np.exp(shifted).sum())
        q = np.exp(log_q)
        probabilities = np.array(record["value_target"])
        utilities = np.array(record["utilities"])
        positive = probabilities[probabilities > 0]
        entropy_nats = float(-(positive * np.log(positive)).sum())
        value_ce = float(-(probabilities * log_q).sum())
        chosen = int(np.argmax(scores))
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(SUPPORT_SIZE, dtype=np.int64)
        ranks[order] = np.arange(1, SUPPORT_SIZE + 1)
        candidate_positions = set(record["candidate_positions"])
        broad = record["regime"] == "broad"
        rows.append({
            "label": model_label(arm, seed),
            "arm": arm,
            "seed": seed,
            "state_key": record["state_key"],
            "turn": record["turn"],
            "candidate_count": record["candidate_count"],
            "regime": record["regime"],
            "chosen_action": record["actions"][chosen],
            "chosen_position": chosen,
            "hard_action": record["hard_action"],
            "hard_top1": chosen == record["hard_index"],
            "hard_rank": int(ranks[record["hard_index"]]),
            "value_ce_nats": value_ce,
            "target_entropy_nats": entropy_nats,
            "kl_to_value_nats": value_ce - entropy_nats,
            "chosen_value_mass": float(probabilities[chosen]),
            "student_candidate_mass": float(
                q[list(candidate_positions)].sum()
            ),
            "chosen_utility": float(utilities[chosen]),
            "best_utility": float(utilities.max()),
            "open_regret_bits": (
                float(utilities.max() - utilities[chosen]) if broad
                else float("nan")
            ),
            "candidate_selected": (
                float("nan") if broad else float(chosen in candidate_positions)
            ),
            "chosen_is_candidate": chosen in candidate_positions,
            "chosen_is_repeat": chosen in set(record["repeat_positions"]),
        })
    frame = pd.DataFrame(rows)
    frame["bucket"] = pd.cut(
        frame["candidate_count"], BUCKETS, labels=BUCKET_LABELS
    )
    return frame


dev_frames = []
dev_score_matrices = {}
for arm, seed, path in EVAL_MODELS:
    label = model_label(arm, seed)
    rows_path = RESULTS_DIR / f"dev-shortlist-{label}.csv"
    scores_path = RESULTS_DIR / f"dev-scores-{label}.npy"
    progress_path = RESULTS_DIR / f"dev-shortlist-{label}-progress.json"
    checkpoint_sha256 = (
        incumbent_hashes[seed]
        if arm == "incumbent"
        else sha256_file(path / "adapter_model.safetensors")
    )
    progress = {
        "label": label,
        "checkpoint_sha256": checkpoint_sha256,
        "target_dev_sha256": target_manifest["dev_sha256"],
    }
    artifact_exists = [
        rows_path.exists(), scores_path.exists(), progress_path.exists()
    ]
    if any(artifact_exists) and not all(artifact_exists):
        raise FileNotFoundError(
            f"incomplete dev artifact set for {label}"
        )
    if rows_path.exists():
        assert json.loads(progress_path.read_text()) == progress, (
            f"cached dev shortlist fingerprint disagrees for {label}"
        )
        matrix = np.load(scores_path)
        assert matrix.shape == (len(dev_targets), SUPPORT_SIZE)
        frame = pd.read_csv(rows_path)
        assert frame["state_key"].tolist() == [
            record["state_key"] for record in dev_targets
        ]
        print(f"{label}: verified cached dev shortlist")
    elif RUN_EVALUATION:
        model = load_eval_adapter(path)
        started = time.perf_counter()
        matrix = score_dev_states(model)
        release_model(model)
        del model
        frame = dev_metric_rows(arm, seed, matrix)
        atomic_npy(matrix, scores_path)
        atomic_csv(frame, rows_path)
        atomic_json(progress, progress_path)
        print(
            f"{label}: scored {len(dev_targets)} dev states in "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )
    else:
        raise FileNotFoundError(
            f"missing dev shortlist for {label} and RUN_EVALUATION=False"
        )
    dev_score_matrices[label] = matrix
    dev_frames.append(frame)

dev_results = pd.concat(dev_frames, ignore_index=True)
dev_results["bucket"] = pd.cut(
    dev_results["candidate_count"], BUCKETS, labels=BUCKET_LABELS
)
print("dev shortlist rows:", len(dev_results))
""")

md("""
## 19.17 Dev shortlist results

Read the arms against their own incumbent, seed by seed. `hard_top1` and
`kl_to_value_nats` measure imitation of the teacher target. `open_regret_bits`
measures how much entropy a broad-state choice throws away against the best
action in the support, which contains the global open optimum by construction.
`candidate_selected` is the sharp-state closure metric: did the model name a
word that can still be the answer?
""")

code("""
def dev_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, observed=True, sort=True)
    summary = grouped.agg(
        states=("state_key", "size"),
        hard_top1=("hard_top1", "mean"),
        median_hard_rank=("hard_rank", "median"),
        value_ce_nats=("value_ce_nats", "mean"),
        kl_to_value_nats=("kl_to_value_nats", "mean"),
        chosen_value_mass=("chosen_value_mass", "mean"),
        student_candidate_mass=("student_candidate_mass", "mean"),
        candidate_choice_rate=("chosen_is_candidate", "mean"),
        repeat_choice_rate=("chosen_is_repeat", "mean"),
    )
    summary["broad_open_regret_bits"] = grouped["open_regret_bits"].mean()
    summary["sharp_candidate_selected"] = grouped["candidate_selected"].mean()
    return summary.reset_index()


dev_by_label = dev_summary(dev_results, ["arm", "seed"])
display(dev_by_label)

dev_by_regime = dev_summary(dev_results, ["arm", "seed", "regime"])
display(dev_by_regime)

dev_by_bucket = dev_summary(dev_results, ["arm", "bucket"])
display(dev_by_bucket)

dev_by_turn = dev_summary(dev_results, ["arm", "turn"])
display(dev_by_turn)
""")

code("""
PAIRED_METRICS = [
    "hard_top1",
    "kl_to_value_nats",
    "open_regret_bits",
    "candidate_selected",
    "student_candidate_mass",
]
PAIRED_DIRECTIONS = {
    "hard_top1": "higher",
    "kl_to_value_nats": "lower",
    "open_regret_bits": "lower",
    "candidate_selected": "higher",
    "student_candidate_mass": "higher",
}


def paired_arm_difference(
    seed: int, left_arm: str, right_arm: str, metric: str
) -> dict:
    left = dev_results.query("seed == @seed and arm == @left_arm")[
        ["state_key", metric]
    ].rename(columns={metric: "left"})
    right = dev_results.query("seed == @seed and arm == @right_arm")[
        ["state_key", metric]
    ].rename(columns={metric: "right"})
    paired = left.merge(
        right, on="state_key", validate="one_to_one"
    ).dropna()
    difference = (
        paired["right"].astype(float) - paired["left"].astype(float)
    )
    improved = (
        difference > 0
        if PAIRED_DIRECTIONS[metric] == "higher"
        else difference < 0
    )
    return {
        "seed": seed,
        "metric": metric,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "states": len(paired),
        "left_mean": float(paired["left"].astype(float).mean()),
        "right_mean": float(paired["right"].astype(float).mean()),
        "right_minus_left": float(difference.mean()),
        "better_when": PAIRED_DIRECTIONS[metric],
        "improved_states": int(improved.sum()),
        "worsened_states": int((~improved & difference.ne(0)).sum()),
    }


dev_paired = pd.DataFrame([
    paired_arm_difference(seed, left_arm, right_arm, metric)
    for seed in SEEDS
    for left_arm, right_arm in [
        ("incumbent", "hard"),
        ("incumbent", "value"),
        ("hard", "value"),
    ]
    for metric in PAIRED_METRICS
])
display(
    dev_paired.pivot_table(
        index=["metric", "left_arm", "right_arm"],
        columns="seed",
        values="right_minus_left",
    ).round(4)
)
print(
    "Paired rows are dev states, not independent training runs. The "
    "replication unit remains the seed: a direction that does not hold for "
    "all three seeds is not replicated."
)
""")

md("""
## 19.18 Gameplay engine copied from Lab 18d

The shortlist metric above is the objective's home turf. The deployment
question is whether an arm plays better full games over the whole 2,315-word
answer list, so the kernels below are copied from Lab 18d rather than rewritten:
same summed-sequence score, same `logits_to_keep=1` prefill, same KV cache with
`CHUNK_SIZE=256`, same `empty_cache` per state, same RAISE opening, same Turns 2
through 6, same free-decoder termination on an invalid word, and the same
strategic diagnostics.

`WORD_TOKENS` is the same tokenization already used for the training actions, so
shortlist scores and gameplay scores are the same quantity measured two ways.
The incumbent arm is **not** replayed here: its rows are loaded from Lab 18d's
persisted results, and Section 19.20 checks that those rows still reproduce the
summaries Lab 18d published.
""")

code("""
WORD_TOKENS = ACTION_TOKENS
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


def candidate_indices(history: list[Turn]) -> np.ndarray:
    words = filter_candidates(ANSWERS, history)
    indices = np.array(
        [WORD_TO_INDEX[word] for word in words], dtype=np.int32
    )
    if len(indices) == 0:
        raise ValueError("game history produced an empty candidate set")
    return indices
""")

code("""
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
    model, seed: int, arm: str, decoder: str, answer: str
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
        prompt = structured_next_guess_prompt(history, len(before))
        history_has_duplicate = (
            len(history) != len({turn.guess for turn in history})
        )
        model_scores = None
        if decoder == "free":
            raw = generate_free(model, prompt)
            guess = parse_guess(raw)
        elif decoder == "answer-constrained":
            model_scores = score_all_words(model, prompt)
            assert LAST_STATE_PEAK_GIB < MEMORY_ABORT_GIB, (
                f"memory regression at seed {seed} {arm} {answer} "
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
            "arm": arm,
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
        "arm": arm,
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
## 19.19 Full-list scoring regression and memory gate

Before any game runs, the seed-42 incumbent must reproduce Lab 18b's first
persisted 2,315-word score vector within float32 tolerance. That pins the
gameplay kernel to a result computed by different code in an earlier lab.

The same model then scores one fixed longest battery prompt 40 times. The shape
never changes, so any growth in driver memory is a leak rather than a larger
workload, and the final third must be flat before 12 game runs start.
""")

code("""
if RUN_EVALUATION:
    checker = load_eval_adapter(INCUMBENTS[42])
    lab18b_battery = pd.read_csv(LAB18B_RESULTS / "battery-states.csv")
    battery_histories = [
        parse_state_key(state_key)
        for state_key in lab18b_battery["state_key"]
    ]
    battery_prompts = [
        structured_next_guess_prompt(
            history, len(candidate_indices(history))
        )
        for history in battery_histories
    ]
    first_scores = score_all_words(checker, battery_prompts[0])
    reference_scores = np.load(
        LAB18B_RESULTS / "scores-B-structured.npy", mmap_mode="r"
    )[0]
    max_abs_diff = float(np.max(np.abs(first_scores - reference_scores)))
    print("Lab 18b score-vector max abs diff:", max_abs_diff)
    assert max_abs_diff < 1e-3

    replay_answer = RESERVED_ANSWERS[0]
    replay_calls, replay_game, _ = play_game(
        checker,
        seed=42,
        arm="incumbent",
        decoder="answer-constrained",
        answer=replay_answer,
    )
    persisted_calls = pd.read_csv(
        LAB18D_RESULTS / "gameplay-calls.csv"
    ).query(
        "seed == 42 and decoder == 'answer-constrained' "
        "and answer == @replay_answer"
    ).sort_values("turn")
    assert [row["guess"] for row in replay_calls] == (
        persisted_calls["guess"].tolist()
    )
    persisted_game = pd.read_csv(
        LAB18D_RESULTS / "gameplay-games.csv"
    ).query(
        "seed == 42 and decoder == 'answer-constrained' "
        "and answer == @replay_answer"
    ).iloc[0]
    assert replay_game["solved"] == bool(persisted_game["solved"])
    assert replay_game["solved_turn"] == persisted_game["solved_turn"]
    print(
        f"seed 42 {replay_answer} constrained trajectory reproduces Lab 18d"
    )

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
    atomic_csv(
        pd.DataFrame({
            "repeat": range(1, len(soak_peaks) + 1),
            "driver_peak_gib": soak_peaks,
        }),
        RESULTS_DIR / "scoring-soak-trace.csv",
    )
    release_model(checker)
    del checker
    print("gameplay kernel verified and memory plateaued")
else:
    print("scoring regression skipped: evaluation disabled")
""")

md("""
## 19.20 Restartable gameplay for the six arms

Each arm plays all 19 reserved answers under both decoders. Artifacts are
rewritten atomically after every completed game, so an interrupted run resumes
at the next unplayed answer instead of replaying finished ones. Constrained
score vectors are checkpointed at the same cadence with their `(answer, turn)`
keys, which makes every ranking in the analysis auditable without rerunning a
model.

The incumbent rows come from Lab 18d's persisted CSVs. Recomputing Lab 18d's
published summary from those rows is the check that the baseline being compared
against is the same baseline Lab 18d reported.
""")

code("""
arm_hashes = {
    f"{arm}-seed{seed}": sha256_file(
        ARM_CHECKPOINTS[(seed, arm)] / "adapter_model.safetensors"
    )
    for seed in SEEDS
    for arm in ARMS
}


def evaluation_paths(seed: int, arm: str, decoder: str) -> dict[str, Path]:
    stem = f"seed{seed}-{arm}-{decoder}"
    return {
        "calls": RESULTS_DIR / f"{stem}-calls.csv",
        "games": RESULTS_DIR / f"{stem}-games.csv",
        "scores": RESULTS_DIR / f"{stem}-scores.npy",
        "score_keys": RESULTS_DIR / f"{stem}-score-keys.csv",
        "progress": RESULTS_DIR / f"{stem}-progress.json",
    }


def evaluate_decoder(
    model, seed: int, arm: str, decoder: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = evaluation_paths(seed, arm, decoder)
    calls_path = paths["calls"]
    games_path = paths["games"]
    progress = {
        "seed": seed,
        "arm": arm,
        "decoder": decoder,
        "checkpoint_sha256": arm_hashes[f"{arm}-seed{seed}"],
        "answers": list(RESERVED_ANSWERS),
    }
    if paths["progress"].exists():
        assert json.loads(paths["progress"].read_text()) == progress, (
            f"stored progress disagrees for seed {seed} {arm} {decoder}"
        )
    else:
        atomic_json(progress, paths["progress"])

    if games_path.exists() and not calls_path.exists():
        raise FileNotFoundError(
            f"games exist without calls for seed {seed} {arm} {decoder}"
        )
    if games_path.exists():
        calls = pd.read_csv(calls_path)
        games = pd.read_csv(games_path)
        assert set(games["answer"]).issubset(RESERVED_SET)
        assert not games["answer"].duplicated().any()
        # Calls are written before the matching game row. If the process dies
        # between those writes, discard that one incomplete game's calls.
        calls = calls.loc[calls["answer"].isin(set(games["answer"]))].copy()
    else:
        calls = pd.DataFrame()
        games = pd.DataFrame()

    score_matrix = np.empty((0, len(ANSWERS)), dtype=np.float32)
    score_keys = pd.DataFrame(columns=["seed", "arm", "answer", "turn"])
    if decoder == "answer-constrained":
        score_exists = paths["scores"].exists()
        keys_exist = paths["score_keys"].exists()
        if score_exists != keys_exist:
            raise FileNotFoundError(
                f"incomplete score artifact pair for seed {seed} {arm}"
            )
        if score_exists:
            score_matrix = np.load(paths["scores"])
            score_keys = pd.read_csv(paths["score_keys"])
            assert len(score_matrix) == len(score_keys)

    completed = set(games["answer"]) if len(games) else set()
    if decoder == "answer-constrained" and len(score_keys):
        keep = score_keys["answer"].isin(completed).to_numpy()
        score_matrix = score_matrix[keep]
        score_keys = score_keys.loc[keep].reset_index(drop=True)

    for answer in RESERVED_ANSWERS:
        if answer in completed:
            continue
        if not RUN_EVALUATION:
            raise FileNotFoundError(
                f"missing game {answer} for seed {seed} {arm} {decoder} "
                "and RUN_EVALUATION=False"
            )
        new_calls, new_game, new_scores = play_game(
            model, seed, arm, decoder, answer
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
                    "arm": row["arm"],
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
            f"seed {seed} {arm} {decoder} {answer}: "
            f"{'SOLVED' if new_game['solved'] else 'FAILED'} "
            f"turn={new_game['solved_turn']} "
            f"calls={new_game['model_calls']}",
            flush=True,
        )

    assert set(games["answer"]) == RESERVED_SET
    assert set(calls["answer"]) == RESERVED_SET
    assert not calls.duplicated(
        ["seed", "arm", "decoder", "answer", "turn"]
    ).any()
    if decoder == "answer-constrained":
        expected_keys = calls[
            ["seed", "arm", "answer", "turn"]
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            score_keys.reset_index(drop=True),
            expected_keys,
            check_dtype=False,
        )
        assert score_matrix.shape == (len(calls), len(ANSWERS))
    return calls, games
""")

code("""
def decoder_complete(seed: int, arm: str, decoder: str) -> bool:
    games_path = evaluation_paths(seed, arm, decoder)["games"]
    if not games_path.exists():
        return False
    return set(pd.read_csv(games_path)["answer"]) == RESERVED_SET


arm_call_frames = []
arm_game_frames = []
for seed in SEEDS:
    for arm in ARMS:
        needed = any(
            not decoder_complete(seed, arm, decoder)
            for decoder in DECODERS
        )
        model = (
            load_eval_adapter(ARM_CHECKPOINTS[(seed, arm)])
            if (needed and RUN_EVALUATION) else None
        )
        for decoder in DECODERS:
            calls, games = evaluate_decoder(model, seed, arm, decoder)
            arm_call_frames.append(calls)
            arm_game_frames.append(games)
        if model is not None:
            release_model(model)
            del model

lab18d_calls = pd.read_csv(LAB18D_RESULTS / "gameplay-calls.csv")
lab18d_games = pd.read_csv(LAB18D_RESULTS / "gameplay-games.csv")
lab18d_summary = pd.read_csv(LAB18D_RESULTS / "game-summary.csv")
assert set(lab18d_games["seed"]) == set(SEEDS)
assert set(RESERVED_ANSWERS) == set(lab18d_games["answer"])

SUMMARY_COLUMNS = [
    "solve_rate",
    "invalid_termination_rate",
    "mean_model_calls",
    "mean_final_candidates",
]
recomputed_18d = lab18d_games.groupby(["seed", "decoder"], sort=True).agg(
    games=("answer", "size"),
    solved=("solved", "sum"),
    solve_rate=("solved", "mean"),
    invalid_termination_rate=("terminated_invalid", "mean"),
    mean_model_calls=("model_calls", "mean"),
    mean_final_candidates=("final_candidate_count", "mean"),
).reset_index()
baseline_check = recomputed_18d.merge(
    lab18d_summary,
    on=["seed", "decoder"],
    suffixes=("_new", "_persisted"),
    validate="one_to_one",
)
for column in SUMMARY_COLUMNS:
    assert np.allclose(
        baseline_check[f"{column}_new"],
        baseline_check[f"{column}_persisted"],
    ), f"Lab 18d {column} did not reproduce from persisted games"
print("Lab 18d incumbent summaries reproduce from persisted game rows")

incumbent_calls = lab18d_calls.loc[
    lab18d_calls["decoder"].isin(DECODERS)
].assign(arm="incumbent")
incumbent_games = lab18d_games.loc[
    lab18d_games["decoder"].isin(DECODERS)
].assign(arm="incumbent")

gameplay_calls = pd.concat(
    [incumbent_calls] + arm_call_frames, ignore_index=True
)
gameplay_games = pd.concat(
    [incumbent_games] + arm_game_frames, ignore_index=True
)
gameplay_calls["arm"] = pd.Categorical(
    gameplay_calls["arm"], ["incumbent"] + ARMS, ordered=True
)
gameplay_games["arm"] = pd.Categorical(
    gameplay_games["arm"], ["incumbent"] + ARMS, ordered=True
)
assert len(gameplay_games) == len(SEEDS) * 3 * len(DECODERS) * len(
    RESERVED_ANSWERS
)
assert not gameplay_games.duplicated(
    ["seed", "arm", "decoder", "answer"]
).any()
print("calls:", len(gameplay_calls), "games:", len(gameplay_games))
""")

md("""
## 19.21 Gameplay results

Each row is one adapter playing 19 reserved answers. The seed is the
replication unit: three seeds give three paired observations of an arm against
its own ancestor, and a direction that appears for one seed only is not a
finding.

Turn 2 is the only state every arm visits with an identical history, because
all games open with RAISE. Later turns are conditional on survival and follow
arm-specific trajectories, so they describe deployed behavior rather than a
controlled one-step contrast.
""")

code("""
game_summary = gameplay_games.groupby(
    ["seed", "arm", "decoder"], observed=True, sort=True
).agg(
    games=("answer", "size"),
    solved=("solved", "sum"),
    solve_rate=("solved", "mean"),
    invalid_termination_rate=("terminated_invalid", "mean"),
    mean_model_calls=("model_calls", "mean"),
    mean_final_candidates=("final_candidate_count", "mean"),
)
game_summary["mean_turns_on_wins"] = gameplay_games.loc[
    gameplay_games["solved"]
].groupby(["seed", "arm", "decoder"], observed=True)["solved_turn"].mean()
game_summary = game_summary.reset_index()
display(game_summary)

display(
    game_summary.pivot_table(
        index=["decoder", "seed"],
        columns="arm",
        values="solved",
        observed=True,
    )
)

paired_solve_rows = []
for seed in SEEDS:
    for decoder in DECODERS:
        for left_arm, right_arm in [
            ("incumbent", "hard"),
            ("incumbent", "value"),
            ("hard", "value"),
        ]:
            left = gameplay_games.query(
                "seed == @seed and decoder == @decoder and arm == @left_arm"
            )[["answer", "solved"]].rename(columns={"solved": "left"})
            right = gameplay_games.query(
                "seed == @seed and decoder == @decoder and arm == @right_arm"
            )[["answer", "solved"]].rename(columns={"solved": "right"})
            paired = left.merge(
                right, on="answer", validate="one_to_one"
            )
            paired_solve_rows.append({
                "seed": seed,
                "decoder": decoder,
                "left_arm": left_arm,
                "right_arm": right_arm,
                "left_solved": int(paired["left"].sum()),
                "right_solved": int(paired["right"].sum()),
                "solve_delta": float(
                    paired["right"].mean() - paired["left"].mean()
                ),
                "left_only": int((paired["left"] & ~paired["right"]).sum()),
                "right_only": int((~paired["left"] & paired["right"]).sum()),
                "both": int((paired["left"] & paired["right"]).sum()),
                "neither": int((~paired["left"] & ~paired["right"]).sum()),
            })
paired_solves = pd.DataFrame(paired_solve_rows)
display(paired_solves)
""")

code("""
action_summary = gameplay_calls.groupby(
    ["seed", "arm", "decoder"], observed=True, sort=True
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
    mean_realized_log2_reduction=("realized_log2_reduction", "mean"),
).reset_index()
display(action_summary)

by_turn = gameplay_calls.groupby(
    ["arm", "decoder", "turn"], observed=True, sort=True
).agg(
    calls=("answer", "size"),
    usable_rate=("usable", "mean"),
    consistency_rate=("history_consistent", "mean"),
    repeat_rate=("repeated", "mean"),
    teacher_match_rate=("teacher_match", "mean"),
    candidate_choice_rate=("chosen_is_candidate", "mean"),
    mean_candidates_before=("candidate_count_before", "mean"),
    mean_candidates_after=("candidate_count_after", "mean"),
    mean_entropy_gap_bits=("entropy_gap_bits", "mean"),
    mean_realized_log2_reduction=("realized_log2_reduction", "mean"),
).reset_index()
display(by_turn)

turn2_summary = gameplay_calls.query("turn == 2").groupby(
    ["seed", "arm", "decoder"], observed=True, sort=True
).agg(
    calls=("answer", "size"),
    usable_rate=("usable", "mean"),
    teacher_match_rate=("teacher_match", "mean"),
    chosen_candidate_rate=("chosen_is_candidate", "mean"),
    mean_open_entropy_regret_bits=("open_entropy_regret_bits", "mean"),
    median_open_entropy_regret_bits=("open_entropy_regret_bits", "median"),
    mean_chosen_entropy_bits=("chosen_entropy_bits", "mean"),
    mean_realized_log2_reduction=("realized_log2_reduction", "mean"),
    mean_candidates_after=("candidate_count_after", "mean"),
    mean_teacher_rank=("model_teacher_rank", "mean"),
    mean_candidate_mass=("candidate_mass", "mean"),
).reset_index()
display(turn2_summary)
""")

md("""
### Late closure

A singleton state has exactly one word left. Naming it wins immediately, and
entropy cannot distinguish it because every action has the same expected
information gain of zero bits over a one-word candidate set. The controlled
game-level metric is whether the model closes on its first singleton
opportunity. Pooling all singleton calls is also reported, but a failure can
create more calls and therefore change its own denominator.
""")

code("""
singleton_calls = gameplay_calls.query("candidate_count_before == 1")
pooled_closure = singleton_calls.groupby(
    ["seed", "arm", "decoder"], observed=True, sort=True
).agg(
    singleton_calls=("answer", "size"),
    sole_candidate_rate=("chosen_is_candidate", "mean"),
    usable_rate=("usable", "mean"),
    repeat_rate=("repeated", "mean"),
).reset_index()
first_singleton = singleton_calls.sort_values(
    ["seed", "arm", "decoder", "answer", "turn"], kind="stable"
).groupby(
    ["seed", "arm", "decoder", "answer"],
    observed=True,
    sort=False,
).head(1)
first_closure = first_singleton.groupby(
    ["seed", "arm", "decoder"], observed=True, sort=True
).agg(
    games_reaching_singleton=("answer", "size"),
    first_singleton_close_rate=("chosen_is_candidate", "mean"),
).reset_index()
closure_summary = pooled_closure.merge(
    first_closure,
    on=["seed", "arm", "decoder"],
    how="outer",
    validate="one_to_one",
)
display(closure_summary)

failures = gameplay_games.loc[~gameplay_games["solved"]].copy()
failures["ended_at_singleton"] = failures["final_candidate_count"] == 1
failure_summary = failures.groupby(
    ["seed", "arm", "decoder"], observed=True, sort=True
).agg(
    failures=("answer", "size"),
    ended_at_singleton=("ended_at_singleton", "sum"),
    mean_final_candidates=("final_candidate_count", "mean"),
).reset_index()
display(failure_summary)
""")

md("""
## 19.22 Preregistered read-out

The rules below were fixed in Section 19.1, before any Lab 19 model was
trained. The primary outcome is answer-constrained solve rate. The two
mechanism metrics are late closure, measured as the sole-candidate selection
rate on each game's first constrained singleton opportunity, and broad action
value, measured as mean Turn 2 open-entropy regret in bits where lower is
better.

A direction counts only when it holds for all three seeds. With three seeds
there is no useful significance test, so replication across seeds is the
standard, and dev states are never treated as independent replications.
""")

code("""
PRIMARY_DECODER = "answer-constrained"


def seed_metric(frame: pd.DataFrame, column: str) -> dict:
    subset = frame.query("decoder == @PRIMARY_DECODER")
    values = {
        (int(row.seed), str(row.arm)): float(getattr(row, column))
        for row in subset.itertuples()
    }
    # A missing cell means the arm never visited that regime, which must read
    # as unknown rather than as a silent zero.
    return {
        (seed, arm): values.get((seed, arm), float("nan"))
        for seed in SEEDS
        for arm in ["incumbent"] + ARMS
    }


solve_by_seed = seed_metric(game_summary, "solve_rate")
closure_by_seed = seed_metric(
    closure_summary, "first_singleton_close_rate"
)
regret_by_seed = seed_metric(turn2_summary, "mean_open_entropy_regret_bits")


def replicated(
    metric: dict, arm: str, baseline: str, direction: str
) -> tuple[int, list[float]]:
    deltas = [
        metric[(seed, arm)] - metric[(seed, baseline)] for seed in SEEDS
    ]
    if direction == "higher":
        return sum(delta > 0 for delta in deltas), deltas
    return sum(delta < 0 for delta in deltas), deltas


verdict_rows = []
for arm, baseline in [
    ("hard", "incumbent"),
    ("value", "incumbent"),
    ("value", "hard"),
    ("hard", "value"),
]:
    for name, metric, direction in [
        ("solve_rate", solve_by_seed, "higher"),
        ("closure_rate", closure_by_seed, "higher"),
        ("turn2_open_regret_bits", regret_by_seed, "lower"),
    ]:
        seeds_better, deltas = replicated(metric, arm, baseline, direction)
        verdict_rows.append({
            "metric": name,
            "better_when": direction,
            "arm": arm,
            "baseline": baseline,
            "seeds_with_data": int(np.isfinite(deltas).sum()),
            "seeds_better": seeds_better,
            "replicated": seeds_better == len(SEEDS),
            **{f"delta_seed{seed}": delta for seed, delta in zip(SEEDS, deltas)},
            "mean_delta": float(np.mean(deltas)),
        })
verdict = pd.DataFrame(verdict_rows)
display(verdict)


def flag(metric_name: str, arm: str, baseline: str) -> bool:
    row = verdict.query(
        "metric == @metric_name and arm == @arm and baseline == @baseline"
    )
    return bool(row.iloc[0]["replicated"]) if len(row) else False


hard_closure_gain = flag("closure_rate", "hard", "incumbent")
value_closure_gain = flag("closure_rate", "value", "incumbent")
hard_regret_gain = flag("turn2_open_regret_bits", "hard", "incumbent")
value_regret_gain = flag("turn2_open_regret_bits", "value", "incumbent")
value_regret_over_hard = flag(
    "turn2_open_regret_bits", "value", "hard"
)
# "Value harms sharp closure" means hard closed better on every seed, not
# merely that value failed to win.
value_closure_loss = (
    replicated(closure_by_seed, "hard", "value", "higher")[0] == len(SEEDS)
)
any_solve_gain = (
    flag("solve_rate", "hard", "incumbent")
    or flag("solve_rate", "value", "incumbent")
)
any_mechanism_gain = (
    hard_closure_gain or value_closure_gain
    or hard_regret_gain or value_regret_gain
)
arms_separated = any(
    flag(metric, left, right)
    for metric in [
        "solve_rate", "closure_rate", "turn2_open_regret_bits"
    ]
    for left, right in [("value", "hard"), ("hard", "value")]
)
shared_mechanism_gain = (
    (hard_closure_gain and value_closure_gain)
    or (hard_regret_gain and value_regret_gain)
)

if not any_solve_gain and not any_mechanism_gain:
    conclusion = (
        "Neither arm produced a replicated improvement over its incumbent on "
        "a preregistered metric. The objective, action support, optimization "
        "budget, and limited broad-state coverage remain live explanations; "
        "this result would not distinguish hard from soft distillation."
    )
elif value_closure_loss:
    conclusion = (
        "The value arm lost sharp closure against the matched hard control on "
        "every seed. Since the two arms differ only in target probabilities, "
        "the soft objective is what interferes with committing to a "
        "determined answer, and the two regimes need separate treatment "
        "rather than one shared target shape."
    )
elif (
    hard_closure_gain
    and value_regret_gain
    and value_regret_over_hard
    and not value_closure_loss
):
    conclusion = (
        "Hard improved closure and value improved broad regret. Relative "
        "value in the target matters: the soft distribution buys broad-state "
        "action value that a one-hot target does not, while explicit "
        "candidate mass buys closure. Both are real and separable."
    )
elif (
    shared_mechanism_gain
    and not arms_separated
):
    conclusion = (
        "Both arms moved together and neither separated from the other. The "
        "gain is most plausibly explained by explicit regime coverage in the "
        "state selection plus 1,029 additional updates, not by the shape of "
        "the teacher distribution."
    )
else:
    conclusion = (
        "The pattern is mixed or unreplicated across seeds. Report the per-"
        "seed deltas above and treat the effect as unresolved rather than "
        "picking the seed that agrees with a preferred story."
    )

print(conclusion)
print()
print("solve rate (answer-constrained) by seed and arm:")
for seed in SEEDS:
    print(
        f"  seed {seed}: "
        + ", ".join(
            f"{arm} {solve_by_seed[(seed, arm)] * len(RESERVED_ANSWERS):.0f}/"
            f"{len(RESERVED_ANSWERS)}"
            for arm in ["incumbent"] + ARMS
        )
    )
print(
    "\\nSeeds are the replication unit. Dev states and per-answer flips are "
    "supporting detail, not independent evidence."
)
""")

md("""
## 19.23 Persist the lab

Everything a later lab needs is written under `results/lab19`: the per-call
gameplay table, the dev shortlist rows and their raw 12-action score matrices,
the training histories, the summary tables, and a manifest recording the source
hashes, incumbent hashes, trained-adapter hashes, and target fingerprints.
""")

code("""
atomic_csv(gameplay_calls, RESULTS_DIR / "gameplay-calls.csv")
atomic_csv(gameplay_games, RESULTS_DIR / "gameplay-games.csv")
atomic_csv(game_summary, RESULTS_DIR / "game-summary.csv")
atomic_csv(paired_solves, RESULTS_DIR / "paired-solves.csv")
atomic_csv(action_summary, RESULTS_DIR / "action-summary.csv")
atomic_csv(by_turn, RESULTS_DIR / "by-turn.csv")
atomic_csv(turn2_summary, RESULTS_DIR / "turn2-summary.csv")
atomic_csv(closure_summary, RESULTS_DIR / "closure-summary.csv")
atomic_csv(failure_summary, RESULTS_DIR / "failure-summary.csv")
atomic_csv(verdict, RESULTS_DIR / "preregistered-verdict.csv")
atomic_csv(dev_results, RESULTS_DIR / "dev-shortlist-rows.csv")
atomic_csv(dev_by_label, RESULTS_DIR / "dev-summary-by-arm.csv")
atomic_csv(dev_by_regime, RESULTS_DIR / "dev-summary-by-regime.csv")
atomic_csv(dev_by_bucket, RESULTS_DIR / "dev-summary-by-bucket.csv")
atomic_csv(dev_by_turn, RESULTS_DIR / "dev-summary-by-turn.csv")
atomic_csv(dev_paired, RESULTS_DIR / "dev-paired.csv")
atomic_csv(training_summary, RESULTS_DIR / "training-summary.csv")

run_manifest = {
    "experiment": "Lab 19 value-aware distillation",
    "model_id": MODEL_ID,
    "representation": "derived_state_v1",
    "seeds": SEEDS,
    "arms": ARMS,
    "decoders": DECODERS,
    "updates_per_arm": UPDATES,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_fraction": WARMUP_FRACTION,
    "teacher_temperature": TEACHER_TEMPERATURE,
    "support_size": SUPPORT_SIZE,
    "broad_states": BROAD_STATES,
    "sharp_states": SHARP_STATES,
    "training_stream_sha256": stream_fingerprint,
    "target_manifest": target_manifest,
    "source_sha256": structured_hashes,
    "incumbent_sha256": {
        str(seed): incumbent_hashes[seed] for seed in SEEDS
    },
    "trained_adapter_sha256": arm_hashes,
    "reserved_answers": list(RESERVED_ANSWERS),
    "opening": OPENING,
    "max_turns": MAX_TURNS,
    "action_space": "2,315 answer words",
    "scoring_rule": "summed log P(word tokens + EOS | structured prompt)",
    "incumbent_gameplay_source": str(LAB18D_RESULTS),
    "conclusion": conclusion,
}
atomic_json(run_manifest, RESULTS_DIR / "lab19-run.json")
print("written to", RESULTS_DIR)
""")

md("""
## Lab 19 checkpoint

Read the result in this order:

1. Did either arm raise answer-constrained solve rate above its own incumbent
   on all three seeds?
2. Did the first-singleton close rate rise? That is the closure defect Lab 18d
   isolated, and the sharp value target is its direct treatment.
3. Did Turn 2 open-entropy regret fall? That is the broad-state action-value
   defect, and Turn 2 is the only paired state across arms.
4. Did the arms separate from each other, or did they move together? Moving
   together points at regime coverage and extra updates rather than at target
   shape.
5. Did the dev shortlist improve while gameplay did not? That is the Lab 09
   pattern repeating: winning a scored shortlist is not the same as choosing
   well from 2,315 words.

What this lab cannot tell you: whether a different support, a different
temperature, or more updates would help. One support rule and one temperature
were fixed in advance precisely so that the hard-versus-value contrast is
interpretable, and the reserved 19 answers were held out of every target,
every audit, and every tuning decision so that the final number means what it
says.
""")


for index, cell in enumerate(cells):
    cell["id"] = f"lab19-{index:02d}-{cell['cell_type']}"

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

path = Path("notebooks/19_value_aware_distillation.ipynb")
path.write_text(json.dumps(notebook, indent=1))
print(f"wrote {path} with {len(cells)} cells")