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

## 19.1 Pre-registered experiment

**Question.** Does distilling a *relative action-value* distribution improve the
constrained policy more than distilling the same teacher's single best action?

**Arms.** Each of the three B-structured incumbents is continued twice.

| arm | teacher target on 12 scored actions |
| --- | --- |
| `hard` | one-hot on the teacher's chosen action |
| `value` | soft distribution over all 12 actions |

Everything else is identical inside a seed pair: the same starting adapter, the
same 1,029 states in the same order, the same 12 actions per state in the same
order, the same update count, optimizer, schedule, and dropout seed. The two
arms differ only in the probability vector on the right-hand side of one
cross-entropy.

| seed | incumbent | source |
| ---: | --- | --- |
| 42 | `qwen3-0.6b-wordle-lora-dataset-b-structured` | Lab 17 |
| 45 | `...-b-structured-seed45` | Lab 18c |
| 47 | `...-b-structured-seed47` | Lab 18c |

**The action support is mined from each seed's own incumbent.** Before any
training, each frozen incumbent ranks all 2,315 answer words at every one of
the 1,029 training states and 466 dev states, using the exact summed
log-probability scorer Lab 18b verified and Lab 18d deployed. Four of the twelve
actions in every support are then the words that *this* incumbent already
prefers. They enter as hard negatives: the target says explicitly how little
probability those words deserve, on the states where the model actually emits
them.

That makes the support a function of `(seed, state)`, not of the state alone.
Seed 45's twelve words at a state may differ from seed 42's. This does not
weaken the contrast the lab is built to make, because that contrast is
within-seed: a seed's `hard` and `value` arms share one frozen support, byte for
byte, in one frozen order, and differ only in target probabilities. Across
seeds the three supports are three *instances* of one policy-adaptive treatment
rule, so a direction that replicates on all three seeds replicates the
intervention rather than one lucky word list.

**Objective.** For a state with actions `a_1..a_12`, the student score is the
summed `log P(word tokens + EOS | structured prompt)` used since Lab 18b, and
the loss is

```text
-(target_probs * log_softmax(student_action_scores)).sum()
```

for both arms. `hard` merely supplies a one-hot vector, so the arms cannot
differ through the loss formula, the action support, or the optimization path.

**Support construction, fixed before any word was mined.** Both regimes reach
exactly 12 distinct actions. Where categories overlap, each ranked list is
walked farther down so the support still fills.

| regime | 12 actions |
| --- | --- |
| broad | 4 highest-entropy open actions, 2 highest-entropy current candidates, 4 highest-ranked incumbent actions, deterministic fill |
| sharp | all 1-2 current candidates, previous guesses that are answer words, 4 highest-ranked incumbent non-candidate actions, lexical-overlap distractors, deterministic fill |

The global open optimum and the canonical candidate teacher survive
deduplication in every broad support, so open regret stays measurable inside the
support. Every selected incumbent action is proven present in the support it was
selected for.

**Teacher targets by regime.**

| regime | states | value target | hard target |
| --- | --- | --- | --- |
| broad, `candidate_count >= 3` | 686 | `softmax(entropy_bits / 0.5)` over the 12 actions | argmax entropy, preferring a tied candidate, then lexicographic |
| sharp, `candidate_count <= 2` | 343 | equal mass on the 1-2 current candidates, zero elsewhere | lexicographically first candidate |

The sharp rule is the whole point of calling this *value-aware* rather than
*entropy-aware*. At a singleton the entropy of every action is 0.0 bits, so an
entropy target is uniform noise precisely where naming the candidate wins the
game. Value says: this word ends it, the others do not.

**Cost, stated up front.** Mining is the dominant expense. Lab 18d measured
roughly 3.4 to 3.8 seconds per full 2,315-word ranking on this hardware, and
this lab needs 3 seeds times 1,495 states, or 4,485 rankings: about 4.5 to 4.8
hours before the first optimizer step. Mining is restartable and hash-guarded,
so an interruption costs at most 25 states.

**Primary outputs.**

1. Held-out dev shortlist metrics for incumbent, `hard`, and `value` on all
   three seeds, each scored on that seed's own frozen support: cross-entropy and
   KL to the value target, `hard` top-1 rate, mean broad open regret in bits,
   and sharp candidate-selection rate.
2. Frozen 19-answer gameplay under exact Lab 18d rules with the free and
   answer-constrained decoders, reported by seed and arm.

**Secondary outputs.** Stratification by regime, candidate bucket, and turn;
Turn 2 open-teacher regret and realized `log2` candidate reduction; repeat and
consistency rates; singleton closure rate; the number of failures that end at a
singleton; and the probability mass each model still places on its own
incumbent's mined favorites. Closure is evaluated per game at the first
singleton opportunity; pooled singleton calls are descriptive because failures
create additional calls.

**Read before seeing results.**

| observation | pre-registered interpretation |
| --- | --- |
| `hard` improves closure while `value` improves broad regret | the *relative* target matters, and the two regimes need different information |
| both arms move together | explicit regime coverage plus 1,029 more updates explains the gain; the soft distribution adds nothing |
| `value` harms sharp closure | the soft broad objective interferes with committing to a determined answer |
| `value` beats `hard` | richer target shape wins against the *same* seed-specific decision boundary, since both arms saw one frozen support built from one incumbent |
| neither arm beats its incumbent | the objective, the 12-action support, or the optimization budget failed; not evidence that distillation cannot work |

Three seeds and 19 answers give a paired diagnostic, not a population solve
rate. The replication unit is the seed. State rows are not independent training
runs, and 1,029 training states are not 1,029 replications of anything.

Because supports differ across seeds, state-level scores are paired only
*within* a seed. A dev state's cross-entropy under seed 42's `hard` arm and
under seed 45's `hard` arm are not two measurements of one quantity; they are
measurements against two different twelve-word boundaries. Seed-level
directions are what replicate.

The source pool is still thinner than deployment at the broadest states. Lab
19 includes every available broad Turn 2 and 11-plus-candidate training state,
but a null Turn 2 result can still mean that the existing curriculum does not
contain enough comparable broad states, not only that the target shape failed.

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
candidate set, or it constrains nothing there.

Lab 19 closes that support hole from both ends. Open-entropy actions and
deterministic filler cover words the teacher likes, and four slots per state are
reserved for the words *the student itself* ranks highest over the full answer
list. A teacher-only support can only ever say "prefer this word"; it is silent
about whatever the student would otherwise have emitted, which is exactly the
failure Lab 12 could not see. Putting the student's own top-ranked words in the
support turns that silence into an explicit instruction, and the softmax
denominator now contains the competitors that actually win at deployment.

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

## 19.3 What kind of distillation this is

It is worth being precise, because "distillation" covers several different
procedures and this lab is not the classical one.

Classical logit distillation trains a *fresh* student from scratch or from a
generic pretrained base, matching a large teacher's full-vocabulary logits over
a corpus. Nothing here does that. Lab 19 takes three already-trained LoRA
adapters and continues them for 1,029 more updates against a symbolic teacher's
*action-value* distribution over twelve named actions. The base weights stay
frozen, only the rank-8 LoRA on `q/k/v/o` moves, and there is no
full-vocabulary matching anywhere: the target lives on a 12-simplex, not on a
151k-vocabulary simplex.

So the right label is **continued LoRA policy and value distillation**. The
teacher is the entropy expert plus the candidate set; the student is an
incumbent policy that already plays; and the quantity being transferred is a
ranking over actions with meaningful relative magnitudes, not a soft copy of
another network's logits.

**Hard-negative distillation.** The four incumbent-mined slots make this a
contrastive procedure as well as an imitative one. In a plain teacher-only
support, the denominator of `log_softmax(student_action_scores)` contains only
words the teacher nominated, so the update can raise the teacher's word without
ever lowering the word the student would really have emitted. Mining each seed's
own top-ranked answers and placing them in the same softmax makes the gradient
push down precisely those words, at precisely the states where they win. That is
the standard hard-negative construction from retrieval and contrastive learning,
applied to a Wordle action space.

It also explains why the support has to be seed-specific. A hard negative is
only hard relative to a particular policy. Seed 45's favourite wrong word is not
seed 42's, and mining a shared support from one seed would give the other two
arms easy negatives and a weaker treatment.

## 19.4 Run controls and memory guard

Run this notebook only through the total-system watchdog:

```
scripts/memguard.py --min-free 64 -- uv run jupyter nbconvert \
    --to notebook --execute --inplace notebooks/19_value_aware_distillation.ipynb
```

The in-process MPS cap turns a runaway allocation into an ordinary exception.
The first expensive thing this notebook does is mine 4,485 full-list rankings,
so its gate runs first and is mandatory: the seed-42 incumbent must reproduce
one persisted Lab 18b score vector, and the same model must hold a flat driver
memory trace across a fixed-shape 40-repeat full-list scoring soak with
inner-loop peak sampling. Mining does not start unless both pass.

Two further gates run before the first optimizer step: a numerical regression of
the batched 12-action scorer against plain single-action forwards, and a
fixed-shape 40-step training soak on the longest state. A third check, replaying
one Lab 18d constrained trajectory, runs before gameplay.

Mined rankings, completed training arms, dev shortlist scores, and gameplay
artifacts are all written atomically. An interruption can lose at most 25 mined
states, the current training arm, one model's dev pass, or one game; everything
already committed is validated by hash and reused.


```python
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
```

    MPS cap: 128 GiB of 464 GiB
    RUN_TRAINING: True
    RUN_EVALUATION: True



```python
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
BROAD_OPEN_SLOTS = 4
BROAD_CANDIDATE_SLOTS = 2
BROAD_INCUMBENT_SLOTS = 4
SHARP_INCUMBENT_SLOTS = 4
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

SPLITS = ["train", "dev"]
MINING_TOP_K = 32
MINING_CHECKPOINT_EVERY = 25
LAB18D_SECONDS_PER_STATE = (3.4, 3.8)

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
    (seed, split): (
        GENERATED_DIR / f"lab19-value-targets-seed{seed}-{split}.jsonl"
    )
    for seed in SEEDS
    for split in SPLITS
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
```

    device: mps


## 19.5 Freeze every source artifact

Lab 19 reads Lab 17's structured JSONL and three frozen adapters. Nothing is
regenerated. The file hashes must match the Dataset B manifest and the manifest
stored inside each incumbent checkpoint, and every incumbent must carry the same
LoRA geometry, or the two arms would not be continuing from a common ancestor.


```python
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
      <th>checkpoint</th>
      <th>prior_steps</th>
      <th>prior_val_loss</th>
      <th>adapter_sha256</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>qwen3-0.6b-wordle-lora-dataset-b-structured</td>
      <td>1029</td>
      <td>1.258621</td>
      <td>8f08ba4787ccaa72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>qwen3-0.6b-wordle-lora-dataset-b-structured-se...</td>
      <td>1029</td>
      <td>1.287198</td>
      <td>a3b849ac3cbc57c0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47</td>
      <td>qwen3-0.6b-wordle-lora-dataset-b-structured-se...</td>
      <td>1029</td>
      <td>1.325974</td>
      <td>52dd5812478f9b41</td>
    </tr>
  </tbody>
</table>
</div>



```python
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
      <th>split</th>
      <th>rows</th>
      <th>next_guess_rows</th>
      <th>unique_states</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train</td>
      <td>8768</td>
      <td>5669</td>
      <td>3099</td>
    </tr>
    <tr>
      <th>1</th>
      <td>validation</td>
      <td>1135</td>
      <td>669</td>
      <td>466</td>
    </tr>
    <tr>
      <th>2</th>
      <td>test</td>
      <td>21</td>
      <td>13</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>


## 19.6 The frozen `derived_state_v1` representation

The input representation does not change in Lab 19. These functions rebuild it
from a state key, and the check below is the proof: every stored `NEXT_GUESS`
prompt in train and dev must be reproduced character for character. The same
builder is used later for gameplay, so training states and deployed states go
through one code path.

Candidates are reconstructed from the state key alone through the pattern
matrix. The hidden `answer` field is never consulted when building actions or
targets, and the reconstructed count must equal the count recorded by Lab 14.


```python
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
    return "\n".join([
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
        "Task: NEXT_GUESS\n"
        "You are playing Wordle.\n"
        "Use the game history to choose the next guess.\n"
        "Return exactly one uppercase five-letter word.\n\n"
        "Derived state:\n"
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
```


```python
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
```

    pattern-matrix candidates match filter_candidates on 24 states
    every stored NEXT_GUESS prompt reproduced: 3565



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
      <th>split</th>
      <th>regime</th>
      <th>states</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train</td>
      <td>broad</td>
      <td>1358</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train</td>
      <td>sharp</td>
      <td>1741</td>
    </tr>
    <tr>
      <th>2</th>
      <td>validation</td>
      <td>broad</td>
      <td>327</td>
    </tr>
    <tr>
      <th>3</th>
      <td>validation</td>
      <td>sharp</td>
      <td>139</td>
    </tr>
  </tbody>
</table>
</div>


## 19.7 Action-value utilities

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


```python
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
```

    vectorized entropy agrees with EntropyExpert on 9 states, max abs diff 8.88e-16


## 19.8 Deterministic training-state selection

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


```python
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
```

    training states removed as reachable in a reserved game: 0


    selected 1029 training states: 686 broad (267 priority-covered, 419 filled) and 343 sharp (229 singleton, 114 two-candidate)



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
      <th>regime</th>
      <th>bucket</th>
      <th>states</th>
      <th>turn2</th>
      <th>mean_turn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>broad</td>
      <td>3-10</td>
      <td>451</td>
      <td>32</td>
      <td>3.184035</td>
    </tr>
    <tr>
      <th>1</th>
      <td>broad</td>
      <td>11+</td>
      <td>235</td>
      <td>140</td>
      <td>2.408511</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sharp</td>
      <td>1</td>
      <td>229</td>
      <td>5</td>
      <td>3.755459</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sharp</td>
      <td>2</td>
      <td>114</td>
      <td>4</td>
      <td>3.684211</td>
    </tr>
  </tbody>
</table>
</div>



```python
dev_order = dev_states.sort_values(
    "state_key", kind="stable"
).reset_index(drop=True)
STATE_FRAMES = {"train": selection, "dev": dev_order}
STATE_COUNTS = {split: len(frame) for split, frame in STATE_FRAMES.items()}
STATE_ORDER_SHA256 = {
    split: sha256_text("|".join(frame["state_key"]))
    for split, frame in STATE_FRAMES.items()
}
assert STATE_COUNTS["train"] == UPDATES
assert STATE_COUNTS["dev"] == len(dev_states)
for split, frame in STATE_FRAMES.items():
    assert frame["state_key"].is_unique

# One state order for every seed and every arm. Only the twelve actions
# attached to a state become seed-specific; the states themselves, and the
# order they are visited in, never do.
print("state order fingerprints (identical for all seeds and arms):")
for split in SPLITS:
    print(
        f"  {split}: {STATE_COUNTS[split]} states, "
        f"{STATE_ORDER_SHA256[split][:16]}"
    )
```

    state order fingerprints (identical for all seeds and arms):
      train: 1029 states, 4f077f386d8df335
      dev: 466 states, 72d28b80be060bac


## 19.9 The scoring kernels

Everything downstream of this point - mining, targets, training, dev scoring,
and gameplay - is defined in terms of one quantity: the summed
`log P(word tokens + EOS | structured prompt)` that Lab 18b verified and Lab
18d deployed. It is computed two ways, and both are defined once, here, before
anything uses either.

`action_scores` is the **training kernel**. The 12 rows of a state share a
prompt and differ only in their action suffix, so one right-padded batch of
shape `(12, prompt_len + width - 1)` covers the state. `logits_to_keep` is
passed the exact response-predicting positions, so the logit tensor is
`(12, width, vocab)` with `width` at most 4 rather than
`(12, prompt_len, vocab)`. Token log-probabilities come from a target gather
minus `logsumexp`, never a full `log_softmax` over the vocabulary, and
`use_cache=False` keeps no keys or values alive across the backward pass. One
state and its 12 actions form one optimizer update.

`score_all_words` is the **deployment kernel**, copied from Lab 18d: one
`logits_to_keep=1` prefill, a KV cache repeated `CHUNK_SIZE=256` ways, one pass
per token-length bucket, and an `empty_cache` per state. It returns all 2,315
scores. Mining calls it 4,485 times and gameplay calls it once per constrained
turn, so mined rankings and gameplay rankings are the same quantity produced by
the same code.

Defining both here closes a trap. This notebook has exactly one tokenizer, one
`ACTION_TOKENS` table, and one full-list scorer. Nothing later redefines them,
so the ranking that chose a state's hard negatives and the ranking that picks a
guess in a game cannot drift apart. `SCORER_SHA256` fingerprints the tokenizer,
the action-token table, and the answer-list order together, and every mined
artifact is stamped with it.


```python
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
probe_prompt = render_prompt(selection.loc[0, "prompt"])
probe_ids = tokenizer(probe_prompt, add_special_tokens=False)["input_ids"]
for word in ANSWERS[:200] + ANSWERS[-200:]:
    joint = tokenizer(
        probe_prompt + word + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    assert joint[:len(probe_ids)] == probe_ids
    assert joint[len(probe_ids):] == ACTION_TOKENS[WORD_TO_INDEX[word]]
print("action token widths:", sorted({len(t) for t in ACTION_TOKENS}))

SCORER_SHA256 = sha256_text("|".join([
    MODEL_ID,
    str(ACTION_WIDTH),
    json.dumps(ACTION_TOKENS),
    json.dumps(ANSWERS),
]))
print("scorer and action-order fingerprint:", SCORER_SHA256[:16])


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
```

    action token widths: [2, 3, 4]
    scorer and action-order fingerprint: 2e35dbf6edcad821



```python
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
    clear_device_cache()
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
```


```python
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
    clear_device_cache()
```

## 19.10 Mandatory pre-mining regression and memory gate

Mining is the single most expensive step in this lab and it writes artifacts
that everything else is built on. It does not start until the deployment kernel
has been proved correct and leak-free on this machine, in this process.

**Correctness.** The seed-42 incumbent scores Lab 18b's first battery state over
all 2,315 answers and must reproduce the persisted Lab 18b score vector within
float32 tolerance. That pins today's kernel to a vector computed by different
code in an earlier lab.

**Memory.** The same model then scores one fixed longest battery prompt 40
times. The shape never changes, so any growth in driver memory is a leak rather
than a larger workload. Peaks are sampled inside the scoring loop while the
chunk activations are still live, not in the quiet moment between repeats, and
the final third of the trace must be flat in both drift and range before 4,485
real rankings are allowed to start.


```python
battery_states = pd.read_csv(LAB18B_RESULTS / "battery-states.csv")
battery_prompts = []
for state_key in battery_states["state_key"]:
    history = parse_state_key(state_key)
    battery_prompts.append(
        structured_next_guess_prompt(history, len(candidate_indices(history)))
    )
battery_prompt_lengths = [
    len(tokenizer(render_prompt(prompt), add_special_tokens=False)["input_ids"])
    for prompt in battery_prompts
]
print(
    f"Lab 18b battery: {len(battery_prompts)} states, "
    f"prompt tokens {min(battery_prompt_lengths)}-"
    f"{max(battery_prompt_lengths)}"
)
```

    Lab 18b battery: 620 states, prompt tokens 111-190



```python
MINING_PATH_KEYS = ["indices", "scores", "keys", "progress"]


def mining_paths(seed: int, split: str) -> dict[str, Path]:
    stem = f"incumbent-ranking-seed{seed}-{split}"
    return {
        "indices": RESULTS_DIR / f"{stem}-indices.npy",
        "scores": RESULTS_DIR / f"{stem}-scores.npy",
        "keys": RESULTS_DIR / f"{stem}-keys.json",
        "progress": RESULTS_DIR / f"{stem}-progress.json",
    }


def mining_guard(seed: int, split: str) -> dict:
    return {
        "seed": seed,
        "split": split,
        "top_k": MINING_TOP_K,
        "states": STATE_COUNTS[split],
        "incumbent": INCUMBENTS[seed].name,
        "incumbent_sha256": incumbent_hashes[seed],
        "state_order_sha256": STATE_ORDER_SHA256[split],
        "scorer_sha256": SCORER_SHA256,
        "ranking_rule": (
            "summed log P(word tokens + EOS | prompt) over all 2,315 answers"
        ),
    }


def mining_completed(seed: int, split: str) -> int:
    paths = mining_paths(seed, split)
    if not paths["progress"].exists():
        return 0
    progress = dict(json.loads(paths["progress"].read_text()))
    completed = int(progress.pop("completed"))
    if progress != mining_guard(seed, split):
        raise ValueError(
            f"stale incumbent-ranking cache for seed {seed} {split}: the "
            "checkpoint, state order, or scorer fingerprint changed. Delete "
            f"{paths['progress'].name} and its arrays to remine."
        )
    return completed


mining_pending = {
    (seed, split): STATE_COUNTS[split] - mining_completed(seed, split)
    for seed in SEEDS
    for split in SPLITS
}
mining_required = any(pending > 0 for pending in mining_pending.values())
pending_states = sum(mining_pending.values())
low, high = LAB18D_SECONDS_PER_STATE
print(
    f"incumbent rankings still to mine: {pending_states} of "
    f"{len(SEEDS) * sum(STATE_COUNTS.values())}"
)
print(
    f"expected cost at Lab 18d's observed {low}-{high} s/state: "
    f"{pending_states * low / 3600:.1f}-{pending_states * high / 3600:.1f} "
    "hours before the first optimizer step"
)
if mining_required and not (RUN_TRAINING or RUN_EVALUATION):
    raise FileNotFoundError(
        "incumbent rankings are incomplete but both run flags are False; "
        "mining needs a model"
    )
```

    incumbent rankings still to mine: 4485 of 4485
    expected cost at Lab 18d's observed 3.4-3.8 s/state: 4.2-4.7 hours before the first optimizer step



```python
if mining_required or RUN_TRAINING or RUN_EVALUATION:
    gate_model = load_eval_adapter(INCUMBENTS[42])

    first_scores = score_all_words(gate_model, battery_prompts[0])
    reference_scores = np.load(
        LAB18B_RESULTS / "scores-B-structured.npy", mmap_mode="r"
    )[0]
    max_abs_diff = float(np.max(np.abs(first_scores - reference_scores)))
    print("Lab 18b first-state score-vector max abs diff:", max_abs_diff)
    assert max_abs_diff < 1e-3

    soak_prompt = battery_prompts[int(np.argmax(battery_prompt_lengths))]
    soak_peaks = []
    for _ in range(40):
        score_all_words(gate_model, soak_prompt)
        soak_peaks.append(LAST_STATE_PEAK_GIB)
        clear_device_cache()
    third = len(soak_peaks) // 3
    creep = (
        np.mean(soak_peaks[-third:])
        - np.mean(soak_peaks[third:2 * third])
    )
    late_range = np.ptp(soak_peaks[-third:])
    print(
        f"full-list scoring soak peak {max(soak_peaks):.2f} GiB, "
        f"creep {creep:+.2f} GiB, final range {late_range:.2f} GiB"
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
    release_model(gate_model)
    del gate_model
    print("full-list scorer verified and memory plateaued; mining may start")
else:
    print("pre-mining gate skipped: every ranking is already mined and cached")
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    Lab 18b first-state score-vector max abs diff: 0.0


    full-list scoring soak peak 54.77 GiB, creep +0.00 GiB, final range 0.00 GiB


    full-list scorer verified and memory plateaued; mining may start


## 19.11 Mine each incumbent's own ranking

For every seed, the frozen incumbent ranks all 2,315 answer words at all 1,029
training states and all 466 dev states. The ranking is the true summed
`log P(word tokens + EOS | prompt)` over the whole answer list, produced by
`score_all_words` - the function the gate above just checked against Lab 18b.
It is not free generation, and it is not a teacher-side proxy. Free generation
would tell us what the model emits, not how it ranks the words it does not emit,
and a teacher proxy would defeat the entire purpose of a hard negative.

Storing 4,485 dense 2,315-vector rows is unnecessary. What the supports need is
enough of the top of each ranking to skip candidates and previous guesses and
still name four actions, so only the top 32 exact indices and their exact scores
are kept, alongside the ordered state keys. Support construction fails loudly if
32 is ever not enough.

**Restartability.** Each `(seed, split)` writes four artifacts atomically at
least every 25 states, with the progress file written last so it is the commit
marker. A restart validates the incumbent checkpoint hash, the source
state-order fingerprint, and the scorer/action-order fingerprint before touching
the arrays, then trims them to the committed count and continues from exactly
there. Any mismatch raises instead of silently reusing a stale cache. Driver
memory is asserted and the cache is cleared after every single state.


```python
def mine_incumbent_ranking(seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    frame = STATE_FRAMES[split]
    paths = mining_paths(seed, split)
    guard = mining_guard(seed, split)
    keys = list(frame["state_key"])
    total = len(frame)
    indices = np.zeros((total, MINING_TOP_K), dtype=np.int32)
    scores = np.zeros((total, MINING_TOP_K), dtype=np.float32)

    completed = mining_completed(seed, split)
    if completed:
        present = [paths[key].exists() for key in MINING_PATH_KEYS]
        if not all(present):
            raise FileNotFoundError(
                f"incomplete incumbent-ranking artifact set for seed {seed} "
                f"{split}"
            )
        stored_indices = np.load(paths["indices"])
        stored_scores = np.load(paths["scores"])
        stored_keys = json.loads(paths["keys"].read_text())
        if min(
            len(stored_indices), len(stored_scores), len(stored_keys)
        ) < completed:
            raise ValueError(
                f"seed {seed} {split} progress claims {completed} states but "
                "the arrays are shorter"
            )
        if stored_keys[:completed] != keys[:completed]:
            raise ValueError(
                f"seed {seed} {split} cached ranking keys disagree with the "
                "frozen state order"
            )
        indices[:completed] = stored_indices[:completed]
        scores[:completed] = stored_scores[:completed]

    def checkpoint(done: int) -> None:
        atomic_npy(indices[:done], paths["indices"])
        atomic_npy(scores[:done], paths["scores"])
        atomic_write(json.dumps(keys[:done]), paths["keys"])
        atomic_json(guard | {"completed": done}, paths["progress"])

    if completed == total:
        print(f"seed {seed} {split}: verified {total} cached rankings")
        return indices, scores

    model = load_eval_adapter(INCUMBENTS[seed])
    started = time.perf_counter()
    peak = 0.0
    try:
        for position in range(completed, total):
            vector = score_all_words(model, frame.loc[position, "prompt"])
            assert LAST_STATE_PEAK_GIB < MEMORY_ABORT_GIB, (
                f"seed {seed} {split} exceeded the memory threshold at state "
                f"{position}: {LAST_STATE_PEAK_GIB:.1f} GiB"
            )
            peak = max(peak, LAST_STATE_PEAK_GIB)
            order = np.argsort(-vector, kind="stable")[:MINING_TOP_K]
            indices[position] = order.astype(np.int32)
            scores[position] = vector[order].astype(np.float32)
            clear_device_cache()
            done = position + 1
            if done % MINING_CHECKPOINT_EVERY == 0 or done == total:
                checkpoint(done)
                rate = (time.perf_counter() - started) / (done - completed)
                print(
                    f"seed {seed} {split} {done:4d}/{total} "
                    f"{rate:.2f} s/state "
                    f"eta {(total - done) * rate / 60:.1f} min "
                    f"peak {peak:.2f} GiB",
                    flush=True,
                )
    finally:
        release_model(model)
        del model
    assert mining_completed(seed, split) == total
    return indices, scores


incumbent_rankings = {}
mining_started = time.perf_counter()
for seed in SEEDS:
    for split in SPLITS:
        incumbent_rankings[(seed, split)] = mine_incumbent_ranking(seed, split)
print(
    f"incumbent rankings ready in "
    f"{(time.perf_counter() - mining_started) / 60:.1f} min"
)

ranking_hashes = {
    (seed, split): sha256_file(mining_paths(seed, split)["indices"])
    for seed in SEEDS
    for split in SPLITS
}
for (seed, split), (indices, scores) in incumbent_rankings.items():
    assert indices.shape == (STATE_COUNTS[split], MINING_TOP_K)
    assert scores.shape == (STATE_COUNTS[split], MINING_TOP_K)
    assert (np.diff(scores, axis=1) <= 1e-6).all(), (
        f"seed {seed} {split} rankings are not sorted by descending score"
    )
    assert all(
        len(set(row.tolist())) == MINING_TOP_K for row in indices
    ), f"seed {seed} {split} rankings contain a duplicated action"
display(pd.DataFrame([
    {
        "seed": seed,
        "split": split,
        "states": STATE_COUNTS[split],
        "top_k": MINING_TOP_K,
        "indices_sha256": ranking_hashes[(seed, split)][:16],
        "mean_top1_score": float(
            incumbent_rankings[(seed, split)][1][:, 0].mean()
        ),
        "mean_top1_minus_top2": float(
            (
                incumbent_rankings[(seed, split)][1][:, 0]
                - incumbent_rankings[(seed, split)][1][:, 1]
            ).mean()
        ),
    }
    for seed in SEEDS
    for split in SPLITS
]))
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 42 train   25/1029 4.17 s/state eta 69.8 min peak 54.74 GiB


    seed 42 train   50/1029 3.74 s/state eta 61.1 min peak 54.74 GiB


    seed 42 train   75/1029 3.60 s/state eta 57.2 min peak 54.74 GiB


    seed 42 train  100/1029 3.51 s/state eta 54.4 min peak 54.74 GiB


    seed 42 train  125/1029 3.50 s/state eta 52.8 min peak 54.74 GiB


    seed 42 train  150/1029 3.48 s/state eta 51.0 min peak 54.74 GiB


    seed 42 train  175/1029 3.45 s/state eta 49.1 min peak 54.74 GiB


    seed 42 train  200/1029 3.44 s/state eta 47.6 min peak 54.74 GiB


    seed 42 train  225/1029 3.43 s/state eta 46.0 min peak 54.74 GiB


    seed 42 train  250/1029 3.43 s/state eta 44.5 min peak 54.74 GiB


    seed 42 train  275/1029 3.42 s/state eta 43.0 min peak 54.74 GiB


    seed 42 train  300/1029 3.41 s/state eta 41.4 min peak 54.74 GiB


    seed 42 train  325/1029 3.41 s/state eta 40.0 min peak 54.74 GiB


    seed 42 train  350/1029 3.40 s/state eta 38.5 min peak 54.74 GiB


    seed 42 train  375/1029 3.40 s/state eta 37.1 min peak 54.74 GiB


    seed 42 train  400/1029 3.40 s/state eta 35.7 min peak 54.74 GiB


    seed 42 train  425/1029 3.40 s/state eta 34.2 min peak 54.74 GiB


    seed 42 train  450/1029 3.40 s/state eta 32.8 min peak 54.74 GiB


    seed 42 train  475/1029 3.39 s/state eta 31.3 min peak 54.74 GiB


    seed 42 train  500/1029 3.39 s/state eta 29.9 min peak 54.74 GiB


    seed 42 train  525/1029 3.39 s/state eta 28.5 min peak 54.74 GiB


    seed 42 train  550/1029 3.39 s/state eta 27.0 min peak 54.74 GiB


    seed 42 train  575/1029 3.39 s/state eta 25.6 min peak 54.74 GiB


    seed 42 train  600/1029 3.38 s/state eta 24.2 min peak 54.74 GiB


    seed 42 train  625/1029 3.38 s/state eta 22.8 min peak 54.74 GiB


    seed 42 train  650/1029 3.38 s/state eta 21.3 min peak 54.74 GiB


    seed 42 train  675/1029 3.38 s/state eta 19.9 min peak 54.74 GiB


    seed 42 train  700/1029 3.37 s/state eta 18.5 min peak 54.74 GiB


    seed 42 train  725/1029 3.37 s/state eta 17.1 min peak 54.74 GiB


    seed 42 train  750/1029 3.37 s/state eta 15.7 min peak 54.74 GiB


    seed 42 train  775/1029 3.37 s/state eta 14.3 min peak 54.74 GiB


    seed 42 train  800/1029 3.37 s/state eta 12.9 min peak 54.74 GiB


    seed 42 train  825/1029 3.37 s/state eta 11.5 min peak 54.74 GiB


    seed 42 train  850/1029 3.37 s/state eta 10.1 min peak 54.74 GiB


    seed 42 train  875/1029 3.37 s/state eta 8.7 min peak 54.74 GiB


    seed 42 train  900/1029 3.37 s/state eta 7.2 min peak 54.74 GiB


    seed 42 train  925/1029 3.37 s/state eta 5.8 min peak 54.74 GiB


    seed 42 train  950/1029 3.37 s/state eta 4.4 min peak 54.74 GiB


    seed 42 train  975/1029 3.37 s/state eta 3.0 min peak 54.74 GiB


    seed 42 train 1000/1029 3.37 s/state eta 1.6 min peak 54.74 GiB


    seed 42 train 1025/1029 3.37 s/state eta 0.2 min peak 54.74 GiB


    seed 42 train 1029/1029 3.37 s/state eta 0.0 min peak 54.74 GiB



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 42 dev   25/466 3.34 s/state eta 24.6 min peak 54.74 GiB


    seed 42 dev   50/466 3.37 s/state eta 23.3 min peak 54.74 GiB


    seed 42 dev   75/466 3.44 s/state eta 22.4 min peak 54.74 GiB


    seed 42 dev  100/466 3.45 s/state eta 21.0 min peak 54.74 GiB


    seed 42 dev  125/466 3.43 s/state eta 19.5 min peak 54.74 GiB


    seed 42 dev  150/466 3.41 s/state eta 18.0 min peak 54.74 GiB


    seed 42 dev  175/466 3.43 s/state eta 16.7 min peak 54.74 GiB


    seed 42 dev  200/466 3.45 s/state eta 15.3 min peak 54.74 GiB


    seed 42 dev  225/466 3.44 s/state eta 13.8 min peak 54.74 GiB


    seed 42 dev  250/466 3.44 s/state eta 12.4 min peak 54.74 GiB


    seed 42 dev  275/466 3.43 s/state eta 10.9 min peak 54.74 GiB


    seed 42 dev  300/466 3.44 s/state eta 9.5 min peak 54.74 GiB


    seed 42 dev  325/466 3.43 s/state eta 8.1 min peak 54.74 GiB


    seed 42 dev  350/466 3.43 s/state eta 6.6 min peak 54.74 GiB


    seed 42 dev  375/466 3.41 s/state eta 5.2 min peak 54.74 GiB


    seed 42 dev  400/466 3.41 s/state eta 3.8 min peak 54.74 GiB


    seed 42 dev  425/466 3.41 s/state eta 2.3 min peak 54.74 GiB


    seed 42 dev  450/466 3.40 s/state eta 0.9 min peak 54.74 GiB


    seed 42 dev  466/466 3.40 s/state eta 0.0 min peak 54.74 GiB



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45 train   25/1029 3.38 s/state eta 56.5 min peak 54.74 GiB


    seed 45 train   50/1029 3.36 s/state eta 54.9 min peak 54.74 GiB


    seed 45 train   75/1029 3.35 s/state eta 53.3 min peak 54.74 GiB


    seed 45 train  100/1029 3.34 s/state eta 51.7 min peak 54.74 GiB


    seed 45 train  125/1029 3.36 s/state eta 50.6 min peak 54.74 GiB


    seed 45 train  150/1029 3.36 s/state eta 49.2 min peak 54.74 GiB


    seed 45 train  175/1029 3.34 s/state eta 47.5 min peak 54.74 GiB


    seed 45 train  200/1029 3.34 s/state eta 46.2 min peak 54.74 GiB


    seed 45 train  225/1029 3.34 s/state eta 44.8 min peak 54.74 GiB


    seed 45 train  250/1029 3.34 s/state eta 43.4 min peak 54.74 GiB


    seed 45 train  275/1029 3.34 s/state eta 41.9 min peak 54.74 GiB


    seed 45 train  300/1029 3.33 s/state eta 40.5 min peak 54.74 GiB


    seed 45 train  325/1029 3.34 s/state eta 39.1 min peak 54.74 GiB


    seed 45 train  350/1029 3.33 s/state eta 37.7 min peak 54.74 GiB


    seed 45 train  375/1029 3.33 s/state eta 36.3 min peak 54.74 GiB


    seed 45 train  400/1029 3.34 s/state eta 35.0 min peak 54.74 GiB


    seed 45 train  425/1029 3.33 s/state eta 33.6 min peak 54.74 GiB


    seed 45 train  450/1029 3.33 s/state eta 32.2 min peak 54.74 GiB


    seed 45 train  475/1029 3.33 s/state eta 30.8 min peak 54.74 GiB


    seed 45 train  500/1029 3.33 s/state eta 29.4 min peak 54.74 GiB


    seed 45 train  525/1029 3.33 s/state eta 28.0 min peak 54.74 GiB


    seed 45 train  550/1029 3.33 s/state eta 26.6 min peak 54.74 GiB


    seed 45 train  575/1029 3.33 s/state eta 25.2 min peak 54.74 GiB


    seed 45 train  600/1029 3.33 s/state eta 23.8 min peak 54.74 GiB


    seed 45 train  625/1029 3.33 s/state eta 22.4 min peak 54.74 GiB


    seed 45 train  650/1029 3.33 s/state eta 21.0 min peak 54.74 GiB


    seed 45 train  675/1029 3.33 s/state eta 19.6 min peak 54.74 GiB


    seed 45 train  700/1029 3.32 s/state eta 18.2 min peak 54.74 GiB


    seed 45 train  725/1029 3.33 s/state eta 16.9 min peak 54.74 GiB


    seed 45 train  750/1029 3.33 s/state eta 15.5 min peak 54.74 GiB


    seed 45 train  775/1029 3.33 s/state eta 14.1 min peak 54.74 GiB


    seed 45 train  800/1029 3.33 s/state eta 12.7 min peak 54.74 GiB


    seed 45 train  825/1029 3.33 s/state eta 11.3 min peak 54.74 GiB


    seed 45 train  850/1029 3.33 s/state eta 9.9 min peak 54.74 GiB


    seed 45 train  875/1029 3.33 s/state eta 8.5 min peak 54.74 GiB


    seed 45 train  900/1029 3.33 s/state eta 7.2 min peak 54.74 GiB


    seed 45 train  925/1029 3.33 s/state eta 5.8 min peak 54.74 GiB


    seed 45 train  950/1029 3.33 s/state eta 4.4 min peak 54.74 GiB


    seed 45 train  975/1029 3.33 s/state eta 3.0 min peak 54.74 GiB


    seed 45 train 1000/1029 3.33 s/state eta 1.6 min peak 54.74 GiB


    seed 45 train 1025/1029 3.33 s/state eta 0.2 min peak 54.74 GiB


    seed 45 train 1029/1029 3.33 s/state eta 0.0 min peak 54.74 GiB



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45 dev   25/466 3.35 s/state eta 24.6 min peak 54.74 GiB


    seed 45 dev   50/466 3.37 s/state eta 23.4 min peak 54.74 GiB


    seed 45 dev   75/466 3.44 s/state eta 22.4 min peak 54.74 GiB


    seed 45 dev  100/466 3.45 s/state eta 21.0 min peak 54.74 GiB


    seed 45 dev  125/466 3.43 s/state eta 19.5 min peak 54.74 GiB


    seed 45 dev  150/466 3.41 s/state eta 18.0 min peak 54.74 GiB


    seed 45 dev  175/466 3.43 s/state eta 16.7 min peak 54.74 GiB


    seed 45 dev  200/466 3.44 s/state eta 15.3 min peak 54.74 GiB


    seed 45 dev  225/466 3.44 s/state eta 13.8 min peak 54.74 GiB


    seed 45 dev  250/466 3.43 s/state eta 12.4 min peak 54.74 GiB


    seed 45 dev  275/466 3.43 s/state eta 10.9 min peak 54.74 GiB


    seed 45 dev  300/466 3.43 s/state eta 9.5 min peak 54.74 GiB


    seed 45 dev  325/466 3.43 s/state eta 8.0 min peak 54.74 GiB


    seed 45 dev  350/466 3.42 s/state eta 6.6 min peak 54.74 GiB


    seed 45 dev  375/466 3.41 s/state eta 5.2 min peak 54.74 GiB


    seed 45 dev  400/466 3.40 s/state eta 3.7 min peak 54.74 GiB


    seed 45 dev  425/466 3.40 s/state eta 2.3 min peak 54.74 GiB


    seed 45 dev  450/466 3.39 s/state eta 0.9 min peak 54.74 GiB


    seed 45 dev  466/466 3.39 s/state eta 0.0 min peak 54.74 GiB



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 47 train   25/1029 3.36 s/state eta 56.1 min peak 54.74 GiB


    seed 47 train   50/1029 3.34 s/state eta 54.4 min peak 54.74 GiB


    seed 47 train   75/1029 3.33 s/state eta 52.9 min peak 54.74 GiB


    seed 47 train  100/1029 3.31 s/state eta 51.2 min peak 54.74 GiB


    seed 47 train  125/1029 3.33 s/state eta 50.2 min peak 54.74 GiB


    seed 47 train  150/1029 3.34 s/state eta 48.9 min peak 54.74 GiB


    seed 47 train  175/1029 3.32 s/state eta 47.3 min peak 54.74 GiB


    seed 47 train  200/1029 3.33 s/state eta 46.0 min peak 54.74 GiB


    seed 47 train  225/1029 3.33 s/state eta 44.6 min peak 54.74 GiB


    seed 47 train  250/1029 3.33 s/state eta 43.3 min peak 54.74 GiB


    seed 47 train  275/1029 3.33 s/state eta 41.8 min peak 54.74 GiB


    seed 47 train  300/1029 3.32 s/state eta 40.4 min peak 54.74 GiB


    seed 47 train  325/1029 3.33 s/state eta 39.1 min peak 54.74 GiB


    seed 47 train  350/1029 3.33 s/state eta 37.7 min peak 54.74 GiB


    seed 47 train  375/1029 3.33 s/state eta 36.3 min peak 54.74 GiB


    seed 47 train  400/1029 3.33 s/state eta 34.9 min peak 54.74 GiB


    seed 47 train  425/1029 3.33 s/state eta 33.5 min peak 54.74 GiB


    seed 47 train  450/1029 3.33 s/state eta 32.1 min peak 54.74 GiB


    seed 47 train  475/1029 3.33 s/state eta 30.7 min peak 54.74 GiB


    seed 47 train  500/1029 3.33 s/state eta 29.3 min peak 54.74 GiB


    seed 47 train  525/1029 3.33 s/state eta 27.9 min peak 54.74 GiB


    seed 47 train  550/1029 3.33 s/state eta 26.6 min peak 54.74 GiB


    seed 47 train  575/1029 3.33 s/state eta 25.2 min peak 54.74 GiB


    seed 47 train  600/1029 3.33 s/state eta 23.8 min peak 54.74 GiB


    seed 47 train  625/1029 3.33 s/state eta 22.4 min peak 54.74 GiB


    seed 47 train  650/1029 3.33 s/state eta 21.0 min peak 54.74 GiB


    seed 47 train  675/1029 3.33 s/state eta 19.6 min peak 54.74 GiB


    seed 47 train  700/1029 3.32 s/state eta 18.2 min peak 54.74 GiB


    seed 47 train  725/1029 3.33 s/state eta 16.9 min peak 54.74 GiB


    seed 47 train  750/1029 3.33 s/state eta 15.5 min peak 54.74 GiB


    seed 47 train  775/1029 3.33 s/state eta 14.1 min peak 54.74 GiB


    seed 47 train  800/1029 3.33 s/state eta 12.7 min peak 54.74 GiB


    seed 47 train  825/1029 3.33 s/state eta 11.3 min peak 54.74 GiB


    seed 47 train  850/1029 3.33 s/state eta 9.9 min peak 54.74 GiB


    seed 47 train  875/1029 3.33 s/state eta 8.5 min peak 54.74 GiB


    seed 47 train  900/1029 3.33 s/state eta 7.2 min peak 54.74 GiB


    seed 47 train  925/1029 3.33 s/state eta 5.8 min peak 54.74 GiB


    seed 47 train  950/1029 3.33 s/state eta 4.4 min peak 54.74 GiB


    seed 47 train  975/1029 3.33 s/state eta 3.0 min peak 54.74 GiB


    seed 47 train 1000/1029 3.33 s/state eta 1.6 min peak 54.74 GiB


    seed 47 train 1025/1029 3.33 s/state eta 0.2 min peak 54.74 GiB


    seed 47 train 1029/1029 3.33 s/state eta 0.0 min peak 54.74 GiB



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 47 dev   25/466 3.34 s/state eta 24.6 min peak 54.74 GiB


    seed 47 dev   50/466 3.38 s/state eta 23.4 min peak 54.74 GiB


    seed 47 dev   75/466 3.44 s/state eta 22.4 min peak 54.74 GiB


    seed 47 dev  100/466 3.45 s/state eta 21.1 min peak 54.74 GiB


    seed 47 dev  125/466 3.43 s/state eta 19.5 min peak 54.74 GiB


    seed 47 dev  150/466 3.41 s/state eta 18.0 min peak 54.74 GiB


    seed 47 dev  175/466 3.44 s/state eta 16.7 min peak 54.74 GiB


    seed 47 dev  200/466 3.44 s/state eta 15.3 min peak 54.74 GiB


    seed 47 dev  225/466 3.44 s/state eta 13.8 min peak 54.74 GiB


    seed 47 dev  250/466 3.43 s/state eta 12.4 min peak 54.74 GiB


    seed 47 dev  275/466 3.43 s/state eta 10.9 min peak 54.74 GiB


    seed 47 dev  300/466 3.43 s/state eta 9.5 min peak 54.74 GiB


    seed 47 dev  325/466 3.42 s/state eta 8.0 min peak 54.74 GiB


    seed 47 dev  350/466 3.42 s/state eta 6.6 min peak 54.74 GiB


    seed 47 dev  375/466 3.40 s/state eta 5.2 min peak 54.74 GiB


    seed 47 dev  400/466 3.40 s/state eta 3.7 min peak 54.74 GiB


    seed 47 dev  425/466 3.40 s/state eta 2.3 min peak 54.74 GiB


    seed 47 dev  450/466 3.39 s/state eta 0.9 min peak 54.74 GiB


    seed 47 dev  466/466 3.44 s/state eta 0.0 min peak 54.74 GiB


    incumbent rankings ready in 251.5 min



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
      <th>split</th>
      <th>states</th>
      <th>top_k</th>
      <th>indices_sha256</th>
      <th>mean_top1_score</th>
      <th>mean_top1_minus_top2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>train</td>
      <td>1029</td>
      <td>32</td>
      <td>db01957518d3e574</td>
      <td>-2.548755</td>
      <td>0.840528</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>dev</td>
      <td>466</td>
      <td>32</td>
      <td>577fed439a9c57db</td>
      <td>-2.809154</td>
      <td>0.758600</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45</td>
      <td>train</td>
      <td>1029</td>
      <td>32</td>
      <td>3f0b03974e76d7e6</td>
      <td>-2.518580</td>
      <td>0.863687</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>dev</td>
      <td>466</td>
      <td>32</td>
      <td>1eccfc54da44f644</td>
      <td>-2.808092</td>
      <td>0.747707</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>train</td>
      <td>1029</td>
      <td>32</td>
      <td>0ecf5139a0384510</td>
      <td>-2.534688</td>
      <td>0.859880</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>dev</td>
      <td>466</td>
      <td>32</td>
      <td>27806b1f007a49c7</td>
      <td>-2.789192</td>
      <td>0.792683</td>
    </tr>
  </tbody>
</table>
</div>


## 19.12 Twelve seed-specific actions per state

Every state is scored on exactly 12 answer-list actions. Ten of them come from
ranked lists and two from a deterministic fill, and the support is frozen the
moment it is built: a seed's `hard` and `value` arms consume the identical list,
in the identical order, for training and for dev scoring.

**Broad states** take the 4 highest-entropy open actions, the 2 highest-entropy
current candidates, the 4 highest-ranked incumbent actions, and a deterministic
fill. Deduplication never drops the global open optimum or the canonical
candidate teacher, because both are the first entry of their own ranked list;
when categories overlap, each list is walked farther down so the support still
reaches exactly 12 distinct words. Open regret therefore stays measurable inside
the support.

**Sharp states** take all 1-2 current candidates, any previous guess that is an
answer word, the 4 highest-ranked incumbent actions that are neither a candidate
nor an already-included previous guess, high lexical-overlap non-candidate
distractors, and a deterministic fill. Those four incumbent actions are exactly
the words this seed would rather emit than close the game, and the sharp target
gives them zero mass.

Only the incumbent block is seed-dependent. The entropy blocks, the distractors
and the fill stream are functions of the state alone, and the fill stream is
seeded from a SHA-256 digest of the state key rather than Python's salted
`hash`, so it is stable across processes. That keeps the cross-seed difference
attributable to one thing: which words that seed's own policy prefers.


```python
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


def build_action_support(state_key: str, ranking: np.ndarray) -> dict:
    history = parse_state_key(state_key)
    candidates = candidate_indices_from_history(history)
    candidate_set = {int(index) for index in candidates}
    regime = "broad" if len(candidates) >= BROAD_THRESHOLD else "sharp"
    previous_guesses = [turn.guess for turn in history]
    previous_indices = [
        WORD_TO_INDEX[word] for word in previous_guesses if word in ANSWER_SET
    ]
    support: list[int] = []

    def add(index) -> bool:
        index = int(index)
        if index in support:
            return False
        support.append(index)
        return True

    def walk(order, quota: int, accept=None) -> list[int]:
        # Overlapping categories are resolved by walking farther down the
        # ranked list, never by shrinking the support.
        taken = []
        for index in order:
            if len(taken) == quota:
                break
            index = int(index)
            if accept is not None and not accept(index):
                continue
            if add(index):
                taken.append(index)
        return taken

    def require_incumbent(taken: list[int], quota: int) -> None:
        if len(taken) < quota:
            raise RuntimeError(
                f"top-{MINING_TOP_K} incumbent ranking exhausted at "
                f"{state_key!r}: {len(taken)} of {quota} hard negatives found. "
                "Increase MINING_TOP_K and remine."
            )

    incumbent_top1 = int(ranking[0])
    if regime == "broad":
        entropies = action_entropies(candidates)
        open_order = utility_order(ALL_INDICES, entropies)
        candidate_order = utility_order(candidates, entropies)
        open_teacher = open_order[0]
        candidate_teacher = candidate_order[0]
        walk(open_order, BROAD_OPEN_SLOTS)
        walk(candidate_order, BROAD_CANDIDATE_SLOTS)
        incumbent_taken = walk(ranking, BROAD_INCUMBENT_SLOTS)
        require_incumbent(incumbent_taken, BROAD_INCUMBENT_SLOTS)
        required = {open_teacher, candidate_teacher, incumbent_top1}
    else:
        entropies = None
        open_teacher = None
        candidate_teacher = int(min(candidate_set, key=lambda i: ANSWERS[i]))
        for index in sorted(candidate_set, key=lambda i: ANSWERS[i]):
            add(index)
        for index in previous_indices:
            add(index)
        previous_set = set(previous_indices)
        incumbent_taken = walk(
            ranking,
            SHARP_INCUMBENT_SLOTS,
            accept=lambda index: (
                index not in candidate_set and index not in previous_set
            ),
        )
        require_incumbent(incumbent_taken, SHARP_INCUMBENT_SLOTS)
        assert len(support) <= SUPPORT_SIZE, (
            f"mandatory sharp actions overflow the support at {state_key!r}"
        )
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
        required = set(candidate_set) | set(incumbent_taken)

    rng = support_rng(state_key)
    for index in rng.permutation(len(ANSWERS)):
        if len(support) >= SUPPORT_SIZE:
            break
        add(index)

    assert len(support) == SUPPORT_SIZE
    assert len(set(support)) == SUPPORT_SIZE
    assert required <= set(support), "required action missing from support"
    assert set(incumbent_taken) <= set(support), (
        "selected incumbent hard negative missing from support"
    )
    assert incumbent_top1 in set(support), (
        "incumbent top-1 action missing from support"
    )
    return {
        "state_key": state_key,
        "regime": regime,
        "candidate_count": len(candidates),
        "candidate_set": candidate_set,
        "support": support,
        "entropies": entropies,
        "open_teacher": open_teacher,
        "candidate_teacher": candidate_teacher,
        "previous_guesses": previous_guesses,
        "incumbent_top1": incumbent_top1,
        "incumbent_taken": incumbent_taken,
    }
```

## 19.13 The value target and its matched one-hot control

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

The utility rule is untouched by mining. A mined hard negative is scored by the
same entropy or candidacy function as every other action in its support, so it
receives whatever probability it deserves and nothing is hand-tuned. In a broad
state a low-entropy incumbent favourite simply lands far down the softmax; in a
sharp state a non-candidate incumbent favourite gets exactly zero.


```python
def build_target_record(
    seed: int, split: str, row: dict, ranking: np.ndarray
) -> dict:
    state_key = row["state_key"]
    support_record = build_action_support(state_key, ranking)
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
    incumbent_top1 = support_record["incumbent_top1"]
    incumbent_positions = [
        support.index(index) for index in support_record["incumbent_taken"]
    ]
    previous_words = set(support_record["previous_guesses"])
    return {
        "seed": seed,
        "split": split,
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
            if ANSWERS[index] in previous_words
        ],
        "incumbent_positions": incumbent_positions,
        "incumbent_actions": [
            ANSWERS[index] for index in support_record["incumbent_taken"]
        ],
        "incumbent_top1_action": ANSWERS[incumbent_top1],
        "incumbent_top1_position": support.index(incumbent_top1),
        "incumbent_top1_is_candidate": incumbent_top1 in candidate_set,
        "incumbent_top1_is_repeat": ANSWERS[incumbent_top1] in previous_words,
        "incumbent_top1_is_open_teacher": (
            incumbent_top1 == support_record["open_teacher"]
        ),
        "incumbent_top1_is_candidate_teacher": (
            incumbent_top1 == support_record["candidate_teacher"]
        ),
        "incumbent_top1_regret_bits": (
            float(
                support_record["entropies"].max()
                - support_record["entropies"][incumbent_top1]
            )
            if support_record["entropies"] is not None
            else float("nan")
        ),
        "incumbent_target_mass": float(
            value_probs[incumbent_positions].sum()
        ),
        "incumbent_top1_target_probability": float(
            value_probs[support.index(incumbent_top1)]
        ),
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
targets_by_seed = {}
for seed in SEEDS:
    for split in SPLITS:
        frame = STATE_FRAMES[split]
        indices, _ = incumbent_rankings[(seed, split)]
        targets_by_seed[(seed, split)] = [
            build_target_record(
                seed, split, frame.loc[position].to_dict(), indices[position]
            )
            for position in range(len(frame))
        ]
all_targets = [
    record
    for seed in SEEDS
    for split in SPLITS
    for record in targets_by_seed[(seed, split)]
]
for seed in SEEDS:
    for split in SPLITS:
        records = targets_by_seed[(seed, split)]
        assert [record["state_key"] for record in records] == list(
            STATE_FRAMES[split]["state_key"]
        ), f"seed {seed} {split} targets drifted from the frozen state order"
print(
    f"built {len(all_targets)} targets over {len(SEEDS)} seeds "
    f"({STATE_COUNTS['train']} train and {STATE_COUNTS['dev']} dev per seed) "
    f"in {time.perf_counter() - started:.1f}s"
)
```

    built 4485 targets over 3 seeds (1029 train and 466 dev per seed) in 18.8s


## 19.14 Persist the target data

The generated targets are an artifact, not a side effect of a run. Each seed
gets its own train and dev file under `data/generated`, named
`lab19-value-targets-seed42-train.jsonl` and so on, because each seed's supports
are its own. One manifest covers all six files and carries every hash the
experiment depends on: the structured source, each incumbent adapter, the mined
ranking arrays, the scorer and action-order fingerprint, the shared state order,
and per-seed fingerprints of the ordered training stream and dev support. If a
manifest already exists it must match exactly, which is what makes a resumed or
repeated run provably the same experiment.


```python
def jsonl_payload(records: list[dict]) -> str:
    return "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )


def support_fingerprint(records: list[dict]) -> str:
    return sha256_text("".join(
        record["state_key"] + "|" + ",".join(record["actions"]) + "\n"
        for record in records
    ))


target_payloads = {
    (seed, split): jsonl_payload(targets_by_seed[(seed, split)])
    for seed in SEEDS
    for split in SPLITS
}
support_fingerprints = {
    (seed, split): support_fingerprint(targets_by_seed[(seed, split)])
    for seed in SEEDS
    for split in SPLITS
}
reference_train = targets_by_seed[(SEEDS[0], "train")]
target_manifest = {
    "experiment": "Lab 19 seed-specific value-aware distillation targets",
    "representation": "derived_state_v1",
    "support_source": "per-seed frozen incumbent full-answer-list ranking",
    "source_sha256": structured_hashes,
    "incumbent_sha256": {
        str(seed): incumbent_hashes[seed] for seed in SEEDS
    },
    "scorer_sha256": SCORER_SHA256,
    "state_order_sha256": STATE_ORDER_SHA256,
    "counts": {
        "seeds": len(SEEDS),
        "train_states": STATE_COUNTS["train"],
        "dev_states": STATE_COUNTS["dev"],
        "broad": sum(1 for r in reference_train if r["regime"] == "broad"),
        "sharp": sum(1 for r in reference_train if r["regime"] == "sharp"),
        "singleton": sum(
            1 for r in reference_train if r["candidate_count"] == 1
        ),
        "two_candidate": sum(
            1 for r in reference_train if r["candidate_count"] == 2
        ),
    },
    "config": {
        "support_size": SUPPORT_SIZE,
        "broad_open_slots": BROAD_OPEN_SLOTS,
        "broad_candidate_slots": BROAD_CANDIDATE_SLOTS,
        "broad_incumbent_slots": BROAD_INCUMBENT_SLOTS,
        "sharp_incumbent_slots": SHARP_INCUMBENT_SLOTS,
        "sharp_random_slots": SHARP_RANDOM_SLOTS,
        "mining_top_k": MINING_TOP_K,
        "temperature": TEACHER_TEMPERATURE,
        "broad_threshold": BROAD_THRESHOLD,
        "priority_candidate_count": PRIORITY_CANDIDATE_COUNT,
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "action_space": "2,315 answer words",
    },
    "seeds": {
        str(seed): {
            "targets_sha256": {
                split: sha256_text(target_payloads[(seed, split)])
                for split in SPLITS
            },
            "support_sha256": {
                split: support_fingerprints[(seed, split)]
                for split in SPLITS
            },
            "ranking_indices_sha256": {
                split: ranking_hashes[(seed, split)] for split in SPLITS
            },
        }
        for seed in SEEDS
    },
}

if TARGET_MANIFEST.exists():
    existing = json.loads(TARGET_MANIFEST.read_text())
    assert existing == target_manifest, (
        "existing Lab 19 target manifest disagrees with regenerated targets"
    )
for seed in SEEDS:
    for split in SPLITS:
        atomic_write(target_payloads[(seed, split)], TARGET_FILES[(seed, split)])
atomic_json(target_manifest, TARGET_MANIFEST)
for seed in SEEDS:
    for split in SPLITS:
        assert sha256_file(TARGET_FILES[(seed, split)]) == (
            target_manifest["seeds"][str(seed)]["targets_sha256"][split]
        )
print(json.dumps(target_manifest["counts"], indent=2))
display(pd.DataFrame([
    {
        "seed": seed,
        "split": split,
        "file": TARGET_FILES[(seed, split)].name,
        "targets_sha256": target_manifest["seeds"][str(seed)][
            "targets_sha256"
        ][split][:16],
        "support_sha256": support_fingerprints[(seed, split)][:16],
    }
    for seed in SEEDS
    for split in SPLITS
]))
distinct_supports = {
    split: len({support_fingerprints[(seed, split)] for seed in SEEDS})
    for split in SPLITS
}
print(
    "distinct support fingerprints across seeds:",
    json.dumps(distinct_supports),
)
```

    {
      "seeds": 3,
      "train_states": 1029,
      "dev_states": 466,
      "broad": 686,
      "sharp": 343,
      "singleton": 229,
      "two_candidate": 114
    }



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
      <th>split</th>
      <th>file</th>
      <th>targets_sha256</th>
      <th>support_sha256</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>train</td>
      <td>lab19-value-targets-seed42-train.jsonl</td>
      <td>44878c552125ebd8</td>
      <td>78ee7d12e12762d9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>dev</td>
      <td>lab19-value-targets-seed42-dev.jsonl</td>
      <td>de423ab5015d744f</td>
      <td>c396167a0e534f48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45</td>
      <td>train</td>
      <td>lab19-value-targets-seed45-train.jsonl</td>
      <td>28dd14a7aa0502bd</td>
      <td>02034321034c69eb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>dev</td>
      <td>lab19-value-targets-seed45-dev.jsonl</td>
      <td>f4a37579737c3190</td>
      <td>30f347131065682d</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>train</td>
      <td>lab19-value-targets-seed47-train.jsonl</td>
      <td>f8043cf7c3b15a64</td>
      <td>dd7360dadce5ceac</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>dev</td>
      <td>lab19-value-targets-seed47-dev.jsonl</td>
      <td>557d7b54cc523954</td>
      <td>99a1d147c7bf0c19</td>
    </tr>
  </tbody>
</table>
</div>


    distinct support fingerprints across seeds: {"train": 3, "dev": 3}


## 19.15 Pre-training target audit

Every check below is a property of the targets alone, evaluated once per seed.
Each can fail in a way that would silently invalidate the experiment: a
duplicated action, a target that does not sum to one, a hard label that is not
the argmax of its own utility, a value target so peaked that the two arms are
the same experiment, a mined hard negative that never made it into the support
it was chosen for, or a held-out state that leaked into training.


```python
target_frame = pd.DataFrame([
    {
        key: value
        for key, value in record.items()
        if key not in {"prompt", "actions", "action_indices", "utilities",
                       "value_target", "candidate_positions", "repeat_positions",
                       "incumbent_positions", "incumbent_actions"}
    }
    | {
        "actions_in_support": len(set(record["actions"])),
        "candidates_in_support": len(record["candidate_positions"]),
        "repeats_in_support": len(record["repeat_positions"]),
        "incumbent_in_support": len(record["incumbent_positions"]),
        "target_sum": float(np.sum(record["value_target"])),
        "effective_actions_1pct": int(
            np.sum(np.array(record["value_target"]) >= 0.01)
        ),
    }
    for record in all_targets
])

for record in all_targets:
    probabilities = np.array(record["value_target"])
    utilities = np.array(record["utilities"])
    assert len(record["actions"]) == SUPPORT_SIZE
    assert len(set(record["actions"])) == SUPPORT_SIZE
    assert all(word in ANSWER_SET for word in record["actions"])
    assert abs(probabilities.sum() - 1.0) < 1e-9
    assert (probabilities >= 0.0).all()
    assert record["actions"][record["hard_index"]] == record["hard_action"]
    assert record["incumbent_top1_action"] in record["actions"]
    assert record["actions"][record["incumbent_top1_position"]] == (
        record["incumbent_top1_action"]
    )
    assert set(record["incumbent_actions"]) <= set(record["actions"])
    assert [
        record["actions"][position]
        for position in record["incumbent_positions"]
    ] == record["incumbent_actions"]
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
        assert len(record["incumbent_actions"]) == BROAD_INCUMBENT_SLOTS
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
        assert len(record["incumbent_actions"]) == SHARP_INCUMBENT_SLOTS
        # Sharp hard negatives are chosen to be wrong: never a candidate, never
        # a previous guess already in the support, and always zero-mass.
        assert not set(record["incumbent_positions"]) & set(
            record["candidate_positions"]
        )
        assert not set(record["incumbent_positions"]) & set(
            record["repeat_positions"]
        )
        assert probabilities[record["incumbent_positions"]].sum() == 0.0
    assert not set(record["candidate_positions"]) & set(
        record["repeat_positions"]
    )

assert target_frame["target_sum"].sub(1.0).abs().max() < 1e-9
assert (target_frame["actions_in_support"] == SUPPORT_SIZE).all()
print(
    "structural target checks passed for", len(target_frame),
    "seed-state targets",
)

regime_audit = target_frame.groupby(
    ["seed", "split", "regime"], sort=True
).agg(
    states=("state_key", "size"),
    mean_candidates=("candidate_count", "mean"),
    median_candidates=("candidate_count", "median"),
    max_candidates=("candidate_count", "max"),
    turn2_states=("turn", lambda values: int((values == 2).sum())),
    mean_turn=("turn", "mean"),
    mean_candidates_in_support=("candidates_in_support", "mean"),
    mean_repeats_in_support=("repeats_in_support", "mean"),
    mean_incumbent_in_support=("incumbent_in_support", "mean"),
).reset_index()
display(regime_audit)
```

    structural target checks passed for 4485 seed-state targets



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
      <th>split</th>
      <th>regime</th>
      <th>states</th>
      <th>mean_candidates</th>
      <th>median_candidates</th>
      <th>max_candidates</th>
      <th>turn2_states</th>
      <th>mean_turn</th>
      <th>mean_candidates_in_support</th>
      <th>mean_repeats_in_support</th>
      <th>mean_incumbent_in_support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>12.908257</td>
      <td>5.0</td>
      <td>752</td>
      <td>25</td>
      <td>3.107034</td>
      <td>2.877676</td>
      <td>0.033639</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>1.187050</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>3.597122</td>
      <td>1.187050</td>
      <td>2.597122</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>15.960641</td>
      <td>5.0</td>
      <td>782</td>
      <td>172</td>
      <td>2.918367</td>
      <td>3.151603</td>
      <td>0.055394</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>1.332362</td>
      <td>1.0</td>
      <td>2</td>
      <td>9</td>
      <td>3.731778</td>
      <td>1.332362</td>
      <td>2.731778</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>12.908257</td>
      <td>5.0</td>
      <td>752</td>
      <td>25</td>
      <td>3.107034</td>
      <td>2.941896</td>
      <td>0.033639</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>1.187050</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>3.597122</td>
      <td>1.187050</td>
      <td>2.597122</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>15.960641</td>
      <td>5.0</td>
      <td>782</td>
      <td>172</td>
      <td>2.918367</td>
      <td>3.188047</td>
      <td>0.053936</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>1.332362</td>
      <td>1.0</td>
      <td>2</td>
      <td>9</td>
      <td>3.731778</td>
      <td>1.332362</td>
      <td>2.731778</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>12.908257</td>
      <td>5.0</td>
      <td>752</td>
      <td>25</td>
      <td>3.107034</td>
      <td>2.932722</td>
      <td>0.045872</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>47</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>1.187050</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>3.597122</td>
      <td>1.187050</td>
      <td>2.597122</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>15.960641</td>
      <td>5.0</td>
      <td>782</td>
      <td>172</td>
      <td>2.918367</td>
      <td>3.188047</td>
      <td>0.051020</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>47</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>1.332362</td>
      <td>1.0</td>
      <td>2</td>
      <td>9</td>
      <td>3.731778</td>
      <td>1.332362</td>
      <td>2.731778</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
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
      <th>bucket</th>
      <th>available</th>
      <th>selected</th>
      <th>selected_share_of_available</th>
      <th>selected_share_of_training</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1405</td>
      <td>229</td>
      <td>0.162989</td>
      <td>0.222546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>336</td>
      <td>114</td>
      <td>0.339286</td>
      <td>0.110787</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3-10</td>
      <td>1123</td>
      <td>451</td>
      <td>0.401603</td>
      <td>0.438290</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11+</td>
      <td>235</td>
      <td>235</td>
      <td>1.000000</td>
      <td>0.228377</td>
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
      <th>turn</th>
      <th>available</th>
      <th>selected</th>
      <th>selected_share_of_available</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>207</td>
      <td>181</td>
      <td>0.874396</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1615</td>
      <td>551</td>
      <td>0.341176</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1007</td>
      <td>230</td>
      <td>0.228401</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>220</td>
      <td>55</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>50</td>
      <td>12</td>
      <td>0.240000</td>
    </tr>
  </tbody>
</table>
</div>


    all available broad Turn 2 states covered: True
    all available 11+ candidate states covered: True



```python
shape_audit = target_frame.groupby(
    ["seed", "split", "regime"], sort=True
).agg(
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

for seed in SEEDS:
    train_frame = target_frame.query("split == 'train' and seed == @seed")
    identical = train_frame["arm_total_variation"] < 1e-12
    identical_states = int(identical.sum())
    print(
        f"seed {seed}: hard and value supply identical targets on "
        f"{identical_states} of {len(train_frame)} training states "
        f"({identical_states / len(train_frame):.1%}); every one is a "
        f"singleton: "
        f"{bool((train_frame.loc[identical, 'candidate_count'] == 1).all())}; "
        "mean total variation elsewhere "
        f"{float(train_frame.loc[~identical, 'arm_total_variation'].mean()):.4f}"
    )
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
      <th>split</th>
      <th>regime</th>
      <th>states</th>
      <th>mean_max_probability</th>
      <th>median_max_probability</th>
      <th>mean_target_entropy_bits</th>
      <th>mean_effective_support</th>
      <th>mean_actions_above_1pct</th>
      <th>mean_hard_value_total_variation</th>
      <th>min_hard_value_total_variation</th>
      <th>median_utility_tie_count</th>
      <th>states_with_five_plus_ties</th>
      <th>hard_candidate_rate</th>
      <th>identical_arm_states</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>0.163936</td>
      <td>0.161682</td>
      <td>3.087782</td>
      <td>8.576690</td>
      <td>9.905199</td>
      <td>0.836064</td>
      <td>0.618688</td>
      <td>4.0</td>
      <td>155</td>
      <td>0.507645</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.906475</td>
      <td>1.000000</td>
      <td>0.187050</td>
      <td>1.187050</td>
      <td>1.187050</td>
      <td>0.093525</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0</td>
      <td>1.000000</td>
      <td>113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>0.173155</td>
      <td>0.175401</td>
      <td>3.063024</td>
      <td>8.432546</td>
      <td>9.908163</td>
      <td>0.826845</td>
      <td>0.607769</td>
      <td>4.0</td>
      <td>241</td>
      <td>0.362974</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>0.833819</td>
      <td>1.000000</td>
      <td>0.332362</td>
      <td>1.332362</td>
      <td>1.332362</td>
      <td>0.166181</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0</td>
      <td>1.000000</td>
      <td>229</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>0.163643</td>
      <td>0.160328</td>
      <td>3.086166</td>
      <td>8.566953</td>
      <td>9.819572</td>
      <td>0.836357</td>
      <td>0.629217</td>
      <td>4.0</td>
      <td>157</td>
      <td>0.507645</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.906475</td>
      <td>1.000000</td>
      <td>0.187050</td>
      <td>1.187050</td>
      <td>1.187050</td>
      <td>0.093525</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0</td>
      <td>1.000000</td>
      <td>113</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>0.172340</td>
      <td>0.174700</td>
      <td>3.070922</td>
      <td>8.482835</td>
      <td>9.978134</td>
      <td>0.827660</td>
      <td>0.607769</td>
      <td>4.0</td>
      <td>244</td>
      <td>0.362974</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>0.833819</td>
      <td>1.000000</td>
      <td>0.332362</td>
      <td>1.332362</td>
      <td>1.332362</td>
      <td>0.166181</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0</td>
      <td>1.000000</td>
      <td>229</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>0.163541</td>
      <td>0.159088</td>
      <td>3.089406</td>
      <td>8.585645</td>
      <td>9.865443</td>
      <td>0.836459</td>
      <td>0.629217</td>
      <td>4.0</td>
      <td>155</td>
      <td>0.507645</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>47</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.906475</td>
      <td>1.000000</td>
      <td>0.187050</td>
      <td>1.187050</td>
      <td>1.187050</td>
      <td>0.093525</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0</td>
      <td>1.000000</td>
      <td>113</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>0.172608</td>
      <td>0.174796</td>
      <td>3.069535</td>
      <td>8.473236</td>
      <td>9.978134</td>
      <td>0.827392</td>
      <td>0.607769</td>
      <td>4.0</td>
      <td>244</td>
      <td>0.362974</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>47</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>0.833819</td>
      <td>1.000000</td>
      <td>0.332362</td>
      <td>1.332362</td>
      <td>1.332362</td>
      <td>0.166181</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0</td>
      <td>1.000000</td>
      <td>229</td>
    </tr>
  </tbody>
</table>
</div>


    seed 42: hard and value supply identical targets on 229 of 1029 training states (22.3%); every one is a singleton: True; mean total variation elsewhere 0.7803
    seed 45: hard and value supply identical targets on 229 of 1029 training states (22.3%); every one is a singleton: True; mean total variation elsewhere 0.7810
    seed 47: hard and value supply identical targets on 229 of 1029 training states (22.3%); every one is a singleton: True; mean total variation elsewhere 0.7807



```python
support_matrix = {
    (seed, split): [
        tuple(record["actions"]) for record in targets_by_seed[(seed, split)]
    ]
    for seed in SEEDS
    for split in SPLITS
}
divergence_rows = []
for split in SPLITS:
    for position in range(STATE_COUNTS[split]):
        supports = [
            set(support_matrix[(seed, split)][position]) for seed in SEEDS
        ]
        shared = set.intersection(*supports)
        union = set.union(*supports)
        divergence_rows.append({
            "split": split,
            "regime": targets_by_seed[(SEEDS[0], split)][position]["regime"],
            "identical_across_seeds": len(
                {support_matrix[(seed, split)][position] for seed in SEEDS}
            ) == 1,
            "shared_actions": len(shared),
            "jaccard": len(shared) / len(union),
        })
divergence = pd.DataFrame(divergence_rows)
display(
    divergence.groupby(["split", "regime"], sort=True).agg(
        states=("jaccard", "size"),
        identical_support_rate=("identical_across_seeds", "mean"),
        mean_shared_actions=("shared_actions", "mean"),
        min_shared_actions=("shared_actions", "min"),
        mean_jaccard=("jaccard", "mean"),
    ).reset_index()
)
print(
    "The entropy, previous-guess, and distractor blocks are computed from the "
    "state alone, so they are identical across seeds. The mined block is not, "
    "and the deterministic fill can be nudged by it: a fill word one seed "
    "already mined is skipped and replaced by the next word in that seed's "
    "fill stream. Everything else is shared, which is why the intersection "
    "cannot fall below the floor asserted next."
)
support_floor = SUPPORT_SIZE - max(
    BROAD_INCUMBENT_SLOTS, SHARP_INCUMBENT_SLOTS
) - SHARP_RANDOM_SLOTS
assert (divergence["shared_actions"] >= support_floor).all()
print(
    f"floor {support_floor}; observed minimum "
    f"{int(divergence['shared_actions'].min())} shared actions"
)
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
      <th>split</th>
      <th>regime</th>
      <th>states</th>
      <th>identical_support_rate</th>
      <th>mean_shared_actions</th>
      <th>min_shared_actions</th>
      <th>mean_jaccard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>0.000000</td>
      <td>10.318043</td>
      <td>8</td>
      <td>0.746989</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.014388</td>
      <td>10.496403</td>
      <td>8</td>
      <td>0.778207</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>0.002915</td>
      <td>10.279883</td>
      <td>8</td>
      <td>0.741908</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>0.029155</td>
      <td>10.489796</td>
      <td>8</td>
      <td>0.774139</td>
    </tr>
  </tbody>
</table>
</div>


    The entropy, previous-guess, and distractor blocks are computed from the state alone, so they are identical across seeds. The mined block is not, and the deterministic fill can be nudged by it: a fill word one seed already mined is skipped and replaced by the next word in that seed's fill stream. Everything else is shared, which is why the intersection cannot fall below the floor asserted next.
    floor 6; observed minimum 8 shared actions



```python
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
        "is_incumbent_pick": [
            position in set(record["incumbent_positions"])
            for position in range(SUPPORT_SIZE)
        ],
    }).sort_values("value_target", ascending=False)


train_targets = targets_by_seed[(SEEDS[0], "train")]
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
        f"--- seed {SEEDS[0]} {label}: turn {record['turn']}, "
        f"{record['candidate_count']} candidates, {record['regime']} ---"
    )
    print(record["state_key"].replace(chr(10), "  |  "))
    display(preview(record))
```

    --- seed 42 broadest Turn 2 state: turn 2, 782 candidates, broad ---
    F J O R D -> B B B B B



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
      <th>action</th>
      <th>utility</th>
      <th>value_target</th>
      <th>hard_target</th>
      <th>is_candidate</th>
      <th>is_previous_guess</th>
      <th>is_incumbent_pick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SLATE</td>
      <td>6.3702</td>
      <td>0.2037</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>STALE</td>
      <td>6.2813</td>
      <td>0.1705</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LEAST</td>
      <td>6.2043</td>
      <td>0.1462</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SHALE</td>
      <td>6.1464</td>
      <td>0.1302</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AISLE</td>
      <td>6.1059</td>
      <td>0.1201</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SCALE</td>
      <td>6.0802</td>
      <td>0.1141</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SLAIN</td>
      <td>5.9009</td>
      <td>0.0797</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BLANK</td>
      <td>4.9707</td>
      <td>0.0124</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LINEN</td>
      <td>4.8800</td>
      <td>0.0103</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BETEL</td>
      <td>4.8096</td>
      <td>0.0090</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SWIRL</td>
      <td>4.2248</td>
      <td>0.0028</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FATTY</td>
      <td>3.7392</td>
      <td>0.0011</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    --- seed 42 mid-size broad state: turn 3, 5 candidates, broad ---
    F J O R D -> B B B B B  |  S L A T E -> B B B G G



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
      <th>action</th>
      <th>utility</th>
      <th>value_target</th>
      <th>hard_target</th>
      <th>is_candidate</th>
      <th>is_previous_guess</th>
      <th>is_incumbent_pick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AUDIO</td>
      <td>2.3219</td>
      <td>0.1618</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AUDIT</td>
      <td>2.3219</td>
      <td>0.1618</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BENCH</td>
      <td>2.3219</td>
      <td>0.1618</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BLUNT</td>
      <td>2.3219</td>
      <td>0.1618</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>QUITE</td>
      <td>2.3219</td>
      <td>0.1618</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BUTTE</td>
      <td>1.9219</td>
      <td>0.0727</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>UNITE</td>
      <td>1.9219</td>
      <td>0.0727</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CHUTE</td>
      <td>1.3710</td>
      <td>0.0242</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BEGET</td>
      <td>0.7219</td>
      <td>0.0066</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>WEAVE</td>
      <td>0.7219</td>
      <td>0.0066</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AMBLE</td>
      <td>0.7219</td>
      <td>0.0066</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ALERT</td>
      <td>-0.0000</td>
      <td>0.0016</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    --- seed 42 two-candidate state: turn 5, 2 candidates, sharp ---
    R A I S E -> B B B B B  |  M U L C H -> B B B B B  |  G O O D Y -> B G G B G  |  B O O B Y -> G G G B G



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
      <th>action</th>
      <th>utility</th>
      <th>value_target</th>
      <th>hard_target</th>
      <th>is_candidate</th>
      <th>is_previous_guess</th>
      <th>is_incumbent_pick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BOOTY</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BOOZY</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RAISE</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MULCH</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GOODY</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BOOBY</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BOWEL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BOBBY</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BOOTH</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BOOZE</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>HORSE</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>EAGER</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    --- seed 42 singleton state: turn 5, 1 candidates, sharp ---
    R A I S E -> B B B B B  |  M U L C H -> B B B B B  |  G O O D Y -> B G B B G  |  B O B B Y -> B G B B G



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
      <th>action</th>
      <th>utility</th>
      <th>value_target</th>
      <th>hard_target</th>
      <th>is_candidate</th>
      <th>is_previous_guess</th>
      <th>is_incumbent_pick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POPPY</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RAISE</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MULCH</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GOODY</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BOBBY</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>WOMEN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>WOODY</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NOVEL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HOWDY</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PUPPY</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RERUN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PRUNE</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python
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
for record in all_targets:
    answer = hidden_answers[record["state_key"]]
    in_support = answer in set(record["actions"])
    answer_visibility_rows.append({
        "seed": record["seed"],
        "split": record["split"],
        "regime": record["regime"],
        "answer_in_support": in_support,
        # The hidden answer may only reach a support by being a state-derived
        # candidate. A mined action can reach it too, but only because that
        # word is a candidate; any other route would mean the answer leaked in.
        "only_via_candidacy": (
            not in_support
            or record["actions"].index(answer)
            in set(record["candidate_positions"])
        ),
    })
answer_visibility = pd.DataFrame(answer_visibility_rows).groupby(
    ["seed", "split", "regime"]
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
      <th>split</th>
      <th>regime</th>
      <th>states</th>
      <th>answer_in_support_rate</th>
      <th>only_via_candidacy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>0.581040</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>1.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>0.527697</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>1.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>0.590214</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>1.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>0.520408</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>1.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>0.577982</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>47</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>1.000000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>0.527697</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>47</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>1.000000</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    The hidden answer is not read when building actions or targets. The table is a visibility diagnostic; the held-out guarantee comes from the reserved-answer reachability assertions above.
    held-out isolation verified: train, dev, test, and every state reachable by a reserved RAISE-opened answer are disjoint


## 19.16 Incumbent-mining audit

The mined block is the new part of this lab, so it gets its own audit. Four
questions matter.

*Did the hard negatives actually land?* Every state must contain its
incumbent's top-ranked word, and every selected incumbent action must be present
in the support it was selected for. Both are asserted, not reported.

*How adaptive is the treatment?* If an incumbent's favourite word were usually
the open teacher or the canonical candidate teacher, the mined slots would be
redundant with the entropy blocks and the intervention would be mostly
cosmetic. The overlap rates below answer that directly, as does the count of
distinct words each seed contributed.

*How wrong is the incumbent?* Broad-state entropy regret of the incumbent's
top-1 word, by regime and candidate bucket, measures how much information the
current policy throws away at the states being trained on. It is also the
quantity the `value` arm is supposed to reduce.

*Are sharp hard negatives really wrong?* In a sharp state every mined action is
required to be a non-candidate that is not an already-included previous guess,
and to receive exactly zero target mass. That combination is what makes the
sharp target a suppression signal rather than a relabelling.


```python
for record in all_targets:
    assert record["incumbent_top1_action"] in record["actions"], (
        "incumbent top-1 action missing from its own support"
    )
    assert set(record["incumbent_actions"]) <= set(record["actions"]), (
        "selected incumbent action missing from its own support"
    )
print(
    f"incumbent top-1 inclusion holds on all {len(all_targets)} seed-state "
    "supports, and every selected incumbent action is present in the support "
    "it was selected for"
)

mining_audit = target_frame.groupby(
    ["seed", "split", "regime"], sort=True
).agg(
    states=("state_key", "size"),
    unique_top1_actions=(
        "incumbent_top1_action", lambda values: int(values.nunique())
    ),
    top1_is_open_teacher=("incumbent_top1_is_open_teacher", "mean"),
    top1_is_candidate_teacher=("incumbent_top1_is_candidate_teacher", "mean"),
    top1_is_candidate=("incumbent_top1_is_candidate", "mean"),
    top1_is_repeat=("incumbent_top1_is_repeat", "mean"),
    mean_top1_target_probability=(
        "incumbent_top1_target_probability", "mean"
    ),
    mean_incumbent_target_mass=("incumbent_target_mass", "mean"),
).reset_index()
display(mining_audit)

retained = pd.DataFrame([
    {
        "seed": record["seed"],
        "split": record["split"],
        "action": action,
    }
    for record in all_targets
    for action in record["incumbent_actions"]
])
display(
    retained.groupby(["seed", "split"], sort=True).agg(
        retained_slots=("action", "size"),
        unique_retained_actions=("action", "nunique"),
    ).reset_index()
)
```

    incumbent top-1 inclusion holds on all 4485 seed-state supports, and every selected incumbent action is present in the support it was selected for



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
      <th>split</th>
      <th>regime</th>
      <th>states</th>
      <th>unique_top1_actions</th>
      <th>top1_is_open_teacher</th>
      <th>top1_is_candidate_teacher</th>
      <th>top1_is_candidate</th>
      <th>top1_is_repeat</th>
      <th>mean_top1_target_probability</th>
      <th>mean_incumbent_target_mass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>198</td>
      <td>0.000000</td>
      <td>0.110092</td>
      <td>0.342508</td>
      <td>0.003058</td>
      <td>0.047025</td>
      <td>0.139274</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>90</td>
      <td>0.000000</td>
      <td>0.244604</td>
      <td>0.251799</td>
      <td>0.071942</td>
      <td>0.230216</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>354</td>
      <td>0.007289</td>
      <td>0.221574</td>
      <td>0.495627</td>
      <td>0.001458</td>
      <td>0.051077</td>
      <td>0.132772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>293</td>
      <td>0.000000</td>
      <td>0.291545</td>
      <td>0.338192</td>
      <td>0.072886</td>
      <td>0.263848</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>197</td>
      <td>0.000000</td>
      <td>0.110092</td>
      <td>0.354740</td>
      <td>0.000000</td>
      <td>0.046173</td>
      <td>0.140298</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>94</td>
      <td>0.000000</td>
      <td>0.251799</td>
      <td>0.266187</td>
      <td>0.057554</td>
      <td>0.233813</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>368</td>
      <td>0.005831</td>
      <td>0.224490</td>
      <td>0.529155</td>
      <td>0.002915</td>
      <td>0.050878</td>
      <td>0.137783</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>282</td>
      <td>0.000000</td>
      <td>0.317784</td>
      <td>0.361516</td>
      <td>0.058309</td>
      <td>0.281341</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>dev</td>
      <td>broad</td>
      <td>327</td>
      <td>191</td>
      <td>0.000000</td>
      <td>0.110092</td>
      <td>0.348624</td>
      <td>0.003058</td>
      <td>0.047751</td>
      <td>0.141407</td>
    </tr>
    <tr>
      <th>9</th>
      <td>47</td>
      <td>dev</td>
      <td>sharp</td>
      <td>139</td>
      <td>88</td>
      <td>0.000000</td>
      <td>0.223022</td>
      <td>0.230216</td>
      <td>0.050360</td>
      <td>0.205036</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47</td>
      <td>train</td>
      <td>broad</td>
      <td>686</td>
      <td>364</td>
      <td>0.008746</td>
      <td>0.224490</td>
      <td>0.517493</td>
      <td>0.000000</td>
      <td>0.050723</td>
      <td>0.135826</td>
    </tr>
    <tr>
      <th>11</th>
      <td>47</td>
      <td>train</td>
      <td>sharp</td>
      <td>343</td>
      <td>293</td>
      <td>0.000000</td>
      <td>0.341108</td>
      <td>0.381924</td>
      <td>0.064140</td>
      <td>0.304665</td>
      <td>0.000000</td>
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
      <th>split</th>
      <th>retained_slots</th>
      <th>unique_retained_actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>dev</td>
      <td>1864</td>
      <td>689</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>train</td>
      <td>4116</td>
      <td>1127</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45</td>
      <td>dev</td>
      <td>1864</td>
      <td>680</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>train</td>
      <td>4116</td>
      <td>1133</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>dev</td>
      <td>1864</td>
      <td>697</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>train</td>
      <td>4116</td>
      <td>1133</td>
    </tr>
  </tbody>
</table>
</div>



```python
broad_targets = target_frame.query("regime == 'broad'").copy()
broad_targets["bucket"] = bucket_of(broad_targets)
regret_audit = broad_targets.groupby(
    ["seed", "split", "bucket"], observed=True, sort=True
).agg(
    states=("state_key", "size"),
    mean_incumbent_top1_regret_bits=("incumbent_top1_regret_bits", "mean"),
    median_incumbent_top1_regret_bits=("incumbent_top1_regret_bits", "median"),
    max_incumbent_top1_regret_bits=("incumbent_top1_regret_bits", "max"),
    zero_regret_rate=(
        "incumbent_top1_regret_bits",
        lambda values: float((values < 1e-9).mean()),
    ),
).reset_index()
display(regret_audit)
print(
    "Regret is measured against the open optimum over all 2,315 answers, so a "
    "zero-regret row means the incumbent's own favourite word already was the "
    "best available split."
)

sharp_targets = [
    record for record in all_targets if record["regime"] == "sharp"
]
sharp_proof = pd.DataFrame([
    {
        "seed": record["seed"],
        "split": record["split"],
        "all_in_support": set(record["incumbent_actions"])
        <= set(record["actions"]),
        "count": len(record["incumbent_actions"]),
        "none_are_candidates": not (
            set(record["incumbent_positions"])
            & set(record["candidate_positions"])
        ),
        "none_are_repeats": not (
            set(record["incumbent_positions"])
            & set(record["repeat_positions"])
        ),
        "zero_target_mass": float(record["incumbent_target_mass"]) == 0.0,
    }
    for record in sharp_targets
])
display(
    sharp_proof.groupby(["seed", "split"], sort=True).agg(
        sharp_states=("count", "size"),
        selected_actions=("count", "sum"),
        all_in_support=("all_in_support", "all"),
        none_are_candidates=("none_are_candidates", "all"),
        none_are_repeats=("none_are_repeats", "all"),
        zero_target_mass=("zero_target_mass", "all"),
    ).reset_index()
)
assert sharp_proof["all_in_support"].all()
assert sharp_proof["none_are_candidates"].all()
assert sharp_proof["none_are_repeats"].all()
assert sharp_proof["zero_target_mass"].all()
assert (sharp_proof["count"] == SHARP_INCUMBENT_SLOTS).all()
print(
    f"proved: {int(sharp_proof['count'].sum())} sharp-state incumbent wrong "
    "actions are in support, are non-candidates, are not repeated previous "
    "guesses, and carry zero target mass"
)
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
      <th>split</th>
      <th>bucket</th>
      <th>states</th>
      <th>mean_incumbent_top1_regret_bits</th>
      <th>median_incumbent_top1_regret_bits</th>
      <th>max_incumbent_top1_regret_bits</th>
      <th>zero_regret_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>dev</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.795448</td>
      <td>0.666667</td>
      <td>2.521641</td>
      <td>0.176030</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>dev</td>
      <td>11+</td>
      <td>60</td>
      <td>1.083800</td>
      <td>1.042778</td>
      <td>2.842684</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>train</td>
      <td>3-10</td>
      <td>451</td>
      <td>0.751539</td>
      <td>0.666667</td>
      <td>2.584963</td>
      <td>0.201774</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>train</td>
      <td>11+</td>
      <td>235</td>
      <td>0.951372</td>
      <td>0.896742</td>
      <td>2.739054</td>
      <td>0.017021</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>dev</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.820579</td>
      <td>0.666667</td>
      <td>2.521641</td>
      <td>0.153558</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>dev</td>
      <td>11+</td>
      <td>60</td>
      <td>1.049462</td>
      <td>0.999133</td>
      <td>2.385981</td>
      <td>0.016667</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>train</td>
      <td>3-10</td>
      <td>451</td>
      <td>0.742244</td>
      <td>0.666667</td>
      <td>2.721928</td>
      <td>0.201774</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>train</td>
      <td>11+</td>
      <td>235</td>
      <td>0.950264</td>
      <td>0.875719</td>
      <td>2.608238</td>
      <td>0.017021</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>dev</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.792178</td>
      <td>0.666667</td>
      <td>2.521641</td>
      <td>0.179775</td>
    </tr>
    <tr>
      <th>9</th>
      <td>47</td>
      <td>dev</td>
      <td>11+</td>
      <td>60</td>
      <td>1.066914</td>
      <td>1.023541</td>
      <td>2.503741</td>
      <td>0.016667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47</td>
      <td>train</td>
      <td>3-10</td>
      <td>451</td>
      <td>0.739368</td>
      <td>0.666667</td>
      <td>2.584963</td>
      <td>0.197339</td>
    </tr>
    <tr>
      <th>11</th>
      <td>47</td>
      <td>train</td>
      <td>11+</td>
      <td>235</td>
      <td>0.972328</td>
      <td>0.911063</td>
      <td>2.845990</td>
      <td>0.025532</td>
    </tr>
  </tbody>
</table>
</div>


    Regret is measured against the open optimum over all 2,315 answers, so a zero-regret row means the incumbent's own favourite word already was the best available split.



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
      <th>split</th>
      <th>sharp_states</th>
      <th>selected_actions</th>
      <th>all_in_support</th>
      <th>none_are_candidates</th>
      <th>none_are_repeats</th>
      <th>zero_target_mass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>dev</td>
      <td>139</td>
      <td>556</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>train</td>
      <td>343</td>
      <td>1372</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45</td>
      <td>dev</td>
      <td>139</td>
      <td>556</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>train</td>
      <td>343</td>
      <td>1372</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>dev</td>
      <td>139</td>
      <td>556</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>train</td>
      <td>343</td>
      <td>1372</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    proved: 5784 sharp-state incumbent wrong actions are in support, are non-candidates, are not repeated previous guesses, and carry zero target mass


## 19.17 Encode every seed's states and targets

Encoding is per seed because the action lists are per seed. Inside a seed the
two arms share one list of encoded tensors and one ordered state stream; only
the target vector differs, which is what makes the step-1 score equality check
after training a real test rather than a formality.

The prompts themselves are seed-independent, so prompt token statistics are
reported once.


```python
started = time.perf_counter()
train_encodings_by_seed = {}
dev_encodings_by_seed = {}
arm_targets_by_seed = {}
for seed in SEEDS:
    train_encodings_by_seed[seed] = [
        encode_state_actions(record["prompt"], record["action_indices"])
        for record in targets_by_seed[(seed, "train")]
    ]
    dev_encodings_by_seed[seed] = [
        encode_state_actions(record["prompt"], record["action_indices"])
        for record in targets_by_seed[(seed, "dev")]
    ]
    value_targets = [
        torch.tensor(record["value_target"], dtype=torch.float32)
        for record in targets_by_seed[(seed, "train")]
    ]
    hard_targets = []
    for record in targets_by_seed[(seed, "train")]:
        one_hot = torch.zeros(SUPPORT_SIZE, dtype=torch.float32)
        one_hot[record["hard_index"]] = 1.0
        hard_targets.append(one_hot)
    arm_targets_by_seed[seed] = {"hard": hard_targets, "value": value_targets}

prompt_token_counts = [
    encoded["prompt_length"] for encoded in train_encodings_by_seed[SEEDS[0]]
]
for seed in SEEDS[1:]:
    assert [
        encoded["prompt_length"] for encoded in train_encodings_by_seed[seed]
    ] == prompt_token_counts, (
        "prompts drifted across seeds; only the actions may be seed-specific"
    )
print(
    f"encoded {len(SEEDS)} x ({STATE_COUNTS['train']} train + "
    f"{STATE_COUNTS['dev']} dev) states in "
    f"{time.perf_counter() - started:.1f}s"
)
print(
    "prompt tokens: min", min(prompt_token_counts),
    "median", int(np.median(prompt_token_counts)),
    "max", max(prompt_token_counts),
)
print(
    "kept logit positions per update:",
    sorted({
        len(e["positions"])
        for seed in SEEDS
        for e in train_encodings_by_seed[seed]
    }),
)
```

    encoded 3 x (1029 train + 466 dev) states in 1.2s
    prompt tokens: min 111 median 141 max 194
    kept logit positions per update: [3, 4]


## 19.18 Numerical regression of the batched scorer

The batched kernel pads, batches, and uses a gather-minus-`logsumexp`
formulation with a position index. The reference below runs one action at a
time, unpadded and unbatched, and takes a full `log_softmax`. If those two paths
disagree, every score, target cross-entropy, and regret in this lab is wrong.

This runs before training, on the seed-42 incumbent, over the widest, the
median, and the narrowest training state, using seed 42's own supports so the
check covers exactly the action lists that seed will train on. Those supports
contain mined actions as well as teacher actions, so the covered tokenizations
are the ones the training loop will really see.


```python
if RUN_TRAINING or RUN_EVALUATION:
    regression_model = load_eval_adapter(INCUMBENTS[42])
    regression_seed = 42
    regression_targets = targets_by_seed[(regression_seed, "train")]
    regression_encodings = train_encodings_by_seed[regression_seed]
    widest = int(np.argmax(prompt_token_counts))
    narrowest = int(np.argmin(prompt_token_counts))
    middle = int(np.argsort(prompt_token_counts)[len(prompt_token_counts) // 2])
    regression_rows = []
    for position in [widest, middle, narrowest]:
        record = regression_targets[position]
        with torch.no_grad():
            batched = action_scores(
                regression_model, to_device(regression_encodings[position])
            ).cpu().numpy()
        for slot, word in enumerate(record["actions"]):
            regression_rows.append({
                "state_position": position,
                "prompt_tokens": prompt_token_counts[position],
                "action": word,
                "is_incumbent_pick": (
                    slot in set(record["incumbent_positions"])
                ),
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
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    batched vs plain single-action scores on 36 actions: max abs diff 4.446e-06



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
      <th>state_position</th>
      <th>prompt_tokens</th>
      <th>action</th>
      <th>is_incumbent_pick</th>
      <th>batched_score</th>
      <th>reference_score</th>
      <th>abs_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>282</td>
      <td>194</td>
      <td>LOFTY</td>
      <td>False</td>
      <td>-5.569310</td>
      <td>-5.569314</td>
      <td>0.000004</td>
    </tr>
    <tr>
      <th>1</th>
      <td>282</td>
      <td>194</td>
      <td>RAISE</td>
      <td>False</td>
      <td>-20.561899</td>
      <td>-20.561904</td>
      <td>0.000004</td>
    </tr>
    <tr>
      <th>2</th>
      <td>282</td>
      <td>194</td>
      <td>MULCH</td>
      <td>False</td>
      <td>-19.110384</td>
      <td>-19.110387</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>3</th>
      <td>282</td>
      <td>194</td>
      <td>BLOWN</td>
      <td>False</td>
      <td>-8.504313</td>
      <td>-8.504318</td>
      <td>0.000004</td>
    </tr>
    <tr>
      <th>4</th>
      <td>282</td>
      <td>194</td>
      <td>GODLY</td>
      <td>False</td>
      <td>-12.592436</td>
      <td>-12.592440</td>
      <td>0.000004</td>
    </tr>
    <tr>
      <th>5</th>
      <td>282</td>
      <td>194</td>
      <td>LOWLY</td>
      <td>True</td>
      <td>-2.793379</td>
      <td>-2.793381</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>6</th>
      <td>282</td>
      <td>194</td>
      <td>LOYAL</td>
      <td>True</td>
      <td>-3.258669</td>
      <td>-3.258671</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>7</th>
      <td>282</td>
      <td>194</td>
      <td>FOLLY</td>
      <td>True</td>
      <td>-3.288851</td>
      <td>-3.288852</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>8</th>
      <td>282</td>
      <td>194</td>
      <td>LOOPY</td>
      <td>True</td>
      <td>-4.310129</td>
      <td>-4.310133</td>
      <td>0.000004</td>
    </tr>
    <tr>
      <th>9</th>
      <td>282</td>
      <td>194</td>
      <td>LEFTY</td>
      <td>False</td>
      <td>-7.870980</td>
      <td>-7.870983</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>10</th>
      <td>282</td>
      <td>194</td>
      <td>SLEET</td>
      <td>False</td>
      <td>-26.177788</td>
      <td>-26.177789</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>11</th>
      <td>282</td>
      <td>194</td>
      <td>CABAL</td>
      <td>False</td>
      <td>-24.639046</td>
      <td>-24.639049</td>
      <td>0.000003</td>
    </tr>
  </tbody>
</table>
</div>


    scoring kernel verified


## 19.19 Fixed-shape training soak

The longest training state is repeated for 40 optimizer steps on a disposable
model. Driver memory is sampled while the forward activations are still live and
again after the backward pass, so the trace measures the real peak rather than
the quiescent value between steps.

The soak model is thrown away and every real run rebuilds its model after
resetting its own seed, so the soak cannot perturb initialization, dropout, or
state order.


```python
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
    clear_device_cache()
    return cpu_scores, loss_value, peak


missing_arms = [
    key for key, path in ARM_CHECKPOINTS.items() if not path.exists()
]
if RUN_TRAINING and missing_arms:
    soak_position = int(np.argmax(prompt_token_counts))
    soak_encoded = to_device(
        train_encodings_by_seed[SEEDS[0]][soak_position]
    )
    soak_target = arm_targets_by_seed[SEEDS[0]]["value"][soak_position].to(
        device
    )
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
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    soak model: 2,293,760 trainable parameters (0.383%)


    training soak peak 13.90 GiB, creep +0.00 GiB, final range 0.00 GiB


    training memory plateaued


## 19.20 Train the six arms

Each arm continues from its frozen incumbent through
`PeftModel.from_pretrained(base, incumbent, is_trainable=True)`, so the LoRA
weights start exactly where Lab 17 or Lab 18c left them. A completed checkpoint
is validated and reused; an `-in-progress` directory stops the notebook for
inspection rather than resuming from an unknown optimizer state.

A seed's two arms read the same `train_encodings_by_seed[seed]` list, in the
same order, and the same `targets_by_seed[(seed, "train")]` records. The only
per-arm object is the probability vector, so the run manifest records the
seed's support fingerprint and its mined-ranking hash alongside the usual
hyperparameters: a checkpoint trained against a different support cannot be
silently reused.

Both cross-entropies are recorded at every step for both arms. The `hard` arm
optimizes `hard_ce` and merely observes `value_ce`; the `value` arm does the
reverse. That makes the two training curves directly comparable instead of two
different quantities plotted on one axis.


```python
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
    seed_manifest = target_manifest["seeds"][str(seed)]
    assert manifest["training_stream_sha256"] == (
        seed_manifest["support_sha256"]["train"]
    )
    assert manifest["target_train_sha256"] == (
        seed_manifest["targets_sha256"]["train"]
    )
    assert manifest["ranking_indices_sha256"] == (
        seed_manifest["ranking_indices_sha256"]["train"]
    )
    assert manifest["scorer_sha256"] == SCORER_SHA256
    assert manifest["state_order_sha256"] == STATE_ORDER_SHA256["train"]


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

    seed_targets = targets_by_seed[(seed, "train")]
    seed_encodings = train_encodings_by_seed[seed]
    targets = arm_targets_by_seed[seed][arm]
    value_reference = arm_targets_by_seed[seed]["value"]
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
        record = seed_targets[position]
        encoded = to_device(seed_encodings[position])
        target = targets[position].to(device)
        scores, loss_value, step_peak = training_step(
            model, optimizer, encoded, target
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        peak_memory = max(peak_memory, step_peak)

        cpu_scores = scores.cpu()
        log_q = torch.log_softmax(cpu_scores, dim=-1)
        value_probs = value_reference[position]
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
            "actions": ",".join(record["actions"]),
            "lr": learning_rate,
            "loss": loss_value,
            "value_ce_nats": value_ce,
            "hard_ce_nats": hard_ce,
            "kl_to_value_nats": value_ce - target_entropy_nats(value_probs),
            "hard_top1": bool(int(cpu_scores.argmax()) == record["hard_index"]),
            "incumbent_top1_rank": int(
                (cpu_scores > cpu_scores[record["incumbent_top1_position"]])
                .sum()
            ) + 1,
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
    seed_manifest = target_manifest["seeds"][str(seed)]
    manifest = {
        "experiment": "Lab 19 seed-specific value-aware distillation",
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
        "support_source": "per-seed frozen incumbent full-answer-list ranking",
        "mining_top_k": MINING_TOP_K,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_stream_sha256": seed_manifest["support_sha256"]["train"],
        "target_train_sha256": seed_manifest["targets_sha256"]["train"],
        "ranking_indices_sha256": (
            seed_manifest["ranking_indices_sha256"]["train"]
        ),
        "scorer_sha256": SCORER_SHA256,
        "state_order_sha256": STATE_ORDER_SHA256["train"],
        "source_sha256": structured_hashes,
        "first_step_scores": first_step_scores,
        "mean_loss": float(history["loss"].mean()),
        "final_100_loss": float(tail["loss"].mean()),
        "final_100_value_ce_nats": float(tail["value_ce_nats"].mean()),
        "final_100_hard_ce_nats": float(tail["hard_ce_nats"].mean()),
        "final_100_hard_top1": float(tail["hard_top1"].mean()),
        "final_100_incumbent_top1_rank": float(
            tail["incumbent_top1_rank"].mean()
        ),
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
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 42 hard step    1/1029 loss=2.4921 value_ce=11.8067 hard_ce=2.4921 lr=1.96e-07 peak=10.85 GiB


    seed 42 hard step   50/1029 loss=6.4057 value_ce=7.2837 hard_ce=6.4057 lr=9.80e-06 peak=11.88 GiB


    seed 42 hard step  100/1029 loss=6.1918 value_ce=6.2729 hard_ce=6.1918 lr=9.94e-06 peak=12.79 GiB


    seed 42 hard step  150/1029 loss=4.3087 value_ce=4.3577 hard_ce=4.3087 lr=9.75e-06 peak=12.88 GiB


    seed 42 hard step  200/1029 loss=3.9082 value_ce=3.9595 hard_ce=3.9082 lr=9.45e-06 peak=12.88 GiB


    seed 42 hard step  250/1029 loss=3.6571 value_ce=3.5631 hard_ce=3.6571 lr=9.02e-06 peak=12.88 GiB


    seed 42 hard step  300/1029 loss=3.4498 value_ce=3.4251 hard_ce=3.4498 lr=8.50e-06 peak=13.91 GiB


    seed 42 hard step  350/1029 loss=3.2107 value_ce=3.1783 hard_ce=3.2107 lr=7.88e-06 peak=13.91 GiB


    seed 42 hard step  400/1029 loss=3.0643 value_ce=3.0484 hard_ce=3.0643 lr=7.19e-06 peak=13.91 GiB


    seed 42 hard step  450/1029 loss=2.9688 value_ce=3.0189 hard_ce=2.9688 lr=6.44e-06 peak=13.91 GiB


    seed 42 hard step  500/1029 loss=3.2031 value_ce=3.1062 hard_ce=3.2031 lr=5.66e-06 peak=13.91 GiB


    seed 42 hard step  550/1029 loss=2.4593 value_ce=2.6685 hard_ce=2.4593 lr=4.86e-06 peak=13.91 GiB


    seed 42 hard step  600/1029 loss=2.7318 value_ce=2.8378 hard_ce=2.7318 lr=4.06e-06 peak=13.91 GiB


    seed 42 hard step  650/1029 loss=2.6050 value_ce=2.9544 hard_ce=2.6050 lr=3.28e-06 peak=13.91 GiB


    seed 42 hard step  700/1029 loss=2.7234 value_ce=2.6645 hard_ce=2.7234 lr=2.56e-06 peak=13.91 GiB


    seed 42 hard step  750/1029 loss=2.7820 value_ce=2.7217 hard_ce=2.7820 lr=1.89e-06 peak=13.91 GiB


    seed 42 hard step  800/1029 loss=2.6067 value_ce=2.6990 hard_ce=2.6067 lr=1.30e-06 peak=13.91 GiB


    seed 42 hard step  850/1029 loss=2.2377 value_ce=2.5541 hard_ce=2.2377 lr=8.13e-07 peak=13.91 GiB


    seed 42 hard step  900/1029 loss=2.6864 value_ce=2.9362 hard_ce=2.6864 lr=4.30e-07 peak=13.91 GiB


    seed 42 hard step  950/1029 loss=2.5215 value_ce=2.6384 hard_ce=2.5215 lr=1.64e-07 peak=13.91 GiB


    seed 42 hard step 1000/1029 loss=2.5559 value_ce=2.7516 hard_ce=2.5559 lr=2.32e-08 peak=13.91 GiB


    seed 42 hard: complete in 11.9 min



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 42 value step    1/1029 loss=11.8067 value_ce=11.8067 hard_ce=2.4921 lr=1.96e-07 peak=10.85 GiB


    seed 42 value step   50/1029 loss=6.8527 value_ce=6.8527 hard_ce=6.0319 lr=9.80e-06 peak=11.88 GiB


    seed 42 value step  100/1029 loss=4.6111 value_ce=4.6111 hard_ce=4.7568 lr=9.94e-06 peak=12.79 GiB


    seed 42 value step  150/1029 loss=3.6292 value_ce=3.6292 hard_ce=3.5550 lr=9.75e-06 peak=12.88 GiB


    seed 42 value step  200/1029 loss=3.3543 value_ce=3.3543 hard_ce=3.3618 lr=9.45e-06 peak=12.88 GiB


    seed 42 value step  250/1029 loss=3.0136 value_ce=3.0136 hard_ce=3.0886 lr=9.02e-06 peak=12.88 GiB


    seed 42 value step  300/1029 loss=2.8814 value_ce=2.8814 hard_ce=2.8068 lr=8.50e-06 peak=13.91 GiB


    seed 42 value step  350/1029 loss=2.8068 value_ce=2.8068 hard_ce=2.7877 lr=7.88e-06 peak=13.91 GiB


    seed 42 value step  400/1029 loss=2.7745 value_ce=2.7745 hard_ce=2.7298 lr=7.19e-06 peak=13.91 GiB


    seed 42 value step  450/1029 loss=2.7010 value_ce=2.7010 hard_ce=2.6071 lr=6.44e-06 peak=13.91 GiB


    seed 42 value step  500/1029 loss=2.7398 value_ce=2.7398 hard_ce=2.6024 lr=5.66e-06 peak=13.91 GiB


    seed 42 value step  550/1029 loss=2.5020 value_ce=2.5020 hard_ce=2.5179 lr=4.86e-06 peak=13.91 GiB


    seed 42 value step  600/1029 loss=2.6066 value_ce=2.6066 hard_ce=2.5648 lr=4.06e-06 peak=13.91 GiB


    seed 42 value step  650/1029 loss=2.7443 value_ce=2.7443 hard_ce=2.5827 lr=3.28e-06 peak=13.91 GiB


    seed 42 value step  700/1029 loss=2.5526 value_ce=2.5526 hard_ce=2.6858 lr=2.56e-06 peak=13.91 GiB


    seed 42 value step  750/1029 loss=2.6620 value_ce=2.6620 hard_ce=2.6881 lr=1.89e-06 peak=13.91 GiB


    seed 42 value step  800/1029 loss=2.7028 value_ce=2.7028 hard_ce=2.7419 lr=1.30e-06 peak=13.91 GiB


    seed 42 value step  850/1029 loss=2.5015 value_ce=2.5015 hard_ce=2.3826 lr=8.13e-07 peak=13.91 GiB


    seed 42 value step  900/1029 loss=2.6624 value_ce=2.6624 hard_ce=2.6577 lr=4.30e-07 peak=13.91 GiB


    seed 42 value step  950/1029 loss=2.4581 value_ce=2.4581 hard_ce=2.2933 lr=1.64e-07 peak=13.91 GiB


    seed 42 value step 1000/1029 loss=2.5605 value_ce=2.5605 hard_ce=2.4504 lr=2.32e-08 peak=13.91 GiB


    seed 42 value: complete in 12.2 min



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45 hard step    1/1029 loss=2.0260 value_ce=12.6055 hard_ce=2.0260 lr=1.96e-07 peak=10.85 GiB


    seed 45 hard step   50/1029 loss=6.5037 value_ce=7.3867 hard_ce=6.5037 lr=9.80e-06 peak=11.88 GiB


    seed 45 hard step  100/1029 loss=6.4268 value_ce=6.4526 hard_ce=6.4268 lr=9.94e-06 peak=12.79 GiB


    seed 45 hard step  150/1029 loss=4.5866 value_ce=4.5606 hard_ce=4.5866 lr=9.75e-06 peak=12.88 GiB


    seed 45 hard step  200/1029 loss=4.1435 value_ce=4.0897 hard_ce=4.1435 lr=9.45e-06 peak=12.88 GiB


    seed 45 hard step  250/1029 loss=3.6359 value_ce=3.5759 hard_ce=3.6359 lr=9.02e-06 peak=12.88 GiB


    seed 45 hard step  300/1029 loss=3.5030 value_ce=3.4408 hard_ce=3.5030 lr=8.50e-06 peak=13.91 GiB


    seed 45 hard step  350/1029 loss=3.3533 value_ce=3.2542 hard_ce=3.3533 lr=7.88e-06 peak=13.91 GiB


    seed 45 hard step  400/1029 loss=3.0672 value_ce=3.0376 hard_ce=3.0672 lr=7.19e-06 peak=13.91 GiB


    seed 45 hard step  450/1029 loss=3.2152 value_ce=3.1665 hard_ce=3.2152 lr=6.44e-06 peak=13.91 GiB


    seed 45 hard step  500/1029 loss=2.9306 value_ce=3.0572 hard_ce=2.9306 lr=5.66e-06 peak=13.91 GiB


    seed 45 hard step  550/1029 loss=2.4353 value_ce=2.6950 hard_ce=2.4353 lr=4.86e-06 peak=13.91 GiB


    seed 45 hard step  600/1029 loss=2.8667 value_ce=2.8705 hard_ce=2.8667 lr=4.06e-06 peak=13.91 GiB


    seed 45 hard step  650/1029 loss=2.7308 value_ce=2.9415 hard_ce=2.7308 lr=3.28e-06 peak=13.91 GiB


    seed 45 hard step  700/1029 loss=2.7941 value_ce=2.7049 hard_ce=2.7941 lr=2.56e-06 peak=13.91 GiB


    seed 45 hard step  750/1029 loss=3.0200 value_ce=3.0252 hard_ce=3.0200 lr=1.89e-06 peak=13.91 GiB


    seed 45 hard step  800/1029 loss=3.0551 value_ce=3.1244 hard_ce=3.0551 lr=1.30e-06 peak=13.91 GiB


    seed 45 hard step  850/1029 loss=2.5239 value_ce=2.8147 hard_ce=2.5239 lr=8.13e-07 peak=13.91 GiB


    seed 45 hard step  900/1029 loss=2.8541 value_ce=2.9772 hard_ce=2.8541 lr=4.30e-07 peak=13.91 GiB


    seed 45 hard step  950/1029 loss=2.7190 value_ce=2.8051 hard_ce=2.7190 lr=1.64e-07 peak=13.91 GiB


    seed 45 hard step 1000/1029 loss=2.6418 value_ce=2.8356 hard_ce=2.6418 lr=2.32e-08 peak=13.91 GiB


    seed 45 hard: complete in 11.8 min



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45 value step    1/1029 loss=12.6055 value_ce=12.6055 hard_ce=2.0260 lr=1.96e-07 peak=10.85 GiB


    seed 45 value step   50/1029 loss=6.9285 value_ce=6.9285 hard_ce=6.0969 lr=9.80e-06 peak=11.89 GiB


    seed 45 value step  100/1029 loss=4.8022 value_ce=4.8022 hard_ce=4.9772 lr=9.94e-06 peak=12.79 GiB


    seed 45 value step  150/1029 loss=3.6760 value_ce=3.6760 hard_ce=3.6629 lr=9.75e-06 peak=12.89 GiB


    seed 45 value step  200/1029 loss=3.5212 value_ce=3.5212 hard_ce=3.6018 lr=9.45e-06 peak=12.89 GiB


    seed 45 value step  250/1029 loss=3.1259 value_ce=3.1259 hard_ce=3.1688 lr=9.02e-06 peak=12.89 GiB


    seed 45 value step  300/1029 loss=2.9549 value_ce=2.9549 hard_ce=2.9399 lr=8.50e-06 peak=13.92 GiB


    seed 45 value step  350/1029 loss=2.8865 value_ce=2.8865 hard_ce=2.9282 lr=7.88e-06 peak=13.92 GiB


    seed 45 value step  400/1029 loss=2.7983 value_ce=2.7983 hard_ce=2.7669 lr=7.19e-06 peak=13.92 GiB


    seed 45 value step  450/1029 loss=2.8300 value_ce=2.8300 hard_ce=2.7912 lr=6.44e-06 peak=13.92 GiB


    seed 45 value step  500/1029 loss=2.7839 value_ce=2.7839 hard_ce=2.6413 lr=5.66e-06 peak=13.92 GiB


    seed 45 value step  550/1029 loss=2.5136 value_ce=2.5136 hard_ce=2.4680 lr=4.86e-06 peak=13.92 GiB


    seed 45 value step  600/1029 loss=2.6173 value_ce=2.6173 hard_ce=2.6124 lr=4.06e-06 peak=13.92 GiB


    seed 45 value step  650/1029 loss=2.7494 value_ce=2.7494 hard_ce=2.6689 lr=3.28e-06 peak=13.92 GiB


    seed 45 value step  700/1029 loss=2.5431 value_ce=2.5431 hard_ce=2.6953 lr=2.56e-06 peak=13.92 GiB


    seed 45 value step  750/1029 loss=2.6838 value_ce=2.6838 hard_ce=2.7094 lr=1.89e-06 peak=13.92 GiB


    seed 45 value step  800/1029 loss=2.6671 value_ce=2.6671 hard_ce=2.7340 lr=1.30e-06 peak=13.92 GiB


    seed 45 value step  850/1029 loss=2.5291 value_ce=2.5291 hard_ce=2.3945 lr=8.13e-07 peak=13.92 GiB


    seed 45 value step  900/1029 loss=2.7136 value_ce=2.7136 hard_ce=2.7194 lr=4.30e-07 peak=13.92 GiB


    seed 45 value step  950/1029 loss=2.5054 value_ce=2.5054 hard_ce=2.4275 lr=1.64e-07 peak=13.92 GiB


    seed 45 value step 1000/1029 loss=2.6313 value_ce=2.6313 hard_ce=2.5433 lr=2.32e-08 peak=13.92 GiB


    seed 45 value: complete in 11.8 min



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 47 hard step    1/1029 loss=1.6415 value_ce=11.7473 hard_ce=1.6415 lr=1.96e-07 peak=10.85 GiB


    seed 47 hard step   50/1029 loss=6.3642 value_ce=7.4173 hard_ce=6.3642 lr=9.80e-06 peak=11.88 GiB


    seed 47 hard step  100/1029 loss=6.3931 value_ce=6.6201 hard_ce=6.3931 lr=9.94e-06 peak=12.79 GiB


    seed 47 hard step  150/1029 loss=4.6191 value_ce=4.5829 hard_ce=4.6191 lr=9.75e-06 peak=12.88 GiB


    seed 47 hard step  200/1029 loss=3.9759 value_ce=4.0283 hard_ce=3.9759 lr=9.45e-06 peak=12.88 GiB


    seed 47 hard step  250/1029 loss=3.8481 value_ce=3.5716 hard_ce=3.8481 lr=9.02e-06 peak=12.88 GiB


    seed 47 hard step  300/1029 loss=3.4650 value_ce=3.3328 hard_ce=3.4650 lr=8.50e-06 peak=13.91 GiB


    seed 47 hard step  350/1029 loss=3.5193 value_ce=3.3163 hard_ce=3.5193 lr=7.88e-06 peak=13.91 GiB


    seed 47 hard step  400/1029 loss=3.4028 value_ce=3.2468 hard_ce=3.4028 lr=7.19e-06 peak=13.91 GiB


    seed 47 hard step  450/1029 loss=3.4063 value_ce=3.2255 hard_ce=3.4063 lr=6.44e-06 peak=13.91 GiB


    seed 47 hard step  500/1029 loss=2.7884 value_ce=3.0787 hard_ce=2.7884 lr=5.66e-06 peak=13.91 GiB


    seed 47 hard step  550/1029 loss=2.7582 value_ce=2.8864 hard_ce=2.7582 lr=4.86e-06 peak=13.91 GiB


    seed 47 hard step  600/1029 loss=2.9022 value_ce=2.8824 hard_ce=2.9022 lr=4.06e-06 peak=13.91 GiB


    seed 47 hard step  650/1029 loss=2.8153 value_ce=3.1154 hard_ce=2.8153 lr=3.28e-06 peak=13.91 GiB


    seed 47 hard step  700/1029 loss=2.8407 value_ce=3.0071 hard_ce=2.8407 lr=2.56e-06 peak=13.91 GiB


    seed 47 hard step  750/1029 loss=3.4216 value_ce=3.3794 hard_ce=3.4216 lr=1.89e-06 peak=13.91 GiB


    seed 47 hard step  800/1029 loss=3.1620 value_ce=3.3903 hard_ce=3.1620 lr=1.30e-06 peak=13.91 GiB


    seed 47 hard step  850/1029 loss=2.9051 value_ce=3.1055 hard_ce=2.9051 lr=8.13e-07 peak=13.91 GiB


    seed 47 hard step  900/1029 loss=3.2706 value_ce=3.1577 hard_ce=3.2706 lr=4.30e-07 peak=13.91 GiB


    seed 47 hard step  950/1029 loss=2.6297 value_ce=2.8607 hard_ce=2.6297 lr=1.64e-07 peak=13.91 GiB


    seed 47 hard step 1000/1029 loss=2.6246 value_ce=2.7963 hard_ce=2.6246 lr=2.32e-08 peak=13.91 GiB


    seed 47 hard: complete in 11.8 min



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 47 value step    1/1029 loss=11.7473 value_ce=11.7473 hard_ce=1.6415 lr=1.96e-07 peak=10.85 GiB


    seed 47 value step   50/1029 loss=6.9241 value_ce=6.9241 hard_ce=5.9440 lr=9.80e-06 peak=11.89 GiB


    seed 47 value step  100/1029 loss=4.7366 value_ce=4.7366 hard_ce=4.7150 lr=9.94e-06 peak=12.79 GiB


    seed 47 value step  150/1029 loss=3.6085 value_ce=3.6085 hard_ce=3.5367 lr=9.75e-06 peak=12.89 GiB


    seed 47 value step  200/1029 loss=3.3698 value_ce=3.3698 hard_ce=3.3564 lr=9.45e-06 peak=12.89 GiB


    seed 47 value step  250/1029 loss=3.0952 value_ce=3.0952 hard_ce=3.2058 lr=9.02e-06 peak=12.89 GiB


    seed 47 value step  300/1029 loss=2.8862 value_ce=2.8862 hard_ce=2.8757 lr=8.50e-06 peak=13.92 GiB


    seed 47 value step  350/1029 loss=2.8701 value_ce=2.8701 hard_ce=2.8858 lr=7.88e-06 peak=13.92 GiB


    seed 47 value step  400/1029 loss=2.7354 value_ce=2.7354 hard_ce=2.8289 lr=7.19e-06 peak=13.92 GiB


    seed 47 value step  450/1029 loss=2.8003 value_ce=2.8003 hard_ce=2.7346 lr=6.44e-06 peak=13.92 GiB


    seed 47 value step  500/1029 loss=2.7213 value_ce=2.7213 hard_ce=2.4896 lr=5.66e-06 peak=13.92 GiB


    seed 47 value step  550/1029 loss=2.5307 value_ce=2.5307 hard_ce=2.4637 lr=4.86e-06 peak=13.92 GiB


    seed 47 value step  600/1029 loss=2.6156 value_ce=2.6156 hard_ce=2.6226 lr=4.06e-06 peak=13.92 GiB


    seed 47 value step  650/1029 loss=2.7838 value_ce=2.7838 hard_ce=2.6443 lr=3.28e-06 peak=13.92 GiB


    seed 47 value step  700/1029 loss=2.5709 value_ce=2.5709 hard_ce=2.7257 lr=2.56e-06 peak=13.92 GiB


    seed 47 value step  750/1029 loss=2.6718 value_ce=2.6718 hard_ce=2.7619 lr=1.89e-06 peak=13.92 GiB


    seed 47 value step  800/1029 loss=2.6436 value_ce=2.6436 hard_ce=2.6852 lr=1.30e-06 peak=13.92 GiB


    seed 47 value step  850/1029 loss=2.4496 value_ce=2.4496 hard_ce=2.3166 lr=8.13e-07 peak=13.92 GiB


    seed 47 value step  900/1029 loss=2.6440 value_ce=2.6440 hard_ce=2.6217 lr=4.30e-07 peak=13.92 GiB


    seed 47 value step  950/1029 loss=2.5193 value_ce=2.5193 hard_ce=2.3410 lr=1.64e-07 peak=13.92 GiB


    seed 47 value step 1000/1029 loss=2.5706 value_ce=2.5706 hard_ce=2.5312 lr=2.32e-08 peak=13.92 GiB


    seed 47 value: complete in 11.8 min



```python
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
    for column in ["state_key", "candidate_count", "regime", "actions"]:
        hard_column = training_history.query(
            "seed == @seed and arm == 'hard'"
        )[column].tolist()
        value_column = training_history.query(
            "seed == @seed and arm == 'value'"
        )[column].tolist()
        assert hard_column == value_column, (
            f"seed {seed} arms disagree on the {column} stream"
        )
print(
    "within every seed the two arms visited identical states in identical "
    "order with identical action lists"
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
        "final_100_incumbent_top1_rank": manifest[
            "final_100_incumbent_top1_rank"
        ],
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
    incumbent_top1_rank=("incumbent_top1_rank", "mean"),
).reset_index()
display(
    curve.pivot_table(
        index="block", columns=["arm"], values=["value_ce_nats", "hard_ce_nats"]
    ).round(3)
)
print(
    "incumbent_top1_rank is the rank the arm gives its own incumbent's "
    "favourite word inside the 12-action support: 1 means the hard negative "
    "still wins, 12 means it has been fully suppressed."
)
```

    matched control verified: within each seed both arms produced identical step-1 action scores, so they differ only in target probabilities
    within every seed the two arms visited identical states in identical order with identical action lists



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
      <th>arm</th>
      <th>updates</th>
      <th>mean_loss</th>
      <th>final_100_loss</th>
      <th>final_100_value_ce_nats</th>
      <th>final_100_hard_ce_nats</th>
      <th>final_100_hard_top1</th>
      <th>final_100_incumbent_top1_rank</th>
      <th>peak_driver_gib</th>
      <th>elapsed_minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>hard</td>
      <td>1029</td>
      <td>3.289295</td>
      <td>2.428761</td>
      <td>2.662365</td>
      <td>2.428761</td>
      <td>0.22</td>
      <td>4.39</td>
      <td>13.911850</td>
      <td>11.865050</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>value</td>
      <td>1029</td>
      <td>3.050270</td>
      <td>2.516200</td>
      <td>2.516201</td>
      <td>2.392135</td>
      <td>0.19</td>
      <td>5.38</td>
      <td>13.911850</td>
      <td>12.205671</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45</td>
      <td>hard</td>
      <td>1029</td>
      <td>3.433237</td>
      <td>2.660110</td>
      <td>2.810817</td>
      <td>2.660110</td>
      <td>0.14</td>
      <td>4.22</td>
      <td>13.911850</td>
      <td>11.792085</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>value</td>
      <td>1029</td>
      <td>3.105565</td>
      <td>2.557522</td>
      <td>2.557522</td>
      <td>2.466212</td>
      <td>0.11</td>
      <td>5.37</td>
      <td>13.919662</td>
      <td>11.769679</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>hard</td>
      <td>1029</td>
      <td>3.549900</td>
      <td>2.743892</td>
      <td>2.845933</td>
      <td>2.743892</td>
      <td>0.16</td>
      <td>4.69</td>
      <td>13.911850</td>
      <td>11.766843</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>value</td>
      <td>1029</td>
      <td>3.070600</td>
      <td>2.518853</td>
      <td>2.518853</td>
      <td>2.450492</td>
      <td>0.13</td>
      <td>5.44</td>
      <td>13.919662</td>
      <td>11.779720</td>
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

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">hard_ce_nats</th>
      <th colspan="2" halign="left">value_ce_nats</th>
    </tr>
    <tr>
      <th>arm</th>
      <th>hard</th>
      <th>value</th>
      <th>hard</th>
      <th>value</th>
    </tr>
    <tr>
      <th>block</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.381</td>
      <td>5.420</td>
      <td>6.906</td>
      <td>5.809</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.257</td>
      <td>3.512</td>
      <td>4.263</td>
      <td>3.526</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.593</td>
      <td>3.014</td>
      <td>3.485</td>
      <td>2.993</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.270</td>
      <td>2.821</td>
      <td>3.180</td>
      <td>2.812</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.085</td>
      <td>2.644</td>
      <td>3.109</td>
      <td>2.763</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.692</td>
      <td>2.542</td>
      <td>2.807</td>
      <td>2.564</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.752</td>
      <td>2.667</td>
      <td>2.898</td>
      <td>2.657</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.008</td>
      <td>2.720</td>
      <td>3.057</td>
      <td>2.672</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.746</td>
      <td>2.515</td>
      <td>2.924</td>
      <td>2.583</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.615</td>
      <td>2.431</td>
      <td>2.781</td>
      <td>2.541</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.879</td>
      <td>2.554</td>
      <td>2.734</td>
      <td>2.502</td>
    </tr>
  </tbody>
</table>
</div>


    incumbent_top1_rank is the rank the arm gives its own incumbent's favourite word inside the 12-action support: 1 means the hard negative still wins, 12 means it has been fully suppressed.


## 19.21 Held-out dev shortlist evaluation

All 466 unique dev `NEXT_GUESS` states are scored by all nine models: three
incumbents and six trained arms. Each model is scored on **its own seed's** dev
support, mined from that seed's incumbent by the same rule that built its
training supports. So seed 42's incumbent, `hard`, and `value` all face one
identical set of twelve words per state, and seed 45's three models face
another.

That is the comparison the lab needs and the only one it claims. Within a seed
the three models are paired state by state on one frozen boundary. Across seeds
the numbers are not paired: a state's cross-entropy under seed 42 and under seed
45 are measurements against different action sets, so cross-seed tables are
descriptive summaries of three separate paired experiments, not one pooled one.

This is a shortlist metric, not a full-lexicon metric. It answers "given twelve
named actions, does the model prefer the valuable one", which is exactly what
the objective optimized. Section 19.24 asks the harder deployment question over
all 2,315 answers.


```python
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
def score_dev_states(model, seed: int) -> np.ndarray:
    encodings = dev_encodings_by_seed[seed]
    matrix = np.zeros((len(encodings), SUPPORT_SIZE), dtype=np.float32)
    peak = 0.0
    for position, encoded in enumerate(encodings):
        matrix[position] = (
            action_scores(model, to_device(encoded)).cpu().numpy()
        )
        peak = max(peak, driver_memory_gib())
        clear_device_cache()
    assert peak < MEMORY_ABORT_GIB, (
        f"dev scoring exceeded the memory threshold: {peak:.1f} GiB"
    )
    return matrix


def dev_metric_rows(arm: str, seed: int, matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for position, record in enumerate(targets_by_seed[(seed, "dev")]):
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
        incumbent_positions = list(record["incumbent_positions"])
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
            "student_incumbent_mass": float(q[incumbent_positions].sum()),
            "incumbent_top1_rank": int(
                ranks[record["incumbent_top1_position"]]
            ),
            "chosen_is_incumbent_pick": chosen in set(incumbent_positions),
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
    seed_dev_targets = targets_by_seed[(seed, "dev")]
    rows_path = RESULTS_DIR / f"dev-shortlist-{label}.csv"
    scores_path = RESULTS_DIR / f"dev-scores-{label}.npy"
    progress_path = RESULTS_DIR / f"dev-shortlist-{label}-progress.json"
    checkpoint_sha256 = (
        incumbent_hashes[seed]
        if arm == "incumbent"
        else sha256_file(path / "adapter_model.safetensors")
    )
    # The cache is only valid for one evaluated adapter *and* one support and
    # target set. Seed-specific supports make the second half mandatory.
    progress = {
        "label": label,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "target_dev_sha256": (
            target_manifest["seeds"][str(seed)]["targets_sha256"]["dev"]
        ),
        "dev_support_sha256": (
            target_manifest["seeds"][str(seed)]["support_sha256"]["dev"]
        ),
        "ranking_indices_sha256": (
            target_manifest["seeds"][str(seed)]["ranking_indices_sha256"]["dev"]
        ),
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
        assert matrix.shape == (len(seed_dev_targets), SUPPORT_SIZE)
        frame = pd.read_csv(rows_path)
        assert frame["state_key"].tolist() == [
            record["state_key"] for record in seed_dev_targets
        ]
        print(f"{label}: verified cached dev shortlist")
    elif RUN_EVALUATION:
        model = load_eval_adapter(path)
        started = time.perf_counter()
        matrix = score_dev_states(model, seed)
        release_model(model)
        del model
        frame = dev_metric_rows(arm, seed, matrix)
        atomic_npy(matrix, scores_path)
        atomic_csv(frame, rows_path)
        atomic_json(progress, progress_path)
        print(
            f"{label}: scored {len(seed_dev_targets)} dev states in "
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
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    incumbent-seed42: scored 466 dev states in 81s



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    incumbent-seed45: scored 466 dev states in 81s



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    incumbent-seed47: scored 466 dev states in 82s



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    hard-seed42: scored 466 dev states in 83s



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    value-seed42: scored 466 dev states in 82s



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    hard-seed45: scored 466 dev states in 82s



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    value-seed45: scored 466 dev states in 82s



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    hard-seed47: scored 466 dev states in 82s



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    value-seed47: scored 466 dev states in 82s


    dev shortlist rows: 4194


## 19.22 Dev shortlist results

Read the arms against their own incumbent, seed by seed. `hard_top1` and
`kl_to_value_nats` measure imitation of the teacher target. `open_regret_bits`
measures how much entropy a broad-state choice throws away against the best
action in the support, which contains the global open optimum by construction.
`candidate_selected` is the sharp-state closure metric: did the model name a
word that can still be the answer?

`student_incumbent_mass` and `incumbent_top1_rank` are the hard-negative
metrics. They ask whether an arm has actually moved probability off the words
its own ancestor preferred, which is the mechanism the mined slots were added to
drive. Both are read within a seed only.

Every table below carries `seed` as a grouping key, including the bucket and
turn breakdowns. Pooling arms across seeds would average scores taken against
three different twelve-word boundaries, so the seed stays visible even where it
makes the table longer.


```python
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
        student_incumbent_mass=("student_incumbent_mass", "mean"),
        median_incumbent_top1_rank=("incumbent_top1_rank", "median"),
        incumbent_pick_rate=("chosen_is_incumbent_pick", "mean"),
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

dev_by_bucket = dev_summary(dev_results, ["arm", "seed", "bucket"])
display(dev_by_bucket)

dev_by_turn = dev_summary(dev_results, ["arm", "seed", "turn"])
display(dev_by_turn)
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
      <th>arm</th>
      <th>seed</th>
      <th>states</th>
      <th>hard_top1</th>
      <th>median_hard_rank</th>
      <th>value_ce_nats</th>
      <th>kl_to_value_nats</th>
      <th>chosen_value_mass</th>
      <th>student_candidate_mass</th>
      <th>student_incumbent_mass</th>
      <th>median_incumbent_top1_rank</th>
      <th>incumbent_pick_rate</th>
      <th>candidate_choice_rate</th>
      <th>repeat_choice_rate</th>
      <th>broad_open_regret_bits</th>
      <th>sharp_candidate_selected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hard</td>
      <td>42</td>
      <td>466</td>
      <td>0.180258</td>
      <td>5.0</td>
      <td>2.787406</td>
      <td>1.246857</td>
      <td>0.158384</td>
      <td>0.307357</td>
      <td>0.446232</td>
      <td>4.0</td>
      <td>0.446352</td>
      <td>0.360515</td>
      <td>0.017167</td>
      <td>0.532215</td>
      <td>0.366906</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hard</td>
      <td>45</td>
      <td>466</td>
      <td>0.145923</td>
      <td>6.0</td>
      <td>2.974566</td>
      <td>1.434803</td>
      <td>0.125328</td>
      <td>0.289058</td>
      <td>0.483196</td>
      <td>4.0</td>
      <td>0.534335</td>
      <td>0.336910</td>
      <td>0.010730</td>
      <td>0.594377</td>
      <td>0.258993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hard</td>
      <td>47</td>
      <td>466</td>
      <td>0.171674</td>
      <td>5.0</td>
      <td>3.030809</td>
      <td>1.489470</td>
      <td>0.144544</td>
      <td>0.312460</td>
      <td>0.433983</td>
      <td>4.0</td>
      <td>0.463519</td>
      <td>0.358369</td>
      <td>0.010730</td>
      <td>0.564770</td>
      <td>0.330935</td>
    </tr>
    <tr>
      <th>3</th>
      <td>incumbent</td>
      <td>42</td>
      <td>466</td>
      <td>0.098712</td>
      <td>6.0</td>
      <td>8.002817</td>
      <td>6.462268</td>
      <td>0.101668</td>
      <td>0.286296</td>
      <td>0.781472</td>
      <td>1.0</td>
      <td>0.763948</td>
      <td>0.315451</td>
      <td>0.023605</td>
      <td>0.848356</td>
      <td>0.251799</td>
    </tr>
    <tr>
      <th>4</th>
      <td>incumbent</td>
      <td>45</td>
      <td>466</td>
      <td>0.098712</td>
      <td>6.0</td>
      <td>8.053456</td>
      <td>6.513693</td>
      <td>0.102143</td>
      <td>0.300284</td>
      <td>0.777522</td>
      <td>1.0</td>
      <td>0.761803</td>
      <td>0.328326</td>
      <td>0.017167</td>
      <td>0.862576</td>
      <td>0.266187</td>
    </tr>
    <tr>
      <th>5</th>
      <td>incumbent</td>
      <td>47</td>
      <td>466</td>
      <td>0.092275</td>
      <td>6.0</td>
      <td>8.053551</td>
      <td>6.512213</td>
      <td>0.094666</td>
      <td>0.295071</td>
      <td>0.780691</td>
      <td>1.0</td>
      <td>0.776824</td>
      <td>0.313305</td>
      <td>0.017167</td>
      <td>0.842588</td>
      <td>0.230216</td>
    </tr>
    <tr>
      <th>6</th>
      <td>value</td>
      <td>42</td>
      <td>466</td>
      <td>0.113734</td>
      <td>6.0</td>
      <td>2.633922</td>
      <td>1.093373</td>
      <td>0.104344</td>
      <td>0.225202</td>
      <td>0.374659</td>
      <td>5.0</td>
      <td>0.343348</td>
      <td>0.248927</td>
      <td>0.010730</td>
      <td>0.462708</td>
      <td>0.143885</td>
    </tr>
    <tr>
      <th>7</th>
      <td>value</td>
      <td>45</td>
      <td>466</td>
      <td>0.118026</td>
      <td>6.0</td>
      <td>2.661771</td>
      <td>1.122008</td>
      <td>0.104401</td>
      <td>0.229438</td>
      <td>0.395777</td>
      <td>5.0</td>
      <td>0.414163</td>
      <td>0.246781</td>
      <td>0.000000</td>
      <td>0.516247</td>
      <td>0.151079</td>
    </tr>
    <tr>
      <th>8</th>
      <td>value</td>
      <td>47</td>
      <td>466</td>
      <td>0.135193</td>
      <td>6.0</td>
      <td>2.617946</td>
      <td>1.076608</td>
      <td>0.124997</td>
      <td>0.234985</td>
      <td>0.369930</td>
      <td>6.0</td>
      <td>0.341202</td>
      <td>0.259657</td>
      <td>0.002146</td>
      <td>0.442360</td>
      <td>0.215827</td>
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
      <th>arm</th>
      <th>seed</th>
      <th>regime</th>
      <th>states</th>
      <th>hard_top1</th>
      <th>median_hard_rank</th>
      <th>value_ce_nats</th>
      <th>kl_to_value_nats</th>
      <th>chosen_value_mass</th>
      <th>student_candidate_mass</th>
      <th>student_incumbent_mass</th>
      <th>median_incumbent_top1_rank</th>
      <th>incumbent_pick_rate</th>
      <th>candidate_choice_rate</th>
      <th>repeat_choice_rate</th>
      <th>broad_open_regret_bits</th>
      <th>sharp_candidate_selected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hard</td>
      <td>42</td>
      <td>broad</td>
      <td>327</td>
      <td>0.100917</td>
      <td>6.0</td>
      <td>3.126477</td>
      <td>0.986189</td>
      <td>0.081978</td>
      <td>0.322370</td>
      <td>0.433447</td>
      <td>4.0</td>
      <td>0.446483</td>
      <td>0.357798</td>
      <td>0.000000</td>
      <td>0.532215</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hard</td>
      <td>42</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.366906</td>
      <td>3.0</td>
      <td>1.989735</td>
      <td>1.860082</td>
      <td>0.338129</td>
      <td>0.272040</td>
      <td>0.476308</td>
      <td>3.0</td>
      <td>0.446043</td>
      <td>0.366906</td>
      <td>0.057554</td>
      <td>NaN</td>
      <td>0.366906</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hard</td>
      <td>45</td>
      <td>broad</td>
      <td>327</td>
      <td>0.097859</td>
      <td>7.0</td>
      <td>3.169220</td>
      <td>1.030053</td>
      <td>0.077685</td>
      <td>0.324916</td>
      <td>0.468266</td>
      <td>4.0</td>
      <td>0.529052</td>
      <td>0.370031</td>
      <td>0.000000</td>
      <td>0.594377</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hard</td>
      <td>45</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.258993</td>
      <td>3.0</td>
      <td>2.516639</td>
      <td>2.386985</td>
      <td>0.237410</td>
      <td>0.204702</td>
      <td>0.518318</td>
      <td>3.0</td>
      <td>0.546763</td>
      <td>0.258993</td>
      <td>0.035971</td>
      <td>NaN</td>
      <td>0.258993</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hard</td>
      <td>47</td>
      <td>broad</td>
      <td>327</td>
      <td>0.113150</td>
      <td>6.0</td>
      <td>3.314170</td>
      <td>1.172757</td>
      <td>0.082134</td>
      <td>0.335446</td>
      <td>0.421364</td>
      <td>4.0</td>
      <td>0.449541</td>
      <td>0.370031</td>
      <td>0.000000</td>
      <td>0.564770</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>hard</td>
      <td>47</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.309353</td>
      <td>2.0</td>
      <td>2.364198</td>
      <td>2.234544</td>
      <td>0.291367</td>
      <td>0.258385</td>
      <td>0.463669</td>
      <td>4.0</td>
      <td>0.496403</td>
      <td>0.330935</td>
      <td>0.035971</td>
      <td>NaN</td>
      <td>0.330935</td>
    </tr>
    <tr>
      <th>6</th>
      <td>incumbent</td>
      <td>42</td>
      <td>broad</td>
      <td>327</td>
      <td>0.036697</td>
      <td>7.0</td>
      <td>10.031501</td>
      <td>7.891213</td>
      <td>0.047025</td>
      <td>0.318472</td>
      <td>0.806152</td>
      <td>1.0</td>
      <td>0.801223</td>
      <td>0.342508</td>
      <td>0.003058</td>
      <td>0.848356</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>incumbent</td>
      <td>42</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.244604</td>
      <td>4.0</td>
      <td>3.230302</td>
      <td>3.100648</td>
      <td>0.230216</td>
      <td>0.210602</td>
      <td>0.723413</td>
      <td>1.0</td>
      <td>0.676259</td>
      <td>0.251799</td>
      <td>0.071942</td>
      <td>NaN</td>
      <td>0.251799</td>
    </tr>
    <tr>
      <th>8</th>
      <td>incumbent</td>
      <td>45</td>
      <td>broad</td>
      <td>327</td>
      <td>0.033639</td>
      <td>7.0</td>
      <td>10.061770</td>
      <td>7.922603</td>
      <td>0.046173</td>
      <td>0.336765</td>
      <td>0.798750</td>
      <td>1.0</td>
      <td>0.798165</td>
      <td>0.354740</td>
      <td>0.000000</td>
      <td>0.862576</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>incumbent</td>
      <td>45</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.251799</td>
      <td>5.0</td>
      <td>3.328859</td>
      <td>3.199206</td>
      <td>0.233813</td>
      <td>0.214464</td>
      <td>0.727581</td>
      <td>1.0</td>
      <td>0.676259</td>
      <td>0.266187</td>
      <td>0.057554</td>
      <td>NaN</td>
      <td>0.266187</td>
    </tr>
    <tr>
      <th>10</th>
      <td>incumbent</td>
      <td>47</td>
      <td>broad</td>
      <td>327</td>
      <td>0.036697</td>
      <td>7.0</td>
      <td>10.060481</td>
      <td>7.919068</td>
      <td>0.047751</td>
      <td>0.331839</td>
      <td>0.802100</td>
      <td>1.0</td>
      <td>0.801223</td>
      <td>0.348624</td>
      <td>0.003058</td>
      <td>0.842588</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>incumbent</td>
      <td>47</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.223022</td>
      <td>5.0</td>
      <td>3.332213</td>
      <td>3.202559</td>
      <td>0.205036</td>
      <td>0.208574</td>
      <td>0.730327</td>
      <td>1.0</td>
      <td>0.719424</td>
      <td>0.230216</td>
      <td>0.050360</td>
      <td>NaN</td>
      <td>0.230216</td>
    </tr>
    <tr>
      <th>12</th>
      <td>value</td>
      <td>42</td>
      <td>broad</td>
      <td>327</td>
      <td>0.103976</td>
      <td>7.0</td>
      <td>2.709471</td>
      <td>0.569183</td>
      <td>0.095181</td>
      <td>0.261179</td>
      <td>0.357015</td>
      <td>6.0</td>
      <td>0.308869</td>
      <td>0.293578</td>
      <td>0.003058</td>
      <td>0.462708</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>value</td>
      <td>42</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.136691</td>
      <td>4.0</td>
      <td>2.456192</td>
      <td>2.326539</td>
      <td>0.125899</td>
      <td>0.140566</td>
      <td>0.416166</td>
      <td>4.0</td>
      <td>0.424460</td>
      <td>0.143885</td>
      <td>0.028777</td>
      <td>NaN</td>
      <td>0.143885</td>
    </tr>
    <tr>
      <th>14</th>
      <td>value</td>
      <td>45</td>
      <td>broad</td>
      <td>327</td>
      <td>0.103976</td>
      <td>6.0</td>
      <td>2.736915</td>
      <td>0.597748</td>
      <td>0.090676</td>
      <td>0.263955</td>
      <td>0.369010</td>
      <td>6.0</td>
      <td>0.382263</td>
      <td>0.287462</td>
      <td>0.000000</td>
      <td>0.516247</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>value</td>
      <td>45</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.151079</td>
      <td>4.0</td>
      <td>2.484993</td>
      <td>2.355340</td>
      <td>0.136691</td>
      <td>0.148234</td>
      <td>0.458745</td>
      <td>4.0</td>
      <td>0.489209</td>
      <td>0.151079</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.151079</td>
    </tr>
    <tr>
      <th>16</th>
      <td>value</td>
      <td>47</td>
      <td>broad</td>
      <td>327</td>
      <td>0.110092</td>
      <td>6.0</td>
      <td>2.706866</td>
      <td>0.565453</td>
      <td>0.098620</td>
      <td>0.266092</td>
      <td>0.343533</td>
      <td>6.0</td>
      <td>0.302752</td>
      <td>0.278287</td>
      <td>0.000000</td>
      <td>0.442360</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>value</td>
      <td>47</td>
      <td>sharp</td>
      <td>139</td>
      <td>0.194245</td>
      <td>3.0</td>
      <td>2.408760</td>
      <td>2.279107</td>
      <td>0.187050</td>
      <td>0.161806</td>
      <td>0.432027</td>
      <td>5.0</td>
      <td>0.431655</td>
      <td>0.215827</td>
      <td>0.007194</td>
      <td>NaN</td>
      <td>0.215827</td>
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
      <th>arm</th>
      <th>seed</th>
      <th>bucket</th>
      <th>states</th>
      <th>hard_top1</th>
      <th>median_hard_rank</th>
      <th>value_ce_nats</th>
      <th>kl_to_value_nats</th>
      <th>chosen_value_mass</th>
      <th>student_candidate_mass</th>
      <th>student_incumbent_mass</th>
      <th>median_incumbent_top1_rank</th>
      <th>incumbent_pick_rate</th>
      <th>candidate_choice_rate</th>
      <th>repeat_choice_rate</th>
      <th>broad_open_regret_bits</th>
      <th>sharp_candidate_selected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hard</td>
      <td>42</td>
      <td>1</td>
      <td>113</td>
      <td>0.380531</td>
      <td>2.0</td>
      <td>1.898066</td>
      <td>1.898066</td>
      <td>0.380531</td>
      <td>0.271249</td>
      <td>0.478661</td>
      <td>3.0</td>
      <td>0.433628</td>
      <td>0.380531</td>
      <td>0.053097</td>
      <td>NaN</td>
      <td>0.380531</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hard</td>
      <td>42</td>
      <td>2</td>
      <td>26</td>
      <td>0.307692</td>
      <td>3.0</td>
      <td>2.388141</td>
      <td>1.694994</td>
      <td>0.153846</td>
      <td>0.275477</td>
      <td>0.466083</td>
      <td>3.5</td>
      <td>0.500000</td>
      <td>0.307692</td>
      <td>0.076923</td>
      <td>NaN</td>
      <td>0.307692</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hard</td>
      <td>42</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.104869</td>
      <td>6.0</td>
      <td>3.157862</td>
      <td>1.000138</td>
      <td>0.080042</td>
      <td>0.306381</td>
      <td>0.434238</td>
      <td>4.0</td>
      <td>0.453184</td>
      <td>0.340824</td>
      <td>0.000000</td>
      <td>0.505618</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hard</td>
      <td>42</td>
      <td>11+</td>
      <td>60</td>
      <td>0.083333</td>
      <td>7.0</td>
      <td>2.986812</td>
      <td>0.924115</td>
      <td>0.090596</td>
      <td>0.393520</td>
      <td>0.429928</td>
      <td>4.0</td>
      <td>0.416667</td>
      <td>0.433333</td>
      <td>0.000000</td>
      <td>0.650571</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hard</td>
      <td>45</td>
      <td>1</td>
      <td>113</td>
      <td>0.265487</td>
      <td>3.0</td>
      <td>2.444080</td>
      <td>2.444080</td>
      <td>0.265487</td>
      <td>0.190115</td>
      <td>0.521509</td>
      <td>4.0</td>
      <td>0.548673</td>
      <td>0.265487</td>
      <td>0.026549</td>
      <td>NaN</td>
      <td>0.265487</td>
    </tr>
    <tr>
      <th>5</th>
      <td>hard</td>
      <td>45</td>
      <td>2</td>
      <td>26</td>
      <td>0.230769</td>
      <td>3.0</td>
      <td>2.831990</td>
      <td>2.138843</td>
      <td>0.115385</td>
      <td>0.268098</td>
      <td>0.504450</td>
      <td>3.0</td>
      <td>0.538462</td>
      <td>0.230769</td>
      <td>0.076923</td>
      <td>NaN</td>
      <td>0.230769</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hard</td>
      <td>45</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.097378</td>
      <td>6.0</td>
      <td>3.195629</td>
      <td>1.041765</td>
      <td>0.072521</td>
      <td>0.301555</td>
      <td>0.475815</td>
      <td>4.0</td>
      <td>0.554307</td>
      <td>0.348315</td>
      <td>0.000000</td>
      <td>0.607084</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hard</td>
      <td>45</td>
      <td>11+</td>
      <td>60</td>
      <td>0.100000</td>
      <td>7.0</td>
      <td>3.051704</td>
      <td>0.977935</td>
      <td>0.100664</td>
      <td>0.428874</td>
      <td>0.434673</td>
      <td>4.0</td>
      <td>0.416667</td>
      <td>0.466667</td>
      <td>0.000000</td>
      <td>0.537834</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>hard</td>
      <td>47</td>
      <td>1</td>
      <td>113</td>
      <td>0.309735</td>
      <td>2.0</td>
      <td>2.268985</td>
      <td>2.268985</td>
      <td>0.309735</td>
      <td>0.243463</td>
      <td>0.468653</td>
      <td>4.0</td>
      <td>0.504425</td>
      <td>0.309735</td>
      <td>0.035398</td>
      <td>NaN</td>
      <td>0.309735</td>
    </tr>
    <tr>
      <th>9</th>
      <td>hard</td>
      <td>47</td>
      <td>2</td>
      <td>26</td>
      <td>0.307692</td>
      <td>3.0</td>
      <td>2.778005</td>
      <td>2.084858</td>
      <td>0.211538</td>
      <td>0.323238</td>
      <td>0.442007</td>
      <td>3.5</td>
      <td>0.461538</td>
      <td>0.423077</td>
      <td>0.038462</td>
      <td>NaN</td>
      <td>0.423077</td>
    </tr>
    <tr>
      <th>10</th>
      <td>hard</td>
      <td>47</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.112360</td>
      <td>6.0</td>
      <td>3.314636</td>
      <td>1.159398</td>
      <td>0.078738</td>
      <td>0.318167</td>
      <td>0.425057</td>
      <td>4.0</td>
      <td>0.460674</td>
      <td>0.355805</td>
      <td>0.000000</td>
      <td>0.559919</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>hard</td>
      <td>47</td>
      <td>11+</td>
      <td>60</td>
      <td>0.116667</td>
      <td>6.0</td>
      <td>3.312095</td>
      <td>1.232205</td>
      <td>0.097242</td>
      <td>0.412334</td>
      <td>0.404931</td>
      <td>5.0</td>
      <td>0.400000</td>
      <td>0.433333</td>
      <td>0.000000</td>
      <td>0.586358</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>incumbent</td>
      <td>42</td>
      <td>1</td>
      <td>113</td>
      <td>0.256637</td>
      <td>4.0</td>
      <td>3.016849</td>
      <td>3.016849</td>
      <td>0.256637</td>
      <td>0.200410</td>
      <td>0.722776</td>
      <td>1.0</td>
      <td>0.663717</td>
      <td>0.256637</td>
      <td>0.079646</td>
      <td>NaN</td>
      <td>0.256637</td>
    </tr>
    <tr>
      <th>13</th>
      <td>incumbent</td>
      <td>42</td>
      <td>2</td>
      <td>26</td>
      <td>0.192308</td>
      <td>4.5</td>
      <td>4.157998</td>
      <td>3.464851</td>
      <td>0.115385</td>
      <td>0.254896</td>
      <td>0.726185</td>
      <td>1.0</td>
      <td>0.730769</td>
      <td>0.230769</td>
      <td>0.038462</td>
      <td>NaN</td>
      <td>0.230769</td>
    </tr>
    <tr>
      <th>14</th>
      <td>incumbent</td>
      <td>42</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.044944</td>
      <td>7.0</td>
      <td>10.396855</td>
      <td>8.239131</td>
      <td>0.049404</td>
      <td>0.314641</td>
      <td>0.788304</td>
      <td>1.0</td>
      <td>0.771536</td>
      <td>0.340824</td>
      <td>0.003745</td>
      <td>0.795448</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>incumbent</td>
      <td>42</td>
      <td>11+</td>
      <td>60</td>
      <td>0.000000</td>
      <td>8.0</td>
      <td>8.405675</td>
      <td>6.342978</td>
      <td>0.036438</td>
      <td>0.335515</td>
      <td>0.885575</td>
      <td>1.0</td>
      <td>0.933333</td>
      <td>0.350000</td>
      <td>0.000000</td>
      <td>1.083800</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>incumbent</td>
      <td>45</td>
      <td>1</td>
      <td>113</td>
      <td>0.247788</td>
      <td>5.0</td>
      <td>3.139999</td>
      <td>3.139999</td>
      <td>0.247788</td>
      <td>0.193963</td>
      <td>0.737524</td>
      <td>1.0</td>
      <td>0.681416</td>
      <td>0.247788</td>
      <td>0.070796</td>
      <td>NaN</td>
      <td>0.247788</td>
    </tr>
    <tr>
      <th>17</th>
      <td>incumbent</td>
      <td>45</td>
      <td>2</td>
      <td>26</td>
      <td>0.269231</td>
      <td>4.0</td>
      <td>4.149677</td>
      <td>3.456530</td>
      <td>0.173077</td>
      <td>0.303565</td>
      <td>0.684370</td>
      <td>1.0</td>
      <td>0.653846</td>
      <td>0.346154</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.346154</td>
    </tr>
    <tr>
      <th>18</th>
      <td>incumbent</td>
      <td>45</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.037453</td>
      <td>7.0</td>
      <td>10.442187</td>
      <td>8.288323</td>
      <td>0.047694</td>
      <td>0.331769</td>
      <td>0.779680</td>
      <td>1.0</td>
      <td>0.779026</td>
      <td>0.348315</td>
      <td>0.000000</td>
      <td>0.820579</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>incumbent</td>
      <td>45</td>
      <td>11+</td>
      <td>60</td>
      <td>0.016667</td>
      <td>8.0</td>
      <td>8.368916</td>
      <td>6.295147</td>
      <td>0.039404</td>
      <td>0.358997</td>
      <td>0.883612</td>
      <td>1.0</td>
      <td>0.883333</td>
      <td>0.383333</td>
      <td>0.000000</td>
      <td>1.049462</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>incumbent</td>
      <td>47</td>
      <td>1</td>
      <td>113</td>
      <td>0.221239</td>
      <td>5.0</td>
      <td>3.161352</td>
      <td>3.161352</td>
      <td>0.221239</td>
      <td>0.193581</td>
      <td>0.734009</td>
      <td>1.0</td>
      <td>0.716814</td>
      <td>0.221239</td>
      <td>0.061947</td>
      <td>NaN</td>
      <td>0.221239</td>
    </tr>
    <tr>
      <th>21</th>
      <td>incumbent</td>
      <td>47</td>
      <td>2</td>
      <td>26</td>
      <td>0.230769</td>
      <td>5.0</td>
      <td>4.074797</td>
      <td>3.381650</td>
      <td>0.134615</td>
      <td>0.273734</td>
      <td>0.714325</td>
      <td>1.0</td>
      <td>0.730769</td>
      <td>0.269231</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.269231</td>
    </tr>
    <tr>
      <th>22</th>
      <td>incumbent</td>
      <td>47</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.041199</td>
      <td>7.0</td>
      <td>10.410915</td>
      <td>8.255676</td>
      <td>0.049883</td>
      <td>0.327919</td>
      <td>0.780239</td>
      <td>1.0</td>
      <td>0.779026</td>
      <td>0.344569</td>
      <td>0.003745</td>
      <td>0.792178</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>incumbent</td>
      <td>47</td>
      <td>11+</td>
      <td>60</td>
      <td>0.016667</td>
      <td>8.0</td>
      <td>8.501054</td>
      <td>6.421164</td>
      <td>0.038265</td>
      <td>0.349283</td>
      <td>0.899379</td>
      <td>1.0</td>
      <td>0.900000</td>
      <td>0.366667</td>
      <td>0.000000</td>
      <td>1.066914</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>value</td>
      <td>42</td>
      <td>1</td>
      <td>113</td>
      <td>0.132743</td>
      <td>4.0</td>
      <td>2.404013</td>
      <td>2.404013</td>
      <td>0.132743</td>
      <td>0.128339</td>
      <td>0.411447</td>
      <td>4.0</td>
      <td>0.424779</td>
      <td>0.132743</td>
      <td>0.017699</td>
      <td>NaN</td>
      <td>0.132743</td>
    </tr>
    <tr>
      <th>25</th>
      <td>value</td>
      <td>42</td>
      <td>2</td>
      <td>26</td>
      <td>0.153846</td>
      <td>5.0</td>
      <td>2.682971</td>
      <td>1.989824</td>
      <td>0.096154</td>
      <td>0.193706</td>
      <td>0.436675</td>
      <td>3.5</td>
      <td>0.423077</td>
      <td>0.192308</td>
      <td>0.076923</td>
      <td>NaN</td>
      <td>0.192308</td>
    </tr>
    <tr>
      <th>26</th>
      <td>value</td>
      <td>42</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.104869</td>
      <td>6.0</td>
      <td>2.696976</td>
      <td>0.539252</td>
      <td>0.093025</td>
      <td>0.240493</td>
      <td>0.352161</td>
      <td>6.0</td>
      <td>0.322097</td>
      <td>0.262172</td>
      <td>0.003745</td>
      <td>0.439592</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>value</td>
      <td>42</td>
      <td>11+</td>
      <td>60</td>
      <td>0.100000</td>
      <td>7.0</td>
      <td>2.765073</td>
      <td>0.702376</td>
      <td>0.104776</td>
      <td>0.353230</td>
      <td>0.378619</td>
      <td>5.5</td>
      <td>0.250000</td>
      <td>0.433333</td>
      <td>0.000000</td>
      <td>0.565575</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>value</td>
      <td>45</td>
      <td>1</td>
      <td>113</td>
      <td>0.150442</td>
      <td>4.0</td>
      <td>2.446423</td>
      <td>2.446423</td>
      <td>0.150442</td>
      <td>0.134579</td>
      <td>0.455547</td>
      <td>5.0</td>
      <td>0.469027</td>
      <td>0.150442</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.150442</td>
    </tr>
    <tr>
      <th>29</th>
      <td>value</td>
      <td>45</td>
      <td>2</td>
      <td>26</td>
      <td>0.153846</td>
      <td>4.0</td>
      <td>2.652626</td>
      <td>1.959478</td>
      <td>0.076923</td>
      <td>0.207583</td>
      <td>0.472644</td>
      <td>4.0</td>
      <td>0.576923</td>
      <td>0.153846</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.153846</td>
    </tr>
    <tr>
      <th>30</th>
      <td>value</td>
      <td>45</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.101124</td>
      <td>7.0</td>
      <td>2.734501</td>
      <td>0.580637</td>
      <td>0.088357</td>
      <td>0.236067</td>
      <td>0.369491</td>
      <td>6.0</td>
      <td>0.397004</td>
      <td>0.247191</td>
      <td>0.000000</td>
      <td>0.499782</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>value</td>
      <td>45</td>
      <td>11+</td>
      <td>60</td>
      <td>0.116667</td>
      <td>6.0</td>
      <td>2.747659</td>
      <td>0.673890</td>
      <td>0.100996</td>
      <td>0.388058</td>
      <td>0.366871</td>
      <td>5.5</td>
      <td>0.316667</td>
      <td>0.466667</td>
      <td>0.000000</td>
      <td>0.589516</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>value</td>
      <td>47</td>
      <td>1</td>
      <td>113</td>
      <td>0.194690</td>
      <td>3.0</td>
      <td>2.359944</td>
      <td>2.359944</td>
      <td>0.194690</td>
      <td>0.146278</td>
      <td>0.432787</td>
      <td>4.0</td>
      <td>0.424779</td>
      <td>0.194690</td>
      <td>0.008850</td>
      <td>NaN</td>
      <td>0.194690</td>
    </tr>
    <tr>
      <th>33</th>
      <td>value</td>
      <td>47</td>
      <td>2</td>
      <td>26</td>
      <td>0.192308</td>
      <td>3.5</td>
      <td>2.620923</td>
      <td>1.927776</td>
      <td>0.153846</td>
      <td>0.229294</td>
      <td>0.428723</td>
      <td>5.0</td>
      <td>0.461538</td>
      <td>0.307692</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.307692</td>
    </tr>
    <tr>
      <th>34</th>
      <td>value</td>
      <td>47</td>
      <td>3-10</td>
      <td>267</td>
      <td>0.108614</td>
      <td>6.0</td>
      <td>2.694742</td>
      <td>0.539504</td>
      <td>0.097405</td>
      <td>0.237949</td>
      <td>0.340473</td>
      <td>6.0</td>
      <td>0.314607</td>
      <td>0.247191</td>
      <td>0.000000</td>
      <td>0.411691</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>value</td>
      <td>47</td>
      <td>11+</td>
      <td>60</td>
      <td>0.116667</td>
      <td>6.0</td>
      <td>2.760819</td>
      <td>0.680929</td>
      <td>0.104027</td>
      <td>0.391330</td>
      <td>0.357153</td>
      <td>6.0</td>
      <td>0.250000</td>
      <td>0.416667</td>
      <td>0.000000</td>
      <td>0.578838</td>
      <td>NaN</td>
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
      <th>arm</th>
      <th>seed</th>
      <th>turn</th>
      <th>states</th>
      <th>hard_top1</th>
      <th>median_hard_rank</th>
      <th>value_ce_nats</th>
      <th>kl_to_value_nats</th>
      <th>chosen_value_mass</th>
      <th>student_candidate_mass</th>
      <th>student_incumbent_mass</th>
      <th>median_incumbent_top1_rank</th>
      <th>incumbent_pick_rate</th>
      <th>candidate_choice_rate</th>
      <th>repeat_choice_rate</th>
      <th>broad_open_regret_bits</th>
      <th>sharp_candidate_selected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hard</td>
      <td>42</td>
      <td>2</td>
      <td>27</td>
      <td>0.074074</td>
      <td>6.0</td>
      <td>2.969055</td>
      <td>1.075959</td>
      <td>0.081029</td>
      <td>0.336520</td>
      <td>0.438067</td>
      <td>5.0</td>
      <td>0.481481</td>
      <td>0.370370</td>
      <td>0.000000</td>
      <td>0.713034</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hard</td>
      <td>42</td>
      <td>3</td>
      <td>314</td>
      <td>0.159236</td>
      <td>5.0</td>
      <td>2.882701</td>
      <td>1.156911</td>
      <td>0.123990</td>
      <td>0.301034</td>
      <td>0.439443</td>
      <td>4.0</td>
      <td>0.439490</td>
      <td>0.340764</td>
      <td>0.006369</td>
      <td>0.493225</td>
      <td>0.308824</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hard</td>
      <td>42</td>
      <td>4</td>
      <td>109</td>
      <td>0.284404</td>
      <td>4.0</td>
      <td>2.498436</td>
      <td>1.425849</td>
      <td>0.289588</td>
      <td>0.324717</td>
      <td>0.465709</td>
      <td>3.0</td>
      <td>0.458716</td>
      <td>0.431193</td>
      <td>0.009174</td>
      <td>0.610435</td>
      <td>0.517857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hard</td>
      <td>42</td>
      <td>5</td>
      <td>12</td>
      <td>0.083333</td>
      <td>4.0</td>
      <td>2.555338</td>
      <td>2.059651</td>
      <td>0.089223</td>
      <td>0.278085</td>
      <td>0.449564</td>
      <td>3.0</td>
      <td>0.416667</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.927694</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hard</td>
      <td>42</td>
      <td>6</td>
      <td>4</td>
      <td>0.000000</td>
      <td>2.5</td>
      <td>2.651229</td>
      <td>2.145193</td>
      <td>0.012619</td>
      <td>0.221659</td>
      <td>0.493531</td>
      <td>3.0</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>0.500000</td>
      <td>0.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>hard</td>
      <td>45</td>
      <td>2</td>
      <td>27</td>
      <td>0.074074</td>
      <td>6.0</td>
      <td>3.041238</td>
      <td>1.134778</td>
      <td>0.087942</td>
      <td>0.376946</td>
      <td>0.413074</td>
      <td>5.0</td>
      <td>0.481481</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.625423</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hard</td>
      <td>45</td>
      <td>3</td>
      <td>314</td>
      <td>0.146497</td>
      <td>6.0</td>
      <td>2.968471</td>
      <td>1.244511</td>
      <td>0.115609</td>
      <td>0.292916</td>
      <td>0.475319</td>
      <td>4.0</td>
      <td>0.515924</td>
      <td>0.347134</td>
      <td>0.000000</td>
      <td>0.562582</td>
      <td>0.279412</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hard</td>
      <td>45</td>
      <td>4</td>
      <td>109</td>
      <td>0.174312</td>
      <td>6.0</td>
      <td>2.920527</td>
      <td>1.849951</td>
      <td>0.170716</td>
      <td>0.265054</td>
      <td>0.514848</td>
      <td>3.0</td>
      <td>0.596330</td>
      <td>0.330275</td>
      <td>0.018349</td>
      <td>0.713369</td>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>8</th>
      <td>hard</td>
      <td>45</td>
      <td>5</td>
      <td>12</td>
      <td>0.083333</td>
      <td>3.5</td>
      <td>3.431413</td>
      <td>2.935726</td>
      <td>0.089223</td>
      <td>0.237886</td>
      <td>0.524295</td>
      <td>3.0</td>
      <td>0.500000</td>
      <td>0.166667</td>
      <td>0.166667</td>
      <td>0.927694</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>hard</td>
      <td>45</td>
      <td>6</td>
      <td>4</td>
      <td>0.000000</td>
      <td>4.0</td>
      <td>3.105012</td>
      <td>2.582339</td>
      <td>0.012105</td>
      <td>0.200600</td>
      <td>0.588984</td>
      <td>2.5</td>
      <td>0.750000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>hard</td>
      <td>47</td>
      <td>2</td>
      <td>27</td>
      <td>0.111111</td>
      <td>7.0</td>
      <td>3.162218</td>
      <td>1.264035</td>
      <td>0.090188</td>
      <td>0.375894</td>
      <td>0.405914</td>
      <td>5.0</td>
      <td>0.407407</td>
      <td>0.370370</td>
      <td>0.000000</td>
      <td>0.666108</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>hard</td>
      <td>47</td>
      <td>3</td>
      <td>314</td>
      <td>0.168790</td>
      <td>5.0</td>
      <td>3.024632</td>
      <td>1.297751</td>
      <td>0.129781</td>
      <td>0.314154</td>
      <td>0.423667</td>
      <td>4.0</td>
      <td>0.436306</td>
      <td>0.363057</td>
      <td>0.000000</td>
      <td>0.527482</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>12</th>
      <td>hard</td>
      <td>47</td>
      <td>4</td>
      <td>109</td>
      <td>0.211009</td>
      <td>5.0</td>
      <td>2.959384</td>
      <td>1.888035</td>
      <td>0.211485</td>
      <td>0.299052</td>
      <td>0.467072</td>
      <td>3.0</td>
      <td>0.541284</td>
      <td>0.357798</td>
      <td>0.018349</td>
      <td>0.674425</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>hard</td>
      <td>47</td>
      <td>5</td>
      <td>12</td>
      <td>0.083333</td>
      <td>3.0</td>
      <td>3.580882</td>
      <td>3.083289</td>
      <td>0.089079</td>
      <td>0.272681</td>
      <td>0.431227</td>
      <td>4.0</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>0.166667</td>
      <td>0.927694</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>hard</td>
      <td>47</td>
      <td>6</td>
      <td>4</td>
      <td>0.000000</td>
      <td>3.5</td>
      <td>2.924810</td>
      <td>2.418774</td>
      <td>0.012619</td>
      <td>0.235993</td>
      <td>0.539841</td>
      <td>2.5</td>
      <td>0.750000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>incumbent</td>
      <td>42</td>
      <td>2</td>
      <td>27</td>
      <td>0.037037</td>
      <td>7.0</td>
      <td>10.998977</td>
      <td>9.105880</td>
      <td>0.063218</td>
      <td>0.362377</td>
      <td>0.885107</td>
      <td>1.0</td>
      <td>0.925926</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>1.231229</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>incumbent</td>
      <td>42</td>
      <td>3</td>
      <td>314</td>
      <td>0.070064</td>
      <td>6.0</td>
      <td>8.285035</td>
      <td>6.559245</td>
      <td>0.072089</td>
      <td>0.280988</td>
      <td>0.813609</td>
      <td>1.0</td>
      <td>0.828025</td>
      <td>0.296178</td>
      <td>0.003185</td>
      <td>0.807050</td>
      <td>0.176471</td>
    </tr>
    <tr>
      <th>17</th>
      <td>incumbent</td>
      <td>42</td>
      <td>4</td>
      <td>109</td>
      <td>0.183486</td>
      <td>6.0</td>
      <td>6.781902</td>
      <td>5.709315</td>
      <td>0.187278</td>
      <td>0.283865</td>
      <td>0.685505</td>
      <td>1.0</td>
      <td>0.587156</td>
      <td>0.357798</td>
      <td>0.064220</td>
      <td>0.859913</td>
      <td>0.339286</td>
    </tr>
    <tr>
      <th>18</th>
      <td>incumbent</td>
      <td>42</td>
      <td>5</td>
      <td>12</td>
      <td>0.250000</td>
      <td>5.0</td>
      <td>5.607882</td>
      <td>5.112194</td>
      <td>0.214223</td>
      <td>0.309786</td>
      <td>0.616861</td>
      <td>1.0</td>
      <td>0.416667</td>
      <td>0.416667</td>
      <td>0.166667</td>
      <td>0.927694</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>incumbent</td>
      <td>42</td>
      <td>6</td>
      <td>4</td>
      <td>0.000000</td>
      <td>6.0</td>
      <td>6.079348</td>
      <td>5.573312</td>
      <td>0.012619</td>
      <td>0.185199</td>
      <td>0.668102</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>incumbent</td>
      <td>45</td>
      <td>2</td>
      <td>27</td>
      <td>0.037037</td>
      <td>7.0</td>
      <td>10.874198</td>
      <td>8.967737</td>
      <td>0.070574</td>
      <td>0.407639</td>
      <td>0.869962</td>
      <td>1.0</td>
      <td>0.851852</td>
      <td>0.444444</td>
      <td>0.000000</td>
      <td>1.037809</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>incumbent</td>
      <td>45</td>
      <td>3</td>
      <td>314</td>
      <td>0.076433</td>
      <td>6.0</td>
      <td>8.328360</td>
      <td>6.604400</td>
      <td>0.077576</td>
      <td>0.294183</td>
      <td>0.806944</td>
      <td>1.0</td>
      <td>0.815287</td>
      <td>0.305732</td>
      <td>0.000000</td>
      <td>0.839447</td>
      <td>0.220588</td>
    </tr>
    <tr>
      <th>22</th>
      <td>incumbent</td>
      <td>45</td>
      <td>4</td>
      <td>109</td>
      <td>0.155963</td>
      <td>6.0</td>
      <td>6.871245</td>
      <td>5.800670</td>
      <td>0.167110</td>
      <td>0.292932</td>
      <td>0.691164</td>
      <td>1.0</td>
      <td>0.633028</td>
      <td>0.348624</td>
      <td>0.055046</td>
      <td>0.888511</td>
      <td>0.303571</td>
    </tr>
    <tr>
      <th>23</th>
      <td>incumbent</td>
      <td>45</td>
      <td>5</td>
      <td>12</td>
      <td>0.333333</td>
      <td>4.5</td>
      <td>5.856088</td>
      <td>5.360401</td>
      <td>0.255890</td>
      <td>0.328241</td>
      <td>0.605673</td>
      <td>1.0</td>
      <td>0.416667</td>
      <td>0.500000</td>
      <td>0.083333</td>
      <td>0.927694</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>incumbent</td>
      <td>45</td>
      <td>6</td>
      <td>4</td>
      <td>0.000000</td>
      <td>6.5</td>
      <td>6.240804</td>
      <td>5.718131</td>
      <td>0.012105</td>
      <td>0.171092</td>
      <td>0.712670</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>incumbent</td>
      <td>47</td>
      <td>2</td>
      <td>27</td>
      <td>0.037037</td>
      <td>7.0</td>
      <td>11.232854</td>
      <td>9.334670</td>
      <td>0.065639</td>
      <td>0.403006</td>
      <td>0.900097</td>
      <td>1.0</td>
      <td>0.925926</td>
      <td>0.370370</td>
      <td>0.000000</td>
      <td>1.183657</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>incumbent</td>
      <td>47</td>
      <td>3</td>
      <td>314</td>
      <td>0.066879</td>
      <td>6.0</td>
      <td>8.294950</td>
      <td>6.568069</td>
      <td>0.071891</td>
      <td>0.288408</td>
      <td>0.808856</td>
      <td>1.0</td>
      <td>0.824841</td>
      <td>0.296178</td>
      <td>0.003185</td>
      <td>0.818718</td>
      <td>0.176471</td>
    </tr>
    <tr>
      <th>27</th>
      <td>incumbent</td>
      <td>47</td>
      <td>4</td>
      <td>109</td>
      <td>0.155963</td>
      <td>6.0</td>
      <td>6.914236</td>
      <td>5.842887</td>
      <td>0.152743</td>
      <td>0.290736</td>
      <td>0.690250</td>
      <td>1.0</td>
      <td>0.642202</td>
      <td>0.339450</td>
      <td>0.045872</td>
      <td>0.792608</td>
      <td>0.267857</td>
    </tr>
    <tr>
      <th>28</th>
      <td>incumbent</td>
      <td>47</td>
      <td>5</td>
      <td>12</td>
      <td>0.333333</td>
      <td>4.5</td>
      <td>5.544565</td>
      <td>5.046971</td>
      <td>0.255746</td>
      <td>0.306501</td>
      <td>0.625785</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>0.416667</td>
      <td>0.083333</td>
      <td>0.927694</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>incumbent</td>
      <td>47</td>
      <td>6</td>
      <td>4</td>
      <td>0.000000</td>
      <td>5.5</td>
      <td>6.216797</td>
      <td>5.710760</td>
      <td>0.012619</td>
      <td>0.173410</td>
      <td>0.692967</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>value</td>
      <td>42</td>
      <td>2</td>
      <td>27</td>
      <td>0.074074</td>
      <td>7.0</td>
      <td>2.699583</td>
      <td>0.806487</td>
      <td>0.084622</td>
      <td>0.289282</td>
      <td>0.358748</td>
      <td>6.0</td>
      <td>0.333333</td>
      <td>0.296296</td>
      <td>0.000000</td>
      <td>0.740601</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>value</td>
      <td>42</td>
      <td>3</td>
      <td>314</td>
      <td>0.114650</td>
      <td>6.0</td>
      <td>2.655279</td>
      <td>0.929489</td>
      <td>0.104299</td>
      <td>0.235356</td>
      <td>0.362273</td>
      <td>5.5</td>
      <td>0.308917</td>
      <td>0.251592</td>
      <td>0.006369</td>
      <td>0.412541</td>
      <td>0.161765</td>
    </tr>
    <tr>
      <th>32</th>
      <td>value</td>
      <td>42</td>
      <td>4</td>
      <td>109</td>
      <td>0.119266</td>
      <td>5.0</td>
      <td>2.536121</td>
      <td>1.463534</td>
      <td>0.113096</td>
      <td>0.187238</td>
      <td>0.406891</td>
      <td>5.0</td>
      <td>0.431193</td>
      <td>0.238532</td>
      <td>0.009174</td>
      <td>0.555662</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>33</th>
      <td>value</td>
      <td>42</td>
      <td>5</td>
      <td>12</td>
      <td>0.000000</td>
      <td>5.0</td>
      <td>2.738473</td>
      <td>2.242786</td>
      <td>0.005890</td>
      <td>0.178736</td>
      <td>0.438239</td>
      <td>4.0</td>
      <td>0.416667</td>
      <td>0.166667</td>
      <td>0.166667</td>
      <td>0.927694</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>value</td>
      <td>42</td>
      <td>6</td>
      <td>4</td>
      <td>0.500000</td>
      <td>4.5</td>
      <td>2.865652</td>
      <td>2.359616</td>
      <td>0.297871</td>
      <td>0.169449</td>
      <td>0.385311</td>
      <td>5.5</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>35</th>
      <td>value</td>
      <td>45</td>
      <td>2</td>
      <td>27</td>
      <td>0.074074</td>
      <td>6.0</td>
      <td>2.689620</td>
      <td>0.783159</td>
      <td>0.089706</td>
      <td>0.313790</td>
      <td>0.334389</td>
      <td>7.0</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.645457</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>value</td>
      <td>45</td>
      <td>3</td>
      <td>314</td>
      <td>0.124204</td>
      <td>6.0</td>
      <td>2.687377</td>
      <td>0.963417</td>
      <td>0.107204</td>
      <td>0.239208</td>
      <td>0.385434</td>
      <td>5.0</td>
      <td>0.388535</td>
      <td>0.280255</td>
      <td>0.000000</td>
      <td>0.493016</td>
      <td>0.191176</td>
    </tr>
    <tr>
      <th>37</th>
      <td>value</td>
      <td>45</td>
      <td>4</td>
      <td>109</td>
      <td>0.110092</td>
      <td>5.0</td>
      <td>2.564187</td>
      <td>1.493612</td>
      <td>0.108373</td>
      <td>0.187019</td>
      <td>0.427220</td>
      <td>5.0</td>
      <td>0.486239</td>
      <td>0.146789</td>
      <td>0.000000</td>
      <td>0.557336</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>value</td>
      <td>45</td>
      <td>5</td>
      <td>12</td>
      <td>0.083333</td>
      <td>4.0</td>
      <td>2.703740</td>
      <td>2.208053</td>
      <td>0.047556</td>
      <td>0.199704</td>
      <td>0.500172</td>
      <td>3.0</td>
      <td>0.583333</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.927694</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>value</td>
      <td>45</td>
      <td>6</td>
      <td>4</td>
      <td>0.250000</td>
      <td>4.5</td>
      <td>2.997007</td>
      <td>2.474334</td>
      <td>0.045922</td>
      <td>0.138184</td>
      <td>0.452061</td>
      <td>3.5</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>value</td>
      <td>47</td>
      <td>2</td>
      <td>27</td>
      <td>0.074074</td>
      <td>6.0</td>
      <td>2.697619</td>
      <td>0.799436</td>
      <td>0.087276</td>
      <td>0.342407</td>
      <td>0.349040</td>
      <td>6.0</td>
      <td>0.259259</td>
      <td>0.296296</td>
      <td>0.000000</td>
      <td>0.773616</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>value</td>
      <td>47</td>
      <td>3</td>
      <td>314</td>
      <td>0.143312</td>
      <td>6.0</td>
      <td>2.638084</td>
      <td>0.911203</td>
      <td>0.123612</td>
      <td>0.242307</td>
      <td>0.353605</td>
      <td>6.0</td>
      <td>0.305732</td>
      <td>0.273885</td>
      <td>0.000000</td>
      <td>0.371461</td>
      <td>0.235294</td>
    </tr>
    <tr>
      <th>42</th>
      <td>value</td>
      <td>47</td>
      <td>4</td>
      <td>109</td>
      <td>0.137615</td>
      <td>5.0</td>
      <td>2.539996</td>
      <td>1.468647</td>
      <td>0.149704</td>
      <td>0.192136</td>
      <td>0.413333</td>
      <td>5.0</td>
      <td>0.440367</td>
      <td>0.229358</td>
      <td>0.009174</td>
      <td>0.605220</td>
      <td>0.232143</td>
    </tr>
    <tr>
      <th>43</th>
      <td>value</td>
      <td>47</td>
      <td>5</td>
      <td>12</td>
      <td>0.000000</td>
      <td>4.5</td>
      <td>2.582437</td>
      <td>2.084844</td>
      <td>0.047413</td>
      <td>0.212333</td>
      <td>0.450322</td>
      <td>4.5</td>
      <td>0.583333</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.927694</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>value</td>
      <td>47</td>
      <td>6</td>
      <td>4</td>
      <td>0.250000</td>
      <td>5.0</td>
      <td>2.730036</td>
      <td>2.224000</td>
      <td>0.047871</td>
      <td>0.170746</td>
      <td>0.368458</td>
      <td>5.0</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
PAIRED_METRICS = [
    "hard_top1",
    "kl_to_value_nats",
    "open_regret_bits",
    "candidate_selected",
    "student_candidate_mass",
    "student_incumbent_mass",
]
PAIRED_DIRECTIONS = {
    "hard_top1": "higher",
    "kl_to_value_nats": "lower",
    "open_regret_bits": "lower",
    "candidate_selected": "higher",
    "student_candidate_mass": "higher",
    "student_incumbent_mass": "lower",
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
      <th></th>
      <th>seed</th>
      <th>42</th>
      <th>45</th>
      <th>47</th>
    </tr>
    <tr>
      <th>metric</th>
      <th>left_arm</th>
      <th>right_arm</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">candidate_selected</th>
      <th>hard</th>
      <th>value</th>
      <td>-0.2230</td>
      <td>-0.1079</td>
      <td>-0.1151</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">incumbent</th>
      <th>hard</th>
      <td>0.1151</td>
      <td>-0.0072</td>
      <td>0.1007</td>
    </tr>
    <tr>
      <th>value</th>
      <td>-0.1079</td>
      <td>-0.1151</td>
      <td>-0.0144</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">hard_top1</th>
      <th>hard</th>
      <th>value</th>
      <td>-0.0665</td>
      <td>-0.0279</td>
      <td>-0.0365</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">incumbent</th>
      <th>hard</th>
      <td>0.0815</td>
      <td>0.0472</td>
      <td>0.0794</td>
    </tr>
    <tr>
      <th>value</th>
      <td>0.0150</td>
      <td>0.0193</td>
      <td>0.0429</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">kl_to_value_nats</th>
      <th>hard</th>
      <th>value</th>
      <td>-0.1535</td>
      <td>-0.3128</td>
      <td>-0.4129</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">incumbent</th>
      <th>hard</th>
      <td>-5.2154</td>
      <td>-5.0789</td>
      <td>-5.0227</td>
    </tr>
    <tr>
      <th>value</th>
      <td>-5.3689</td>
      <td>-5.3917</td>
      <td>-5.4356</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">open_regret_bits</th>
      <th>hard</th>
      <th>value</th>
      <td>-0.0695</td>
      <td>-0.0781</td>
      <td>-0.1224</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">incumbent</th>
      <th>hard</th>
      <td>-0.3161</td>
      <td>-0.2682</td>
      <td>-0.2778</td>
    </tr>
    <tr>
      <th>value</th>
      <td>-0.3856</td>
      <td>-0.3463</td>
      <td>-0.4002</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">student_candidate_mass</th>
      <th>hard</th>
      <th>value</th>
      <td>-0.0822</td>
      <td>-0.0596</td>
      <td>-0.0775</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">incumbent</th>
      <th>hard</th>
      <td>0.0211</td>
      <td>-0.0112</td>
      <td>0.0174</td>
    </tr>
    <tr>
      <th>value</th>
      <td>-0.0611</td>
      <td>-0.0708</td>
      <td>-0.0601</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">student_incumbent_mass</th>
      <th>hard</th>
      <th>value</th>
      <td>-0.0716</td>
      <td>-0.0874</td>
      <td>-0.0641</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">incumbent</th>
      <th>hard</th>
      <td>-0.3352</td>
      <td>-0.2943</td>
      <td>-0.3467</td>
    </tr>
    <tr>
      <th>value</th>
      <td>-0.4068</td>
      <td>-0.3817</td>
      <td>-0.4108</td>
    </tr>
  </tbody>
</table>
</div>


    Paired rows are dev states, not independent training runs. The replication unit remains the seed: a direction that does not hold for all three seeds is not replicated.


## 19.23 Gameplay engine reusing the verified Lab 18d kernel

The shortlist metric above is the objective's home turf. The deployment
question is whether an arm plays better full games over the whole 2,315-word
answer list, so gameplay reuses the exact kernel Section 19.9 defined and
Section 19.10 gated, rather than a second copy of it:
same summed-sequence score, same `logits_to_keep=1` prefill, same KV cache with
`CHUNK_SIZE=256`, same `empty_cache` per state, same RAISE opening, same Turns 2
through 6, same free-decoder termination on an invalid word, and the same
strategic diagnostics.

There is deliberately only one `score_all_words` in this notebook. It ranked
every incumbent's answer list during mining, it passed the Lab 18b numerical
regression and the 40-repeat memory soak before a single word was mined, and it
is the same function that plays the games below. `WORD_TOKENS is ACTION_TOKENS`,
so shortlist scores and gameplay scores are the same quantity measured two ways.
The incumbent arm is **not** replayed here: its rows are loaded from Lab 18d's
persisted results, and Section 19.24 checks that those rows still reproduce the
summaries Lab 18d published.


```python
# No kernel is defined here. Gameplay reuses the objects Section 19.9 built and
# Section 19.10 gated; these asserts fail loudly if that stopped being true.
assert WORD_TOKENS is ACTION_TOKENS
assert score_all_words.__module__ == "__main__"
assert set(LENGTH_BUCKETS) == {
    len(tokens) for tokens in ACTION_TOKENS
}
assert sum(
    len(indices) for indices, _ in LENGTH_BUCKETS.values()
) % CHUNK_SIZE == 0
print(
    "gameplay will reuse the mining scorer: "
    f"scorer_sha256={SCORER_SHA256[:16]} over {len(ANSWERS)} answers in "
    f"{len(LENGTH_BUCKETS)} length buckets"
)
```

    gameplay will reuse the mining scorer: scorer_sha256=2e35dbf6edcad821 over 2315 answers in 3 length buckets



```python
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
```

## 19.24 Lab 18d trajectory replay

The kernel's numerical regression against Lab 18b and its 40-repeat memory soak
already ran in Section 19.10, because mining could not be allowed to start on an
unverified or leaking scorer. Repeating them here would burn model time to
re-answer a settled question.

What is still open is the *game loop* wrapped around that kernel. So before any
game runs, the seed-42 incumbent replays one complete constrained trajectory and
must reproduce Lab 18d's persisted guesses and solve turn exactly. That pins the
turn logic, the feedback filter, and the RAISE opening to a result an earlier lab
published, and it is the check that would catch a game-loop change even though
the scorer is unchanged.


```python
if RUN_EVALUATION:
    checker = load_eval_adapter(INCUMBENTS[42])
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
    assert Path(RESULTS_DIR / "scoring-soak-trace.csv").exists(), (
        "the Section 19.10 memory soak trace is missing; the kernel gate "
        "must run before gameplay"
    )
    release_model(checker)
    del checker
    print("game loop verified against Lab 18d")
else:
    print("trajectory replay skipped: evaluation disabled")
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 42 SHORE constrained trajectory reproduces Lab 18d


    game loop verified against Lab 18d


## 19.25 Restartable gameplay for the six arms

Each arm plays all 19 reserved answers under both decoders. Artifacts are
rewritten atomically after every completed game, so an interrupted run resumes
at the next unplayed answer instead of replaying finished ones. Constrained
score vectors are checkpointed at the same cadence with their `(answer, turn)`
keys, which makes every ranking in the analysis auditable without rerunning a
model.

The incumbent rows come from Lab 18d's persisted CSVs. Recomputing Lab 18d's
published summary from those rows is the check that the baseline being compared
against is the same baseline Lab 18d reported.


```python
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
```


```python
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
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 42 hard free SHORE: FAILED turn=None calls=1


    seed 42 hard free MIGHT: FAILED turn=None calls=1


    seed 42 hard free BRICK: FAILED turn=None calls=1


    seed 42 hard free GHOST: FAILED turn=None calls=1


    seed 42 hard free KNIFE: FAILED turn=None calls=1


    seed 42 hard free DOUBT: FAILED turn=None calls=1


    seed 42 hard free FLING: FAILED turn=None calls=1


    seed 42 hard free ROUND: FAILED turn=None calls=1


    seed 42 hard free CHAMP: FAILED turn=None calls=1


    seed 42 hard free WASTE: FAILED turn=None calls=1


    seed 42 hard free BLIND: FAILED turn=None calls=1


    seed 42 hard free POINT: FAILED turn=None calls=1


    seed 42 hard free SLATE: FAILED turn=None calls=1


    seed 42 hard free CRANE: FAILED turn=None calls=1


    seed 42 hard free APPLE: FAILED turn=None calls=1


    seed 42 hard free SHEEP: FAILED turn=None calls=1


    seed 42 hard free BANAL: FAILED turn=None calls=1


    seed 42 hard free ALLEY: FAILED turn=None calls=1


    seed 42 hard free AUDIO: FAILED turn=None calls=1


    seed 42 hard answer-constrained SHORE: FAILED turn=None calls=5


    seed 42 hard answer-constrained MIGHT: FAILED turn=None calls=5


    seed 42 hard answer-constrained BRICK: FAILED turn=None calls=5


    seed 42 hard answer-constrained GHOST: FAILED turn=None calls=5


    seed 42 hard answer-constrained KNIFE: FAILED turn=None calls=5


    seed 42 hard answer-constrained DOUBT: FAILED turn=None calls=5


    seed 42 hard answer-constrained FLING: FAILED turn=None calls=5


    seed 42 hard answer-constrained ROUND: FAILED turn=None calls=5


    seed 42 hard answer-constrained CHAMP: SOLVED turn=6 calls=5


    seed 42 hard answer-constrained WASTE: FAILED turn=None calls=5


    seed 42 hard answer-constrained BLIND: SOLVED turn=6 calls=5


    seed 42 hard answer-constrained POINT: FAILED turn=None calls=5


    seed 42 hard answer-constrained SLATE: SOLVED turn=3 calls=2


    seed 42 hard answer-constrained CRANE: SOLVED turn=6 calls=5


    seed 42 hard answer-constrained APPLE: FAILED turn=None calls=5


    seed 42 hard answer-constrained SHEEP: FAILED turn=None calls=5


    seed 42 hard answer-constrained BANAL: FAILED turn=None calls=5


    seed 42 hard answer-constrained ALLEY: FAILED turn=None calls=5


    seed 42 hard answer-constrained AUDIO: FAILED turn=None calls=5



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 42 value free SHORE: FAILED turn=None calls=1


    seed 42 value free MIGHT: FAILED turn=None calls=1


    seed 42 value free BRICK: FAILED turn=None calls=1


    seed 42 value free GHOST: FAILED turn=None calls=1


    seed 42 value free KNIFE: FAILED turn=None calls=1


    seed 42 value free DOUBT: FAILED turn=None calls=1


    seed 42 value free FLING: FAILED turn=None calls=1


    seed 42 value free ROUND: FAILED turn=None calls=1


    seed 42 value free CHAMP: FAILED turn=None calls=1


    seed 42 value free WASTE: FAILED turn=None calls=1


    seed 42 value free BLIND: FAILED turn=None calls=1


    seed 42 value free POINT: FAILED turn=None calls=1


    seed 42 value free SLATE: FAILED turn=None calls=1


    seed 42 value free CRANE: FAILED turn=None calls=1


    seed 42 value free APPLE: FAILED turn=None calls=1


    seed 42 value free SHEEP: FAILED turn=None calls=1


    seed 42 value free BANAL: FAILED turn=None calls=1


    seed 42 value free ALLEY: FAILED turn=None calls=1


    seed 42 value free AUDIO: FAILED turn=None calls=1


    seed 42 value answer-constrained SHORE: FAILED turn=None calls=5


    seed 42 value answer-constrained MIGHT: FAILED turn=None calls=5


    seed 42 value answer-constrained BRICK: FAILED turn=None calls=5


    seed 42 value answer-constrained GHOST: FAILED turn=None calls=5


    seed 42 value answer-constrained KNIFE: FAILED turn=None calls=5


    seed 42 value answer-constrained DOUBT: FAILED turn=None calls=5


    seed 42 value answer-constrained FLING: FAILED turn=None calls=5


    seed 42 value answer-constrained ROUND: FAILED turn=None calls=5


    seed 42 value answer-constrained CHAMP: FAILED turn=None calls=5


    seed 42 value answer-constrained WASTE: FAILED turn=None calls=5


    seed 42 value answer-constrained BLIND: FAILED turn=None calls=5


    seed 42 value answer-constrained POINT: FAILED turn=None calls=5


    seed 42 value answer-constrained SLATE: FAILED turn=None calls=5


    seed 42 value answer-constrained CRANE: FAILED turn=None calls=5


    seed 42 value answer-constrained APPLE: FAILED turn=None calls=5


    seed 42 value answer-constrained SHEEP: FAILED turn=None calls=5


    seed 42 value answer-constrained BANAL: FAILED turn=None calls=5


    seed 42 value answer-constrained ALLEY: FAILED turn=None calls=5


    seed 42 value answer-constrained AUDIO: FAILED turn=None calls=5



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45 hard free SHORE: FAILED turn=None calls=1


    seed 45 hard free MIGHT: FAILED turn=None calls=1


    seed 45 hard free BRICK: FAILED turn=None calls=1


    seed 45 hard free GHOST: FAILED turn=None calls=1


    seed 45 hard free KNIFE: FAILED turn=None calls=1


    seed 45 hard free DOUBT: FAILED turn=None calls=1


    seed 45 hard free FLING: FAILED turn=None calls=1


    seed 45 hard free ROUND: FAILED turn=None calls=1


    seed 45 hard free CHAMP: FAILED turn=None calls=1


    seed 45 hard free WASTE: FAILED turn=None calls=1


    seed 45 hard free BLIND: FAILED turn=None calls=1


    seed 45 hard free POINT: FAILED turn=None calls=1


    seed 45 hard free SLATE: FAILED turn=None calls=1


    seed 45 hard free CRANE: FAILED turn=None calls=1


    seed 45 hard free APPLE: FAILED turn=None calls=1


    seed 45 hard free SHEEP: FAILED turn=None calls=1


    seed 45 hard free BANAL: FAILED turn=None calls=1


    seed 45 hard free ALLEY: FAILED turn=None calls=1


    seed 45 hard free AUDIO: FAILED turn=None calls=1


    seed 45 hard answer-constrained SHORE: FAILED turn=None calls=5


    seed 45 hard answer-constrained MIGHT: FAILED turn=None calls=5


    seed 45 hard answer-constrained BRICK: SOLVED turn=6 calls=5


    seed 45 hard answer-constrained GHOST: FAILED turn=None calls=5


    seed 45 hard answer-constrained KNIFE: FAILED turn=None calls=5


    seed 45 hard answer-constrained DOUBT: FAILED turn=None calls=5


    seed 45 hard answer-constrained FLING: FAILED turn=None calls=5


    seed 45 hard answer-constrained ROUND: FAILED turn=None calls=5


    seed 45 hard answer-constrained CHAMP: FAILED turn=None calls=5


    seed 45 hard answer-constrained WASTE: SOLVED turn=6 calls=5


    seed 45 hard answer-constrained BLIND: FAILED turn=None calls=5


    seed 45 hard answer-constrained POINT: FAILED turn=None calls=5


    seed 45 hard answer-constrained SLATE: SOLVED turn=5 calls=4


    seed 45 hard answer-constrained CRANE: FAILED turn=None calls=5


    seed 45 hard answer-constrained APPLE: FAILED turn=None calls=5


    seed 45 hard answer-constrained SHEEP: FAILED turn=None calls=5


    seed 45 hard answer-constrained BANAL: FAILED turn=None calls=5


    seed 45 hard answer-constrained ALLEY: FAILED turn=None calls=5


    seed 45 hard answer-constrained AUDIO: SOLVED turn=5 calls=4



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45 value free SHORE: FAILED turn=None calls=1


    seed 45 value free MIGHT: FAILED turn=None calls=1


    seed 45 value free BRICK: FAILED turn=None calls=1


    seed 45 value free GHOST: FAILED turn=None calls=1


    seed 45 value free KNIFE: FAILED turn=None calls=1


    seed 45 value free DOUBT: FAILED turn=None calls=1


    seed 45 value free FLING: FAILED turn=None calls=1


    seed 45 value free ROUND: FAILED turn=None calls=1


    seed 45 value free CHAMP: FAILED turn=None calls=1


    seed 45 value free WASTE: FAILED turn=None calls=1


    seed 45 value free BLIND: FAILED turn=None calls=1


    seed 45 value free POINT: FAILED turn=None calls=1


    seed 45 value free SLATE: FAILED turn=None calls=1


    seed 45 value free CRANE: FAILED turn=None calls=1


    seed 45 value free APPLE: FAILED turn=None calls=1


    seed 45 value free SHEEP: FAILED turn=None calls=1


    seed 45 value free BANAL: FAILED turn=None calls=1


    seed 45 value free ALLEY: FAILED turn=None calls=1


    seed 45 value free AUDIO: FAILED turn=None calls=1


    seed 45 value answer-constrained SHORE: FAILED turn=None calls=5


    seed 45 value answer-constrained MIGHT: FAILED turn=None calls=5


    seed 45 value answer-constrained BRICK: FAILED turn=None calls=5


    seed 45 value answer-constrained GHOST: FAILED turn=None calls=5


    seed 45 value answer-constrained KNIFE: FAILED turn=None calls=5


    seed 45 value answer-constrained DOUBT: FAILED turn=None calls=5


    seed 45 value answer-constrained FLING: FAILED turn=None calls=5


    seed 45 value answer-constrained ROUND: FAILED turn=None calls=5


    seed 45 value answer-constrained CHAMP: FAILED turn=None calls=5


    seed 45 value answer-constrained WASTE: FAILED turn=None calls=5


    seed 45 value answer-constrained BLIND: FAILED turn=None calls=5


    seed 45 value answer-constrained POINT: FAILED turn=None calls=5


    seed 45 value answer-constrained SLATE: FAILED turn=None calls=5


    seed 45 value answer-constrained CRANE: FAILED turn=None calls=5


    seed 45 value answer-constrained APPLE: FAILED turn=None calls=5


    seed 45 value answer-constrained SHEEP: FAILED turn=None calls=5


    seed 45 value answer-constrained BANAL: FAILED turn=None calls=5


    seed 45 value answer-constrained ALLEY: FAILED turn=None calls=5


    seed 45 value answer-constrained AUDIO: FAILED turn=None calls=5



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 47 hard free SHORE: FAILED turn=None calls=1


    seed 47 hard free MIGHT: FAILED turn=None calls=1


    seed 47 hard free BRICK: FAILED turn=None calls=1


    seed 47 hard free GHOST: FAILED turn=None calls=1


    seed 47 hard free KNIFE: FAILED turn=None calls=1


    seed 47 hard free DOUBT: FAILED turn=None calls=1


    seed 47 hard free FLING: FAILED turn=None calls=1


    seed 47 hard free ROUND: FAILED turn=None calls=1


    seed 47 hard free CHAMP: FAILED turn=None calls=1


    seed 47 hard free WASTE: FAILED turn=None calls=1


    seed 47 hard free BLIND: FAILED turn=None calls=1


    seed 47 hard free POINT: FAILED turn=None calls=1


    seed 47 hard free SLATE: FAILED turn=None calls=1


    seed 47 hard free CRANE: FAILED turn=None calls=1


    seed 47 hard free APPLE: FAILED turn=None calls=1


    seed 47 hard free SHEEP: FAILED turn=None calls=1


    seed 47 hard free BANAL: FAILED turn=None calls=1


    seed 47 hard free ALLEY: FAILED turn=None calls=1


    seed 47 hard free AUDIO: FAILED turn=None calls=1


    seed 47 hard answer-constrained SHORE: FAILED turn=None calls=5


    seed 47 hard answer-constrained MIGHT: FAILED turn=None calls=5


    seed 47 hard answer-constrained BRICK: FAILED turn=None calls=5


    seed 47 hard answer-constrained GHOST: FAILED turn=None calls=5


    seed 47 hard answer-constrained KNIFE: FAILED turn=None calls=5


    seed 47 hard answer-constrained DOUBT: FAILED turn=None calls=5


    seed 47 hard answer-constrained FLING: FAILED turn=None calls=5


    seed 47 hard answer-constrained ROUND: FAILED turn=None calls=5


    seed 47 hard answer-constrained CHAMP: FAILED turn=None calls=5


    seed 47 hard answer-constrained WASTE: FAILED turn=None calls=5


    seed 47 hard answer-constrained BLIND: SOLVED turn=6 calls=5


    seed 47 hard answer-constrained POINT: FAILED turn=None calls=5


    seed 47 hard answer-constrained SLATE: FAILED turn=None calls=5


    seed 47 hard answer-constrained CRANE: FAILED turn=None calls=5


    seed 47 hard answer-constrained APPLE: FAILED turn=None calls=5


    seed 47 hard answer-constrained SHEEP: FAILED turn=None calls=5


    seed 47 hard answer-constrained BANAL: FAILED turn=None calls=5


    seed 47 hard answer-constrained ALLEY: FAILED turn=None calls=5


    seed 47 hard answer-constrained AUDIO: FAILED turn=None calls=5



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 47 value free SHORE: FAILED turn=None calls=1


    seed 47 value free MIGHT: FAILED turn=None calls=1


    seed 47 value free BRICK: FAILED turn=None calls=1


    seed 47 value free GHOST: FAILED turn=None calls=1


    seed 47 value free KNIFE: FAILED turn=None calls=1


    seed 47 value free DOUBT: FAILED turn=None calls=1


    seed 47 value free FLING: FAILED turn=None calls=1


    seed 47 value free ROUND: FAILED turn=None calls=1


    seed 47 value free CHAMP: FAILED turn=None calls=1


    seed 47 value free WASTE: FAILED turn=None calls=1


    seed 47 value free BLIND: FAILED turn=None calls=1


    seed 47 value free POINT: FAILED turn=None calls=1


    seed 47 value free SLATE: FAILED turn=None calls=1


    seed 47 value free CRANE: FAILED turn=None calls=1


    seed 47 value free APPLE: FAILED turn=None calls=1


    seed 47 value free SHEEP: FAILED turn=None calls=1


    seed 47 value free BANAL: FAILED turn=None calls=1


    seed 47 value free ALLEY: FAILED turn=None calls=1


    seed 47 value free AUDIO: FAILED turn=None calls=1


    seed 47 value answer-constrained SHORE: FAILED turn=None calls=5


    seed 47 value answer-constrained MIGHT: FAILED turn=None calls=5


    seed 47 value answer-constrained BRICK: FAILED turn=None calls=5


    seed 47 value answer-constrained GHOST: FAILED turn=None calls=5


    seed 47 value answer-constrained KNIFE: FAILED turn=None calls=5


    seed 47 value answer-constrained DOUBT: FAILED turn=None calls=5


    seed 47 value answer-constrained FLING: FAILED turn=None calls=5


    seed 47 value answer-constrained ROUND: FAILED turn=None calls=5


    seed 47 value answer-constrained CHAMP: FAILED turn=None calls=5


    seed 47 value answer-constrained WASTE: FAILED turn=None calls=5


    seed 47 value answer-constrained BLIND: FAILED turn=None calls=5


    seed 47 value answer-constrained POINT: FAILED turn=None calls=5


    seed 47 value answer-constrained SLATE: FAILED turn=None calls=5


    seed 47 value answer-constrained CRANE: FAILED turn=None calls=5


    seed 47 value answer-constrained APPLE: FAILED turn=None calls=5


    seed 47 value answer-constrained SHEEP: FAILED turn=None calls=5


    seed 47 value answer-constrained BANAL: FAILED turn=None calls=5


    seed 47 value answer-constrained ALLEY: FAILED turn=None calls=5


    seed 47 value answer-constrained AUDIO: FAILED turn=None calls=5


    Lab 18d incumbent summaries reproduce from persisted game rows
    calls: 1011 games: 342


## 19.26 Gameplay results

Each row is one adapter playing 19 reserved answers. The seed is the
replication unit: three seeds give three paired observations of an arm against
its own ancestor, and a direction that appears for one seed only is not a
finding.

Turn 2 is the only state every arm visits with an identical history, because
all games open with RAISE. Later turns are conditional on survival and follow
arm-specific trajectories, so they describe deployed behavior rather than a
controlled one-step contrast.

Gameplay is the one measurement in this lab that is directly comparable across
seeds. It ranges over all 2,315 answers with no support at all, so the
seed-specific twelve-word training boundary does not enter the metric. Two arms
trained against different mined supports are still being asked the same
question here.


```python
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
      <th>arm</th>
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
      <td>incumbent</td>
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
      <th>1</th>
      <td>42</td>
      <td>incumbent</td>
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
      <th>2</th>
      <td>42</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>4</td>
      <td>0.210526</td>
      <td>0.000000</td>
      <td>4.842105</td>
      <td>1.210526</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>1.315789</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>42</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>incumbent</td>
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
      <th>7</th>
      <td>45</td>
      <td>incumbent</td>
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
      <th>8</th>
      <td>45</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>4</td>
      <td>0.210526</td>
      <td>0.000000</td>
      <td>4.894737</td>
      <td>1.000000</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>45</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>1.315789</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>45</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>47</td>
      <td>incumbent</td>
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
      <th>13</th>
      <td>47</td>
      <td>incumbent</td>
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
      <th>14</th>
      <td>47</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>1</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>1.210526</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>47</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>47</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>1.421053</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>47</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
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
      <th>arm</th>
      <th>incumbent</th>
      <th>hard</th>
      <th>value</th>
    </tr>
    <tr>
      <th>decoder</th>
      <th>seed</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">answer-constrained</th>
      <th>42</th>
      <td>10.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>10.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">free</th>
      <th>42</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>left_arm</th>
      <th>right_arm</th>
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
      <td>incumbent</td>
      <td>hard</td>
      <td>5</td>
      <td>0</td>
      <td>-0.263158</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>free</td>
      <td>incumbent</td>
      <td>value</td>
      <td>5</td>
      <td>0</td>
      <td>-0.263158</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>free</td>
      <td>hard</td>
      <td>value</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>incumbent</td>
      <td>hard</td>
      <td>10</td>
      <td>4</td>
      <td>-0.315789</td>
      <td>6</td>
      <td>0</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>incumbent</td>
      <td>value</td>
      <td>10</td>
      <td>0</td>
      <td>-0.526316</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>42</td>
      <td>answer-constrained</td>
      <td>hard</td>
      <td>value</td>
      <td>4</td>
      <td>0</td>
      <td>-0.210526</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>free</td>
      <td>incumbent</td>
      <td>hard</td>
      <td>3</td>
      <td>0</td>
      <td>-0.157895</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>free</td>
      <td>incumbent</td>
      <td>value</td>
      <td>3</td>
      <td>0</td>
      <td>-0.157895</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>8</th>
      <td>45</td>
      <td>free</td>
      <td>hard</td>
      <td>value</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>incumbent</td>
      <td>hard</td>
      <td>10</td>
      <td>4</td>
      <td>-0.315789</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>incumbent</td>
      <td>value</td>
      <td>10</td>
      <td>0</td>
      <td>-0.526316</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>11</th>
      <td>45</td>
      <td>answer-constrained</td>
      <td>hard</td>
      <td>value</td>
      <td>4</td>
      <td>0</td>
      <td>-0.210526</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>12</th>
      <td>47</td>
      <td>free</td>
      <td>incumbent</td>
      <td>hard</td>
      <td>3</td>
      <td>0</td>
      <td>-0.157895</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>free</td>
      <td>incumbent</td>
      <td>value</td>
      <td>3</td>
      <td>0</td>
      <td>-0.157895</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>14</th>
      <td>47</td>
      <td>free</td>
      <td>hard</td>
      <td>value</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>15</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>incumbent</td>
      <td>hard</td>
      <td>10</td>
      <td>1</td>
      <td>-0.473684</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>incumbent</td>
      <td>value</td>
      <td>10</td>
      <td>0</td>
      <td>-0.526316</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>17</th>
      <td>47</td>
      <td>answer-constrained</td>
      <td>hard</td>
      <td>value</td>
      <td>1</td>
      <td>0</td>
      <td>-0.052632</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



```python
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
      <th>arm</th>
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
      <td>incumbent</td>
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
      <th>1</th>
      <td>42</td>
      <td>incumbent</td>
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
      <th>2</th>
      <td>42</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>92</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.086957</td>
      <td>0.086957</td>
      <td>0.141304</td>
      <td>0.043478</td>
      <td>13.782609</td>
      <td>3.826087</td>
      <td>1.030033</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0.842105</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>49.421053</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>95</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.021053</td>
      <td>0.021053</td>
      <td>0.031579</td>
      <td>0.000000</td>
      <td>15.568421</td>
      <td>5.947368</td>
      <td>0.986980</td>
    </tr>
    <tr>
      <th>5</th>
      <td>42</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0.631579</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>49.421053</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>incumbent</td>
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
      <th>7</th>
      <td>45</td>
      <td>incumbent</td>
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
      <th>8</th>
      <td>45</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>93</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.096774</td>
      <td>0.096774</td>
      <td>0.075269</td>
      <td>0.032258</td>
      <td>13.354839</td>
      <td>3.462366</td>
      <td>1.057506</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0.947368</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>49.421053</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>45</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>95</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.021053</td>
      <td>0.021053</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.452632</td>
      <td>4.831579</td>
      <td>0.997506</td>
    </tr>
    <tr>
      <th>11</th>
      <td>45</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0.684211</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>49.421053</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>47</td>
      <td>incumbent</td>
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
    <tr>
      <th>13</th>
      <td>47</td>
      <td>incumbent</td>
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
      <th>14</th>
      <td>47</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>95</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.063158</td>
      <td>0.063158</td>
      <td>0.042105</td>
      <td>0.010526</td>
      <td>13.915789</td>
      <td>4.273684</td>
      <td>1.001875</td>
    </tr>
    <tr>
      <th>15</th>
      <td>47</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0.947368</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>49.421053</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>47</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>95</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.010526</td>
      <td>0.010526</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.168421</td>
      <td>5.568421</td>
      <td>0.968696</td>
    </tr>
    <tr>
      <th>17</th>
      <td>47</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0.842105</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>49.421053</td>
      <td>0.000000</td>
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
      <th>arm</th>
      <th>decoder</th>
      <th>turn</th>
      <th>calls</th>
      <th>usable_rate</th>
      <th>consistency_rate</th>
      <th>repeat_rate</th>
      <th>teacher_match_rate</th>
      <th>candidate_choice_rate</th>
      <th>mean_candidates_before</th>
      <th>mean_candidates_after</th>
      <th>mean_entropy_gap_bits</th>
      <th>mean_realized_log2_reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>2</td>
      <td>57</td>
      <td>0.263158</td>
      <td>0.263158</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.263158</td>
      <td>49.421053</td>
      <td>7.596491</td>
      <td>0.944820</td>
      <td>2.830786</td>
    </tr>
    <tr>
      <th>1</th>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>54</td>
      <td>0.351852</td>
      <td>0.351852</td>
      <td>0.000000</td>
      <td>0.074074</td>
      <td>0.351852</td>
      <td>7.962963</td>
      <td>2.574074</td>
      <td>0.557501</td>
      <td>1.462133</td>
    </tr>
    <tr>
      <th>2</th>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>4</td>
      <td>50</td>
      <td>0.280000</td>
      <td>0.280000</td>
      <td>0.180000</td>
      <td>0.160000</td>
      <td>0.280000</td>
      <td>2.700000</td>
      <td>1.720000</td>
      <td>0.386302</td>
      <td>0.513146</td>
    </tr>
    <tr>
      <th>3</th>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>5</td>
      <td>41</td>
      <td>0.439024</td>
      <td>0.439024</td>
      <td>0.243902</td>
      <td>0.268293</td>
      <td>0.439024</td>
      <td>1.878049</td>
      <td>1.439024</td>
      <td>0.184999</td>
      <td>0.359438</td>
    </tr>
    <tr>
      <th>4</th>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>6</td>
      <td>31</td>
      <td>0.225806</td>
      <td>0.225806</td>
      <td>0.419355</td>
      <td>0.225806</td>
      <td>0.225806</td>
      <td>1.580645</td>
      <td>1.451613</td>
      <td>0.238588</td>
      <td>0.088867</td>
    </tr>
    <tr>
      <th>5</th>
      <td>incumbent</td>
      <td>free</td>
      <td>2</td>
      <td>57</td>
      <td>0.157895</td>
      <td>0.157895</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.157895</td>
      <td>49.421053</td>
      <td>32.070175</td>
      <td>0.912549</td>
      <td>1.197461</td>
    </tr>
    <tr>
      <th>6</th>
      <td>incumbent</td>
      <td>free</td>
      <td>3</td>
      <td>25</td>
      <td>0.280000</td>
      <td>0.280000</td>
      <td>0.000000</td>
      <td>0.120000</td>
      <td>0.280000</td>
      <td>8.160000</td>
      <td>4.600000</td>
      <td>0.192900</td>
      <td>1.097987</td>
    </tr>
    <tr>
      <th>7</th>
      <td>incumbent</td>
      <td>free</td>
      <td>4</td>
      <td>10</td>
      <td>0.400000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.400000</td>
      <td>0.400000</td>
      <td>1.500000</td>
      <td>1.300000</td>
      <td>0.407407</td>
      <td>0.158496</td>
    </tr>
    <tr>
      <th>8</th>
      <td>incumbent</td>
      <td>free</td>
      <td>5</td>
      <td>5</td>
      <td>0.600000</td>
      <td>0.600000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.600000</td>
      <td>1.600000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>incumbent</td>
      <td>free</td>
      <td>6</td>
      <td>2</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>2</td>
      <td>57</td>
      <td>0.087719</td>
      <td>0.087719</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.087719</td>
      <td>49.421053</td>
      <td>10.701754</td>
      <td>1.255357</td>
      <td>2.344359</td>
    </tr>
    <tr>
      <th>11</th>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>57</td>
      <td>0.122807</td>
      <td>0.122807</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.122807</td>
      <td>10.701754</td>
      <td>3.543860</td>
      <td>0.833306</td>
      <td>1.570689</td>
    </tr>
    <tr>
      <th>12</th>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>4</td>
      <td>56</td>
      <td>0.035714</td>
      <td>0.035714</td>
      <td>0.053571</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>3.589286</td>
      <td>2.375000</td>
      <td>0.492913</td>
      <td>0.491131</td>
    </tr>
    <tr>
      <th>13</th>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>5</td>
      <td>56</td>
      <td>0.035714</td>
      <td>0.035714</td>
      <td>0.125000</td>
      <td>0.035714</td>
      <td>0.035714</td>
      <td>2.375000</td>
      <td>1.303571</td>
      <td>0.224858</td>
      <td>0.551308</td>
    </tr>
    <tr>
      <th>14</th>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>6</td>
      <td>54</td>
      <td>0.129630</td>
      <td>0.129630</td>
      <td>0.240741</td>
      <td>0.092593</td>
      <td>0.129630</td>
      <td>1.314815</td>
      <td>1.148148</td>
      <td>0.096148</td>
      <td>0.125091</td>
    </tr>
    <tr>
      <th>15</th>
      <td>hard</td>
      <td>free</td>
      <td>2</td>
      <td>57</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>49.421053</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>value</td>
      <td>answer-constrained</td>
      <td>2</td>
      <td>57</td>
      <td>0.035088</td>
      <td>0.035088</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035088</td>
      <td>49.421053</td>
      <td>13.491228</td>
      <td>1.789505</td>
      <td>1.892447</td>
    </tr>
    <tr>
      <th>17</th>
      <td>value</td>
      <td>answer-constrained</td>
      <td>3</td>
      <td>57</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017544</td>
      <td>13.491228</td>
      <td>7.228070</td>
      <td>1.369171</td>
      <td>1.095719</td>
    </tr>
    <tr>
      <th>18</th>
      <td>value</td>
      <td>answer-constrained</td>
      <td>4</td>
      <td>57</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017544</td>
      <td>7.228070</td>
      <td>3.298246</td>
      <td>0.621002</td>
      <td>1.089657</td>
    </tr>
    <tr>
      <th>19</th>
      <td>value</td>
      <td>answer-constrained</td>
      <td>5</td>
      <td>57</td>
      <td>0.017544</td>
      <td>0.017544</td>
      <td>0.035088</td>
      <td>0.000000</td>
      <td>0.017544</td>
      <td>3.298246</td>
      <td>1.877193</td>
      <td>0.395929</td>
      <td>0.581388</td>
    </tr>
    <tr>
      <th>20</th>
      <td>value</td>
      <td>answer-constrained</td>
      <td>6</td>
      <td>57</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017544</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.877193</td>
      <td>1.350877</td>
      <td>0.180471</td>
      <td>0.262759</td>
    </tr>
    <tr>
      <th>21</th>
      <td>value</td>
      <td>free</td>
      <td>2</td>
      <td>57</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>49.421053</td>
      <td>NaN</td>
      <td>0.000000</td>
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
      <th>arm</th>
      <th>decoder</th>
      <th>calls</th>
      <th>usable_rate</th>
      <th>teacher_match_rate</th>
      <th>chosen_candidate_rate</th>
      <th>mean_open_entropy_regret_bits</th>
      <th>median_open_entropy_regret_bits</th>
      <th>mean_chosen_entropy_bits</th>
      <th>mean_realized_log2_reduction</th>
      <th>mean_candidates_after</th>
      <th>mean_teacher_rank</th>
      <th>mean_candidate_mass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.210526</td>
      <td>0.0</td>
      <td>0.210526</td>
      <td>1.158016</td>
      <td>1.135290</td>
      <td>2.792719</td>
      <td>2.807769</td>
      <td>7.578947</td>
      <td>281.263158</td>
      <td>0.290236</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>incumbent</td>
      <td>free</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.131867</td>
      <td>0.923485</td>
      <td>39.736842</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.052632</td>
      <td>0.0</td>
      <td>0.052632</td>
      <td>1.531053</td>
      <td>1.508056</td>
      <td>2.419681</td>
      <td>2.408087</td>
      <td>10.526316</td>
      <td>369.526316</td>
      <td>0.081272</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.052632</td>
      <td>0.0</td>
      <td>0.052632</td>
      <td>2.389393</td>
      <td>2.588626</td>
      <td>1.561341</td>
      <td>1.657780</td>
      <td>15.105263</td>
      <td>618.894737</td>
      <td>0.042210</td>
    </tr>
    <tr>
      <th>5</th>
      <td>42</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.368421</td>
      <td>0.0</td>
      <td>0.368421</td>
      <td>1.051316</td>
      <td>1.052189</td>
      <td>2.899418</td>
      <td>3.317823</td>
      <td>5.157895</td>
      <td>243.578947</td>
      <td>0.288871</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>incumbent</td>
      <td>free</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.106453</td>
      <td>1.555673</td>
      <td>21.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>45</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>1.445854</td>
      <td>1.417210</td>
      <td>2.504881</td>
      <td>2.557809</td>
      <td>9.631579</td>
      <td>437.526316</td>
      <td>0.077681</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>45</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.052632</td>
      <td>0.0</td>
      <td>0.052632</td>
      <td>1.760048</td>
      <td>1.475489</td>
      <td>2.190687</td>
      <td>2.276944</td>
      <td>11.736842</td>
      <td>604.473684</td>
      <td>0.045061</td>
    </tr>
    <tr>
      <th>11</th>
      <td>45</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>47</td>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.210526</td>
      <td>0.0</td>
      <td>0.210526</td>
      <td>1.446406</td>
      <td>1.485475</td>
      <td>2.504328</td>
      <td>2.366765</td>
      <td>10.052632</td>
      <td>285.947368</td>
      <td>0.269467</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>incumbent</td>
      <td>free</td>
      <td>19</td>
      <td>0.157895</td>
      <td>0.0</td>
      <td>0.157895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.252880</td>
      <td>1.113225</td>
      <td>35.473684</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>47</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.052632</td>
      <td>0.0</td>
      <td>0.052632</td>
      <td>1.610442</td>
      <td>1.534860</td>
      <td>2.340293</td>
      <td>2.067181</td>
      <td>11.947368</td>
      <td>480.684211</td>
      <td>0.071871</td>
    </tr>
    <tr>
      <th>15</th>
      <td>47</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>47</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2.040350</td>
      <td>2.192424</td>
      <td>1.910384</td>
      <td>1.742616</td>
      <td>13.631579</td>
      <td>682.368421</td>
      <td>0.036878</td>
    </tr>
    <tr>
      <th>17</th>
      <td>47</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>49.421053</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


### Late closure

A singleton state has exactly one word left. Naming it wins immediately, and
entropy cannot distinguish it because every action has the same expected
information gain of zero bits over a one-word candidate set. The controlled
game-level metric is whether the model closes on its first singleton
opportunity. Pooling all singleton calls is also reported, but a failure can
create more calls and therefore change its own denominator.


```python
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
      <th>arm</th>
      <th>decoder</th>
      <th>singleton_calls</th>
      <th>sole_candidate_rate</th>
      <th>usable_rate</th>
      <th>repeat_rate</th>
      <th>games_reaching_singleton</th>
      <th>first_singleton_close_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>25</td>
      <td>0.240000</td>
      <td>0.240000</td>
      <td>0.400000</td>
      <td>13</td>
      <td>0.384615</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>incumbent</td>
      <td>free</td>
      <td>5</td>
      <td>0.600000</td>
      <td>0.600000</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>36</td>
      <td>0.111111</td>
      <td>0.111111</td>
      <td>0.305556</td>
      <td>16</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>26</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.038462</td>
      <td>13</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>0.315789</td>
      <td>0.315789</td>
      <td>0.263158</td>
      <td>11</td>
      <td>0.454545</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45</td>
      <td>incumbent</td>
      <td>free</td>
      <td>5</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>37</td>
      <td>0.054054</td>
      <td>0.054054</td>
      <td>0.189189</td>
      <td>15</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>33</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47</td>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>21</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>12</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>47</td>
      <td>incumbent</td>
      <td>free</td>
      <td>2</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>34</td>
      <td>0.029412</td>
      <td>0.029412</td>
      <td>0.088235</td>
      <td>16</td>
      <td>0.062500</td>
    </tr>
    <tr>
      <th>11</th>
      <td>47</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>22</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11</td>
      <td>0.000000</td>
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
      <th>arm</th>
      <th>decoder</th>
      <th>failures</th>
      <th>ended_at_singleton</th>
      <th>mean_final_candidates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>9</td>
      <td>7</td>
      <td>1.444444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>incumbent</td>
      <td>free</td>
      <td>14</td>
      <td>1</td>
      <td>50.214286</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>15</td>
      <td>12</td>
      <td>1.266667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>49.421053</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>16</td>
      <td>1.315789</td>
    </tr>
    <tr>
      <th>5</th>
      <td>42</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>49.421053</td>
    </tr>
    <tr>
      <th>6</th>
      <td>45</td>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>9</td>
      <td>5</td>
      <td>1.666667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>45</td>
      <td>incumbent</td>
      <td>free</td>
      <td>16</td>
      <td>3</td>
      <td>22.875000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>45</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>15</td>
      <td>15</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>49.421053</td>
    </tr>
    <tr>
      <th>10</th>
      <td>45</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>17</td>
      <td>1.315789</td>
    </tr>
    <tr>
      <th>11</th>
      <td>45</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>49.421053</td>
    </tr>
    <tr>
      <th>12</th>
      <td>47</td>
      <td>incumbent</td>
      <td>answer-constrained</td>
      <td>9</td>
      <td>6</td>
      <td>1.444444</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>incumbent</td>
      <td>free</td>
      <td>16</td>
      <td>0</td>
      <td>40.875000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>47</td>
      <td>hard</td>
      <td>answer-constrained</td>
      <td>18</td>
      <td>16</td>
      <td>1.222222</td>
    </tr>
    <tr>
      <th>15</th>
      <td>47</td>
      <td>hard</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>49.421053</td>
    </tr>
    <tr>
      <th>16</th>
      <td>47</td>
      <td>value</td>
      <td>answer-constrained</td>
      <td>19</td>
      <td>14</td>
      <td>1.421053</td>
    </tr>
    <tr>
      <th>17</th>
      <td>47</td>
      <td>value</td>
      <td>free</td>
      <td>19</td>
      <td>0</td>
      <td>49.421053</td>
    </tr>
  </tbody>
</table>
</div>


## 19.27 Preregistered read-out

The rules below were fixed in Section 19.1, before any Lab 19 model was
trained. The primary outcome is answer-constrained solve rate. The two
mechanism metrics are late closure, measured as the sole-candidate selection
rate on each game's first constrained singleton opportunity, and broad action
value, measured as mean Turn 2 open-entropy regret in bits where lower is
better.

A direction counts only when it holds for all three seeds. With three seeds
there is no useful significance test, so replication across seeds is the
standard, and dev states are never treated as independent replications.

The `hard` versus `value` contrast is causal inside a seed and only inside a
seed. Within one seed the arms share one incumbent, one mined support, one state
order, one encoding, and one RNG stream, so target shape is the only thing that
differs and any gap is attributable to it. Across seeds the supports differ,
because each was mined from a different incumbent's own preferences. The three
seeds are therefore three instances of the same policy-adaptive treatment rather
than three copies of one fixed treatment, and replication means the intervention
reproduced against three independently drawn decision boundaries.

That makes a `value` win a specific claim: richer target shape beat one-hot
*against the same seed-specific decision boundary*, on the words that seed's own
incumbent was actually most tempted by. It is not a claim about a shared
twelve-word action set, because there is no shared action set.


```python
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
    "\nSeeds are the replication unit. Dev states and per-answer flips are "
    "supporting detail, not independent evidence."
)
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
      <th>metric</th>
      <th>better_when</th>
      <th>arm</th>
      <th>baseline</th>
      <th>seeds_with_data</th>
      <th>seeds_better</th>
      <th>replicated</th>
      <th>delta_seed42</th>
      <th>delta_seed45</th>
      <th>delta_seed47</th>
      <th>mean_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>solve_rate</td>
      <td>higher</td>
      <td>hard</td>
      <td>incumbent</td>
      <td>3</td>
      <td>0</td>
      <td>False</td>
      <td>-0.315789</td>
      <td>-0.315789</td>
      <td>-0.473684</td>
      <td>-0.368421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>closure_rate</td>
      <td>higher</td>
      <td>hard</td>
      <td>incumbent</td>
      <td>3</td>
      <td>0</td>
      <td>False</td>
      <td>-0.197115</td>
      <td>-0.387879</td>
      <td>-0.437500</td>
      <td>-0.340831</td>
    </tr>
    <tr>
      <th>2</th>
      <td>turn2_open_regret_bits</td>
      <td>lower</td>
      <td>hard</td>
      <td>incumbent</td>
      <td>3</td>
      <td>0</td>
      <td>False</td>
      <td>0.373037</td>
      <td>0.394538</td>
      <td>0.164035</td>
      <td>0.310537</td>
    </tr>
    <tr>
      <th>3</th>
      <td>solve_rate</td>
      <td>higher</td>
      <td>value</td>
      <td>incumbent</td>
      <td>3</td>
      <td>0</td>
      <td>False</td>
      <td>-0.526316</td>
      <td>-0.526316</td>
      <td>-0.526316</td>
      <td>-0.526316</td>
    </tr>
    <tr>
      <th>4</th>
      <td>closure_rate</td>
      <td>higher</td>
      <td>value</td>
      <td>incumbent</td>
      <td>3</td>
      <td>0</td>
      <td>False</td>
      <td>-0.384615</td>
      <td>-0.454545</td>
      <td>-0.500000</td>
      <td>-0.446387</td>
    </tr>
    <tr>
      <th>5</th>
      <td>turn2_open_regret_bits</td>
      <td>lower</td>
      <td>value</td>
      <td>incumbent</td>
      <td>3</td>
      <td>0</td>
      <td>False</td>
      <td>1.231377</td>
      <td>0.708732</td>
      <td>0.593944</td>
      <td>0.844684</td>
    </tr>
    <tr>
      <th>6</th>
      <td>solve_rate</td>
      <td>higher</td>
      <td>value</td>
      <td>hard</td>
      <td>3</td>
      <td>0</td>
      <td>False</td>
      <td>-0.210526</td>
      <td>-0.210526</td>
      <td>-0.052632</td>
      <td>-0.157895</td>
    </tr>
    <tr>
      <th>7</th>
      <td>closure_rate</td>
      <td>higher</td>
      <td>value</td>
      <td>hard</td>
      <td>3</td>
      <td>0</td>
      <td>False</td>
      <td>-0.187500</td>
      <td>-0.066667</td>
      <td>-0.062500</td>
      <td>-0.105556</td>
    </tr>
    <tr>
      <th>8</th>
      <td>turn2_open_regret_bits</td>
      <td>lower</td>
      <td>value</td>
      <td>hard</td>
      <td>3</td>
      <td>0</td>
      <td>False</td>
      <td>0.858340</td>
      <td>0.314194</td>
      <td>0.429908</td>
      <td>0.534147</td>
    </tr>
    <tr>
      <th>9</th>
      <td>solve_rate</td>
      <td>higher</td>
      <td>hard</td>
      <td>value</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>0.210526</td>
      <td>0.210526</td>
      <td>0.052632</td>
      <td>0.157895</td>
    </tr>
    <tr>
      <th>10</th>
      <td>closure_rate</td>
      <td>higher</td>
      <td>hard</td>
      <td>value</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>0.187500</td>
      <td>0.066667</td>
      <td>0.062500</td>
      <td>0.105556</td>
    </tr>
    <tr>
      <th>11</th>
      <td>turn2_open_regret_bits</td>
      <td>lower</td>
      <td>hard</td>
      <td>value</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>-0.858340</td>
      <td>-0.314194</td>
      <td>-0.429908</td>
      <td>-0.534147</td>
    </tr>
  </tbody>
</table>
</div>


    Neither arm produced a replicated improvement over its incumbent on a preregistered metric. The objective, action support, optimization budget, and limited broad-state coverage remain live explanations; this result would not distinguish hard from soft distillation.
    
    solve rate (answer-constrained) by seed and arm:
      seed 42: incumbent 10/19, hard 4/19, value 0/19
      seed 45: incumbent 10/19, hard 4/19, value 0/19
      seed 47: incumbent 10/19, hard 1/19, value 0/19
    
    Seeds are the replication unit. Dev states and per-answer flips are supporting detail, not independent evidence.


## 19.28 Persist the lab

Everything a later lab needs is written under `results/lab19`: the per-call
gameplay table, the dev shortlist rows and their raw 12-action score matrices,
the training histories, the summary tables, the mined incumbent rankings, the
mining and divergence audits, and a manifest recording the source hashes,
incumbent hashes, trained-adapter hashes, scorer fingerprint, state-order
fingerprint, and per-seed support and target fingerprints.

A later lab that wants to reuse a seed's action boundary has everything it
needs: the mined top-32 arrays under `incumbent-ranking-seed*`, the state order
they were mined in, and the hash of the scorer that produced them.


```python
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
atomic_csv(mining_audit, RESULTS_DIR / "mining-audit.csv")
atomic_csv(retained, RESULTS_DIR / "mining-retained-actions.csv")
atomic_csv(regret_audit, RESULTS_DIR / "mining-regret-audit.csv")
atomic_csv(sharp_proof, RESULTS_DIR / "mining-sharp-proof.csv")
atomic_csv(divergence, RESULTS_DIR / "support-divergence.csv")

run_manifest = {
    "experiment": "Lab 19 seed-specific value-aware distillation",
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
    "support_source": "per-seed frozen incumbent full-answer-list ranking",
    "support_slots": {
        "broad_open_teacher": BROAD_OPEN_SLOTS,
        "broad_candidate_teacher": BROAD_CANDIDATE_SLOTS,
        "broad_incumbent": BROAD_INCUMBENT_SLOTS,
        "sharp_incumbent": SHARP_INCUMBENT_SLOTS,
        "sharp_random": SHARP_RANDOM_SLOTS,
    },
    "mining_top_k": MINING_TOP_K,
    "mining_checkpoint_every": MINING_CHECKPOINT_EVERY,
    "scorer_sha256": SCORER_SHA256,
    "state_order_sha256": STATE_ORDER_SHA256,
    "training_stream_sha256": {
        str(seed): support_fingerprints[(seed, "train")] for seed in SEEDS
    },
    "dev_stream_sha256": {
        str(seed): support_fingerprints[(seed, "dev")] for seed in SEEDS
    },
    "ranking_sha256": {
        f"seed{seed}-{split}": ranking_hashes[(seed, split)]
        for seed in SEEDS
        for split in SPLITS
    },
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
```

    written to ../results/lab19


## Lab 19 checkpoint

The action boundary in this lab is incumbent-adaptive. Each seed's twelve words
per state were mined from that seed's own frozen incumbent by ranking all 2,315
answers with the verified scorer, so the three seeds do not share an action set.
What *is* shared is the thing the causal claim needs: within a seed, the `hard`
and `value` arms saw one identical support, in one identical state order, with
one identical encoding and RNG stream, and differed only in target
probabilities.

So read the seed as the replication unit and read state-level pairing as
within-seed only. Cross-seed tables summarize three separate paired experiments.
Gameplay is the exception: it ranges over all 2,315 answers with no support, so
it is directly comparable everywhere.

Read the result in this order:

1. Did either arm raise answer-constrained solve rate above its own incumbent
   on all three seeds?
2. Did the first-singleton close rate rise? That is the closure defect Lab 18d
   isolated, and the sharp value target is its direct treatment.
3. Did Turn 2 open-entropy regret fall? That is the broad-state action-value
   defect, and Turn 2 is the only paired state across arms.
4. Did `student_incumbent_mass` fall and `incumbent_top1_rank` rise? Those are
   the hard negatives working: probability moving off the exact words that
   seed's ancestor preferred. If they did not move, the mined slots were not
   the operative part of the update.
5. Did the arms separate from each other, or did they move together? Moving
   together points at regime coverage and extra updates rather than at target
   shape.
6. Did the dev shortlist improve while gameplay did not? That is the Lab 09
   pattern repeating: winning a scored shortlist is not the same as choosing
   well from 2,315 words. The mined supports make this test sharper than Lab
   09's, because a shortlist win now has to be won against the student's own
   ancestor's favourites rather than against filler.

What this lab cannot tell you: whether a different support, a different
temperature, or more updates would help. One support rule and one temperature
were fixed in advance precisely so that the hard-versus-value contrast is
interpretable, and it cannot tell you what would happen against a support mined
from some *other* policy's preferences, because the support is a function of the
incumbent by design. The reserved 19 answers were held out of every target,
every mined ranking, every audit, and every tuning decision so that the final
number means what it says.
