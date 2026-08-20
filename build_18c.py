"""Generate notebooks/18c_seed_replication.ipynb."""

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
# Lab 18c - Does the constrained policy replicate across seeds?

Lab 18b found that free generation hid substantial learned policy signal.
B-structured rose from 14.5% usable under free generation to 30.3% when all
2,315 answer words were ranked directly. The base model remained at 0.16%, so
the adapter learned state-conditioned ranking rather than borrowing competence
from the answer list.

That conclusion still rests on one training seed. This lab reproduces the
B-structured run twice, changing only the training seed, and evaluates all three
adapters on the same 620 held-out states.
""")

md("""
## 18c.1 Pre-registered experiment

**Question.** Does the roughly 30% constrained-policy capability replicate, and
how variable is the weaker entropy-ranking signal?

**Arms.**

| label | seed | source |
| --- | ---: | --- |
| B-seed-42 | 42 | frozen Lab 17 checkpoint |
| B-seed-45 | 45 | trained here |
| B-seed-47 | 47 | trained here |

The seed controls initialization, LoRA dropout, and shuffled row order. Everything
else is fixed to Lab 17: Dataset B, `derived_state_v1`, model, optimizer, LoRA
configuration, batch sizes, schedule, and 1,029 optimizer steps.

Lab 17 seeds epoch shuffles with `seed + epoch` and the run spans two epochs.
Adjacent seeds would therefore share an entire epoch order. Seeds 45 and 47 are
the nearest pair whose epoch-shuffle seeds `{45, 46}` and `{47, 48}` are
disjoint from seed 42's `{42, 43}` and from each other.

**Primary outputs.**

1. Tier 1 usable rate over all 2,315 answer words.
2. Mean candidate-rank percentile. Lower is better; 0.5 is state-blind.
3. Tier 2 teacher-match rate among consistent candidates.
4. Free-generation usable rate as a secondary continuity measure.

Report every seed, the mean, sample standard deviation, minimum, maximum, range,
and paired state-level differences between seeds. Three seeds estimate whether
the effects under discussion are larger than ordinary run-to-run spread; they
do not provide a precise population variance.

The sample standard deviation is reported with its 95% upper confidence bound
for the population standard deviation. With only three seeds that upper bound
is 6.29 times the observed sample standard deviation. A small point estimate
does not prove that seed noise is small.

**Read before seeing results.**

- Tier 1 near 30% for all seeds means the hidden constrained policy replicates.
- Tier 1 observed spread comparable to the effects claimed since Lab 15 means
  those single-seed comparisons cannot support directional conclusions. If only
  the 95% upper bound on seed standard deviation overlaps those effects, the
  three-seed result is inconclusive rather than evidence of low variance.
- Stable Tier 1 with unstable or chance-level Tier 2 means legality learning
  replicates but strategic ranking does not.
- Stable Tier 1 and Tier 2 support an efficient constrained decoder and full
  gameplay before another training intervention.
- Dataset G is not retrained. Its seed-42 Lab 18b result is only a reference;
  no B-versus-G claim becomes a seed-level comparison until G is replicated.

This lab does not redesign Lab 19. It establishes the variance that any revised
ranking or distillation experiment must beat.
""")

md("""
## 18c.2 Run controls and memory guard

Run this notebook only through the total-system watchdog:

```
scripts/memguard.py --min-free 64 -- uv run jupyter nbconvert \\
    --to notebook --execute --inplace notebooks/18c_seed_replication.ipynb
```

The in-process MPS cap turns a runaway allocation into a normal exception. Two
fixed-shape gates run before full scale: a 40-step training soak on one longest
batch and a 40-state scoring soak on one longest prompt. Largest real prompts
are checked separately for scoring headroom.
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
from itertools import combinations
from pathlib import Path
import gc
import hashlib
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from IPython.display import display
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_wordle.game import Turn, filter_candidates, is_consistent
from tiny_wordle.hardware import preferred_device, trainable_parameter_count
from tiny_wordle.benchmark import parse_guess

MODEL_ID = "Qwen/Qwen3-0.6B"
REFERENCE_SEED = 42
NEW_SEEDS = [45, 47]
ALL_SEEDS = [REFERENCE_SEED, *NEW_SEEDS]

MAX_LENGTH = 256
BATCH_SIZE = 16
TRAIN_MICROBATCH_SIZE = 4
VAL_BATCH_SIZE = 8
COMMON_STEPS = 1029
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.05
LOG_EVERY = 25
EVAL_EVERY = 200
CHUNK_SIZE = 256
MEMORY_ABORT_GIB = MEMORY_CAP_GIB * 0.75

DATA_DIR = Path("../data")
GENERATED_DIR = DATA_DIR / "generated"
CHECKPOINT_ROOT = Path("../checkpoints")
RESULTS_DIR = Path("../results/lab18c")
LAB18_RESULTS = Path("../results/lab18")
LAB18B_RESULTS = Path("../results/lab18b")
REFERENCE_CHECKPOINT = (
    CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured"
)
SEED_CHECKPOINTS = {
    42: REFERENCE_CHECKPOINT,
    45: CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed45",
    47: CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed47",
}
STRUCTURED_FILES = {
    "train": GENERATED_DIR / "wordle-part2-structured-train.jsonl",
    "validation": GENERATED_DIR / "wordle-part2-structured-dev.jsonl",
    "test": GENERATED_DIR / "wordle-part2-structured-test.jsonl",
}
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
""")

md("""
## 18c.3 Freeze the data and reference configuration

Lab 18c reads the structured JSONL files written by Lab 17. It does not
regenerate or transform a row. Every file hash must match both the Dataset B
manifest and the seed-42 checkpoint manifest.
""")

code("""
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


structured_manifest = json.loads(
    (GENERATED_DIR / "wordle-part2-structured-manifest.json").read_text()
)
reference_manifest = json.loads(
    (REFERENCE_CHECKPOINT / "lab17-run.json").read_text()
)
structured_hashes = {
    split: sha256_file(path) for split, path in STRUCTURED_FILES.items()
}
assert structured_hashes == structured_manifest["structured_sha256"]
assert structured_hashes == reference_manifest["structured_data_sha256"]
assert reference_manifest["seed"] == REFERENCE_SEED
assert reference_manifest["base_model"] == MODEL_ID
assert reference_manifest["optimizer_steps"] == COMMON_STEPS
assert reference_manifest["effective_batch_size"] == BATCH_SIZE
assert reference_manifest["train_microbatch_size"] == TRAIN_MICROBATCH_SIZE

structured_rows = {
    split: [json.loads(line) for line in path.read_text().splitlines()]
    for split, path in STRUCTURED_FILES.items()
}
batches_per_epoch = len(structured_rows["train"]) // BATCH_SIZE
epochs_used = math.ceil(COMMON_STEPS / batches_per_epoch)
assert epochs_used == 2
epoch_shuffle_seeds = {
    seed: {seed + epoch for epoch in range(epochs_used)}
    for seed in ALL_SEEDS
}
for left_seed, right_seed in combinations(ALL_SEEDS, 2):
    assert epoch_shuffle_seeds[left_seed].isdisjoint(
        epoch_shuffle_seeds[right_seed]
    ), (
        f"seeds {left_seed} and {right_seed} share an epoch shuffle: "
        f"{epoch_shuffle_seeds[left_seed] & epoch_shuffle_seeds[right_seed]}"
    )

structured_dataset = DatasetDict({
    split: Dataset.from_list(rows) for split, rows in structured_rows.items()
})
print("structured hashes match seed 42")
display(pd.DataFrame([
    {"split": split, "rows": len(rows)}
    for split, rows in structured_rows.items()
]))
""")

md("""
## 18c.4 Tokenization and the exact Lab 17 training kernel

The functions below are the Lab 17 path with one mechanical change: every
function receives its seed explicitly so the two runs cannot share hidden global
state. Response-only logits and `use_cache=False` prevent full-vocabulary logits
from being materialized across prompt positions.
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


def encode_example(example: dict) -> dict:
    prompt_text = render_prompt(example["prompt"])
    full_text = prompt_text + example["response"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) >= MAX_LENGTH:
        raise ValueError(f"sequence length {len(full_ids)} reached {MAX_LENGTH}")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}


def collate_batch(rows: list[dict]) -> dict[str, torch.Tensor]:
    encoded = [encode_example(row) for row in rows]
    max_len = max(len(item["input_ids"]) for item in encoded)
    inputs, labels, attention = [], [], []
    for item in encoded:
        pad = max_len - len(item["input_ids"])
        inputs.append([PAD_ID] * pad + item["input_ids"])
        labels.append([-100] * pad + item["labels"])
        attention.append([0] * pad + [1] * len(item["input_ids"]))
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def batch_stream(split, seed: int):
    epoch = 0
    while True:
        loader = DataLoader(
            split,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_batch,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        for batch in loader:
            yield epoch, batch
        epoch += 1


lengths = {
    split: [len(encode_example(row)["input_ids"]) for row in dataset]
    for split, dataset in structured_dataset.items()
}
assert max(max(values) for values in lengths.values()) < MAX_LENGTH
display(pd.DataFrame([
    {
        "split": split,
        "rows": len(values),
        "mean_tokens": np.mean(values),
        "max_tokens": max(values),
    }
    for split, values in lengths.items()
]))
""")

code("""
WARMUP_STEPS = max(1, int(COMMON_STEPS * WARMUP_FRACTION))


def lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )


def reset_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_lora_model(seed: int):
    reset_seeds(seed)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)
    base.config.use_cache = False
    model = get_peft_model(base, lora_config())
    trainable, total = trainable_parameter_count(model)
    print(
        f"seed {seed}: {trainable:,} trainable parameters "
        f"({trainable / total:.3%})"
    )
    return model


def load_adapter(path: Path):
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


def response_loss(
    model, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, int]:
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
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return loss, supervised_tokens


@torch.no_grad()
def evaluate_loss(model, split) -> float:
    loader = DataLoader(
        split,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        loss, count = response_loss(model, batch)
        weighted_loss += float(loss.detach().cpu()) * count
        supervised_tokens += count
    model.train()
    return weighted_loss / supervised_tokens


def lr_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(
        1, COMMON_STEPS - WARMUP_STEPS
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))
""")

md("""
## 18c.5 Fixed-shape training soak

The longest training row is repeated into one fixed batch for 40 optimizer
steps on a disposable seed-45 model. This isolates allocator growth from shape
changes. The real runs rebuild their models after resetting their own seeds, so
the soak cannot alter initialization, dropout, or row order.
""")

code("""
def train_microbatches(model, optimizer, batch) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    supervised_in_batch = int(batch["labels"].ne(-100).sum())
    weighted_loss = 0.0
    peak = 0.0
    for start_index in range(0, BATCH_SIZE, TRAIN_MICROBATCH_SIZE):
        microbatch = {
            key: value[start_index:start_index + TRAIN_MICROBATCH_SIZE]
            for key, value in batch.items()
        }
        loss, microbatch_tokens = response_loss(model, microbatch)
        loss_weight = microbatch_tokens / supervised_in_batch
        (loss * loss_weight).backward()
        weighted_loss += float(loss.detach().cpu()) * microbatch_tokens
        peak = max(peak, driver_memory_gib())
    torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), max_norm=1.0
    )
    optimizer.step()
    return weighted_loss / supervised_in_batch, peak


if RUN_TRAINING and any(not SEED_CHECKPOINTS[s].exists() for s in NEW_SEEDS):
    longest_index = int(np.argmax(lengths["train"]))
    longest_row = structured_dataset["train"][longest_index]
    soak_batch = collate_batch([longest_row] * BATCH_SIZE)
    soak_batch = {key: value.to(device) for key, value in soak_batch.items()}
    soak_model = build_lora_model(NEW_SEEDS[0])
    soak_optimizer = AdamW(
        (p for p in soak_model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    soak_peaks = []
    for step in range(40):
        _, peak = train_microbatches(
            soak_model, soak_optimizer, soak_batch
        )
        soak_peaks.append(peak)
    third = len(soak_peaks) // 3
    middle = np.mean(soak_peaks[third:2 * third])
    late = np.mean(soak_peaks[-third:])
    creep = late - middle
    late_range = np.ptp(soak_peaks[-third:])
    print(
        f"training soak peak {max(soak_peaks):.2f} GiB, "
        f"creep {creep:+.2f} GiB, final range {late_range:.2f} GiB"
    )
    assert creep < 0.5
    assert late_range < 0.5
    assert max(soak_peaks) < MEMORY_ABORT_GIB
    del soak_optimizer, soak_batch
    release_model(soak_model)
    print("training memory plateaued")
else:
    print("training soak skipped: training disabled or both checkpoints exist")
""")

md("""
## 18c.6 Train seeds 45 and 47

Each completed seed is promoted atomically from an `-in-progress` directory.
A valid completed checkpoint is reused on restart. Any in-progress directory
stops the notebook for inspection rather than silently resuming from an unknown
optimizer state.
""")

code("""
def train_seed(seed: int) -> dict:
    checkpoint = SEED_CHECKPOINTS[seed]
    in_progress = checkpoint.with_name(checkpoint.name + "-in-progress")
    if checkpoint.exists():
        manifest_path = checkpoint / "lab18c-run.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"checkpoint exists without Lab 18c manifest: {checkpoint}"
            )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["seed"] == seed
        assert manifest["structured_data_sha256"] == structured_hashes
        print(f"seed {seed}: verified existing checkpoint")
        return manifest
    if in_progress.exists():
        raise FileExistsError(
            f"incomplete seed {seed} checkpoint needs inspection: {in_progress}"
        )
    if not RUN_TRAINING:
        raise FileNotFoundError(
            f"seed {seed} checkpoint missing and RUN_TRAINING=False"
        )

    model = build_lora_model(seed)
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_multiplier
    )
    stream = batch_stream(structured_dataset["train"], seed)
    baseline_val_loss = evaluate_loss(
        model, structured_dataset["validation"]
    )
    records = []
    processed_input_tokens = 0
    processed_supervised_tokens = 0
    peak_memory = 0.0
    started = time.perf_counter()

    for step in range(1, COMMON_STEPS + 1):
        epoch, batch = next(stream)
        batch = {key: value.to(device) for key, value in batch.items()}
        processed_input_tokens += int(batch["attention_mask"].sum())
        processed_supervised_tokens += int(batch["labels"].ne(-100).sum())

        loss_value, step_peak = train_microbatches(
            model, optimizer, batch
        )
        peak_memory = max(peak_memory, step_peak)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        record = {
            "step": step,
            "data_epoch": epoch + 1,
            "train_loss": loss_value,
            "lr": lr,
            "input_tokens": processed_input_tokens,
            "supervised_tokens": processed_supervised_tokens,
            "val_loss": None,
            "driver_peak_gib": step_peak,
        }
        if step % EVAL_EVERY == 0 or step == COMMON_STEPS:
            record["val_loss"] = evaluate_loss(
                model, structured_dataset["validation"]
            )
            model.save_pretrained(in_progress)
        records.append(record)
        if step == 1 or step % LOG_EVERY == 0:
            print(
                f"seed {seed} step {step:4d}/{COMMON_STEPS} "
                f"loss={loss_value:.4f} lr={lr:.2e} "
                f"epoch={epoch + 1} peak={peak_memory:.2f} GiB"
            )
        if record["val_loss"] is not None:
            print(f"  validation loss={record['val_loss']:.4f}")
        assert peak_memory < MEMORY_ABORT_GIB, (
            f"seed {seed} exceeded memory threshold at step {step}: "
            f"{peak_memory:.1f} GiB"
        )

    model.save_pretrained(in_progress)
    tokenizer.save_pretrained(in_progress)
    history = pd.DataFrame(records)
    history.to_csv(in_progress / "training-history.csv", index=False)
    manifest = {
        "experiment": "Lab 18c B-structured seed replication",
        "representation": "derived_state_v1",
        "base_model": MODEL_ID,
        "seed": seed,
        "optimizer_steps": COMMON_STEPS,
        "effective_batch_size": BATCH_SIZE,
        "train_microbatch_size": TRAIN_MICROBATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "processed_input_tokens": processed_input_tokens,
        "processed_supervised_tokens": processed_supervised_tokens,
        "baseline_val_loss": baseline_val_loss,
        "final_val_loss": next(
            row["val_loss"]
            for row in reversed(records)
            if row["val_loss"] is not None
        ),
        "peak_driver_gib": peak_memory,
        "elapsed_seconds": time.perf_counter() - started,
        "structured_data_sha256": structured_hashes,
    }
    (in_progress / "lab18c-run.json").write_text(
        json.dumps(manifest, indent=2)
    )
    in_progress.rename(checkpoint)
    del optimizer
    release_model(model)
    print(f"seed {seed}: complete")
    return manifest


training_manifests = {
    42: reference_manifest,
    **{seed: train_seed(seed) for seed in NEW_SEEDS},
}
training_summary = pd.DataFrame([
    {
        "seed": seed,
        "optimizer_steps": manifest["optimizer_steps"],
        "input_tokens": manifest["processed_input_tokens"],
        "supervised_tokens": manifest["processed_supervised_tokens"],
        "baseline_val_loss": manifest["baseline_val_loss"],
        "final_val_loss": manifest["final_val_loss"],
        "elapsed_minutes": manifest["elapsed_seconds"] / 60,
    }
    for seed, manifest in training_manifests.items()
])
display(training_summary)
""")

md("""
## 18c.7 Rebuild the exact Lab 18b battery

The state order and teacher targets come from Lab 18b's persisted battery. The
structured prompts are rebuilt through the unchanged Lab 17 representation, and
the answer order must match the saved seed-42 score matrix.
""")

code("""
ANSWERS = [
    line.strip().upper()
    for line in (DATA_DIR / "wordle-answers-original.txt").read_text().splitlines()
    if line.strip()
]
ANSWER_SET = set(ANSWERS)
WORD_TO_INDEX = {word: index for index, word in enumerate(ANSWERS)}
saved_answer_order = pd.read_csv(LAB18B_RESULTS / "answer-order.csv")["word"].tolist()
assert ANSWERS == saved_answer_order


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
            raise ValueError(
                f"impossible count constraint for {letter}"
            )
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


def raw_policy_prompt(state_key: str) -> str:
    return (
        "Task: NEXT_GUESS\\n"
        "You are playing Wordle.\\n"
        "Use the game history to choose the next guess.\\n"
        "Return exactly one uppercase five-letter word.\\n\\n"
        f"History:\\n{state_key}"
    )


def transform_prompt(
    prompt: str, state_key: str, candidate_count: int
) -> str:
    marker = "\\n\\nHistory:\\n"
    prefix, remainder = prompt.split(marker, 1)
    if "\\n\\n" in remainder:
        raw_history, suffix = remainder.split("\\n\\n", 1)
        suffix = "\\n\\n" + suffix
    else:
        raw_history, suffix = remainder, ""
    assert raw_history == state_key
    history = parse_state_key(state_key)
    return prefix + "\\n\\nDerived state:\\n" + render_structured_state(
        history, candidate_count
    ) + suffix


battery = pd.read_csv(LAB18B_RESULTS / "battery-states.csv")
battery["history"] = battery["state_key"].map(parse_state_key)
battery["candidates"] = [
    filter_candidates(ANSWERS, history) for history in battery["history"]
]
battery["structured_prompt"] = [
    transform_prompt(
        raw_policy_prompt(row.state_key),
        row.state_key,
        int(row.candidate_count),
    )
    for row in battery.itertuples()
]
assert len(battery) == 620
assert all(
    len(candidates) == count
    for candidates, count in zip(
        battery["candidates"], battery["candidate_count"]
    )
)
assert all(
    row.expected in set(row.candidates) for row in battery.itertuples()
)
print("battery states:", len(battery))
""")

md("""
## 18c.8 Free-generation continuity metric

Seed 42 is loaded from Lab 18. Seeds 45 and 47 use the same greedy generation,
parser, and usability rule. These results remain secondary because Lab 18b
showed that malformed strings hide learned ranking signal.
""")

code("""
@torch.no_grad()
def generate_prompt(model, prompt: str) -> str:
    batch = tokenizer(
        render_prompt(prompt), return_tensors="pt"
    ).to(device)
    output = model.generate(
        **batch, max_new_tokens=16, do_sample=False
    )
    new_tokens = output[0, batch["input_ids"].shape[1]:]
    return tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()


def evaluate_free_generation(model, seed: int) -> pd.DataFrame:
    rows = []
    for state in battery.itertuples():
        raw = generate_prompt(model, state.structured_prompt)
        guess = parse_guess(raw)
        in_lexicon = bool(guess and guess in ANSWER_SET)
        consistent = bool(
            in_lexicon and is_consistent(guess, state.history)
        )
        repeated = bool(
            guess and guess in {turn.guess for turn in state.history}
        )
        rows.append({
            "seed": seed,
            "state_key": state.state_key,
            "expected": state.expected,
            "actual": guess or raw.strip().upper(),
            "format_valid": guess is not None,
            "in_answer_lexicon": in_lexicon,
            "history_consistent": consistent,
            "repeated": repeated,
            "usable": bool(in_lexicon and consistent and not repeated),
            "teacher_match": (
                guess or raw.strip().upper()
            ) == state.expected,
        })
    return pd.DataFrame(rows)


seed42_free = pd.read_csv(
    LAB18_RESULTS / "b-state-battery-results.csv"
).copy()
assert seed42_free["state_key"].tolist() == battery["state_key"].tolist()
seed42_free["seed"] = 42
free_generation = {42: seed42_free}
""")

md("""
## 18c.9 Constrained scoring engine

This is Lab 18b's verified kernel: summed
`log P(word tokens + EOS | prompt)`, `logits_to_keep=1` on the prefill, target
gather minus `logsumexp`, KV-cache reuse, and an allocator flush after every
state. Seed 42 loads the persisted Lab 18b matrix rather than recomputing it.
""")

code("""
WORD_TOKENS = [
    tokenizer.encode(word, add_special_tokens=False)
    + [tokenizer.eos_token_id]
    for word in ANSWERS
]
probe_prompt = render_prompt(battery["structured_prompt"].iloc[0])
probe_ids = tokenizer(
    probe_prompt, add_special_tokens=False
)["input_ids"]
for word, tokens in zip(ANSWERS, WORD_TOKENS):
    joint = tokenizer(
        probe_prompt + word + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]
    assert joint[:len(probe_ids)] == probe_ids
    assert joint[len(probe_ids):] == tokens

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


def score_battery(
    model, seed: int, score_path: Path
) -> np.ndarray:
    partial_path = score_path.with_suffix(".partial.npy")
    progress_path = score_path.with_suffix(".progress.json")
    if partial_path.exists() or progress_path.exists():
        if not partial_path.exists() or not progress_path.exists():
            raise FileNotFoundError(
                f"incomplete score checkpoint pair for seed {seed}"
            )
        matrix = np.load(partial_path)
        progress = json.loads(progress_path.read_text())
        start_position = int(progress["completed_states"])
        assert progress["seed"] == seed
        assert matrix.shape == (len(battery), len(ANSWERS))
        assert 0 <= start_position < len(battery)
        print(
            f"seed {seed}: resuming scores at state {start_position}"
        )
    else:
        matrix = np.zeros(
            (len(battery), len(ANSWERS)), dtype=np.float32
        )
        start_position = 0

    started = time.time()
    peak = 0.0
    prompts = battery["structured_prompt"]
    for position in range(start_position, len(prompts)):
        prompt = prompts.iloc[position]
        matrix[position] = score_all_words(model, prompt)
        peak = max(peak, LAST_STATE_PEAK_GIB)
        completed = position + 1
        if completed % 100 == 0:
            np.save(partial_path, matrix)
            progress_path.write_text(json.dumps({
                "seed": seed,
                "completed_states": completed,
            }))
            elapsed = time.time() - started
            print(
                f"seed {seed} {completed:4d}/{len(battery)} "
                f"{elapsed / (completed - start_position):.2f}s/state "
                f"peak {peak:.2f} GiB",
                flush=True,
            )
        assert peak < MEMORY_ABORT_GIB
    print(
        f"seed {seed} scored in {(time.time() - started) / 60:.1f} min, "
        f"peak {peak:.2f} GiB"
    )
    return matrix
""")

md("""
## 18c.10 Evaluate the new seeds

Each free-generation CSV and score matrix is written immediately. On restart,
artifacts are validated against the 620 by 2,315 shape before reuse. Scoring
also checkpoints its partial matrix every 100 states. The fixed-shape scoring
soak runs once before the first missing matrix.
""")

code("""
score_matrices = {
    42: np.load(LAB18B_RESULTS / "scores-B-structured.npy")
}
assert score_matrices[42].shape == (len(battery), len(ANSWERS))
scoring_soak_done = False

if RUN_EVALUATION:
    for seed in NEW_SEEDS:
        free_path = RESULTS_DIR / f"free-generation-seed{seed}.csv"
        score_path = RESULTS_DIR / f"scores-seed{seed}.npy"
        need_free = not free_path.exists()
        need_scores = not score_path.exists()

        if not need_free:
            cached_free = pd.read_csv(free_path)
            assert cached_free["state_key"].tolist() == battery["state_key"].tolist()
            free_generation[seed] = cached_free
        if not need_scores:
            cached_scores = np.load(score_path)
            assert cached_scores.shape == (len(battery), len(ANSWERS))
            score_matrices[seed] = cached_scores

        if not need_free and not need_scores:
            print(f"seed {seed}: verified cached evaluations")
            continue

        model = load_adapter(SEED_CHECKPOINTS[seed])
        if need_free:
            free_generation[seed] = evaluate_free_generation(model, seed)
            free_generation[seed].to_csv(free_path, index=False)
            print(f"seed {seed}: free generation saved")

        if need_scores:
            if not scoring_soak_done:
                prompt_tokens = battery["structured_prompt"].map(
                    lambda prompt: len(
                        tokenizer(render_prompt(prompt)).input_ids
                    )
                )
                longest_prompt = battery.loc[
                    prompt_tokens.idxmax(), "structured_prompt"
                ]
                soak_peaks = []
                for _ in range(40):
                    score_all_words(model, longest_prompt)
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
                scoring_soak_done = True

            score_matrices[seed] = score_battery(
                model, seed, score_path
            )
            np.save(score_path, score_matrices[seed])
            score_path.with_suffix(".partial.npy").unlink(
                missing_ok=True
            )
            score_path.with_suffix(".progress.json").unlink(
                missing_ok=True
            )
        release_model(model)
else:
    for seed in NEW_SEEDS:
        free_generation[seed] = pd.read_csv(
            RESULTS_DIR / f"free-generation-seed{seed}.csv"
        )
        score_matrices[seed] = np.load(
            RESULTS_DIR / f"scores-seed{seed}.npy"
        )

print({
    seed: matrix.shape for seed, matrix in score_matrices.items()
})
""")

md("""
## 18c.11 Compute the four pre-registered outputs

Tier 1 chooses the highest-scoring answer. Tier 2 chooses the highest-scoring
consistent candidate. Candidate-rank percentile uses the whole answer ranking,
so it measures state conditioning without depending on one argmax.
""")

code("""
ANSWER_ARRAY = np.array(ANSWERS)
CANDIDATE_INDICES = [
    np.array(
        [WORD_TO_INDEX[word] for word in candidates],
        dtype=np.int64,
    )
    for candidates in battery["candidates"]
]


def tier_frame(seed: int, matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for position, state in enumerate(battery.itertuples()):
        scores = matrix[position]
        candidate_ids = CANDIDATE_INDICES[position]
        previous = {turn.guess for turn in state.history}
        tier1_word = ANSWER_ARRAY[int(scores.argmax())]
        tier2_word = ANSWER_ARRAY[
            candidate_ids[int(scores[candidate_ids].argmax())]
        ]
        tier1_consistent = is_consistent(
            tier1_word, state.history
        )
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)
        rows.append({
            "seed": seed,
            "state_key": state.state_key,
            "turn": state.turn,
            "candidate_bucket": state.candidate_bucket,
            "candidate_count": state.candidate_count,
            "expected": state.expected,
            "tier1_word": tier1_word,
            "tier1_usable": bool(
                tier1_consistent and tier1_word not in previous
            ),
            "tier1_teacher_match": tier1_word == state.expected,
            "tier2_word": tier2_word,
            "tier2_teacher_match": tier2_word == state.expected,
            "tier2_chance": 1.0 / len(candidate_ids),
            "teacher_rank": int(
                ranks[WORD_TO_INDEX[state.expected]]
            ),
            "best_candidate_rank": int(
                ranks[candidate_ids].min()
            ),
            "candidate_rank_percentile": float(
                ranks[candidate_ids].mean() / len(scores)
            ),
        })
    return pd.DataFrame(rows)


tier_results = pd.concat([
    tier_frame(seed, score_matrices[seed]) for seed in ALL_SEEDS
], ignore_index=True)
free_results = pd.concat([
    free_generation[seed].assign(seed=seed)
    for seed in ALL_SEEDS
], ignore_index=True)

per_seed = tier_results.groupby("seed").agg(
    tier1_usable=("tier1_usable", "mean"),
    candidate_rank_percentile=("candidate_rank_percentile", "mean"),
    tier2_teacher_match=("tier2_teacher_match", "mean"),
    tier2_chance=("tier2_chance", "mean"),
    median_teacher_rank=("teacher_rank", "median"),
)
chance_stats = tier_results.groupby("seed")["tier2_chance"].agg(
    chance_expected="sum",
    chance_variance=lambda values: float(
        (values * (1.0 - values)).sum()
    ),
)
per_seed = per_seed.join(chance_stats)
per_seed["tier2_correct"] = (
    tier_results.groupby("seed")["tier2_teacher_match"].sum()
)
per_seed["tier2_z_vs_chance"] = (
    per_seed["tier2_correct"] - per_seed["chance_expected"]
) / np.sqrt(per_seed["chance_variance"])
per_seed["free_generation_usable"] = (
    free_results.groupby("seed")["usable"].mean()
)
per_seed = per_seed.reset_index()

seed42_lab18b = pd.read_csv(
    LAB18B_RESULTS / "tier-results.csv"
)
seed42_lab18b = seed42_lab18b.loc[
    seed42_lab18b["model"] == "B-structured"
]
seed42_row = per_seed.loc[per_seed["seed"] == 42].iloc[0]
assert np.isclose(
    seed42_row["tier1_usable"],
    seed42_lab18b["tier1_usable"].mean(),
)
assert np.isclose(
    seed42_row["tier2_teacher_match"],
    seed42_lab18b["tier2_teacher_match"].mean(),
)
assert np.isclose(
    seed42_row["candidate_rank_percentile"],
    seed42_lab18b["candidate_rank_percentile"].mean(),
)
print("seed 42 reproduces all Lab 18b primary metrics")
display(per_seed)
""")

md("""
## 18c.12 Between-seed spread and paired state differences

The seed is the unit of replication. The table with three rows is therefore the
main result. State-level pairing explains where seeds disagree but does not turn
620 states into 620 independent training runs. Seed 42 is the pre-existing
reference that motivated the replication, not a newly sampled arm. The two new
seeds provide the direct replication check; all-three mean and spread remain
descriptive.
""")

code("""
PRIMARY_METRICS = [
    "tier1_usable",
    "candidate_rank_percentile",
    "tier2_teacher_match",
    "free_generation_usable",
]
spread = pd.DataFrame([
    {
        "metric": metric,
        "mean": per_seed[metric].mean(),
        "sample_std": per_seed[metric].std(ddof=1),
        "sigma_upper_95": per_seed[metric].std(ddof=1) * 6.285,
        "minimum": per_seed[metric].min(),
        "maximum": per_seed[metric].max(),
        "range": per_seed[metric].max() - per_seed[metric].min(),
    }
    for metric in PRIMARY_METRICS
])
display(spread)


def paired_seed_metric(
    frame: pd.DataFrame,
    left_seed: int,
    right_seed: int,
    metric: str,
    binary: bool,
    bootstrap_samples: int = 10_000,
) -> dict:
    left = (
        frame.loc[frame["seed"] == left_seed, ["state_key", metric]]
        .rename(columns={metric: "left"})
    )
    right = (
        frame.loc[frame["seed"] == right_seed, ["state_key", metric]]
        .rename(columns={metric: "right"})
    )
    paired = left.merge(
        right, on="state_key", validate="one_to_one"
    )
    difference = (
        paired["right"].astype(float)
        - paired["left"].astype(float)
    ).to_numpy()
    rng = np.random.default_rng(
        18_000 + left_seed * 100 + right_seed
    )
    indices = rng.integers(
        0,
        len(difference),
        size=(bootstrap_samples, len(difference)),
    )
    low, high = np.quantile(
        difference[indices].mean(axis=1), [0.025, 0.975]
    )
    result = {
        "metric": metric,
        "left_seed": left_seed,
        "right_seed": right_seed,
        "right_minus_left": difference.mean(),
        "paired_ci_low": low,
        "paired_ci_high": high,
    }
    if binary:
        left_values = paired["left"].astype(bool)
        right_values = paired["right"].astype(bool)
        result["left_only"] = int(
            (left_values & ~right_values).sum()
        )
        result["right_only"] = int(
            (~left_values & right_values).sum()
        )
    return result


paired_rows = []
for left_seed, right_seed in combinations(ALL_SEEDS, 2):
    for metric in ["tier1_usable", "tier2_teacher_match"]:
        paired_rows.append(paired_seed_metric(
            tier_results,
            left_seed,
            right_seed,
            metric,
            binary=True,
        ))
    paired_rows.append(paired_seed_metric(
        tier_results,
        left_seed,
        right_seed,
        "candidate_rank_percentile",
        binary=False,
    ))
    paired_rows.append(paired_seed_metric(
        free_results,
        left_seed,
        right_seed,
        "usable",
        binary=True,
    ) | {"metric": "free_generation_usable"})

paired_seeds = pd.DataFrame(paired_rows)
display(paired_seeds)
""")

md("""
## 18c.13 References and decision table

The G seed-42 result is shown only to locate it relative to B's seed spread. The
three B seeds determine replication. A single G seed cannot estimate Dataset G
variance.
""")

code("""
lab18b_headline = pd.read_csv(
    LAB18B_RESULTS / "headline-summary.csv"
)
g_reference = lab18b_headline.loc[
    lab18b_headline["model"] == "G-structured"
].iloc[0]

references = pd.DataFrame([
    {
        "reference": "G-structured seed 42",
        "tier1_usable": g_reference["tier1_usable"],
        "tier2_teacher_match": g_reference["tier2_teacher_match"],
        "free_generation_usable": g_reference["tier0_usable"],
    },
    {
        "reference": "B seed range minimum",
        "tier1_usable": per_seed["tier1_usable"].min(),
        "tier2_teacher_match": per_seed["tier2_teacher_match"].min(),
        "free_generation_usable": per_seed["free_generation_usable"].min(),
    },
    {
        "reference": "B seed range maximum",
        "tier1_usable": per_seed["tier1_usable"].max(),
        "tier2_teacher_match": per_seed["tier2_teacher_match"].max(),
        "free_generation_usable": per_seed["free_generation_usable"].max(),
    },
])
display(references)

print(
    "Tier 1 capability replicated above every seed's free generation:",
    per_seed["tier1_usable"].min()
    > per_seed["free_generation_usable"].max(),
)
print(
    "G seed-42 Tier 1 lies inside the B seed range:",
    per_seed["tier1_usable"].min()
    <= g_reference["tier1_usable"]
    <= per_seed["tier1_usable"].max(),
)
print(
    "Every B seed beats Tier 2 chance at z >= 1.96:",
    bool(
        (per_seed["tier2_z_vs_chance"] >= 1.96).all()
    ),
)
""")

md("""
## 18c.14 Persist the replication record

The score matrices remain the source for any later decoder analysis. CSVs keep
the state-level decisions, per-seed summary, spread, and paired comparisons.
""")

code("""
tier_results.to_csv(
    RESULTS_DIR / "tier-results.csv", index=False
)
free_results.to_csv(
    RESULTS_DIR / "free-generation-results.csv", index=False
)
training_summary.to_csv(
    RESULTS_DIR / "training-summary.csv", index=False
)
per_seed.to_csv(
    RESULTS_DIR / "per-seed-summary.csv", index=False
)
spread.to_csv(
    RESULTS_DIR / "between-seed-spread.csv", index=False
)
paired_seeds.to_csv(
    RESULTS_DIR / "paired-seed-differences.csv", index=False
)
references.to_csv(
    RESULTS_DIR / "reference-comparison.csv", index=False
)

run_manifest = {
    "experiment": "Lab 18c B-structured seed replication",
    "model_id": MODEL_ID,
    "seeds": ALL_SEEDS,
    "new_seeds": NEW_SEEDS,
    "states": len(battery),
    "answers": len(ANSWERS),
    "optimizer_steps": COMMON_STEPS,
    "training_data_sha256": structured_hashes,
    "primary_metrics": PRIMARY_METRICS,
    "scoring_rule": "summed log P(word tokens + EOS | prompt)",
}
(RESULTS_DIR / "lab18c-run.json").write_text(
    json.dumps(run_manifest, indent=2)
)
print("written to", RESULTS_DIR)
""")

md("""
## Lab 18c checkpoint

Read the result in this order:

1. Did Tier 1 remain above free generation for every seed?
2. Is Tier 1 spread smaller or larger than the effects claimed in Labs 15-18?
3. Does candidate-rank percentile remain near 0.03?
4. Does every seed beat the Tier 2 chance baseline, or is entropy ranking
   seed-sensitive?
5. Does the single G result sit inside B's seed range?

Only then decide whether to build constrained gameplay or redesign the next
training objective around relative action values.
""")


for index, cell in enumerate(cells):
    cell["id"] = f"lab18c-{index:02d}-{cell['cell_type']}"


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

path = Path("notebooks/18c_seed_replication.ipynb")
path.write_text(json.dumps(notebook, indent=1))
print(f"wrote {path} with {len(cells)} cells")
