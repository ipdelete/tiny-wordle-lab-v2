# Lab 18c - Does the constrained policy replicate across seeds?

Lab 18b found that free generation hid substantial learned policy signal.
B-structured rose from 14.5% usable under free generation to 30.3% when all
2,315 answer words were ranked directly. The base model remained at 0.16%, so
the adapter learned state-conditioned ranking rather than borrowing competence
from the answer list.

That conclusion still rests on one training seed. This lab reproduces the
B-structured run twice, changing only the training seed, and evaluates all three
adapters on the same 620 held-out states.

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

## 18c.2 Run controls and memory guard

Run this notebook only through the total-system watchdog:

```
scripts/memguard.py --min-free 64 -- uv run jupyter nbconvert \
    --to notebook --execute --inplace notebooks/18c_seed_replication.ipynb
```

The in-process MPS cap turns a runaway allocation into a normal exception. Two
fixed-shape gates run before full scale: a 40-step training soak on one longest
batch and a 40-state scoring soak on one longest prompt. Largest real prompts
are checked separately for scoring headroom.


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
```

    device: mps


## 18c.3 Freeze the data and reference configuration

Lab 18c reads the structured JSONL files written by Lab 17. It does not
regenerate or transform a row. Every file hash must match both the Dataset B
manifest and the seed-42 checkpoint manifest.


```python
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
```

    structured hashes match seed 42



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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train</td>
      <td>8768</td>
    </tr>
    <tr>
      <th>1</th>
      <td>validation</td>
      <td>1135</td>
    </tr>
    <tr>
      <th>2</th>
      <td>test</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>


## 18c.4 Tokenization and the exact Lab 17 training kernel

The functions below are the Lab 17 path with one mechanical change: every
function receives its seed explicitly so the two runs cannot share hidden global
state. Response-only logits and `use_cache=False` prevent full-vocabulary logits
from being materialized across prompt positions.


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
      <th>mean_tokens</th>
      <th>max_tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train</td>
      <td>8768</td>
      <td>149.172787</td>
      <td>214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>validation</td>
      <td>1135</td>
      <td>149.962115</td>
      <td>205</td>
    </tr>
    <tr>
      <th>2</th>
      <td>test</td>
      <td>21</td>
      <td>132.333333</td>
      <td>161</td>
    </tr>
  </tbody>
</table>
</div>



```python
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
```

## 18c.5 Fixed-shape training soak

The longest training row is repeated into one fixed batch for 40 optimizer
steps on a disposable seed-45 model. This isolates allocator growth from shape
changes. The real runs rebuild their models after resetting their own seeds, so
the soak cannot alter initialization, dropout, or row order.


```python
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
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45: 2,293,760 trainable parameters (0.383%)


    training soak peak 6.96 GiB, creep +0.00 GiB, final range 0.00 GiB


    training memory plateaued


## 18c.6 Train seeds 45 and 47

Each completed seed is promoted atomically from an `-in-progress` directory.
A valid completed checkpoint is reused on restart. Any in-progress directory
stops the notebook for inspection rather than silently resuming from an unknown
optimizer state.


```python
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
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45: 2,293,760 trainable parameters (0.383%)


    seed 45 step    1/1029 loss=6.5411 lr=1.96e-06 epoch=1 peak=6.15 GiB


    seed 45 step   25/1029 loss=2.4537 lr=4.90e-05 epoch=1 peak=6.18 GiB


    seed 45 step   50/1029 loss=2.2371 lr=9.80e-05 epoch=1 peak=8.84 GiB


    seed 45 step   75/1029 loss=2.0706 lr=9.99e-05 epoch=1 peak=10.90 GiB


    seed 45 step  100/1029 loss=1.5869 lr=9.94e-05 epoch=1 peak=10.91 GiB


    seed 45 step  125/1029 loss=1.5725 lr=9.86e-05 epoch=1 peak=10.91 GiB


    seed 45 step  150/1029 loss=1.1404 lr=9.75e-05 epoch=1 peak=10.91 GiB


    seed 45 step  175/1029 loss=1.4213 lr=9.61e-05 epoch=1 peak=10.91 GiB


    seed 45 step  200/1029 loss=1.7351 lr=9.45e-05 epoch=1 peak=10.91 GiB
      validation loss=1.3787


    seed 45 step  225/1029 loss=1.3660 lr=9.25e-05 epoch=1 peak=10.91 GiB


    seed 45 step  250/1029 loss=1.1446 lr=9.02e-05 epoch=1 peak=10.91 GiB


    seed 45 step  275/1029 loss=1.2274 lr=8.77e-05 epoch=1 peak=10.91 GiB


    seed 45 step  300/1029 loss=1.4398 lr=8.50e-05 epoch=1 peak=11.91 GiB


    seed 45 step  325/1029 loss=1.2499 lr=8.20e-05 epoch=1 peak=11.91 GiB


    seed 45 step  350/1029 loss=0.8882 lr=7.88e-05 epoch=1 peak=11.91 GiB


    seed 45 step  375/1029 loss=1.1557 lr=7.54e-05 epoch=1 peak=11.91 GiB


    seed 45 step  400/1029 loss=1.2414 lr=7.19e-05 epoch=1 peak=11.91 GiB
      validation loss=1.3393


    seed 45 step  425/1029 loss=1.3497 lr=6.82e-05 epoch=1 peak=11.91 GiB


    seed 45 step  450/1029 loss=1.1174 lr=6.44e-05 epoch=1 peak=11.91 GiB


    seed 45 step  475/1029 loss=0.8389 lr=6.05e-05 epoch=1 peak=11.91 GiB


    seed 45 step  500/1029 loss=0.7054 lr=5.66e-05 epoch=1 peak=11.91 GiB


    seed 45 step  525/1029 loss=0.6990 lr=5.26e-05 epoch=1 peak=11.91 GiB


    seed 45 step  550/1029 loss=0.6345 lr=4.86e-05 epoch=2 peak=11.91 GiB


    seed 45 step  575/1029 loss=0.8990 lr=4.46e-05 epoch=2 peak=11.91 GiB


    seed 45 step  600/1029 loss=0.6830 lr=4.06e-05 epoch=2 peak=11.91 GiB
      validation loss=1.2972


    seed 45 step  625/1029 loss=1.2372 lr=3.67e-05 epoch=2 peak=11.91 GiB


    seed 45 step  650/1029 loss=0.7438 lr=3.28e-05 epoch=2 peak=11.91 GiB


    seed 45 step  675/1029 loss=1.2603 lr=2.91e-05 epoch=2 peak=11.91 GiB


    seed 45 step  700/1029 loss=1.0458 lr=2.56e-05 epoch=2 peak=11.91 GiB


    seed 45 step  725/1029 loss=1.1277 lr=2.21e-05 epoch=2 peak=11.91 GiB


    seed 45 step  750/1029 loss=0.9762 lr=1.89e-05 epoch=2 peak=11.91 GiB


    seed 45 step  775/1029 loss=0.8232 lr=1.59e-05 epoch=2 peak=11.91 GiB


    seed 45 step  800/1029 loss=1.0565 lr=1.30e-05 epoch=2 peak=11.91 GiB
      validation loss=1.3094


    seed 45 step  825/1029 loss=0.5016 lr=1.05e-05 epoch=2 peak=11.91 GiB


    seed 45 step  850/1029 loss=0.8558 lr=8.13e-06 epoch=2 peak=11.91 GiB


    seed 45 step  875/1029 loss=0.8791 lr=6.07e-06 epoch=2 peak=11.91 GiB


    seed 45 step  900/1029 loss=0.9335 lr=4.30e-06 epoch=2 peak=11.91 GiB


    seed 45 step  925/1029 loss=0.7087 lr=2.82e-06 epoch=2 peak=11.91 GiB


    seed 45 step  950/1029 loss=0.3844 lr=1.64e-06 epoch=2 peak=11.91 GiB


    seed 45 step  975/1029 loss=0.8141 lr=7.78e-07 epoch=2 peak=11.91 GiB


    seed 45 step 1000/1029 loss=0.7428 lr=2.32e-07 epoch=2 peak=11.91 GiB
      validation loss=1.2873


    seed 45 step 1025/1029 loss=0.5837 lr=6.45e-09 epoch=2 peak=11.91 GiB


      validation loss=1.2872


    seed 45: complete



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 47: 2,293,760 trainable parameters (0.383%)


    seed 47 step    1/1029 loss=7.4695 lr=1.96e-06 epoch=1 peak=6.23 GiB


    seed 47 step   25/1029 loss=2.5874 lr=4.90e-05 epoch=1 peak=8.29 GiB


    seed 47 step   50/1029 loss=2.1051 lr=9.80e-05 epoch=1 peak=8.29 GiB


    seed 47 step   75/1029 loss=1.6094 lr=9.99e-05 epoch=1 peak=8.29 GiB


    seed 47 step  100/1029 loss=1.7141 lr=9.94e-05 epoch=1 peak=9.36 GiB


    seed 47 step  125/1029 loss=1.5052 lr=9.86e-05 epoch=1 peak=9.36 GiB


    seed 47 step  150/1029 loss=1.4804 lr=9.75e-05 epoch=1 peak=9.36 GiB


    seed 47 step  175/1029 loss=1.0814 lr=9.61e-05 epoch=1 peak=9.36 GiB


    seed 47 step  200/1029 loss=1.0362 lr=9.45e-05 epoch=1 peak=9.36 GiB
      validation loss=1.4572


    seed 47 step  225/1029 loss=1.4309 lr=9.25e-05 epoch=1 peak=9.36 GiB


    seed 47 step  250/1029 loss=0.6673 lr=9.02e-05 epoch=1 peak=9.36 GiB


    seed 47 step  275/1029 loss=0.9616 lr=8.77e-05 epoch=1 peak=9.36 GiB


    seed 47 step  300/1029 loss=1.0604 lr=8.50e-05 epoch=1 peak=9.36 GiB


    seed 47 step  325/1029 loss=1.0371 lr=8.20e-05 epoch=1 peak=9.36 GiB


    seed 47 step  350/1029 loss=0.6667 lr=7.88e-05 epoch=1 peak=9.36 GiB


    seed 47 step  375/1029 loss=1.0616 lr=7.54e-05 epoch=1 peak=9.36 GiB


    seed 47 step  400/1029 loss=1.4509 lr=7.19e-05 epoch=1 peak=9.36 GiB
      validation loss=1.3139


    seed 47 step  425/1029 loss=1.1704 lr=6.82e-05 epoch=1 peak=9.36 GiB


    seed 47 step  450/1029 loss=1.4225 lr=6.44e-05 epoch=1 peak=9.36 GiB


    seed 47 step  475/1029 loss=1.3490 lr=6.05e-05 epoch=1 peak=9.36 GiB


    seed 47 step  500/1029 loss=1.1045 lr=5.66e-05 epoch=1 peak=9.36 GiB


    seed 47 step  525/1029 loss=1.2190 lr=5.26e-05 epoch=1 peak=9.36 GiB


    seed 47 step  550/1029 loss=0.8346 lr=4.86e-05 epoch=2 peak=9.36 GiB


    seed 47 step  575/1029 loss=1.2140 lr=4.46e-05 epoch=2 peak=9.36 GiB


    seed 47 step  600/1029 loss=0.8481 lr=4.06e-05 epoch=2 peak=9.36 GiB
      validation loss=1.3415


    seed 47 step  625/1029 loss=0.9774 lr=3.67e-05 epoch=2 peak=9.36 GiB


    seed 47 step  650/1029 loss=0.6680 lr=3.28e-05 epoch=2 peak=9.36 GiB


    seed 47 step  675/1029 loss=0.9485 lr=2.91e-05 epoch=2 peak=9.36 GiB


    seed 47 step  700/1029 loss=1.1229 lr=2.56e-05 epoch=2 peak=9.36 GiB


    seed 47 step  725/1029 loss=0.9336 lr=2.21e-05 epoch=2 peak=9.36 GiB


    seed 47 step  750/1029 loss=1.0381 lr=1.89e-05 epoch=2 peak=9.36 GiB


    seed 47 step  775/1029 loss=0.9744 lr=1.59e-05 epoch=2 peak=9.36 GiB


    seed 47 step  800/1029 loss=1.0351 lr=1.30e-05 epoch=2 peak=9.36 GiB
      validation loss=1.3265


    seed 47 step  825/1029 loss=0.6512 lr=1.05e-05 epoch=2 peak=9.36 GiB


    seed 47 step  850/1029 loss=0.6455 lr=8.13e-06 epoch=2 peak=9.36 GiB


    seed 47 step  875/1029 loss=0.8282 lr=6.07e-06 epoch=2 peak=9.36 GiB


    seed 47 step  900/1029 loss=0.6421 lr=4.30e-06 epoch=2 peak=9.36 GiB


    seed 47 step  925/1029 loss=1.0550 lr=2.82e-06 epoch=2 peak=9.36 GiB


    seed 47 step  950/1029 loss=0.5107 lr=1.64e-06 epoch=2 peak=9.36 GiB


    seed 47 step  975/1029 loss=0.5942 lr=7.78e-07 epoch=2 peak=9.36 GiB


    seed 47 step 1000/1029 loss=0.5151 lr=2.32e-07 epoch=2 peak=9.36 GiB
      validation loss=1.3263


    seed 47 step 1025/1029 loss=0.9518 lr=6.45e-09 epoch=2 peak=9.36 GiB


      validation loss=1.3260


    seed 47: complete



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
      <th>optimizer_steps</th>
      <th>input_tokens</th>
      <th>supervised_tokens</th>
      <th>baseline_val_loss</th>
      <th>final_val_loss</th>
      <th>elapsed_minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>1029</td>
      <td>2456934</td>
      <td>50308</td>
      <td>7.327902</td>
      <td>1.258621</td>
      <td>19.610658</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>1029</td>
      <td>2455477</td>
      <td>50289</td>
      <td>7.327902</td>
      <td>1.287198</td>
      <td>19.782918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47</td>
      <td>1029</td>
      <td>2456443</td>
      <td>50308</td>
      <td>7.327902</td>
      <td>1.325974</td>
      <td>19.749741</td>
    </tr>
  </tbody>
</table>
</div>


## 18c.7 Rebuild the exact Lab 18b battery

The state order and teacher targets come from Lab 18b's persisted battery. The
structured prompts are rebuilt through the unchanged Lab 17 representation, and
the answer order must match the saved seed-42 score matrix.


```python
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
    return "\n".join([
        f"GREENS: {greens}",
        f"LETTER_COUNTS: {', '.join(counts) or 'NONE'}",
        f"EXCLUDED_POSITIONS: {', '.join(excluded) or 'NONE'}",
        f"ABSENT_LETTERS: {' '.join(absent) or 'NONE'}",
        f"PREVIOUS_GUESSES: {', '.join(state['previous_guesses']) or 'NONE'}",
        f"CANDIDATE_COUNT: {candidate_count}",
    ])


def raw_policy_prompt(state_key: str) -> str:
    return (
        "Task: NEXT_GUESS\n"
        "You are playing Wordle.\n"
        "Use the game history to choose the next guess.\n"
        "Return exactly one uppercase five-letter word.\n\n"
        f"History:\n{state_key}"
    )


def transform_prompt(
    prompt: str, state_key: str, candidate_count: int
) -> str:
    marker = "\n\nHistory:\n"
    prefix, remainder = prompt.split(marker, 1)
    if "\n\n" in remainder:
        raw_history, suffix = remainder.split("\n\n", 1)
        suffix = "\n\n" + suffix
    else:
        raw_history, suffix = remainder, ""
    assert raw_history == state_key
    history = parse_state_key(state_key)
    return prefix + "\n\nDerived state:\n" + render_structured_state(
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
```

    battery states: 620


## 18c.8 Free-generation continuity metric

Seed 42 is loaded from Lab 18. Seeds 45 and 47 use the same greedy generation,
parser, and usability rule. These results remain secondary because Lab 18b
showed that malformed strings hide learned ranking signal.


```python
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
```

## 18c.9 Constrained scoring engine

This is Lab 18b's verified kernel: summed
`log P(word tokens + EOS | prompt)`, `logits_to_keep=1` on the prefill, target
gather minus `logsumexp`, KV-cache reuse, and an allocator flush after every
state. Seed 42 loads the persisted Lab 18b matrix rather than recomputing it.


```python
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
```

## 18c.10 Evaluate the new seeds

Each free-generation CSV and score matrix is written immediately. On restart,
artifacts are validated against the 620 by 2,315 shape before reuse. Scoring
also checkpoints its partial matrix every 100 states. The fixed-shape scoring
soak runs once before the first missing matrix.


```python
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
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 45: free generation saved


    scoring soak peak 56.04 GiB, creep +0.00 GiB, final range 0.00 GiB


    seed 45  100/620 3.32s/state peak 54.78 GiB


    seed 45  200/620 3.32s/state peak 54.78 GiB


    seed 45  300/620 3.36s/state peak 54.78 GiB


    seed 45  400/620 3.37s/state peak 54.78 GiB


    seed 45  500/620 3.39s/state peak 54.78 GiB


    seed 45  600/620 3.43s/state peak 54.78 GiB


    seed 45 scored in 35.5 min, peak 54.78 GiB



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    seed 47: free generation saved


    seed 47  100/620 3.34s/state peak 54.76 GiB


    seed 47  200/620 3.33s/state peak 54.76 GiB


    seed 47  300/620 3.37s/state peak 54.76 GiB


    seed 47  400/620 3.37s/state peak 54.76 GiB


    seed 47  500/620 3.40s/state peak 54.76 GiB


    seed 47  600/620 3.43s/state peak 54.77 GiB


    seed 47 scored in 35.5 min, peak 54.77 GiB


    {42: (620, 2315), 45: (620, 2315), 47: (620, 2315)}


## 18c.11 Compute the four pre-registered outputs

Tier 1 chooses the highest-scoring answer. Tier 2 chooses the highest-scoring
consistent candidate. Candidate-rank percentile uses the whole answer ranking,
so it measures state conditioning without depending on one argmax.


```python
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
```

    seed 42 reproduces all Lab 18b primary metrics



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
      <th>tier1_usable</th>
      <th>candidate_rank_percentile</th>
      <th>tier2_teacher_match</th>
      <th>tier2_chance</th>
      <th>median_teacher_rank</th>
      <th>chance_expected</th>
      <th>chance_variance</th>
      <th>tier2_correct</th>
      <th>tier2_z_vs_chance</th>
      <th>free_generation_usable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>0.303226</td>
      <td>0.028345</td>
      <td>0.574194</td>
      <td>0.521896</td>
      <td>9.0</td>
      <td>323.575555</td>
      <td>68.384429</td>
      <td>356</td>
      <td>3.920974</td>
      <td>0.145161</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>0.309677</td>
      <td>0.029124</td>
      <td>0.583871</td>
      <td>0.521896</td>
      <td>8.0</td>
      <td>323.575555</td>
      <td>68.384429</td>
      <td>362</td>
      <td>4.646533</td>
      <td>0.162903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47</td>
      <td>0.300000</td>
      <td>0.028912</td>
      <td>0.575806</td>
      <td>0.521896</td>
      <td>9.0</td>
      <td>323.575555</td>
      <td>68.384429</td>
      <td>357</td>
      <td>4.041900</td>
      <td>0.162903</td>
    </tr>
  </tbody>
</table>
</div>


## 18c.12 Between-seed spread and paired state differences

The seed is the unit of replication. The table with three rows is therefore the
main result. State-level pairing explains where seeds disagree but does not turn
620 states into 620 independent training runs. Seed 42 is the pre-existing
reference that motivated the replication, not a newly sampled arm. The two new
seeds provide the direct replication check; all-three mean and spread remain
descriptive.


```python
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
      <th>mean</th>
      <th>sample_std</th>
      <th>sigma_upper_95</th>
      <th>minimum</th>
      <th>maximum</th>
      <th>range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tier1_usable</td>
      <td>0.304301</td>
      <td>0.004928</td>
      <td>0.030969</td>
      <td>0.300000</td>
      <td>0.309677</td>
      <td>0.009677</td>
    </tr>
    <tr>
      <th>1</th>
      <td>candidate_rank_percentile</td>
      <td>0.028794</td>
      <td>0.000403</td>
      <td>0.002531</td>
      <td>0.028345</td>
      <td>0.029124</td>
      <td>0.000779</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tier2_teacher_match</td>
      <td>0.577957</td>
      <td>0.005185</td>
      <td>0.032586</td>
      <td>0.574194</td>
      <td>0.583871</td>
      <td>0.009677</td>
    </tr>
    <tr>
      <th>3</th>
      <td>free_generation_usable</td>
      <td>0.156989</td>
      <td>0.010243</td>
      <td>0.064379</td>
      <td>0.145161</td>
      <td>0.162903</td>
      <td>0.017742</td>
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
      <th>metric</th>
      <th>left_seed</th>
      <th>right_seed</th>
      <th>right_minus_left</th>
      <th>paired_ci_low</th>
      <th>paired_ci_high</th>
      <th>left_only</th>
      <th>right_only</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tier1_usable</td>
      <td>42</td>
      <td>45</td>
      <td>0.006452</td>
      <td>-0.019355</td>
      <td>0.032258</td>
      <td>32.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tier2_teacher_match</td>
      <td>42</td>
      <td>45</td>
      <td>0.009677</td>
      <td>-0.009677</td>
      <td>0.029032</td>
      <td>16.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>candidate_rank_percentile</td>
      <td>42</td>
      <td>45</td>
      <td>0.000779</td>
      <td>-0.000244</td>
      <td>0.001974</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>free_generation_usable</td>
      <td>42</td>
      <td>45</td>
      <td>0.017742</td>
      <td>-0.004839</td>
      <td>0.040323</td>
      <td>20.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tier1_usable</td>
      <td>42</td>
      <td>47</td>
      <td>-0.003226</td>
      <td>-0.029032</td>
      <td>0.022581</td>
      <td>37.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tier2_teacher_match</td>
      <td>42</td>
      <td>47</td>
      <td>0.001613</td>
      <td>-0.017742</td>
      <td>0.020968</td>
      <td>19.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>candidate_rank_percentile</td>
      <td>42</td>
      <td>47</td>
      <td>0.000567</td>
      <td>-0.000400</td>
      <td>0.001626</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>free_generation_usable</td>
      <td>42</td>
      <td>47</td>
      <td>0.017742</td>
      <td>-0.004839</td>
      <td>0.040323</td>
      <td>20.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>tier1_usable</td>
      <td>45</td>
      <td>47</td>
      <td>-0.009677</td>
      <td>-0.035484</td>
      <td>0.016129</td>
      <td>37.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>tier2_teacher_match</td>
      <td>45</td>
      <td>47</td>
      <td>-0.008065</td>
      <td>-0.025806</td>
      <td>0.009677</td>
      <td>19.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>candidate_rank_percentile</td>
      <td>45</td>
      <td>47</td>
      <td>-0.000213</td>
      <td>-0.001200</td>
      <td>0.000711</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>free_generation_usable</td>
      <td>45</td>
      <td>47</td>
      <td>0.000000</td>
      <td>-0.020968</td>
      <td>0.022581</td>
      <td>24.0</td>
      <td>24.0</td>
    </tr>
  </tbody>
</table>
</div>


## 18c.13 References and decision table

The G seed-42 result is shown only to locate it relative to B's seed spread. The
three B seeds determine replication. A single G seed cannot estimate Dataset G
variance.


```python
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
      <th>reference</th>
      <th>tier1_usable</th>
      <th>tier2_teacher_match</th>
      <th>free_generation_usable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>G-structured seed 42</td>
      <td>0.300000</td>
      <td>0.545161</td>
      <td>0.159677</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B seed range minimum</td>
      <td>0.300000</td>
      <td>0.574194</td>
      <td>0.145161</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B seed range maximum</td>
      <td>0.309677</td>
      <td>0.583871</td>
      <td>0.162903</td>
    </tr>
  </tbody>
</table>
</div>


    Tier 1 capability replicated above every seed's free generation: True
    G seed-42 Tier 1 lies inside the B seed range: True
    Every B seed beats Tier 2 chance at z >= 1.96: True


## 18c.14 Persist the replication record

The score matrices remain the source for any later decoder analysis. CSVs keep
the state-level decisions, per-seed summary, spread, and paired comparisons.


```python
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
```

    written to ../results/lab18c


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
