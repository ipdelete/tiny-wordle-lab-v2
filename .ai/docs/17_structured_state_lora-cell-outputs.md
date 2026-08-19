# Lab 17 - Train on explicit Wordle state

**Goal:** test whether duplicate-aware derived-state features help Qwen3-0.6B bind feedback constraints to its next action.

Lab 16 found that Model B changed its action on 16.1% of parsed training-format perturbation pairs but achieved 0% paired consistency. This lab changes representation during training while freezing Dataset B's states, targets, row weights, and splits.

```text
B-raw         = frozen Lab 15 Dataset B adapter + raw feedback history
B-structured  = fresh identical LoRA adapter + explicit derived state
```

## 17.1 Pre-registered experiment

The only training-data intervention is the prompt. The structured corpus preserves every Dataset B response and all non-prompt metadata, while replacing raw feedback with deterministically derived constraints and candidate count. Candidate count adds a symbolic-solver feature at training and inference time, so a positive result supports the derived-state package rather than a pure re-encoding claim.

Both adapters use the Lab 15 base model, LoRA configuration, optimizer, schedule, seed, effective batch size, 1,029 optimizer steps, and final-step checkpoint rule. B-structured splits each 16-example optimizer batch into four-example gradient microbatches to fit the longer sequences in device memory. Equal steps and the same shuffled row stream hold examples and optimizer exposure fixed. Structured prompts are longer, so this notebook reports the input-token ratio rather than claiming equal token exposure.

The primary metric is paired consistency across all 34 Lab 16 perturbation pairs; unparseable outputs count as inconsistent. Branch consistency across all 68 branches measures partial progress. Parse-conditioned consistency and sensitivity remain diagnostic only. Paired consistency uses a Wilson interval, branch consistency bootstraps the 34 paired units, and an exact paired test prevents a small nonzero count from being treated as decisive evidence.

## 17.2 Run controls

Preflight builds and validates the transformed corpus without loading a model. Training and evaluation remain explicit expensive actions.


```python
RUN_TRAINING = True
RUN_EVALUATION = True
RUN_FROZEN_TRANSFER_DIAGNOSTIC = False

print("RUN_TRAINING:", RUN_TRAINING)
print("RUN_EVALUATION:", RUN_EVALUATION)
print("RUN_FROZEN_TRANSFER_DIAGNOSTIC:", RUN_FROZEN_TRANSFER_DIAGNOSTIC)
```

    RUN_TRAINING: True
    RUN_EVALUATION: True
    RUN_FROZEN_TRANSFER_DIAGNOSTIC: False



```python
from collections import Counter, defaultdict
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
from datasets import Dataset, DatasetDict, load_dataset
from IPython.display import display
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_wordle.benchmark import DEFAULT_EVAL_ANSWERS, generate_raw_guess, parse_guess
from tiny_wordle.game import Turn, filter_candidates, is_consistent, score_string
from tiny_wordle.hardware import preferred_device, trainable_parameter_count

MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 42
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

DATA_DIR = Path("../data")
GENERATED_DIR = DATA_DIR / "generated"
CHECKPOINT_ROOT = Path("../checkpoints")
RESULTS_DIR = Path("../results/lab17")
RAW_CHECKPOINT = CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b"
STRUCTURED_CHECKPOINT = CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured"
LAB15_RESULTS = Path("../results/lab15")
LAB16_RESULTS = Path("../results/lab16")

device = preferred_device()
torch.set_float32_matmul_precision("high")
print("device:", device)
```

    device: mps


## 17.3 Derive duplicate-aware constraints

The representation is computed from feedback, not from the hidden answer. For each letter, matched yellow and green occurrences establish a minimum count. A black duplicate beyond those matches establishes a maximum count. Every non-green occurrence excludes that position.


```python
ANSWERS = [
    line.strip().upper()
    for line in (DATA_DIR / "wordle-answers-original.txt").read_text().splitlines()
    if line.strip()
]
ANSWER_SET = set(ANSWERS)

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
        for position, (letter, mark) in enumerate(zip(turn.guess, turn.feedback), 1):
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
        "excluded": {letter: sorted(positions) for letter, positions in excluded.items()},
        "previous_guesses": [turn.guess for turn in history],
    }

def render_structured_state(history: list[Turn], candidate_count: int) -> str:
    state = derive_constraints(history)
    greens = " ".join(letter or "_" for letter in state["greens"])
    present_letters = sorted(
        letter for letter, count in state["minimum"].items() if count > 0
    )
    counts = []
    for letter in present_letters:
        low = state["minimum"][letter]
        high = state["maximum"].get(letter, 5)
        counts.append(f"{letter}={low}..{high}" if high < 5 else f"{letter}>={low}")
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

def transform_prompt(prompt: str, state_key: str, candidate_count: int) -> str:
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
```

## 17.4 Transform Dataset B without changing the experiment rows

The original files are fingerprinted before and after transformation. Responses and every non-prompt field must remain identical.


```python
RAW_FILES = {
    "train": GENERATED_DIR / "wordle-part2-policy-train.jsonl",
    "validation": GENERATED_DIR / "wordle-part2-policy-dev.jsonl",
    "test": GENERATED_DIR / "wordle-part2-policy-test.jsonl",
}
STRUCTURED_FILES = {
    "train": GENERATED_DIR / "wordle-part2-structured-train.jsonl",
    "validation": GENERATED_DIR / "wordle-part2-structured-dev.jsonl",
    "test": GENERATED_DIR / "wordle-part2-structured-test.jsonl",
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

raw_hashes_before = {split: sha256_file(path) for split, path in RAW_FILES.items()}
raw_rows = {
    split: [json.loads(line) for line in path.read_text().splitlines()]
    for split, path in RAW_FILES.items()
}
structured_rows = {}
for split, rows in raw_rows.items():
    transformed = []
    for row in rows:
        updated = dict(row)
        updated["prompt"] = transform_prompt(
            row["prompt"], row["state_key"], int(row["candidate_count"])
        )
        updated["representation"] = "derived_state_v1"
        transformed.append(updated)
    structured_rows[split] = transformed

for split in raw_rows:
    assert len(raw_rows[split]) == len(structured_rows[split])
    for raw, structured in zip(raw_rows[split], structured_rows[split]):
        assert raw["response"] == structured["response"]
        assert raw["state_key"] == structured["state_key"]
        assert raw["task"] == structured["task"]
        assert {
            key: value for key, value in structured.items()
            if key not in {"prompt", "representation"}
        } == {key: value for key, value in raw.items() if key != "prompt"}

for split, path in STRUCTURED_FILES.items():
    with path.open("w") as handle:
        for row in structured_rows[split]:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

assert raw_hashes_before == {
    split: sha256_file(path) for split, path in RAW_FILES.items()
}
structured_prompt_targets = defaultdict(set)
for rows in structured_rows.values():
    for row in rows:
        structured_prompt_targets[row["prompt"]].add(row["response"])
assert not any(len(targets) > 1 for targets in structured_prompt_targets.values())

structured_manifest = {
    "source": "Lab 14 Dataset B",
    "representation": "derived_state_v1",
    "raw_sha256": raw_hashes_before,
    "structured_sha256": {
        split: sha256_file(path) for split, path in STRUCTURED_FILES.items()
    },
    "counts": {split: len(rows) for split, rows in structured_rows.items()},
    "changed_fields": ["prompt", "representation"],
}
(GENERATED_DIR / "wordle-part2-structured-manifest.json").write_text(
    json.dumps(structured_manifest, indent=2)
)
display(pd.DataFrame({
    split: {"rows": len(rows), "tasks": dict(Counter(r["task"] for r in rows))}
    for split, rows in structured_rows.items()
}).T)
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
      <th>rows</th>
      <th>tasks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>8768</td>
      <td>{'NEXT_GUESS': 5669, 'VALID_CANDIDATE': 1494, ...</td>
    </tr>
    <tr>
      <th>validation</th>
      <td>1135</td>
      <td>{'NEXT_GUESS': 669, 'CHOOSE_VALID': 247, 'VALI...</td>
    </tr>
    <tr>
      <th>test</th>
      <td>21</td>
      <td>{'NEXT_GUESS': 13, 'CHOOSE_VALID': 4, 'VALID_C...</td>
    </tr>
  </tbody>
</table>
</div>


## 17.5 Verify the representation against candidate filtering

The structured fields must describe the same state as the raw feedback. Candidate count is recomputed for every unique history. Duplicate-aware count bounds are checked against every remaining candidate.


```python
unique_states = {}
for rows in raw_rows.values():
    for row in rows:
        unique_states.setdefault(row["state_key"], int(row["candidate_count"]))

for state_key, expected_count in unique_states.items():
    history = parse_state_key(state_key)
    candidates = filter_candidates(ANSWERS, history)
    assert len(candidates) == expected_count
    constraints = derive_constraints(history)
    for candidate in candidates:
        counts = Counter(candidate)
        for letter, low in constraints["minimum"].items():
            assert counts[letter] >= low
        for letter, high in constraints["maximum"].items():
            assert counts[letter] <= high
        for position, letter in enumerate(constraints["greens"]):
            if letter:
                assert candidate[position] == letter
        for letter, positions in constraints["excluded"].items():
            assert all(candidate[position - 1] != letter for position in positions)

print("verified unique states:", len(unique_states))

broad_example = max(unique_states, key=unique_states.get)
duplicate_example = next(
    state_key for state_key in unique_states
    if any(
        high < 5 and derive_constraints(parse_state_key(state_key))["minimum"].get(letter, 0) > 0
        for letter, high in derive_constraints(parse_state_key(state_key))["maximum"].items()
    )
)
for label, state_key in [("broad", broad_example), ("duplicate-aware", duplicate_example)]:
    print(f"\n{label} example:\n{render_structured_state(parse_state_key(state_key), unique_states[state_key])}")
```

    verified unique states: 3573
    
    broad example:
    GREENS: _ _ _ _ _
    LETTER_COUNTS: NONE
    EXCLUDED_POSITIONS: D@5, F@1, J@2, O@3, R@4
    ABSENT_LETTERS: D F J O R
    PREVIOUS_GUESSES: FJORD
    CANDIDATE_COUNT: 782
    
    duplicate-aware example:
    GREENS: _ _ A S E
    LETTER_COUNTS: A>=1, E=1..1, S>=1
    EXCLUDED_POSITIONS: A@2, C@1, E@2, I@3, R@1
    ABSENT_LETTERS: C I R
    PREVIOUS_GUESSES: RAISE, CEASE
    CANDIDATE_COUNT: 2


## 17.6 Tokenize and pre-register exposure

Only assistant response tokens contribute to loss. The same shuffled Dataset B row stream and 1,029 steps are used as Lab 15. The token ratio measures the added context.


```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
PAD_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

structured_dataset = DatasetDict({
    split: Dataset.from_list(rows) for split, rows in structured_rows.items()
})

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
            split, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
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

stream = batch_stream(structured_dataset["train"], SEED)
planned_structured_tokens = 0
for _ in range(COMMON_STEPS):
    _, batch = next(stream)
    planned_structured_tokens += int(batch["attention_mask"].sum())

raw_manifest = json.loads((RAW_CHECKPOINT / "lab15-run.json").read_text())
raw_tokens = int(raw_manifest["processed_input_tokens"])
print("raw input tokens:", raw_tokens)
print("structured planned input tokens:", planned_structured_tokens)
print("structured/raw token ratio:", f"{planned_structured_tokens / raw_tokens:.3f}")
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

    raw input tokens: 1300155
    structured planned input tokens: 2456934
    structured/raw token ratio: 1.890



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


## 17.7 Train the structured adapter

This is the only new model. B-raw remains the frozen Lab 15 checkpoint.


```python
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
WARMUP_STEPS = max(1, int(COMMON_STEPS * WARMUP_FRACTION))

def reset_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def build_lora_model():
    reset_seeds()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)
    base.config.use_cache = False
    model = get_peft_model(base, LORA_CONFIG)
    trainable, total = trainable_parameter_count(model)
    print("trainable parameters:", f"{trainable:,}")
    print("trainable share:", f"{trainable / total:.3%}")
    return model

def release_model(model):
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

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
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return loss, supervised_tokens

@torch.no_grad()
def evaluate_loss(model, split) -> float:
    loader = DataLoader(
        split, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_batch
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
    progress = (step - WARMUP_STEPS) / max(1, COMMON_STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
```


```python
training_history = pd.DataFrame()
if RUN_TRAINING:
    in_progress = STRUCTURED_CHECKPOINT.with_name(
        STRUCTURED_CHECKPOINT.name + "-in-progress"
    )
    collisions = [path for path in [STRUCTURED_CHECKPOINT, in_progress] if path.exists()]
    if collisions:
        raise FileExistsError(f"existing Lab 17 paths: {collisions}")

    model = build_lora_model()
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_multiplier
    )
    stream = batch_stream(structured_dataset["train"], SEED)
    baseline_val_loss = evaluate_loss(model, structured_dataset["validation"])
    records = []
    processed_input_tokens = 0
    processed_supervised_tokens = 0
    start = time.perf_counter()

    for step in range(1, COMMON_STEPS + 1):
        epoch, batch = next(stream)
        batch = {key: value.to(device) for key, value in batch.items()}
        processed_input_tokens += int(batch["attention_mask"].sum())
        processed_supervised_tokens += int(batch["labels"].ne(-100).sum())
        optimizer.zero_grad(set_to_none=True)
        supervised_in_batch = int(batch["labels"].ne(-100).sum())
        weighted_loss = 0.0
        for start_index in range(0, BATCH_SIZE, TRAIN_MICROBATCH_SIZE):
            microbatch = {
                key: value[start_index:start_index + TRAIN_MICROBATCH_SIZE]
                for key, value in batch.items()
            }
            loss, microbatch_tokens = response_loss(model, microbatch)
            loss_weight = microbatch_tokens / supervised_in_batch
            (loss * loss_weight).backward()
            weighted_loss += float(loss.detach().cpu()) * microbatch_tokens
        loss_value = weighted_loss / supervised_in_batch
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0
        )
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        record = {
            "step": step,
            "data_epoch": epoch + 1,
            "train_loss": loss_value,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "input_tokens": processed_input_tokens,
            "supervised_tokens": processed_supervised_tokens,
            "val_loss": None,
        }
        if device.type == "mps":
            record["mps_allocated_gib"] = torch.mps.current_allocated_memory() / 2**30
            record["mps_driver_gib"] = torch.mps.driver_allocated_memory() / 2**30
        if step % EVAL_EVERY == 0 or step == COMMON_STEPS:
            record["val_loss"] = evaluate_loss(
                model, structured_dataset["validation"]
            )
            model.save_pretrained(in_progress)
        records.append(record)
        if step == 1 or step % LOG_EVERY == 0:
            print(
                f"step {step:4d}/{COMMON_STEPS} loss={record['train_loss']:.4f} "
                f"lr={record['lr']:.2e} epoch={record['data_epoch']} "
                f"mps_driver_gib={record.get('mps_driver_gib', float('nan')):.2f}"
            )
        if record["val_loss"] is not None:
            print(f"  validation loss={record['val_loss']:.4f}")

    model.save_pretrained(in_progress)
    tokenizer.save_pretrained(in_progress)
    training_history = pd.DataFrame(records)
    training_history.to_csv(in_progress / "training-history.csv", index=False)
    run_manifest = {
        "representation": "derived_state_v1",
        "base_model": MODEL_ID,
        "seed": SEED,
        "optimizer_steps": COMMON_STEPS,
        "effective_batch_size": BATCH_SIZE,
        "train_microbatch_size": TRAIN_MICROBATCH_SIZE,
        "processed_input_tokens": processed_input_tokens,
        "processed_supervised_tokens": processed_supervised_tokens,
        "raw_input_tokens": raw_tokens,
        "input_token_ratio": processed_input_tokens / raw_tokens,
        "baseline_val_loss": baseline_val_loss,
        "final_val_loss": next(
            row["val_loss"] for row in reversed(records) if row["val_loss"] is not None
        ),
        "elapsed_seconds": time.perf_counter() - start,
        "structured_data_sha256": structured_manifest["structured_sha256"],
    }
    (in_progress / "lab17-run.json").write_text(json.dumps(run_manifest, indent=2))
    in_progress.rename(STRUCTURED_CHECKPOINT)
    release_model(model)
else:
    print("Training skipped. Set RUN_TRAINING=True to create B-structured.")
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    trainable parameters: 2,293,760
    trainable share: 0.383%


    step    1/1029 loss=8.3887 lr=1.96e-06 epoch=1 mps_driver_gib=5.75


    step   25/1029 loss=2.4148 lr=4.90e-05 epoch=1 mps_driver_gib=9.68


    step   50/1029 loss=2.0803 lr=9.80e-05 epoch=1 mps_driver_gib=9.68


    step   75/1029 loss=2.0009 lr=9.99e-05 epoch=1 mps_driver_gib=10.69


    step  100/1029 loss=1.5594 lr=9.94e-05 epoch=1 mps_driver_gib=10.69


    step  125/1029 loss=1.3642 lr=9.86e-05 epoch=1 mps_driver_gib=10.69


    step  150/1029 loss=1.4488 lr=9.75e-05 epoch=1 mps_driver_gib=10.69


    step  175/1029 loss=1.5379 lr=9.61e-05 epoch=1 mps_driver_gib=10.69


    step  200/1029 loss=1.2118 lr=9.45e-05 epoch=1 mps_driver_gib=10.69
      validation loss=1.4125


    step  225/1029 loss=1.0268 lr=9.25e-05 epoch=1 mps_driver_gib=10.69


    step  250/1029 loss=1.1941 lr=9.02e-05 epoch=1 mps_driver_gib=10.69


    step  275/1029 loss=1.4611 lr=8.77e-05 epoch=1 mps_driver_gib=10.69


    step  300/1029 loss=1.0930 lr=8.50e-05 epoch=1 mps_driver_gib=10.69


    step  325/1029 loss=0.8745 lr=8.20e-05 epoch=1 mps_driver_gib=10.69


    step  350/1029 loss=1.2648 lr=7.88e-05 epoch=1 mps_driver_gib=10.69


    step  375/1029 loss=1.2866 lr=7.54e-05 epoch=1 mps_driver_gib=10.69


    step  400/1029 loss=1.0674 lr=7.19e-05 epoch=1 mps_driver_gib=10.69
      validation loss=1.3284


    step  425/1029 loss=1.1378 lr=6.82e-05 epoch=1 mps_driver_gib=10.69


    step  450/1029 loss=1.1028 lr=6.44e-05 epoch=1 mps_driver_gib=10.69


    step  475/1029 loss=0.5680 lr=6.05e-05 epoch=1 mps_driver_gib=10.69


    step  500/1029 loss=0.8518 lr=5.66e-05 epoch=1 mps_driver_gib=10.69


    step  525/1029 loss=1.0570 lr=5.26e-05 epoch=1 mps_driver_gib=10.69


    step  550/1029 loss=1.3903 lr=4.86e-05 epoch=2 mps_driver_gib=10.69


    step  575/1029 loss=0.3826 lr=4.46e-05 epoch=2 mps_driver_gib=10.69


    step  600/1029 loss=1.1928 lr=4.06e-05 epoch=2 mps_driver_gib=10.69
      validation loss=1.2728


    step  625/1029 loss=0.9851 lr=3.67e-05 epoch=2 mps_driver_gib=10.69


    step  650/1029 loss=0.8051 lr=3.28e-05 epoch=2 mps_driver_gib=10.69


    step  675/1029 loss=1.3597 lr=2.91e-05 epoch=2 mps_driver_gib=10.69


    step  700/1029 loss=1.0486 lr=2.56e-05 epoch=2 mps_driver_gib=10.69


    step  725/1029 loss=0.8410 lr=2.21e-05 epoch=2 mps_driver_gib=10.69


    step  750/1029 loss=0.7306 lr=1.89e-05 epoch=2 mps_driver_gib=10.69


    step  775/1029 loss=0.6615 lr=1.59e-05 epoch=2 mps_driver_gib=10.69


    step  800/1029 loss=0.8722 lr=1.30e-05 epoch=2 mps_driver_gib=10.69
      validation loss=1.2793


    step  825/1029 loss=0.5772 lr=1.05e-05 epoch=2 mps_driver_gib=10.69


    step  850/1029 loss=0.6323 lr=8.13e-06 epoch=2 mps_driver_gib=10.69


    step  875/1029 loss=0.8727 lr=6.07e-06 epoch=2 mps_driver_gib=10.69


    step  900/1029 loss=0.5214 lr=4.30e-06 epoch=2 mps_driver_gib=10.69


    step  925/1029 loss=0.5434 lr=2.82e-06 epoch=2 mps_driver_gib=10.69


    step  950/1029 loss=1.0257 lr=1.64e-06 epoch=2 mps_driver_gib=10.69


    step  975/1029 loss=0.6081 lr=7.78e-07 epoch=2 mps_driver_gib=10.69


    step 1000/1029 loss=0.5193 lr=2.32e-07 epoch=2 mps_driver_gib=10.69
      validation loss=1.2585


    step 1025/1029 loss=0.7135 lr=6.45e-09 epoch=2 mps_driver_gib=10.69


      validation loss=1.2586


## 17.8 Reuse the exact Lab 16 perturbation pairs

B-raw baselines come from the frozen Lab 16 result. B-structured receives the same child states rendered through `derived_state_v1`.


```python
pair_design = pd.read_csv(LAB16_RESULTS / "perturbation-pairs.csv")
lab16_results = pd.read_csv(LAB16_RESULTS / "perturbation-results.csv")
raw_pair_baseline = lab16_results.loc[
    (lab16_results["model"] == "B")
    & (lab16_results["interface"] == "training")
].copy()
assert len(pair_design) == len(raw_pair_baseline) == 34

def history_from_raw_prompt(prompt: str) -> list[Turn]:
    state_key = prompt.split("\n\nHistory:\n", 1)[1]
    return parse_state_key(state_key)

for side in ["a", "b"]:
    pair_design[f"history_{side}"] = pair_design[f"prompt_{side}"].map(
        history_from_raw_prompt
    )
    pair_design[f"structured_prompt_{side}"] = [
        transform_prompt(prompt, prompt.split("\n\nHistory:\n", 1)[1], count)
        for prompt, count in zip(
            pair_design[f"prompt_{side}"], pair_design[f"candidates_{side}"]
        )
    ]

display(pair_design.groupby(["pair_scope", "feedback_change_type"]).size()
 .rename("pairs").to_frame())
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
      <th>pairs</th>
    </tr>
    <tr>
      <th>pair_scope</th>
      <th>feedback_change_type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">broad</th>
      <th>B/G</th>
      <td>7</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">mixed</th>
      <th>B/G</th>
      <td>3</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">narrow</th>
      <th>B/G</th>
      <td>2</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


## 17.9 Evaluate paired and branch consistency

Paired consistency is the headline. Branch consistency gives credit for partial improvement. Sensitivity without consistency is not success.


```python
@torch.no_grad()
def generate_prompt(model, prompt: str) -> str:
    batch = tokenizer(render_prompt(prompt), return_tensors="pt").to(device)
    output = model.generate(**batch, max_new_tokens=16, do_sample=False)
    new_tokens = output[0, batch["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def load_adapter(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing adapter {path}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)
    return PeftModel.from_pretrained(base, path).to(device)

def evaluate_structured_pairs(model, label: str) -> pd.DataFrame:
    rows = []
    model.eval()
    for pair in pair_design.itertuples():
        sides = {}
        for side in ["a", "b"]:
            raw = generate_prompt(model, getattr(pair, f"structured_prompt_{side}"))
            parsed = parse_guess(raw)
            history = getattr(pair, f"history_{side}")
            sides[side] = {
                "action": parsed or raw.strip().upper(),
                "parsed": parsed,
                "consistent": is_consistent(parsed, history) if parsed else False,
            }
        both_parse = bool(sides["a"]["parsed"] and sides["b"]["parsed"])
        rows.append({
            "model": label,
            "pair_id": pair.pair_id,
            "pair_scope": pair.pair_scope,
            "feedback_change_type": pair.feedback_change_type,
            "action_a": sides["a"]["action"],
            "action_b": sides["b"]["action"],
            "both_parse": both_parse,
            "action_changed": sides["a"]["action"] != sides["b"]["action"],
            "consistent_a": sides["a"]["consistent"],
            "consistent_b": sides["b"]["consistent"],
            "both_consistent": sides["a"]["consistent"] and sides["b"]["consistent"],
            "consistent_branches": int(sides["a"]["consistent"]) + int(sides["b"]["consistent"]),
        })
    return pd.DataFrame(rows)

structured_pair_results = pd.DataFrame()
if RUN_EVALUATION:
    model = load_adapter(STRUCTURED_CHECKPOINT)
    structured_pair_results = evaluate_structured_pairs(model, "B-structured")
    release_model(model)
else:
    print("Evaluation skipped. Set RUN_EVALUATION=True after training.")
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]



```python
def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    margin = z * ((rate * (1 - rate) / trials + z**2 / (4 * trials**2)) ** 0.5) / denominator
    return center - margin, center + margin

def exact_paired_p_value(raw: pd.Series, structured: pd.Series) -> float:
    raw_only = int((raw & ~structured).sum())
    structured_only = int((~raw & structured).sum())
    discordant = raw_only + structured_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(raw_only, structured_only) + 1)
    ) / 2**discordant
    return min(1.0, 2 * tail)

def paired_bootstrap_branch_interval(
    frame: pd.DataFrame, samples: int = 10_000, seed: int = SEED
) -> tuple[float, float]:
    branch_results = frame[["consistent_a", "consistent_b"]].to_numpy(
        dtype=float
    )
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(branch_results), size=(samples, len(branch_results))
    )
    sampled_rates = branch_results[sampled_indices].mean(axis=(1, 2))
    return tuple(np.quantile(sampled_rates, [0.025, 0.975]))

def pair_summary(frame: pd.DataFrame, label: str) -> dict:
    parsed = frame.loc[frame["both_parse"]]
    paired_successes = int(frame["both_consistent"].sum())
    branch_successes = int(
        frame["consistent_a"].sum() + frame["consistent_b"].sum()
    )
    paired_low, paired_high = wilson_interval(paired_successes, len(frame))
    branch_low, branch_high = paired_bootstrap_branch_interval(frame)
    return {
        "model": label,
        "pairs": len(frame),
        "parsed_pairs": len(parsed),
        "both_parse_rate": frame["both_parse"].mean(),
        "sensitivity": parsed["action_changed"].mean() if len(parsed) else float("nan"),
        "paired_consistent": paired_successes,
        "paired_consistency": paired_successes / len(frame),
        "paired_ci_low": paired_low,
        "paired_ci_high": paired_high,
        "consistent_branches": branch_successes,
        "branches": len(frame) * 2,
        "branch_consistency": branch_successes / (len(frame) * 2),
        "branch_ci_low": branch_low,
        "branch_ci_high": branch_high,
        "parsed_paired_consistency": parsed["both_consistent"].mean() if len(parsed) else float("nan"),
        "parsed_branch_consistency": (
            parsed["consistent_a"].sum() + parsed["consistent_b"].sum()
        ) / (len(parsed) * 2) if len(parsed) else float("nan"),
    }

if RUN_EVALUATION:
    summaries = pd.DataFrame([
        pair_summary(raw_pair_baseline, "B-raw"),
        pair_summary(structured_pair_results, "B-structured"),
    ])
    display(summaries)
    paired_comparison = raw_pair_baseline[["pair_id", "both_consistent"]].merge(
        structured_pair_results[["pair_id", "both_consistent"]],
        on="pair_id", suffixes=("_raw", "_structured"), validate="one_to_one",
    )
    print("exact paired-consistency p-value:", exact_paired_p_value(
        paired_comparison["both_consistent_raw"],
        paired_comparison["both_consistent_structured"],
    ))
    pair_breakdown = pd.concat([
        raw_pair_baseline.assign(comparison_model="B-raw"),
        structured_pair_results.assign(comparison_model="B-structured"),
    ])
    display(pair_breakdown.loc[pair_breakdown["both_parse"]].groupby(
        ["comparison_model", "pair_scope", "feedback_change_type"]
    ).agg(
        pairs=("pair_id", "size"),
        sensitivity=("action_changed", "mean"),
        parsed_paired_consistency=("both_consistent", "mean"),
        consistent_branches=("consistent_branches", "sum"),
    ))
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
      <th>model</th>
      <th>pairs</th>
      <th>parsed_pairs</th>
      <th>both_parse_rate</th>
      <th>sensitivity</th>
      <th>paired_consistent</th>
      <th>paired_consistency</th>
      <th>paired_ci_low</th>
      <th>paired_ci_high</th>
      <th>consistent_branches</th>
      <th>branches</th>
      <th>branch_consistency</th>
      <th>branch_ci_low</th>
      <th>branch_ci_high</th>
      <th>parsed_paired_consistency</th>
      <th>parsed_branch_consistency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B-raw</td>
      <td>34</td>
      <td>31</td>
      <td>0.911765</td>
      <td>0.16129</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.101518</td>
      <td>4</td>
      <td>68</td>
      <td>0.058824</td>
      <td>0.014706</td>
      <td>0.117647</td>
      <td>0.000000</td>
      <td>0.064516</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B-structured</td>
      <td>34</td>
      <td>24</td>
      <td>0.705882</td>
      <td>0.75000</td>
      <td>4</td>
      <td>0.117647</td>
      <td>0.046714</td>
      <td>0.266212</td>
      <td>19</td>
      <td>68</td>
      <td>0.279412</td>
      <td>0.161765</td>
      <td>0.397059</td>
      <td>0.166667</td>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
</div>


    exact paired-consistency p-value: 0.125



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
      <th></th>
      <th>pairs</th>
      <th>sensitivity</th>
      <th>parsed_paired_consistency</th>
      <th>consistent_branches</th>
    </tr>
    <tr>
      <th>comparison_model</th>
      <th>pair_scope</th>
      <th>feedback_change_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="9" valign="top">B-raw</th>
      <th rowspan="3" valign="top">broad</th>
      <th>B/G</th>
      <td>7</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>2</td>
      <td>0.50</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
      <td>0.50</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">mixed</th>
      <th>B/G</th>
      <td>3</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>6</td>
      <td>0.50</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">narrow</th>
      <th>B/G</th>
      <td>2</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>4</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>3</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">B-structured</th>
      <th rowspan="3" valign="top">broad</th>
      <th>B/G</th>
      <td>4</td>
      <td>1.00</td>
      <td>0.250000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>1</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">mixed</th>
      <th>B/G</th>
      <td>3</td>
      <td>1.00</td>
      <td>0.333333</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>5</td>
      <td>1.00</td>
      <td>0.400000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>1</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">narrow</th>
      <th>B/G</th>
      <td>2</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>4</td>
      <td>0.75</td>
      <td>0.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


## 17.10 Optional frozen-adapter transfer diagnostic

This asks whether B-raw can use an unseen structured prompt without training. It is not part of the primary representation comparison.


```python
frozen_transfer_results = pd.DataFrame()
if RUN_FROZEN_TRANSFER_DIAGNOSTIC:
    raw_model = load_adapter(RAW_CHECKPOINT)
    frozen_transfer_results = evaluate_structured_pairs(raw_model, "B-raw-structured-zero-shot")
    release_model(raw_model)
    display(pd.DataFrame([
        pair_summary(frozen_transfer_results, "B-raw-structured-zero-shot")
    ]))
else:
    print("Frozen transfer diagnostic skipped.")
```

    Frozen transfer diagnostic skipped.


## 17.11 Fixed-state, auxiliary, and gameplay guardrails

The paired experiment carries the derived-state claim. The Lab 15 fixed states and 19 fixed-opening games check whether any gain transfers beyond the perturbation set. Auxiliary tasks receive the decoded constraints they ask about, so their results are a representation-advantaged guardrail, not an independent transfer test.


```python
lab15_policy = pd.read_csv(LAB15_RESULTS / "policy-results.csv")
lab15_auxiliary = pd.read_csv(LAB15_RESULTS / "auxiliary-results.csv")
lab15_gameplay_calls = pd.read_csv(LAB15_RESULTS / "gameplay-calls.csv")
lab15_gameplay_games = pd.read_csv(LAB15_RESULTS / "gameplay-games.csv")

fixed_states = (
    lab15_policy.loc[
        (lab15_policy["model"] == "B") & (lab15_policy["interface"] == "training")
    ][["state_key", "answer", "turn", "candidate_count", "difficulty", "expected"]]
    .drop_duplicates("state_key").reset_index(drop=True)
)
fixed_states["history"] = fixed_states["state_key"].map(parse_state_key)
fixed_states["structured_prompt"] = [
    transform_prompt(
        "Task: NEXT_GUESS\nYou are playing Wordle.\nUse the game history to choose the next guess.\nReturn exactly one uppercase five-letter word.\n\nHistory:\n" + row.state_key,
        row.state_key,
        int(row.candidate_count),
    )
    for row in fixed_states.itertuples()
]

def evaluate_fixed_states(model) -> pd.DataFrame:
    rows = []
    for state in fixed_states.itertuples():
        raw = generate_prompt(model, state.structured_prompt)
        guess = parse_guess(raw)
        consistent = bool(
            guess and guess in ANSWER_SET and is_consistent(guess, state.history)
        )
        repeated = guess in {turn.guess for turn in state.history} if guess else False
        rows.append({
            "model": "B-structured",
            "state_key": state.state_key,
            "difficulty": state.difficulty,
            "actual": guess or raw.strip().upper(),
            "format_valid": guess is not None,
            "in_answer_lexicon": guess in ANSWER_SET if guess else False,
            "history_consistent": consistent,
            "repeated": repeated,
            "usable": bool(guess and guess in ANSWER_SET and consistent and not repeated),
            "exact": (guess or raw.strip().upper()) == state.expected,
        })
    return pd.DataFrame(rows)

structured_fixed_results = pd.DataFrame()
if RUN_EVALUATION:
    model = load_adapter(STRUCTURED_CHECKPOINT)
    structured_fixed_results = evaluate_fixed_states(model)
    release_model(model)
    raw_fixed = lab15_policy.loc[
        (lab15_policy["model"] == "B") & (lab15_policy["interface"] == "training")
    ]
    display(pd.DataFrame([
        {"model": "B-raw", "usable_rate": raw_fixed["usable"].mean()},
        {"model": "B-structured", "usable_rate": structured_fixed_results["usable"].mean()},
    ]))
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]



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
      <th>model</th>
      <th>usable_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B-raw</td>
      <td>0.063830</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B-structured</td>
      <td>0.234043</td>
    </tr>
  </tbody>
</table>
</div>



```python
a_test_rows = [
    json.loads(line)
    for line in (GENERATED_DIR / "wordle-sft-test.jsonl").read_text().splitlines()
]
all_train_prompts = {
    row["prompt"]
    for path in [
        GENERATED_DIR / "wordle-sft-train.jsonl",
        GENERATED_DIR / "wordle-part2-policy-train.jsonl",
    ]
    for row in (json.loads(line) for line in path.read_text().splitlines())
}
structured_aux_eval = []
for row in a_test_rows:
    if row["task"] == "NEXT_GUESS" or row["prompt"] in all_train_prompts:
        continue
    updated = dict(row)
    prompt_remainder = row["prompt"].split("\n\nHistory:\n", 1)[1]
    state_key = prompt_remainder.split("\n\n", 1)[0]
    candidate_count = len(filter_candidates(ANSWERS, parse_state_key(state_key)))
    updated["prompt"] = transform_prompt(
        row["prompt"], state_key, candidate_count
    )
    structured_aux_eval.append(updated)
assert len(structured_aux_eval) == 147
print("structured auxiliary guardrail rows:", len(structured_aux_eval))

def evaluate_structured_auxiliary(model) -> pd.DataFrame:
    rows = []
    for record in structured_aux_eval:
        actual = generate_prompt(model, record["prompt"]).strip().upper()
        rows.append({
            "model": "B-structured",
            "task": record["task"],
            "expected": record["response"].strip().upper(),
            "actual": actual,
            "correct": actual == record["response"].strip().upper(),
        })
    return pd.DataFrame(rows)

def format_training_history(history: list[Turn]) -> str:
    return "\n".join(
        f"{' '.join(turn.guess)} -> {' '.join(turn.feedback)}"
        for turn in history
    )

def structured_next_guess_prompt(history: list[Turn]) -> str:
    state_key = format_training_history(history)
    candidate_count = len(filter_candidates(ANSWERS, history))
    return transform_prompt(
        raw_next_guess_prompt(history), state_key, candidate_count
    )

def raw_next_guess_prompt(history: list[Turn]) -> str:
    state_key = format_training_history(history)
    return (
        "Task: NEXT_GUESS\n"
        "You are playing Wordle.\n"
        "Use the game history to choose the next guess.\n"
        "Return exactly one uppercase five-letter word.\n\n"
        f"History:\n{state_key}"
    )

def evaluate_gameplay(
    model, label: str, prompt_builder
) -> tuple[pd.DataFrame, pd.DataFrame]:
    call_rows, game_rows = [], []
    for answer in DEFAULT_EVAL_ANSWERS:
        history = [Turn("RAISE", score_string(answer, "RAISE"))]
        seen_game_guesses = {"RAISE"}
        seen_outputs = set()
        solved_turn = None
        for turn_number in range(2, 7):
            candidates_before = filter_candidates(ANSWERS, history)
            raw = generate_prompt(model, prompt_builder(history))
            guess = parse_guess(raw)
            format_valid = guess is not None
            in_answer_lexicon = bool(guess and guess in ANSWER_SET)
            repeated = bool(guess and guess in seen_game_guesses)
            output_repeated = bool(guess and guess in seen_outputs)
            if guess:
                seen_outputs.add(guess)
            consistent = bool(
                in_answer_lexicon and is_consistent(guess, history)
            )
            usable = bool(
                in_answer_lexicon and not repeated and consistent
            )
            feedback = score_string(answer, guess) if in_answer_lexicon else None
            if in_answer_lexicon:
                seen_game_guesses.add(guess)
                history.append(Turn(guess, feedback))
                candidates_after = filter_candidates(ANSWERS, history)
            else:
                candidates_after = candidates_before
            call_rows.append({
                "model": label,
                "answer": answer,
                "turn": turn_number,
                "raw": raw,
                "guess": guess,
                "format_valid": format_valid,
                "in_answer_lexicon": in_answer_lexicon,
                "repeated": repeated,
                "output_repeated": output_repeated,
                "history_consistent": consistent,
                "usable": usable,
                "candidate_count_before": len(candidates_before),
                "candidate_count_after": len(candidates_after),
            })
            if feedback == "GGGGG":
                solved_turn = turn_number
                break
        game_rows.append({
            "model": label,
            "answer": answer,
            "solved": solved_turn is not None,
            "solved_turn": solved_turn,
        })
    return pd.DataFrame(call_rows), pd.DataFrame(game_rows)

structured_aux_results = pd.DataFrame()
raw_gameplay_calls = pd.DataFrame()
raw_gameplay_games = pd.DataFrame()
structured_gameplay_calls = pd.DataFrame()
structured_gameplay_games = pd.DataFrame()
if RUN_EVALUATION:
    model = load_adapter(STRUCTURED_CHECKPOINT)
    structured_aux_results = evaluate_structured_auxiliary(model)
    structured_gameplay_calls, structured_gameplay_games = (
        evaluate_gameplay(model, "B-structured", structured_next_guess_prompt)
    )
    release_model(model)
    raw_model = load_adapter(RAW_CHECKPOINT)
    raw_gameplay_calls, raw_gameplay_games = evaluate_gameplay(
        raw_model, "B-raw", raw_next_guess_prompt
    )
    release_model(raw_model)

    raw_aux = lab15_auxiliary.loc[lab15_auxiliary["model"] == "B"]
    display(pd.concat([
        raw_aux.groupby("task")["correct"].mean().rename("B-raw"),
        structured_aux_results.groupby("task")["correct"].mean().rename(
            "B-structured"
        ),
    ], axis=1))

    gameplay_summary = pd.DataFrame([
        {
            "model": "B-raw",
            "solve_rate": raw_gameplay_games["solved"].mean(),
            "usable_rate": raw_gameplay_calls["usable"].mean(),
            "parsed_output_repeat_rate": raw_gameplay_calls["output_repeated"].mean(),
            "history_consistency_rate": raw_gameplay_calls["history_consistent"].mean(),
        },
        {
            "model": "B-structured",
            "solve_rate": structured_gameplay_games["solved"].mean(),
            "usable_rate": structured_gameplay_calls["usable"].mean(),
            "parsed_output_repeat_rate": structured_gameplay_calls[
                "output_repeated"
            ].mean(),
            "history_consistency_rate": structured_gameplay_calls[
                "history_consistent"
            ].mean(),
        },
    ])
    display(gameplay_summary)
else:
    print("Auxiliary and gameplay evaluation skipped.")
```

    structured auxiliary guardrail rows: 147



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]



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
      <th>B-raw</th>
      <th>B-structured</th>
    </tr>
    <tr>
      <th>task</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CHOOSE_VALID</th>
      <td>0.653061</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>VALID_CANDIDATE</th>
      <td>0.765306</td>
      <td>0.918367</td>
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
      <th>model</th>
      <th>solve_rate</th>
      <th>usable_rate</th>
      <th>parsed_output_repeat_rate</th>
      <th>history_consistency_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B-raw</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.421053</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B-structured</td>
      <td>0.263158</td>
      <td>0.119048</td>
      <td>0.273810</td>
      <td>0.119048</td>
    </tr>
  </tbody>
</table>
</div>


## 17.12 Persist results and interpret without moving the goalposts

| Result | Interpretation |
| --- | --- |
| Paired and branch consistency rise with paired evidence beyond run noise | Derived-state features were a bottleneck under this training setup |
| Sensitivity rises but consistency does not | Explicit fields attract attention but do not teach constraint semantics |
| Fixed states improve but pairs do not | The model memorizes familiar structured states without causal state binding |
| Nothing improves | Derived-state features alone are insufficient; test targeted constraint learning or capacity |

Do not change Dataset B targets or add new examples in this notebook.


```python
if RUN_EVALUATION:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    structured_pair_results.to_csv(RESULTS_DIR / "pair-results.csv", index=False)
    structured_fixed_results.to_csv(RESULTS_DIR / "fixed-state-results.csv", index=False)
    structured_aux_results.to_csv(RESULTS_DIR / "auxiliary-results.csv", index=False)
    raw_gameplay_calls.to_csv(RESULTS_DIR / "raw-gameplay-calls.csv", index=False)
    raw_gameplay_games.to_csv(RESULTS_DIR / "raw-gameplay-games.csv", index=False)
    structured_gameplay_calls.to_csv(RESULTS_DIR / "gameplay-calls.csv", index=False)
    structured_gameplay_games.to_csv(RESULTS_DIR / "gameplay-games.csv", index=False)
    summaries.to_csv(RESULTS_DIR / "pair-summary.csv", index=False)
    print("saved Lab 17 results to", RESULTS_DIR)

if RUN_FROZEN_TRANSFER_DIAGNOSTIC:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frozen_transfer_results.to_csv(
        RESULTS_DIR / "frozen-transfer-results.csv", index=False
    )
    print("saved frozen-transfer diagnostic to", RESULTS_DIR)
```

    saved Lab 17 results to ../results/lab17


# Lab 17 checkpoint

Record:

1. proof that Dataset B rows, targets, weights, and splits stayed fixed;
2. structured representation examples, including duplicate-letter states;
3. structured/raw input-token exposure ratio;
4. paired consistency and branch consistency for B-raw and B-structured;
5. sensitivity by broad, mixed, narrow, and feedback-change type;
6. fixed-state usable rate;
7. auxiliary accuracy and fixed-opening gameplay guardrails;
8. the pre-registered interpretation supported by the result.
