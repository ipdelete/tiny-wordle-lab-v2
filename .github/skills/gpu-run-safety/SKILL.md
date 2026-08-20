---
name: gpu-run-safety
description: Guard long GPU/MPS runs against unbounded memory growth. Use before executing any notebook or script that trains, evaluates, or scores a model, or when a run has crashed the host or been killed for memory.
---

# GPU run safety

Three runs in this project have grown memory without bound. The last exhausted
512 GiB and took the machine down. Every one was a new GPU loop taken straight to
full scale.

The gate below is the point of this skill. The tooling only makes it cheap.

## The gate

**A new or modified GPU loop does not run at full scale until it has passed a
bounded soak.** No exceptions for "it's only evaluation" or "it's the same kernel
as last time". Two of the three crashes were evaluation, not training.

Full scale means the real N: every state, every epoch, every model.

## Procedure

### 1. Cap the process

First cell of any notebook, top of any script:

```python
MEMORY_CAP_GIB = 24.0
import torch
if torch.backends.mps.is_available():
    total = torch.mps.recommended_max_memory() / 1024**3
    torch.mps.set_per_process_memory_fraction(MEMORY_CAP_GIB / total)
```

Past the cap PyTorch raises `RuntimeError: MPS backend out of memory` with a
stack trace at the offending line. A crash becomes a bug report.

The cap does more than abort. Under watermark pressure the MPS allocator starts
recycling cached blocks instead of requesting new ones, so it also suppresses the
leak below. Measured on the Lab 18b kernel with `empty_cache()` deliberately
removed: unbounded growth without the cap, a flat 11.75 GiB plateau over 60
iterations with a 24 GiB cap.

Do not let that become the fix. Recycling under pressure is a side effect, and it
still runs at four times the steady state. Fix the leak and keep the cap.

Set the cap to roughly twice the expected steady state, not to what is
available. A generous cap defeats the purpose.

CUDA equivalent is `torch.cuda.set_per_process_memory_fraction(fraction)`.

### 2. Soak at small N

Run 20 to 40 iterations and print memory every iteration:

```python
def driver_memory_gib() -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1024**3
    return float("nan")
```

Flat is the pass condition. Not "slowly rising", not "rising then plateauing
maybe". Growth over 25 iterations must be under about 1 GiB.

Assert it, so the soak fails loudly rather than being eyeballed:

```python
assert growth < 1.0, f"memory grows {growth:.2f} GiB over 25 states, do not run full scale"
```

Keep the soak in the notebook. It is the regression test.

### 3. Run under the watchdog

Never launch a long run bare:

```bash
scripts/memguard.py -- uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/NN_name.ipynb
```

`scripts/memguard.py` samples total system memory once a second and kills the
whole process group before the host is at risk. Defaults are 48 GiB minimum
available and 8 GiB maximum swap; tune with `--min-free` and `--max-swap`.

On a trip it exits 137 and writes `/tmp/memguard-reason.txt` and
`/tmp/memguard-trace.csv`. Read both before changing any code.

Exit 137 means the watchdog fired. Treat it as a memory bug, not a flake, and
never rerun without diagnosing.

## Diagnosing a leak

Reproduce in a bounded subprocess with a hard abort, never in the main session.
A watchdog thread calling `os._exit()` past a threshold keeps the host safe while
you iterate. Note that the memory cap from step 1 will mask a leak by forcing
block recycling, so turn it off or raise it well above steady state while
bisecting, and keep the abort threshold low.

Then bisect by variant rather than guessing. The measured result from the Lab 18b
investigation:

| variant | outcome |
| --- | --- |
| baseline | +11 GiB per state, unbounded |
| plus `torch.mps.synchronize()` | +11 GiB per state, unbounded |
| fixed-length prompts, constant tensor shapes | unbounded |
| plus `torch.mps.empty_cache()` | flat at 2.71 GiB |

Two hypotheses died there. It was not pending async frees, and it was not shape
variance. Do not assume either without testing.

## Known causes in this repo

**The MPS allocator does not reuse large blocks.** A large per-iteration
allocation, such as a KV cache expanded to a batch dimension, is cached and never
handed back. The pool grows once per iteration until the machine dies. Call
`torch.mps.empty_cache()` at the end of each iteration. It costs nothing
measurable: 1.8 s per state either way.

**Full-vocabulary logits across all positions.** Qwen3's vocabulary is 151,936.
Materializing logits for every prompt position builds enormous tensors. This is
Lab 09's bug and it reappeared in Lab 18b's prefill.

- Generation and scoring: pass `logits_to_keep=1` so only the final position is
  produced.
- Training: slice logits to the response-predicting positions before the loss,
  and set `use_cache=False`.
- Prefer `gather` minus `logsumexp` over `log_softmax` when you need only a few
  token log-probabilities. `log_softmax` materializes a second full-size copy.

## After any fix

Verify numbers, not just memory. A memory fix that changes results is a
different bug. Compare against a plain single-sequence forward pass and expect
float32 agreement, roughly 1e-5.

Then rerun the soak before going to full scale.
