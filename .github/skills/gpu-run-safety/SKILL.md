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

## Check this first: full-vocabulary logits

Full-vocabulary logits contributed to all three crashes, but they were not the
only cause. Check them before anything else.

Qwen3's vocabulary is 151,936. A logits tensor is
`batch x positions x 151,936 x 4 bytes`. At batch 12 and 109 positions that is
**795 GiB**. The model will happily ask for it.

You almost never want logits at every position. You want them at the positions
you are about to score or train on.

| lab | shape requested | fix |
| --- | --- | --- |
| 09 distillation | `[12, 109, 151936]` | `logits_to_keep` on response positions, `use_cache=False` |
| 17 structured | same pattern on the first run | same, plus gradient microbatching |
| 18b scoring | `[1, 129, 151936]` in the prefill | `logits_to_keep=1` |

The fixes:

- **Scoring or generation.** Pass `logits_to_keep=1`. Only the final position can
  predict the next token, so everything else is waste.
- **Training.** Slice to the response-predicting positions before computing the
  loss, and set `use_cache=False`. Prompt positions contribute nothing to a
  response-only loss.
- **Reading a few token probabilities.** Use `gather` minus `logsumexp`, not
  `log_softmax`. `log_softmax` materializes a second full-size copy of the
  logits.

This is not only a memory fix. Removing the unused prefill logits in Lab 18b made
the kernel **8x faster**, 1.72 s/state to 0.22 s/state, because the model had
been computing and discarding a 129 by 151,936 projection every single state.

So: before running anything, grep your loop for `.logits` and ask what shape it
is and which positions you actually need.

## Procedure

### 1. Cap the process

First cell of any notebook, top of any script:

```python
MEMORY_CAP_GIB = 128.0
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

Size the cap to give the kernel room to run at full speed, then leave the rest of
the machine a wide margin. The cap exists to catch a runaway, not to enforce
frugality. A 512 GiB host running a 24 GiB working set has no problem. Capping
that same kernel at 36 GiB buys nothing and turns ordinary allocator variance
into a false alarm. Roughly 4x the observed peak, with at least a few hundred GiB
left for the system, is the right shape.

CUDA equivalent is `torch.cuda.set_per_process_memory_fraction(fraction)`.

### 2. Soak at small N

Repeat one fixed worst-case input for 40 or more iterations and record the peak
**inside** the iteration, not between iterations:

```python
def driver_memory_gib() -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1024**3
    return float("nan")
```

Sampling between iterations is the trap that let Lab 18b through. The probe read
a reassuring flat 2.71 GiB after each state's `empty_cache()`, while the real
mid-state peak was 23.6 GiB and raised at the cap on the next run. Sample where
the big tensors are live, inside the inner loop.

Note which figure you are reading. `current_allocated_memory` is live tensors,
`driver_allocated_memory` is the pool. They can differ by 3x, and the cap is
enforced against the pool. Soak on the pool figure.

Keep shapes constant during this test. Changing prompt length, batch size, or
sequence length changes the legitimate working set and can look like a leak.
Lab 18b first soaked states in battery order, where mean turn depth rose from
2.9 to 3.7. It then shuffled them, but still mixed shapes. Neither run isolated
allocator growth.

**Test for a plateau, not for a small number.** A high steady working set is fine.
A working set that keeps climbing is not, because it has no ceiling. Discard the
first third as warmup, then compare the middle and final thirds. Also require a
narrow range within the final third:

```python
third = len(peaks) // 3
creep = sum(peaks[-third:]) / third - sum(peaks[third:2 * third]) / third
assert creep < 0.5, f"still climbing {creep:+.2f} GiB after warmup, diagnose first"
assert max(peaks[-third:]) - min(peaks[-third:]) < 0.5, "working set has not plateaued"
assert max(peaks) < ABORT_GIB, f"peak {max(peaks):.1f} GiB too close to the cap"
```

Measure headroom separately by running the largest real inputs after the fixed
soak. That test asks whether the cap covers the workload; it does not diagnose a
leak.

If headroom is genuinely short, measure the trade before shrinking the batch.
In Lab 18b, halving the chunk from 256 to 128 halved the *live* peak from 12.9 to
6.9 GiB but barely moved the *pool*, which is what the cap sees. It cost 0.07 s
per state and bought nothing. Raising the cap was the correct move.

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
| plus `torch.mps.empty_cache()` | flat between states |

Two hypotheses died there. It was not pending async frees, and it was not shape
variance. Do not assume either without testing.

And note what that table missed. Every row was measured between states, so the
whole exercise reported a 2.71 GiB steady state for a kernel whose real peak was
23.6 GiB. The bisection was correct and the measurement was not. When a leak is
fixed, re-measure the peak before declaring the run safe.

## Known causes in this repo

**Full-vocabulary logits.** See the section above. Present in all three crashes,
so it is the first thing to check, not the last.

**The MPS allocator does not reuse large blocks.** A large per-iteration
allocation, such as a KV cache expanded to a batch dimension, is cached and never
handed back. The pool grows once per iteration until the machine dies. Call
`torch.mps.empty_cache()` at the end of each iteration. It costs nothing
measurable.

**Grouped-query attention doubles the cache during the forward.** Qwen3 has 16
query heads against 8 key/value heads, so SDPA's `repeat_kv` materializes a
doubled copy of the expanded cache on every forward. It does not appear in the
cache size you calculated, and it is transient, so it is invisible to any
between-iteration measurement. Peak scales linearly with the batch, so shrink the
chunk if you need headroom.

## After any fix

Verify numbers, not just memory. A memory fix that changes results is a
different bug. Compare against a plain single-sequence forward pass and expect
float32 agreement, roughly 1e-5.

Then rerun the soak before going to full scale.
