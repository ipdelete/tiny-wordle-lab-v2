# Measures from Karpathy's autoresearch

This note records how
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) measures and
selects experiments, then maps that design to the GPT-OSS-20B Wordle baseline.
The source revision reviewed was
[`228791f`](https://github.com/karpathy/autoresearch/commit/228791fb499afffb54b46200aca536f79142f117).

## One metric decides

Autoresearch has one optimization target:

```text
val_bpb
```

Validation bits per byte measures cross-entropy normalized by the byte length
of each target token:

```text
val_bpb = total validation nats / (ln(2) * total target bytes)
```

Lower is better. Byte normalization permits comparisons when an experiment
changes vocabulary size or tokenization behavior.

The agent keeps an experiment only when validation BPB improves. The other
measurements explain the result but do not compete with the primary metric.

Source:
[`prepare.py`, evaluation](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/prepare.py#L340-L365).

## Fixed experimental budget

Each experiment receives five minutes of steady-state training. Startup and
compilation do not count.

The fixed budget makes architecture, optimizer, model size, and batch-size
changes answer the same question:

> Which configuration achieves the lowest validation BPB on this hardware in
> five minutes?

Time is a control, not an outcome. A run that exceeds ten minutes is treated as
a failure because its stopping logic or startup behavior is outside the
experiment contract.

Source:
[`program.md`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md).

## Run diagnostics

Each completed run prints:

| Measure | Role |
| --- | --- |
| `val_bpb` | Primary quality metric |
| `training_seconds` | Confirms the training budget |
| `total_seconds` | Includes startup, compilation, and evaluation |
| `peak_vram_mb` | Resource guardrail |
| `mfu_percent` | Hardware utilization |
| `total_tokens_M` | Training throughput |
| `num_steps` | Optimizer updates completed |
| `num_params_M` | Model size |
| `depth` | Main architecture-size parameter |

Only validation BPB decides whether an experiment advances. Peak VRAM is a
soft constraint. The other values help explain why one configuration trained
better within the fixed period.

Source:
[`train.py`, final summary](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/train.py#L610-L630).

## Experiment ledger

Autoresearch records each attempt in `results.tsv`:

```text
commit    val_bpb    memory_gb    status    description
```

Status is `keep`, `discard`, or `crash`. Each idea gets a commit before its
run. Improved commits remain on the experiment branch. Rejected commits are
recorded and then removed from the active branch.

The analysis notebook derives:

- counts of kept, discarded, and crashed experiments;
- keep rate;
- the running best BPB;
- total improvement from baseline;
- improvement from each accepted experiment;
- experiments required to reach each improvement;
- a frontier plot of accepted and rejected runs.

Source:
[`analysis.ipynb`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/analysis.ipynb).

## Simplicity remains qualitative

Autoresearch uses a second selection rule that is deliberately not folded into
BPB:

- Equal quality favors simpler code.
- A tiny gain may not justify a large or fragile change.
- Equal quality with less code is worth keeping.

This avoids inventing a numerical complexity score that would be hard to
defend. Quality remains measurable; code complexity remains a judgment.

## Mapping the design to Wordle

The prompted GPT-OSS-20B baseline has a different purpose. It characterizes a
stochastic sequential policy rather than optimizing language-model pretraining
under a wall-clock budget.

The same hierarchy still applies:

1. Use one unambiguous result to rank policies.
2. Freeze the conditions that make results comparable.
3. Record diagnostics that explain failures without letting them redefine the
   winner.

### Primary result

The existing evaluator uses the right ordering:

```text
solved games, descending
penalized turns, ascending
```

A solved game costs its turn count. A failed game costs seven turns. More
solves always win; penalized turns resolve ties.

This should remain a lexicographic comparison rather than a collection of
equally weighted behavior metrics.

### Fixed controls

Every comparable baseline should freeze and record:

- model name and revision;
- prompt content and SHA-256;
- answer-list content and SHA-256;
- legal vocabulary;
- number of repeats;
- decoding temperature and seeds;
- six action opportunities;
- failure cost;
- evaluator revision.

A fixed answer and rollout budget is more appropriate than fixed wall time.
Model latency should not affect policy quality.

### Resource diagnostics

Record:

- elapsed time;
- model calls;
- input and output tokens;
- truncated responses;
- calls per second;
- tokens per game.

These numbers measure operational cost. They do not decide which policy plays
better.

### Behavioral diagnostics

Record:

- legal-action rate;
- repeated-action rate;
- feedback consistency;
- singleton closure;
- results by candidate-count regime;
- results for repeated-letter answers;
- results by word-frequency band;
- candidate count after each turn;
- information gain or entropy regret.

These measures locate failure modes. They should enter a training reward only
after an experiment shows that the corresponding failure needs a direct
learning signal.

## Proposed Wordle ledger

A compact experiment ledger could contain:

```text
commit
prompt_sha256
model
seeds
games
solved
penalized_turns
illegal_actions
repeat_actions
status
description
```

Detailed trajectories remain in `games.jsonl`. The ledger tracks comparisons,
not every observation.

Useful statuses are:

- `baseline`
- `keep`
- `discard`
- `crash`

A candidate advances when it solves more frozen validation games, or ties on
solves and uses fewer penalized turns.

## Where Wordle needs stronger evidence

Autoresearch evaluates one fixed validation loss under a controlled local
training process. Wordle gameplay adds stochastic decoding and sequential
state changes. Equal aggregate scores can conceal different game outcomes.

Wordle experiments should also report:

- results across repeated seeds;
- paired outcomes for each answer;
- candidate-only and incumbent-only wins;
- subgroup results;
- variation across repeats;
- one untouched final holdout.

The baseline notebook should be broad in what it measures. Later autonomous
training experiments should remain narrow in how they promote a checkpoint.

## Takeaway

Autoresearch does not optimize every number it prints. It uses one metric for
selection, a fixed budget for comparability, resource measurements for
guardrails, and a small ledger for experimental memory.

The Wordle lab should do the same:

> Rank policies by solved games and penalized turns. Use the remaining
> measurements to explain what the policy learned and where it still fails.
