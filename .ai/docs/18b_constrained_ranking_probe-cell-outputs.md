# Lab 18b - Does the answer lexicon explain the gap?

Lab 18 returned a null. Dataset G moved the 620-state battery from 90/620 to
99/620 usable, a delta of +1.45 points with a 95% CI of -1.45 to +4.19 and an
exact p of 0.374. Three data interventions in a row (Lab 14 allocation, Lab 17
representation, Lab 18 distribution) have produced shrinking effects.

Before running a fourth, this lab asks whether we have been measuring the right
failure. On the same 620 states, B-structured parses as five letters 92.7% of the
time but emits an actual answer-list word only 46.1% of the time. The rest are
strings like `BAGGE`, `WESHT`, and `GALEL`. Meanwhile the auxiliary tasks, where
the model picks from a supplied list, score 95.9% and 91.8%. Given options the
model is nearly perfect. Asked to produce one, it invents a word about half the
time.

This lab trains nothing. It scores every one of the 2,315 answer words under each
existing adapter and asks what the policy looks like when the lexicon is imposed
at decode time instead of left to the model.

## 18b.1 Pre-registered experiment

**Hypothesis.** If the answer lexicon is imposed at decode time, state-conditioned
correctness rises substantially. If it does not, the policy is weak on its own
terms and no further data intervention is worth running.

**Design.** For each of the same 620 held-out states used in Lab 18, compute the
summed log-probability of every answer word under each model and keep the whole
620 by 2315 matrix. Every readout below is a filter over that one matrix, so the
analysis reruns without touching the GPU.

**Tiers.**

| Tier | Action space | Question |
| --- | --- | --- |
| 0 | free generation (from Lab 18) | current measured behaviour |
| 1 | argmax over all 2,315 answers | is lexicon grounding the bottleneck |
| 2 | argmax over the consistent candidate set | can it pick the *best* legal word |

Tier 2 is history-consistent by construction, so its metric is teacher match.
That separates "knows which words are legal" from "knows which legal word is
best".

**Distributional readouts.** Argmax throws away most of the signal. We also record
the rank of the teacher word in the full 2,315 ranking, mean reciprocal rank,
top-k hit rates, and the mean rank percentile of the consistent candidate set. A
state-blind ranking puts that percentile at exactly 0.5; values near 0 mean the
feedback pulls legal words to the top of the whole list even when the argmax is
wrong. That percentile is the cleanest state-conditioning measure here because it
uses the entire ranking and never depends on an argmax.

**Models.** `base` (Qwen3-0.6B, no adapter), `B-structured`, `G-structured`. The
base control decides whether any advantage comes from our training or from the
lexicon restriction alone.

**Scoring rule.** Summed `log P(word tokens + EOS | prompt)`, which is the true
string likelihood and what a real constrained decoder maximizes. Length-normalized
mean log-probability is recorded as a sensitivity check, because answer words
tokenize into 1, 2, or 3 tokens and unnormalized sums tilt toward shorter
tokenizations.

**Statistics.** All tiers are evaluated on the identical 620 states, so every
comparison is paired: flip counts, paired bootstrap 95% CI, and exact McNemar.

**Pre-registered readings.**

- Tier 1 is close to Tier 0 -> grounding is not the bottleneck, the policy is
  genuinely weak, and Part II's data labs are at their ceiling. This is a stop
  signal and we report it as one.
- Tier 1 is far above Tier 0 -> we have been scoring a decoding failure since Lab
  15, and Labs 15 through 18 need rereading under the constrained interface.
- Tier 1 is far above Tier 0 but `base` matches the adapters -> the lexicon
  constraint did the work, not the training.
- Tier 2 teacher match near chance (`1 / candidates`) -> the model learned
  consistency filtering and never learned the entropy policy. Different problem
  than the one we have been attacking.
- Candidate rank percentile near 0.5 -> the likelihood surface is state-blind.
  Lab 16's finding reappearing in the distribution rather than in the argmax.

**What this lab cannot show.** Nothing here says a constrained decoder would win a
game. Tier 1 removes an entire failure mode by fiat. A positive result licenses
changing the evaluation and decoding interface, not a claim about the policy's
unaided competence.

## 18b.2 Run controls and memory guard

Three labs in this project have hit unbounded memory growth on MPS, and the last
one took the host down. The point of the guard is to stop a runaway, not to run
lean. This host has 512 GiB and a fast working set is worth paying for, so the
cap below is set well above what the kernel needs rather than trimmed to it.

Layer 1 is the cell below: a hard per-process cap on the MPS allocator. Past the
cap PyTorch raises a normal `RuntimeError` with a stack trace instead of
consuming the machine.

Layer 2 lives outside this notebook. Run it through the watchdog, never bare:

```
scripts/memguard.py -- uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/18b_constrained_ranking_probe.ipynb
```


```python
RUN_SCORING = True
SCORE_B_RAW = False  # optional, needs the raw interface and doubles nothing else

# Sized to leave the kernel room to run at full speed, not to minimise usage.
# 128 GiB still leaves ~380 GiB for the rest of the machine.
MEMORY_CAP_GIB = 128.0

import torch

if torch.backends.mps.is_available():
    total_gib = torch.mps.recommended_max_memory() / 1024**3
    torch.mps.set_per_process_memory_fraction(MEMORY_CAP_GIB / total_gib)
    print(f"MPS cap: {MEMORY_CAP_GIB:.0f} GiB of {total_gib:.0f} GiB")
    print("past the cap PyTorch raises RuntimeError instead of taking the host down")

print("RUN_SCORING:", RUN_SCORING)
print("SCORE_B_RAW:", SCORE_B_RAW)
```

    MPS cap: 128 GiB of 464 GiB
    past the cap PyTorch raises RuntimeError instead of taking the host down
    RUN_SCORING: True
    SCORE_B_RAW: False



```python
from collections import defaultdict
from pathlib import Path
import gc
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_wordle.expert import EntropyExpert
from tiny_wordle.game import Turn, filter_candidates, is_consistent
from tiny_wordle.hardware import preferred_device

MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 42
CHUNK_SIZE = 256
MEMORY_ABORT_GIB = MEMORY_CAP_GIB * 0.75

DATA_DIR = Path("../data")
CHECKPOINT_ROOT = Path("../checkpoints")
RESULTS_DIR = Path("../results/lab18b")
LAB18_RESULTS = Path("../results/lab18")
B_RAW_CHECKPOINT = CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b"
B_STRUCTURED_CHECKPOINT = CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured"
G_STRUCTURED_CHECKPOINT = CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-g-structured"
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


## 18b.3 Reuse the Lab 17 structured representation verbatim

These functions are copied unchanged from Labs 17 and 18. The prompts this lab
scores must be byte-identical to the prompts the adapters were trained on and
were evaluated on in Lab 18, so the representation is rebuilt through the same
`raw_policy_prompt` plus `transform_prompt` path rather than written afresh.


```python
ANSWERS = [
    line.strip().upper()
    for line in (DATA_DIR / "wordle-answers-original.txt").read_text().splitlines()
    if line.strip()
]
ANSWER_SET = set(ANSWERS)
PATTERNS = np.load(DATA_DIR / "wordle-patterns-original-2315.npy")
expert = EntropyExpert(ANSWERS, PATTERNS)
WORD_TO_INDEX = expert.word_to_index
assert len(ANSWERS) == 2315 and PATTERNS.shape == (2315, 2315)


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


def raw_policy_prompt(state_key: str) -> str:
    return (
        "Task: NEXT_GUESS\nYou are playing Wordle.\n"
        "Use the game history to choose the next guess.\n"
        "Return exactly one uppercase five-letter word.\n\n"
        f"History:\n{state_key}"
    )


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

## 18b.4 Rebuild the exact 620-state battery

The battery is read back from the Lab 18 result artifacts rather than
regenerated, which guarantees the same states in the same order. The Lab 18
free-generation outcomes come along as the Tier 0 baseline.


```python
b_tier0 = pd.read_csv(LAB18_RESULTS / "b-state-battery-results.csv")
g_tier0 = pd.read_csv(LAB18_RESULTS / "state-battery-results.csv")
assert len(b_tier0) == len(g_tier0) == 620
assert list(b_tier0["state_key"]) == list(g_tier0["state_key"])

battery = g_tier0[
    ["state_key", "source", "turn", "candidate_bucket", "expected"]
].copy()
battery["history"] = battery["state_key"].map(parse_state_key)
battery["candidates"] = [
    filter_candidates(ANSWERS, history) for history in battery["history"]
]
battery["candidate_count"] = battery["candidates"].map(len)
battery["structured_prompt"] = [
    transform_prompt(
        raw_policy_prompt(row.state_key), row.state_key, int(row.candidate_count)
    )
    for row in battery.itertuples()
]
battery["raw_prompt"] = battery["state_key"].map(raw_policy_prompt)

# The teacher target must be inside the candidate set, and the recomputed
# candidate counts must reproduce the Lab 18 buckets exactly.
assert all(
    row.expected in set(row.candidates) for row in battery.itertuples()
), "teacher target outside candidate set"
rebuilt_bucket = pd.cut(
    battery["candidate_count"], [0, 2, 10, 50, 200, float("inf")],
    labels=["1-2", "3-10", "11-50", "51-200", "201+"],
).astype(str)
assert (rebuilt_bucket == battery["candidate_bucket"]).all(), "bucket mismatch vs Lab 18"

print("battery states:", len(battery))
print("mean candidates:", round(battery["candidate_count"].mean(), 1))
display(pd.crosstab(battery["turn"], battery["candidate_bucket"]))
print()
print("Tier 0 (free generation, from Lab 18)")
display(pd.DataFrame([
    {
        "model": name,
        "format_valid": frame["format_valid"].mean(),
        "in_answer_lexicon": frame["in_answer_lexicon"].mean(),
        "history_consistent": frame["history_consistent"].mean(),
        "usable": frame["usable"].mean(),
        "teacher_match": frame["teacher_match"].mean(),
    }
    for name, frame in [("B-structured", b_tier0), ("G-structured", g_tier0)]
]))
```

    battery states: 620
    mean candidates: 7.4



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
      <th>candidate_bucket</th>
      <th>1-2</th>
      <th>11-50</th>
      <th>201+</th>
      <th>3-10</th>
      <th>51-200</th>
    </tr>
    <tr>
      <th>turn</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>12</td>
      <td>2</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>96</td>
      <td>39</td>
      <td>0</td>
      <td>206</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Tier 0 (free generation, from Lab 18)



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
      <th>format_valid</th>
      <th>in_answer_lexicon</th>
      <th>history_consistent</th>
      <th>usable</th>
      <th>teacher_match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B-structured</td>
      <td>0.927419</td>
      <td>0.461290</td>
      <td>0.145161</td>
      <td>0.145161</td>
      <td>0.074194</td>
    </tr>
    <tr>
      <th>1</th>
      <td>G-structured</td>
      <td>0.885484</td>
      <td>0.477419</td>
      <td>0.159677</td>
      <td>0.159677</td>
      <td>0.077419</td>
    </tr>
  </tbody>
</table>
</div>


## 18b.5 Tokenize the answer lexicon

Training used `render_prompt(prompt) + response + eos_token` tokenized as one
string. Scoring a word standalone is only valid if the tokenizer produces the
same ids at the join, so that is asserted for all 2,315 words against a real
prompt rather than assumed.

Words are bucketed by token length. A word of `L` tokens needs only `L-1`
forwarded positions, because the first token's distribution comes from the
prompt's final position. That drops the work from 9,260 scored positions per
state to 5,246.


```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


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

# Standalone encodings must match the joint prompt+response tokenization used in
# training, otherwise every score would be computed off a different token path.
probe_prompt = render_prompt(battery["structured_prompt"].iloc[0])
probe_ids = tokenizer(probe_prompt, add_special_tokens=False)["input_ids"]
for word, tokens in zip(ANSWERS, WORD_TOKENS):
    joint = tokenizer(
        probe_prompt + word + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    assert joint[:len(probe_ids)] == probe_ids and joint[len(probe_ids):] == tokens

length_counts = pd.Series([len(t) for t in WORD_TOKENS]).value_counts().sort_index()
print("word+eos token lengths:")
print(length_counts.to_string())
print("scored positions per state:",
      int(sum((len(t) - 1) for t in WORD_TOKENS)), "of", len(ANSWERS) * 4, "naive")

# Pad each bucket up to a whole number of chunks so the expanded KV cache keeps a
# constant batch dimension and can be reused across every chunk of a state.
LENGTH_BUCKETS = {}
for length in sorted({len(t) for t in WORD_TOKENS}):
    indices = [i for i, t in enumerate(WORD_TOKENS) if len(t) == length]
    padding = (-len(indices)) % CHUNK_SIZE
    padded = indices + [indices[-1]] * padding
    LENGTH_BUCKETS[length] = (
        torch.tensor(padded),
        torch.tensor([WORD_TOKENS[i] for i in padded], device=device),
    )
print("buckets:", {k: int(v[0].shape[0]) for k, v in LENGTH_BUCKETS.items()})
```

    word+eos token lengths:
    2     156
    3    1387
    4     772
    scored positions per state: 5246 of 9260 naive
    buckets: {2: 256, 3: 1536, 4: 1024}


## 18b.6 Scoring engine

One prompt forward per state produces a KV cache, which is expanded once to the
chunk batch size and reused for every chunk by cropping the appended positions
back off. Log-probabilities are gathered per chunk and reduced immediately.

### Memory discipline

An earlier version of this kernel exhausted 512 GiB of unified memory and took
the host down. The measured cause, in a bounded probe:

| variant | behaviour |
| --- | --- |
| baseline | +11 GiB per state, unbounded |
| plus `torch.mps.synchronize()` | +11 GiB per state, unbounded |
| fixed-length prompts (constant cache shape) | unbounded |
| plus `torch.mps.empty_cache()` | flat between states |

So it was neither pending async frees nor shape variance. The MPS caching
allocator retains the expanded KV-cache block and does not reuse it, so the pool
grows once per state until the machine dies. Only an explicit allocator flush
returns it.

That fix was necessary but not sufficient. The first guarded run still raised at
the cap, inside `repeat_kv`: Qwen3 has 16 query heads against 8 key/value heads,
so SDPA materializes a doubled copy of the expanded cache on every forward. The
probe had sampled memory only *between* states, after the flush, and reported a
reassuring 2.71 GiB while the true mid-state peak was 23.6 GiB.

Peak scales with the chunk batch. Live-tensor peak, measured:

| chunk | peak | s/state |
| --- | --- | --- |
| 256 | 12.9 GiB | 0.22 |
| 128 | 6.9 GiB | 0.29 |
| 64 | 4.4 GiB | 0.47 |

Those are live-tensor figures. The allocator *pool* runs a good deal higher,
around 24 GiB at either 256 or 128, because blocks from the three word-length
buckets accumulate within a state and are only released at the end of it.
Shrinking the chunk barely moves the pool, so there is no speed to be gained by
trading it away.

`CHUNK_SIZE = 256` is the fast setting and it fits under a 128 GiB cap with
about 5x headroom. Running at 24 GiB on a 512 GiB host is not a problem. Running
at 24 GiB and climbing is, which is what the soak gate below actually tests.

Four disciplines are applied, and the peak column printed during scoring is the
regression test:

1. `torch.mps.empty_cache()` after every state. Stops the pool from growing.
2. `logits_to_keep=1` on the prefill. Lab 09's rule: never materialize
   full-vocabulary logits across all prompt positions. Without it the prompt
   forward builds a 129 by 151,936 tensor, which also cost 8x in wall clock.
3. `gather` minus `logsumexp` instead of `log_softmax`, avoiding a second
   full-size copy of the chunk logits.
4. Peaks sampled inside the chunk loop, not between states, so the number the
   soak gate checks is the number that can actually trip the cap.


```python
def load_model(path: Path | None):
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    if path is None:
        return base.eval()
    if not path.exists():
        raise FileNotFoundError(f"missing adapter {path}")
    return PeftModel.from_pretrained(base, path).to(device).eval()


def release_model(model):
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


LAST_STATE_PEAK_GIB = 0.0


@torch.no_grad()
def score_all_words(model, prompt_text: str) -> np.ndarray:
    """Summed log P(word + EOS | prompt) for all 2,315 answers."""
    global LAST_STATE_PEAK_GIB
    input_ids = tokenizer(
        render_prompt(prompt_text), return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    prefill = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    final_logits = prefill.logits[0, -1].float()
    first_token_logprobs = final_logits - final_logits.logsumexp(-1)
    cache = prefill.past_key_values
    cache.batch_repeat_interleave(CHUNK_SIZE)

    peak = 0.0
    scores = torch.zeros(len(ANSWERS), dtype=torch.float32)
    for length, (indices, tokens) in LENGTH_BUCKETS.items():
        for start in range(0, len(indices), CHUNK_SIZE):
            chunk = tokens[start:start + CHUNK_SIZE]
            total = first_token_logprobs[chunk[:, 0]].clone()
            if length > 1:
                step = length - 1
                output = model(
                    input_ids=chunk[:, :step],
                    past_key_values=cache,
                    use_cache=True,
                )
                chunk_logits = output.logits.float()
                token_logits = chunk_logits.gather(
                    2, chunk[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                total = total + (
                    token_logits - chunk_logits.logsumexp(-1)
                ).sum(dim=1)
                # Sampled inside the chunk loop on purpose. Sampling only
                # between states misses the transient peak entirely, which is
                # how a 23.6 GiB peak passed a soak that reported 2.71 GiB.
                peak = max(peak, driver_memory_gib())
                cache.crop(-step)
                del output, chunk_logits, token_logits
            scores[indices[start:start + CHUNK_SIZE]] = total.cpu()

    LAST_STATE_PEAK_GIB = peak
    del cache, prefill, final_logits, first_token_logprobs
    # Load-bearing. Without this the allocator pool grows ~11 GiB per state.
    if device.type == "mps":
        torch.mps.empty_cache()
    return scores.numpy()


def score_battery(model, prompt_column: str, label: str) -> np.ndarray:
    matrix = np.zeros((len(battery), len(ANSWERS)), dtype=np.float32)
    started = time.time()
    peak_memory = 0.0
    for position, prompt_text in enumerate(battery[prompt_column]):
        matrix[position] = score_all_words(model, prompt_text)
        peak_memory = max(peak_memory, LAST_STATE_PEAK_GIB)
        if position % 100 == 0:
            elapsed = time.time() - started
            print(
                f"  {label} {position:4d}/{len(battery)}"
                f"  {elapsed / max(position, 1):.2f}s/state"
                f"  peak {peak_memory:.2f} GiB",
                flush=True,
            )
        # Memory must stay flat. A steady climb means the allocator flush
        # stopped working and we abort before the host is affected.
        assert peak_memory < MEMORY_ABORT_GIB, (
            f"memory regression: {peak_memory:.1f} GiB at state {position}"
        )
    print(
        f"  {label} done in {(time.time() - started) / 60:.1f} min,"
        f" peak {peak_memory:.2f} GiB"
    )
    return matrix
```

### Kernel correctness and memory soak

The chunked kernel is verified against a plain single-sequence forward pass on a
handful of words, then soaked for 60 repetitions of one fixed worst-case prompt
with the memory trace asserted flat. Both gates run before any full battery is
scored. If the soak fails the notebook stops here rather than continuing into
the run that previously killed the host.


```python
@torch.no_grad()
def score_one_word_reference(model, prompt_text: str, word_index: int) -> float:
    prompt_ids = tokenizer(
        render_prompt(prompt_text), return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    tokens = WORD_TOKENS[word_index]
    sequence = torch.cat(
        [prompt_ids, torch.tensor([tokens], device=device)], dim=1
    )
    logprobs = torch.log_softmax(model(input_ids=sequence).logits[0].float(), dim=-1)
    offset = prompt_ids.shape[1] - 1
    result = float(sum(logprobs[offset + i, t] for i, t in enumerate(tokens)))
    del logprobs
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


if RUN_SCORING:
    checker = load_model(B_STRUCTURED_CHECKPOINT)
    probe = battery["structured_prompt"].iloc[0]
    fast = score_all_words(checker, probe)
    checks = []
    for word_index in [0, 7, 1000, 1500, 2314]:
        reference = score_one_word_reference(checker, probe, word_index)
        checks.append({
            "word": ANSWERS[word_index],
            "chunked": fast[word_index],
            "reference": reference,
            "abs_diff": abs(fast[word_index] - reference),
        })
    checks = pd.DataFrame(checks)
    display(checks)
    assert checks["abs_diff"].max() < 1e-3, "chunked kernel disagrees with reference"
    print("kernel verified")

    # Memory soak. Peaks are sampled inside the chunk loop, because the first
    # attempt at this run sampled only between states, reported a flat 2.71
    # GiB, and then hit 23.6 GiB mid-state and raised.
    #
    # The gate tests for a plateau, not for a small number. A high steady
    # working set is fine on a 512 GiB machine. What is not fine is a working
    # set that keeps climbing, because that has no ceiling.
    #
    # Leak detection and peak measurement are separate tests. Repeating one
    # fixed longest prompt holds every tensor shape constant, so any sustained
    # growth belongs to the allocator rather than to a changing workload.
    # Distinct long prompts are scored afterward only to measure worst-case
    # headroom.
    prompt_lengths = battery["structured_prompt"].map(
        lambda prompt: len(tokenizer(prompt).input_ids)
    )
    fixed_prompt = battery.loc[prompt_lengths.idxmax(), "structured_prompt"]
    fixed_prompt_tokens = int(prompt_lengths.max())

    SOAK_STATES = 60
    print(
        f"\nmemory soak: fixed {fixed_prompt_tokens}-token prompt, "
        f"{SOAK_STATES} repetitions"
    )
    peaks = []
    for _ in range(SOAK_STATES):
        score_all_words(checker, fixed_prompt)
        peaks.append(LAST_STATE_PEAK_GIB)

    longest = (
        battery.assign(_tokens=prompt_lengths)
        .nlargest(10, "_tokens")["structured_prompt"]
        .tolist()
    )
    worst_peaks = []
    for prompt in longest:
        score_all_words(checker, prompt)
        worst_peaks.append(LAST_STATE_PEAK_GIB)
    release_model(checker)

    third = SOAK_STATES // 3
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    soak_trace = pd.DataFrame({
        "position": range(SOAK_STATES),
        "prompt_tokens": fixed_prompt_tokens,
        "peak_gib": peaks,
    })
    soak_trace.to_csv(RESULTS_DIR / "soak-trace.csv", index=False)
    middle = sum(peaks[third:2 * third]) / third
    late = sum(peaks[-third:]) / third
    creep = late - middle
    late_range = max(peaks[-third:]) - min(peaks[-third:])
    print(f"  per-state peak: first {peaks[0]:.2f} GiB, max {max(peaks):.2f} GiB")
    print(
        f"  middle third mean {middle:.2f} GiB, final third mean {late:.2f} GiB,"
        f" creep {creep:+.2f} GiB, final range {late_range:.2f} GiB"
    )
    print(f"  10 longest prompts in battery: max {max(worst_peaks):.2f} GiB")
    print(f"  cap {MEMORY_CAP_GIB:.0f} GiB, headroom"
          f" {MEMORY_CAP_GIB / max(worst_peaks + peaks):.1f}x")
    assert creep < 0.5, (
        f"working set still climbing {creep:+.2f} GiB per {third} states after warmup."
        " This has no ceiling. Diagnose before running the full battery."
    )
    assert late_range < 0.5, (
        f"working set has not plateaued: final {third} peaks span"
        f" {late_range:.2f} GiB"
    )
    assert max(worst_peaks + peaks) < MEMORY_ABORT_GIB, (
        f"peak {max(worst_peaks + peaks):.1f} GiB exceeds abort threshold"
        f" {MEMORY_ABORT_GIB:.1f} GiB"
    )
    print("  working set plateaued and within headroom, safe to score the full battery")
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
      <th>word</th>
      <th>chunked</th>
      <th>reference</th>
      <th>abs_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ABACK</td>
      <td>-22.005051</td>
      <td>-22.005041</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABLED</td>
      <td>-20.635523</td>
      <td>-20.635511</td>
      <td>0.000011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HUMAN</td>
      <td>-13.124020</td>
      <td>-13.124040</td>
      <td>0.000020</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PURSE</td>
      <td>-23.597862</td>
      <td>-23.597805</td>
      <td>0.000057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ZONAL</td>
      <td>-21.232304</td>
      <td>-21.232281</td>
      <td>0.000023</td>
    </tr>
  </tbody>
</table>
</div>


    kernel verified
    
    memory soak: fixed 178-token prompt, 60 repetitions


      per-state peak: first 54.77 GiB, max 54.77 GiB
      middle third mean 54.77 GiB, final third mean 54.77 GiB, creep +0.00 GiB, final range 0.00 GiB
      10 longest prompts in battery: max 54.77 GiB
      cap 128 GiB, headroom 2.3x
      working set plateaued and within headroom, safe to score the full battery


## 18b.7 Score every model


```python
SCORING_MODELS = [
    ("base", None, "structured_prompt"),
    ("B-structured", B_STRUCTURED_CHECKPOINT, "structured_prompt"),
    ("G-structured", G_STRUCTURED_CHECKPOINT, "structured_prompt"),
]
if SCORE_B_RAW:
    SCORING_MODELS.append(("B-raw", B_RAW_CHECKPOINT, "raw_prompt"))

score_matrices = {}
if RUN_SCORING:
    battery[["state_key", "turn", "candidate_bucket", "candidate_count", "expected"]].to_csv(
        RESULTS_DIR / "battery-states.csv", index=False
    )
    pd.Series(ANSWERS, name="word").to_csv(RESULTS_DIR / "answer-order.csv", index=False)
    for label, path, prompt_column in SCORING_MODELS:
        print(f"scoring {label} ...", flush=True)
        model = load_model(path)
        score_matrices[label] = score_battery(model, prompt_column, label)
        release_model(model)
        np.save(RESULTS_DIR / f"scores-{label}.npy", score_matrices[label])
else:
    for label, _, _ in SCORING_MODELS:
        score_matrices[label] = np.load(RESULTS_DIR / f"scores-{label}.npy")
    print("loaded cached score matrices")

print({label: matrix.shape for label, matrix in score_matrices.items()})
```

    scoring base ...



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


      base    0/620  2.87s/state  peak 33.76 GiB


      base  100/620  3.33s/state  peak 54.75 GiB


      base  200/620  3.30s/state  peak 54.75 GiB


      base  300/620  3.33s/state  peak 54.75 GiB


      base  400/620  3.34s/state  peak 54.75 GiB


      base  500/620  3.35s/state  peak 54.75 GiB


      base  600/620  3.39s/state  peak 54.76 GiB


      base done in 35.0 min, peak 54.76 GiB


    scoring B-structured ...



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


      B-structured    0/620  2.86s/state  peak 33.77 GiB


      B-structured  100/620  3.38s/state  peak 54.76 GiB


      B-structured  200/620  3.36s/state  peak 54.76 GiB


      B-structured  300/620  3.39s/state  peak 54.76 GiB


      B-structured  400/620  3.40s/state  peak 54.76 GiB


      B-structured  500/620  3.42s/state  peak 54.76 GiB


      B-structured  600/620  3.45s/state  peak 54.77 GiB


      B-structured done in 35.7 min, peak 54.77 GiB


    scoring G-structured ...



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


      G-structured    0/620  2.91s/state  peak 33.77 GiB


      G-structured  100/620  3.38s/state  peak 54.76 GiB


      G-structured  200/620  3.37s/state  peak 54.76 GiB


      G-structured  300/620  3.40s/state  peak 54.76 GiB


      G-structured  400/620  3.41s/state  peak 54.76 GiB


      G-structured  500/620  3.43s/state  peak 54.76 GiB


      G-structured  600/620  3.46s/state  peak 54.77 GiB


      G-structured done in 35.8 min, peak 54.77 GiB


    {'base': (620, 2315), 'B-structured': (620, 2315), 'G-structured': (620, 2315)}


## 18b.8 Tier 1 and Tier 2

Tier 1 takes the top-scoring word over the whole answer list. Format validity and
lexicon membership are true by construction, so the metrics that survive are
history consistency, repetition, usability, and teacher match.

Tier 2 restricts the argmax to the consistent candidate set. Consistency is then
guaranteed, so teacher match is the only informative metric, and it measures
whether the model prefers the entropy-optimal legal word over the other legal
words.


```python
CANDIDATE_INDICES = [
    np.array([WORD_TO_INDEX[word] for word in candidates], dtype=np.int64)
    for candidates in battery["candidates"]
]
ANSWER_ARRAY = np.array(ANSWERS)


def tier_frame(label: str, matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for position, state in enumerate(battery.itertuples()):
        scores = matrix[position]
        candidate_ids = CANDIDATE_INDICES[position]
        previous = {turn.guess for turn in state.history}

        tier1_word = ANSWER_ARRAY[int(scores.argmax())]
        tier2_word = ANSWER_ARRAY[candidate_ids[int(scores[candidate_ids].argmax())]]
        tier1_consistent = is_consistent(tier1_word, state.history)

        # Descending rank of the teacher word, 1-based.
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)
        teacher_rank = int(ranks[WORD_TO_INDEX[state.expected]])
        best_candidate_rank = int(ranks[candidate_ids].min())

        # Share of total probability mass on the consistent candidate set,
        # against the uniform expectation of len(candidates) / 2315.
        shifted = scores - scores.max()
        weights = np.exp(shifted)
        candidate_mass = float(weights[candidate_ids].sum() / weights.sum())
        uniform_mass = len(candidate_ids) / len(ANSWERS)

        # Mean rank percentile of the candidate set. Exactly 0.5 for a
        # state-blind ranking, near 0 when constraints pull candidates to the
        # top. Unlike mass_lift this is not dominated by the single top word.
        candidate_rank_percentile = float(ranks[candidate_ids].mean() / len(scores))

        rows.append({
            "model": label,
            "state_key": state.state_key,
            "turn": state.turn,
            "candidate_bucket": state.candidate_bucket,
            "candidate_count": state.candidate_count,
            "expected": state.expected,
            "tier1_word": tier1_word,
            "tier1_history_consistent": bool(tier1_consistent),
            "tier1_repeated": bool(tier1_word in previous),
            "tier1_usable": bool(tier1_consistent and tier1_word not in previous),
            "tier1_teacher_match": bool(tier1_word == state.expected),
            "tier2_word": tier2_word,
            "tier2_teacher_match": bool(tier2_word == state.expected),
            "tier2_chance": 1.0 / len(candidate_ids),
            "teacher_rank": teacher_rank,
            "teacher_reciprocal_rank": 1.0 / teacher_rank,
            "best_candidate_rank": best_candidate_rank,
            "candidate_mass": candidate_mass,
            "uniform_mass": uniform_mass,
            "mass_lift": candidate_mass / uniform_mass,
            "candidate_rank_percentile": candidate_rank_percentile,
        })
    return pd.DataFrame(rows)


tier_results = pd.concat(
    [tier_frame(label, matrix) for label, matrix in score_matrices.items()],
    ignore_index=True,
)

tier0_rates = {
    "B-structured": b_tier0["usable"].mean(),
    "G-structured": g_tier0["usable"].mean(),
}
headline = tier_results.groupby("model", sort=False).agg(
    states=("state_key", "size"),
    tier1_consistency=("tier1_history_consistent", "mean"),
    tier1_usable=("tier1_usable", "mean"),
    tier1_teacher_match=("tier1_teacher_match", "mean"),
    tier2_teacher_match=("tier2_teacher_match", "mean"),
    tier2_chance=("tier2_chance", "mean"),
)
headline.insert(1, "tier0_usable", headline.index.map(tier0_rates))
display(headline)
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
      <th>states</th>
      <th>tier0_usable</th>
      <th>tier1_consistency</th>
      <th>tier1_usable</th>
      <th>tier1_teacher_match</th>
      <th>tier2_teacher_match</th>
      <th>tier2_chance</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>base</th>
      <td>620</td>
      <td>NaN</td>
      <td>0.001613</td>
      <td>0.001613</td>
      <td>0.000000</td>
      <td>0.522581</td>
      <td>0.521896</td>
    </tr>
    <tr>
      <th>B-structured</th>
      <td>620</td>
      <td>0.145161</td>
      <td>0.303226</td>
      <td>0.303226</td>
      <td>0.167742</td>
      <td>0.574194</td>
      <td>0.521896</td>
    </tr>
    <tr>
      <th>G-structured</th>
      <td>620</td>
      <td>0.159677</td>
      <td>0.300000</td>
      <td>0.300000</td>
      <td>0.161290</td>
      <td>0.545161</td>
      <td>0.521896</td>
    </tr>
  </tbody>
</table>
</div>


### Tier 1 versus Tier 0, the primary comparison

Same 620 states, same model, one interface change. This is the number the lab was
built to produce.


```python
def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    margin = z * ((rate * (1 - rate) / trials + z**2 / (4 * trials**2)) ** 0.5) / denominator
    return center - margin, center + margin


def exact_paired_p_value(left: pd.Series, right: pd.Series) -> float:
    left_only = int((left & ~right).sum())
    right_only = int((~left & right).sum())
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)
    ) / 2**discordant
    return min(1.0, 2 * tail)


def paired_metric(
    left: pd.Series, right: pd.Series, name: str, bootstrap_samples: int = 10_000
) -> dict:
    left_values = left.to_numpy(dtype=bool)
    right_values = right.to_numpy(dtype=bool)
    differences = right_values.astype(float) - left_values.astype(float)
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
    deltas = differences[sampled].mean(axis=1)
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    return {
        "comparison": name,
        "states": len(differences),
        "left_rate": left_values.mean(),
        "right_rate": right_values.mean(),
        "delta": differences.mean(),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "left_only": int((left_values & ~right_values).sum()),
        "right_only": int((~left_values & right_values).sum()),
        "both": int((left_values & right_values).sum()),
        "neither": int((~left_values & ~right_values).sum()),
        "exact_p_value": exact_paired_p_value(pd.Series(left_values), pd.Series(right_values)),
    }


tier_indexed = tier_results.set_index(["model", "state_key"])
comparisons = []
for label, tier0_frame in [("B-structured", b_tier0), ("G-structured", g_tier0)]:
    tier1 = tier_indexed.loc[label].loc[list(battery["state_key"])]
    for metric, tier0_column, tier1_column in [
        ("usable", "usable", "tier1_usable"),
        ("history_consistent", "history_consistent", "tier1_history_consistent"),
        ("teacher_match", "teacher_match", "tier1_teacher_match"),
    ]:
        comparisons.append(paired_metric(
            tier0_frame.set_index("state_key").loc[list(battery["state_key"])][tier0_column],
            tier1[tier1_column],
            f"{label}: tier0 -> tier1 {metric}",
        ))

tier0_vs_tier1 = pd.DataFrame(comparisons)
display(tier0_vs_tier1)
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
      <th>comparison</th>
      <th>states</th>
      <th>left_rate</th>
      <th>right_rate</th>
      <th>delta</th>
      <th>delta_ci_low</th>
      <th>delta_ci_high</th>
      <th>left_only</th>
      <th>right_only</th>
      <th>both</th>
      <th>neither</th>
      <th>exact_p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B-structured: tier0 -&gt; tier1 usable</td>
      <td>620</td>
      <td>0.145161</td>
      <td>0.303226</td>
      <td>0.158065</td>
      <td>0.125806</td>
      <td>0.190323</td>
      <td>10</td>
      <td>108</td>
      <td>80</td>
      <td>422</td>
      <td>6.450875e-22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B-structured: tier0 -&gt; tier1 history_consistent</td>
      <td>620</td>
      <td>0.145161</td>
      <td>0.303226</td>
      <td>0.158065</td>
      <td>0.125806</td>
      <td>0.190323</td>
      <td>10</td>
      <td>108</td>
      <td>80</td>
      <td>422</td>
      <td>6.450875e-22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B-structured: tier0 -&gt; tier1 teacher_match</td>
      <td>620</td>
      <td>0.074194</td>
      <td>0.167742</td>
      <td>0.093548</td>
      <td>0.069355</td>
      <td>0.119355</td>
      <td>5</td>
      <td>63</td>
      <td>41</td>
      <td>511</td>
      <td>7.651063e-14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>G-structured: tier0 -&gt; tier1 usable</td>
      <td>620</td>
      <td>0.159677</td>
      <td>0.300000</td>
      <td>0.140323</td>
      <td>0.108065</td>
      <td>0.172581</td>
      <td>14</td>
      <td>101</td>
      <td>85</td>
      <td>420</td>
      <td>1.982660e-17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>G-structured: tier0 -&gt; tier1 history_consistent</td>
      <td>620</td>
      <td>0.159677</td>
      <td>0.300000</td>
      <td>0.140323</td>
      <td>0.108065</td>
      <td>0.172581</td>
      <td>14</td>
      <td>101</td>
      <td>85</td>
      <td>420</td>
      <td>1.982660e-17</td>
    </tr>
    <tr>
      <th>5</th>
      <td>G-structured: tier0 -&gt; tier1 teacher_match</td>
      <td>620</td>
      <td>0.077419</td>
      <td>0.161290</td>
      <td>0.083871</td>
      <td>0.059677</td>
      <td>0.109677</td>
      <td>8</td>
      <td>60</td>
      <td>40</td>
      <td>512</td>
      <td>5.747761e-11</td>
    </tr>
  </tbody>
</table>
</div>


### Does the adapter beat the base model under the same constraint?

If `base` ranks as well as the adapters, the lexicon restriction did the work and
the fine-tuning added nothing that survives constrained decoding.


```python
base_vs_adapter = []
base_tier = tier_indexed.loc["base"].loc[list(battery["state_key"])]
for label in [name for name, _, _ in SCORING_MODELS if name != "base"]:
    adapter_tier = tier_indexed.loc[label].loc[list(battery["state_key"])]
    for metric in ["tier1_usable", "tier1_teacher_match", "tier2_teacher_match"]:
        base_vs_adapter.append(paired_metric(
            base_tier[metric], adapter_tier[metric], f"base -> {label}: {metric}"
        ))
display(pd.DataFrame(base_vs_adapter))
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
      <th>comparison</th>
      <th>states</th>
      <th>left_rate</th>
      <th>right_rate</th>
      <th>delta</th>
      <th>delta_ci_low</th>
      <th>delta_ci_high</th>
      <th>left_only</th>
      <th>right_only</th>
      <th>both</th>
      <th>neither</th>
      <th>exact_p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>base -&gt; B-structured: tier1_usable</td>
      <td>620</td>
      <td>0.001613</td>
      <td>0.303226</td>
      <td>0.301613</td>
      <td>0.266129</td>
      <td>0.338710</td>
      <td>0</td>
      <td>187</td>
      <td>1</td>
      <td>432</td>
      <td>1.019579e-56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>base -&gt; B-structured: tier1_teacher_match</td>
      <td>620</td>
      <td>0.000000</td>
      <td>0.167742</td>
      <td>0.167742</td>
      <td>0.138710</td>
      <td>0.198387</td>
      <td>0</td>
      <td>104</td>
      <td>0</td>
      <td>516</td>
      <td>9.860761e-32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>base -&gt; B-structured: tier2_teacher_match</td>
      <td>620</td>
      <td>0.522581</td>
      <td>0.574194</td>
      <td>0.051613</td>
      <td>0.014516</td>
      <td>0.088710</td>
      <td>51</td>
      <td>83</td>
      <td>273</td>
      <td>213</td>
      <td>7.178115e-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>base -&gt; G-structured: tier1_usable</td>
      <td>620</td>
      <td>0.001613</td>
      <td>0.300000</td>
      <td>0.298387</td>
      <td>0.262903</td>
      <td>0.333871</td>
      <td>0</td>
      <td>185</td>
      <td>1</td>
      <td>434</td>
      <td>4.078315e-56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>base -&gt; G-structured: tier1_teacher_match</td>
      <td>620</td>
      <td>0.000000</td>
      <td>0.161290</td>
      <td>0.161290</td>
      <td>0.133871</td>
      <td>0.190323</td>
      <td>0</td>
      <td>100</td>
      <td>0</td>
      <td>520</td>
      <td>1.577722e-30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>base -&gt; G-structured: tier2_teacher_match</td>
      <td>620</td>
      <td>0.522581</td>
      <td>0.545161</td>
      <td>0.022581</td>
      <td>-0.012903</td>
      <td>0.058065</td>
      <td>52</td>
      <td>66</td>
      <td>272</td>
      <td>230</td>
      <td>2.312657e-01</td>
    </tr>
  </tbody>
</table>
</div>


## 18b.9 Distributional readouts

Argmax discards almost everything the model expressed. These readouts use the
whole ranking.

`candidate_rank_percentile` is the mean rank of the consistent candidate set
divided by 2,315. A state-blind ranking sits at exactly 0.5. Values near 0 mean
the feedback is pulling legal words to the top of the whole list even when the
single argmax is wrong. This is the primary state-conditioning readout because it
is not dominated by one word.

`mass_lift` is the softmax mass on the candidate set divided by
`candidates / 2315`. It is reported alongside, but note it is close to
winner-take-all: real log-likelihoods span tens of nats, so the softmax
concentrates on a couple of words and the lift behaves almost like a restatement
of Tier 1. Read the percentile first.


```python
rank_summary = tier_results.groupby("model", sort=False).agg(
    median_teacher_rank=("teacher_rank", "median"),
    mean_reciprocal_rank=("teacher_reciprocal_rank", "mean"),
    median_best_candidate_rank=("best_candidate_rank", "median"),
    mean_candidate_rank_percentile=("candidate_rank_percentile", "mean"),
    median_mass_lift=("mass_lift", "median"),
    mean_candidate_mass=("candidate_mass", "mean"),
)
for k in [1, 5, 10, 50, 100]:
    rank_summary[f"teacher_top{k}"] = tier_results.groupby("model", sort=False).apply(
        lambda frame, k=k: (frame["teacher_rank"] <= k).mean(), include_groups=False
    )
display(rank_summary)

print("Tier 1 output diversity (state-blindness check)")
display(tier_results.groupby("model", sort=False).agg(
    distinct_tier1_words=("tier1_word", "nunique"),
    most_common_word=("tier1_word", lambda s: s.value_counts().idxmax()),
    most_common_share=("tier1_word", lambda s: s.value_counts().iloc[0] / len(s)),
))

print("By candidate bucket")
display(tier_results.groupby(["model", "candidate_bucket"], observed=True, sort=False).agg(
    states=("state_key", "size"),
    tier1_usable=("tier1_usable", "mean"),
    tier2_teacher_match=("tier2_teacher_match", "mean"),
    tier2_chance=("tier2_chance", "mean"),
    mean_candidate_rank_percentile=("candidate_rank_percentile", "mean"),
    median_best_candidate_rank=("best_candidate_rank", "median"),
).round(4))
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
      <th>median_teacher_rank</th>
      <th>mean_reciprocal_rank</th>
      <th>median_best_candidate_rank</th>
      <th>mean_candidate_rank_percentile</th>
      <th>median_mass_lift</th>
      <th>mean_candidate_mass</th>
      <th>teacher_top1</th>
      <th>teacher_top5</th>
      <th>teacher_top10</th>
      <th>teacher_top50</th>
      <th>teacher_top100</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>base</th>
      <td>861.0</td>
      <td>0.005614</td>
      <td>390.5</td>
      <td>0.432916</td>
      <td>0.003096</td>
      <td>0.001794</td>
      <td>0.000000</td>
      <td>0.003226</td>
      <td>0.004839</td>
      <td>0.048387</td>
      <td>0.091935</td>
    </tr>
    <tr>
      <th>B-structured</th>
      <td>9.0</td>
      <td>0.286921</td>
      <td>3.0</td>
      <td>0.028345</td>
      <td>60.979155</td>
      <td>0.187974</td>
      <td>0.167742</td>
      <td>0.414516</td>
      <td>0.533871</td>
      <td>0.791935</td>
      <td>0.866129</td>
    </tr>
    <tr>
      <th>G-structured</th>
      <td>10.0</td>
      <td>0.272507</td>
      <td>4.0</td>
      <td>0.031830</td>
      <td>57.651174</td>
      <td>0.186717</td>
      <td>0.161290</td>
      <td>0.375806</td>
      <td>0.501613</td>
      <td>0.785484</td>
      <td>0.875806</td>
    </tr>
  </tbody>
</table>
</div>


    Tier 1 output diversity (state-blindness check)



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
      <th>distinct_tier1_words</th>
      <th>most_common_word</th>
      <th>most_common_share</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>base</th>
      <td>12</td>
      <td>GREEN</td>
      <td>0.396774</td>
    </tr>
    <tr>
      <th>B-structured</th>
      <td>342</td>
      <td>CABLE</td>
      <td>0.019355</td>
    </tr>
    <tr>
      <th>G-structured</th>
      <td>348</td>
      <td>FLAKE</td>
      <td>0.017742</td>
    </tr>
  </tbody>
</table>
</div>


    By candidate bucket



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
      <th>states</th>
      <th>tier1_usable</th>
      <th>tier2_teacher_match</th>
      <th>tier2_chance</th>
      <th>mean_candidate_rank_percentile</th>
      <th>median_best_candidate_rank</th>
    </tr>
    <tr>
      <th>model</th>
      <th>candidate_bucket</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">base</th>
      <th>11-50</th>
      <td>51</td>
      <td>0.0000</td>
      <td>0.0784</td>
      <td>0.0616</td>
      <td>0.5165</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>272</td>
      <td>0.0000</td>
      <td>0.2537</td>
      <td>0.2439</td>
      <td>0.4542</td>
      <td>327.0</td>
    </tr>
    <tr>
      <th>1-2</th>
      <td>288</td>
      <td>0.0000</td>
      <td>0.8681</td>
      <td>0.8819</td>
      <td>0.3948</td>
      <td>626.0</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>7</td>
      <td>0.0000</td>
      <td>0.1429</td>
      <td>0.0134</td>
      <td>0.5436</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>201+</th>
      <td>2</td>
      <td>0.5000</td>
      <td>0.0000</td>
      <td>0.0014</td>
      <td>0.5086</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">B-structured</th>
      <th>11-50</th>
      <td>51</td>
      <td>0.3137</td>
      <td>0.0980</td>
      <td>0.0616</td>
      <td>0.0728</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>272</td>
      <td>0.3419</td>
      <td>0.3272</td>
      <td>0.2439</td>
      <td>0.0302</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1-2</th>
      <td>288</td>
      <td>0.2569</td>
      <td>0.9097</td>
      <td>0.8819</td>
      <td>0.0137</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>7</td>
      <td>0.4286</td>
      <td>0.0000</td>
      <td>0.0134</td>
      <td>0.1562</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>201+</th>
      <td>2</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0014</td>
      <td>0.2968</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">G-structured</th>
      <th>11-50</th>
      <td>51</td>
      <td>0.2941</td>
      <td>0.0784</td>
      <td>0.0616</td>
      <td>0.0858</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>272</td>
      <td>0.3456</td>
      <td>0.2647</td>
      <td>0.2439</td>
      <td>0.0335</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1-2</th>
      <td>288</td>
      <td>0.2535</td>
      <td>0.9097</td>
      <td>0.8819</td>
      <td>0.0151</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>7</td>
      <td>0.4286</td>
      <td>0.0000</td>
      <td>0.0134</td>
      <td>0.1792</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>201+</th>
      <td>2</td>
      <td>0.5000</td>
      <td>0.0000</td>
      <td>0.0014</td>
      <td>0.3290</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>


### Is Tier 2 above chance?

A model that has learned nothing about which legal word is best still scores
`1 / candidates` by picking arbitrarily. The comparison below is against that
per-state chance rate, summed into an expected count, not against zero.


```python
tier2_rows = []
for label in score_matrices:
    frame = tier_indexed.loc[label]
    observed = int(frame["tier2_teacher_match"].sum())
    expected = float(frame["tier2_chance"].sum())
    # Poisson-binomial mean/variance under independent per-state chance.
    variance = float((frame["tier2_chance"] * (1 - frame["tier2_chance"])).sum())
    low, high = wilson_interval(observed, len(frame))
    tier2_rows.append({
        "model": label,
        "states": len(frame),
        "tier2_correct": observed,
        "chance_expected": round(expected, 1),
        "lift_over_chance": observed / expected if expected else float("nan"),
        "z_vs_chance": (observed - expected) / math.sqrt(variance) if variance else float("nan"),
        "rate": observed / len(frame),
        "ci_low": low,
        "ci_high": high,
    })
display(pd.DataFrame(tier2_rows))
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
      <th>states</th>
      <th>tier2_correct</th>
      <th>chance_expected</th>
      <th>lift_over_chance</th>
      <th>z_vs_chance</th>
      <th>rate</th>
      <th>ci_low</th>
      <th>ci_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>base</td>
      <td>620</td>
      <td>324</td>
      <td>323.6</td>
      <td>1.001312</td>
      <td>0.051327</td>
      <td>0.522581</td>
      <td>0.483245</td>
      <td>0.561638</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B-structured</td>
      <td>620</td>
      <td>356</td>
      <td>323.6</td>
      <td>1.100207</td>
      <td>3.920974</td>
      <td>0.574194</td>
      <td>0.534932</td>
      <td>0.612541</td>
    </tr>
    <tr>
      <th>2</th>
      <td>G-structured</td>
      <td>620</td>
      <td>338</td>
      <td>323.6</td>
      <td>1.044578</td>
      <td>1.744297</td>
      <td>0.545161</td>
      <td>0.505806</td>
      <td>0.583960</td>
    </tr>
  </tbody>
</table>
</div>


### Sensitivity: length-normalized scoring

Answer words tokenize into 1, 2, or 3 tokens, so summed log-probability slightly
favours shorter tokenizations. Summed likelihood stays primary because it is what
a constrained decoder maximizes, but if the normalized ranking tells a different
story that belongs in the write-up.


```python
TOKEN_LENGTHS = np.array([len(t) for t in WORD_TOKENS], dtype=np.float32)

normalized_rows = []
for label, matrix in score_matrices.items():
    normalized = matrix / TOKEN_LENGTHS
    usable = []
    teacher = []
    for position, state in enumerate(battery.itertuples()):
        word = ANSWER_ARRAY[int(normalized[position].argmax())]
        previous = {turn.guess for turn in state.history}
        usable.append(bool(is_consistent(word, state.history) and word not in previous))
        teacher.append(bool(word == state.expected))
    summed = tier_indexed.loc[label]
    normalized_rows.append({
        "model": label,
        "summed_tier1_usable": summed["tier1_usable"].mean(),
        "normalized_tier1_usable": float(np.mean(usable)),
        "summed_tier1_teacher_match": summed["tier1_teacher_match"].mean(),
        "normalized_tier1_teacher_match": float(np.mean(teacher)),
    })
display(pd.DataFrame(normalized_rows).round(4))
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
      <th>summed_tier1_usable</th>
      <th>normalized_tier1_usable</th>
      <th>summed_tier1_teacher_match</th>
      <th>normalized_tier1_teacher_match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>base</td>
      <td>0.0016</td>
      <td>0.0016</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B-structured</td>
      <td>0.3032</td>
      <td>0.3113</td>
      <td>0.1677</td>
      <td>0.1710</td>
    </tr>
    <tr>
      <th>2</th>
      <td>G-structured</td>
      <td>0.3000</td>
      <td>0.3048</td>
      <td>0.1613</td>
      <td>0.1645</td>
    </tr>
  </tbody>
</table>
</div>


## 18b.10 Persist results


```python
tier_results.to_csv(RESULTS_DIR / "tier-results.csv", index=False)
tier0_vs_tier1.to_csv(RESULTS_DIR / "tier0-vs-tier1-paired.csv", index=False)
pd.DataFrame(base_vs_adapter).to_csv(RESULTS_DIR / "base-vs-adapter-paired.csv", index=False)
headline.to_csv(RESULTS_DIR / "headline-summary.csv")
rank_summary.to_csv(RESULTS_DIR / "rank-summary.csv")
pd.DataFrame(tier2_rows).to_csv(RESULTS_DIR / "tier2-vs-chance.csv", index=False)

(RESULTS_DIR / "lab18b-run.json").write_text(json.dumps({
    "model_id": MODEL_ID,
    "states": len(battery),
    "answers": len(ANSWERS),
    "chunk_size": CHUNK_SIZE,
    "scoring_rule": "summed log P(word tokens + EOS | prompt)",
    "models": [name for name, _, _ in SCORING_MODELS],
    "tier0_source": "results/lab18 free-generation battery",
}, indent=2))
print("written to", RESULTS_DIR)
print(sorted(p.name for p in RESULTS_DIR.iterdir()))
```

    written to ../results/lab18b
    ['answer-order.csv', 'base-vs-adapter-paired.csv', 'battery-states.csv', 'headline-summary.csv', 'lab18b-run.json', 'rank-summary.csv', 'scores-B-structured.npy', 'scores-G-structured.npy', 'scores-base.npy', 'soak-trace.csv', 'tier-results.csv', 'tier0-vs-tier1-paired.csv', 'tier2-vs-chance.csv']


## 18b.11 Read the result against the pre-registration

Fill this in from the tables above without moving the goalposts. The four
readings were fixed in 18b.1 before scoring:

1. Tier 1 close to Tier 0 -> grounding is not the bottleneck, the policy is weak,
   Part II's data labs are at their ceiling. Stop signal.
2. Tier 1 far above Tier 0 -> we have been scoring a decoding failure since Lab
   15. Labs 15 through 18 need rereading under the constrained interface.
3. Tier 1 far above Tier 0 but `base` matches the adapters -> the lexicon
   constraint did the work, not the training.
4. Tier 2 near `1 / candidates` -> the model learned consistency filtering and
   never learned the entropy policy.

Whatever the outcome, Tier 1 removes a failure mode by fiat and says nothing
about unaided competence. A positive result licenses changing the evaluation and
decoding interface, and it sets the interface Lab 18c replicates on. It does not
license a claim that the model can play Wordle.
