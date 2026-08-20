"""Generate notebooks/18b_constrained_ranking_probe.ipynb."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md("""
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
""")

md("""
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
""")

md("""
## 18b.2 Run controls and memory guard

Three labs in this project have hit unbounded memory growth on MPS, and the last
one took the host down. Layer 1 of the guard is the cell below: a hard
per-process cap on the MPS allocator. Past the cap PyTorch raises a normal
`RuntimeError` with a stack trace instead of consuming the machine.

Layer 2 lives outside this notebook. Run it through the watchdog, never bare:

```
scripts/memguard.py -- uv run jupyter nbconvert --to notebook --execute --inplace \\
    notebooks/18b_constrained_ranking_probe.ipynb
```
""")

code("""
RUN_SCORING = True
SCORE_B_RAW = False  # optional, needs the raw interface and doubles nothing else

MEMORY_CAP_GIB = 24.0

import torch

if torch.backends.mps.is_available():
    total_gib = torch.mps.recommended_max_memory() / 1024**3
    torch.mps.set_per_process_memory_fraction(MEMORY_CAP_GIB / total_gib)
    print(f"MPS cap: {MEMORY_CAP_GIB:.0f} GiB of {total_gib:.0f} GiB")
    print("past the cap PyTorch raises RuntimeError instead of taking the host down")

print("RUN_SCORING:", RUN_SCORING)
print("SCORE_B_RAW:", SCORE_B_RAW)
""")

code("""
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
""")

md("""
## 18b.3 Reuse the Lab 17 structured representation verbatim

These functions are copied unchanged from Labs 17 and 18. The prompts this lab
scores must be byte-identical to the prompts the adapters were trained on and
were evaluated on in Lab 18, so the representation is rebuilt through the same
`raw_policy_prompt` plus `transform_prompt` path rather than written afresh.
""")

code("""
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
        "Task: NEXT_GUESS\\nYou are playing Wordle.\\n"
        "Use the game history to choose the next guess.\\n"
        "Return exactly one uppercase five-letter word.\\n\\n"
        f"History:\\n{state_key}"
    )


def transform_prompt(prompt: str, state_key: str, candidate_count: int) -> str:
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
""")

md("""
## 18b.4 Rebuild the exact 620-state battery

The battery is read back from the Lab 18 result artifacts rather than
regenerated, which guarantees the same states in the same order. The Lab 18
free-generation outcomes come along as the Tier 0 baseline.
""")

code("""
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
""")

md("""
## 18b.5 Tokenize the answer lexicon

Training used `render_prompt(prompt) + response + eos_token` tokenized as one
string. Scoring a word standalone is only valid if the tokenizer produces the
same ids at the join, so that is asserted for all 2,315 words against a real
prompt rather than assumed.

Words are bucketed by token length. A word of `L` tokens needs only `L-1`
forwarded positions, because the first token's distribution comes from the
prompt's final position. That drops the work from 9,260 scored positions per
state to 5,246.
""")

code("""
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
""")

md("""
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
| plus `torch.mps.empty_cache()` | flat at 2.71 GiB |

So it was neither pending async frees nor shape variance. The MPS caching
allocator retains the roughly 7.6 GiB expanded KV-cache block and does not reuse
it, so the pool grows once per state until the machine dies. Only an explicit
allocator flush returns it. The flush costs nothing measurable: 1.8 s/state
either way.

Three disciplines are applied, and the memory column printed during scoring is
the regression test:

1. `torch.mps.empty_cache()` after every state. This is the one that matters.
2. `logits_to_keep=1` on the prefill. Lab 09's rule: never materialize
   full-vocabulary logits across all prompt positions. Without it the prompt
   forward alone builds a 129 by 151,936 tensor.
3. `gather` minus `logsumexp` instead of `log_softmax`, avoiding a second
   full-size copy of the chunk logits.

Peak full-vocabulary logits are therefore one chunk by at most three positions,
about 0.5 GiB, instead of the multi-terabyte tensor a naive implementation would
request.
""")

code("""
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


@torch.no_grad()
def score_all_words(model, prompt_text: str) -> np.ndarray:
    \"\"\"Summed log P(word + EOS | prompt) for all 2,315 answers.\"\"\"
    input_ids = tokenizer(
        render_prompt(prompt_text), return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    prefill = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    final_logits = prefill.logits[0, -1].float()
    first_token_logprobs = final_logits - final_logits.logsumexp(-1)
    cache = prefill.past_key_values
    cache.batch_repeat_interleave(CHUNK_SIZE)

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
                cache.crop(-step)
                del output, chunk_logits, token_logits
            scores[indices[start:start + CHUNK_SIZE]] = total.cpu()

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
        peak_memory = max(peak_memory, driver_memory_gib())
        if position % 100 == 0:
            elapsed = time.time() - started
            print(
                f"  {label} {position:4d}/{len(battery)}"
                f"  {elapsed / max(position, 1):.2f}s/state"
                f"  driver {driver_memory_gib():.2f} GiB",
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
""")

md("""
### Kernel correctness and memory soak

The chunked kernel is verified against a plain single-sequence forward pass on a
handful of words, then soaked for 25 states with the memory trace asserted flat.
Both gates run before any full battery is scored. If the soak fails the notebook
stops here rather than continuing into the run that previously killed the host.
""")

code("""
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

    # Memory soak. The full run is 1,860 states; if the allocator flush is
    # working, 25 states are enough to see a flat line, and a climb here stops
    # the notebook long before the host is at risk.
    print("\\nmemory soak over 25 states")
    baseline = driver_memory_gib()
    trace = []
    for position in range(25):
        score_all_words(checker, battery["structured_prompt"].iloc[position])
        trace.append(driver_memory_gib())
    release_model(checker)
    growth = trace[-1] - trace[0]
    print(f"  first {trace[0]:.2f} GiB, last {trace[-1]:.2f} GiB, growth {growth:+.2f} GiB")
    assert growth < 1.0, f"memory grows {growth:.2f} GiB over 25 states, do not run the full battery"
    print("  memory flat, safe to score the full battery")
""")

md("## 18b.7 Score every model")

code("""
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
""")

md("""
## 18b.8 Tier 1 and Tier 2

Tier 1 takes the top-scoring word over the whole answer list. Format validity and
lexicon membership are true by construction, so the metrics that survive are
history consistency, repetition, usability, and teacher match.

Tier 2 restricts the argmax to the consistent candidate set. Consistency is then
guaranteed, so teacher match is the only informative metric, and it measures
whether the model prefers the entropy-optimal legal word over the other legal
words.
""")

code("""
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
""")

md("""
### Tier 1 versus Tier 0, the primary comparison

Same 620 states, same model, one interface change. This is the number the lab was
built to produce.
""")

code("""
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
""")

md("""
### Does the adapter beat the base model under the same constraint?

If `base` ranks as well as the adapters, the lexicon restriction did the work and
the fine-tuning added nothing that survives constrained decoding.
""")

code("""
base_vs_adapter = []
base_tier = tier_indexed.loc["base"].loc[list(battery["state_key"])]
for label in [name for name, _, _ in SCORING_MODELS if name != "base"]:
    adapter_tier = tier_indexed.loc[label].loc[list(battery["state_key"])]
    for metric in ["tier1_usable", "tier1_teacher_match", "tier2_teacher_match"]:
        base_vs_adapter.append(paired_metric(
            base_tier[metric], adapter_tier[metric], f"base -> {label}: {metric}"
        ))
display(pd.DataFrame(base_vs_adapter))
""")

md("""
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
""")

code("""
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
""")

md("""
### Is Tier 2 above chance?

A model that has learned nothing about which legal word is best still scores
`1 / candidates` by picking arbitrarily. The comparison below is against that
per-state chance rate, summed into an expected count, not against zero.
""")

code("""
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
""")

md("""
### Sensitivity: length-normalized scoring

Answer words tokenize into 1, 2, or 3 tokens, so summed log-probability slightly
favours shorter tokenizations. Summed likelihood stays primary because it is what
a constrained decoder maximizes, but if the normalized ranking tells a different
story that belongs in the write-up.
""")

code("""
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
""")

md("## 18b.10 Persist results")

code("""
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
""")

md("""
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
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = Path("notebooks/18b_constrained_ranking_probe.ipynb")
path.write_text(json.dumps(notebook, indent=1) + "\n")
print("wrote", path, "with", len(cells), "cells")
