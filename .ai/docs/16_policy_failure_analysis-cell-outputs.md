# Lab 16 - Is the model ignoring state or misreading it?

**Goal:** explain why Dataset B changed the output distribution without producing a usable broad-candidate policy.

Lab 15 is frozen. This notebook does not retrain either adapter or redesign Dataset B. It analyzes the saved Lab 15 calls, repairs the repeat diagnostic, classifies constraint failures, and tests whether valid changes to Wordle state change the generated action.

The central distinction is:

- **state insensitivity:** the action does not respond to a changed state;
- **state misinterpretation:** the action changes but violates the new state's constraints.

## 16.1 Frozen inputs and run control

The stored Lab 15 CSV files support the failure census without loading a model. The paired perturbation experiment is gated because it performs hundreds of generations across the two adapters.


```python
RUN_PERTURBATION_EVALUATION = True
EVALUATE_BASE_MODEL = True
PAIR_LIMIT_PER_STRATUM = 10

print("RUN_PERTURBATION_EVALUATION:", RUN_PERTURBATION_EVALUATION)
print("EVALUATE_BASE_MODEL:", EVALUATE_BASE_MODEL)
print("PAIR_LIMIT_PER_STRATUM:", PAIR_LIMIT_PER_STRATUM)
```

    RUN_PERTURBATION_EVALUATION: True
    EVALUATE_BASE_MODEL: True
    PAIR_LIMIT_PER_STRATUM: 10



```python
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import gc
import json
import random

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_wordle.benchmark import DEFAULT_EVAL_ANSWERS, generate_raw_guess, parse_guess
from tiny_wordle.expert import EntropyExpert, decode_feedback
from tiny_wordle.game import Turn, is_consistent, score_string
from tiny_wordle.hardware import preferred_device

MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 42
DATA_DIR = Path("../data")
GENERATED_DIR = DATA_DIR / "generated"
LAB15_RESULTS = Path("../results/lab15")
LAB16_RESULTS = Path("../results/lab16")
CHECKPOINTS = {
    "A": Path("../checkpoints/qwen3-0.6b-wordle-lora-dataset-a"),
    "B": Path("../checkpoints/qwen3-0.6b-wordle-lora-dataset-b"),
}

device = preferred_device()
torch.set_float32_matmul_precision("high")
print("device:", device)
```

    device: mps


## 16.2 Load the Lab 15 evidence

The analysis uses raw calls rather than reconstructing behavior from summary percentages.


```python
required_results = {
    "policy": LAB15_RESULTS / "policy-results.csv",
    "auxiliary": LAB15_RESULTS / "auxiliary-results.csv",
    "gameplay_calls": LAB15_RESULTS / "gameplay-calls.csv",
    "gameplay_games": LAB15_RESULTS / "gameplay-games.csv",
    "summary": LAB15_RESULTS / "summary.json",
}
missing = [path for path in required_results.values() if not path.exists()]
assert not missing, f"Run Lab 15 first; missing {missing}"

policy = pd.read_csv(required_results["policy"])
auxiliary = pd.read_csv(required_results["auxiliary"])
gameplay_calls = pd.read_csv(required_results["gameplay_calls"])
gameplay_games = pd.read_csv(required_results["gameplay_games"])
lab15_summary = json.loads(required_results["summary"].read_text())

assert set(policy["model"]) == {"base", "A", "B"}
assert set(policy["interface"]) == {"training", "deployment"}
assert len(policy) == 47 * 2 * 3
assert len(gameplay_games) == 19 * 3

display(pd.DataFrame(lab15_summary["primary"]))
display(pd.DataFrame(lab15_summary["solve"]))
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
      <th>calls</th>
      <th>usable_calls</th>
      <th>usable_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>47</td>
      <td>0</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>47</td>
      <td>3</td>
      <td>0.06383</td>
    </tr>
    <tr>
      <th>2</th>
      <td>base</td>
      <td>47</td>
      <td>0</td>
      <td>0.00000</td>
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
      <th>games</th>
      <th>solved</th>
      <th>solve_rate</th>
      <th>mean_turns_on_wins</th>
      <th>teacher_ceiling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.947368</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.947368</td>
    </tr>
    <tr>
      <th>2</th>
      <td>base</td>
      <td>19</td>
      <td>1</td>
      <td>0.052632</td>
      <td>2.0</td>
      <td>0.947368</td>
    </tr>
  </tbody>
</table>
</div>


## 16.3 Reclassify every fixed-state failure

Lab 15's `repeated` field requires answer-lexicon membership. Here, `output_repeated` asks the simpler diagnostic question: did the model emit a word already present in the supplied history? Constraint labels compare the feedback that the generated word would imply with the feedback in the state.


```python
def parse_state_key(state_key: str) -> list[Turn]:
    history = []
    for line in state_key.splitlines():
        guess_text, feedback_text = line.split(" -> ")
        history.append(Turn(
            guess=guess_text.replace(" ", ""),
            feedback=feedback_text.replace(" ", ""),
        ))
    return history

def feedback_violation_types(candidate: str, history: list[Turn]) -> str:
    violations = set()
    for turn in history:
        implied = score_string(candidate, turn.guess)
        for expected, actual in zip(turn.feedback, implied):
            if expected == actual:
                continue
            if expected == "G":
                violations.add("green_not_preserved")
            elif expected == "Y" and actual == "G":
                violations.add("yellow_reused_position")
            elif expected == "Y":
                violations.add("yellow_count_missing")
            elif expected == "B":
                violations.add("gray_or_excess_letter")
    return ";".join(sorted(violations))

fixed_failures = policy.copy()
fixed_failures["history"] = fixed_failures["state_key"].map(parse_state_key)
fixed_failures["history_depth"] = fixed_failures["history"].map(len)
fixed_failures["output_repeated"] = [
    bool(row.format_valid) and row.actual in {turn.guess for turn in row.history}
    for row in fixed_failures.itertuples()
]
fixed_failures["violation_types"] = [
    feedback_violation_types(row.actual, row.history)
    if bool(row.format_valid) else ""
    for row in fixed_failures.itertuples()
]

def fixed_failure_mode(row) -> str:
    if bool(row.usable):
        return "usable"
    if not bool(row.format_valid):
        return "invalid_format"
    if bool(row.output_repeated):
        return "repeated_history_guess"
    if not bool(row.in_answer_lexicon):
        return "outside_answer_lexicon"
    return "constraint_violation"

fixed_failures["failure_mode"] = [
    fixed_failure_mode(row) for row in fixed_failures.itertuples()
]

display(fixed_failures.groupby(
    ["model", "interface", "failure_mode"]
).size().rename("calls").to_frame())
display(fixed_failures.loc[
    fixed_failures["failure_mode"] == "constraint_violation"
].groupby(["model", "interface", "violation_types"]).size()
 .rename("calls").to_frame())
display(fixed_failures.groupby(["model", "interface"]).agg(
    output_repeat_rate=("output_repeated", "mean"),
    any_constraint_violation=(
        "violation_types", lambda values: values.ne("").mean()
    ),
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
      <th></th>
      <th></th>
      <th>calls</th>
    </tr>
    <tr>
      <th>model</th>
      <th>interface</th>
      <th>failure_mode</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">A</th>
      <th rowspan="4" valign="top">deployment</th>
      <th>constraint_violation</th>
      <td>25</td>
    </tr>
    <tr>
      <th>invalid_format</th>
      <td>8</td>
    </tr>
    <tr>
      <th>outside_answer_lexicon</th>
      <td>4</td>
    </tr>
    <tr>
      <th>repeated_history_guess</th>
      <td>10</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">training</th>
      <th>constraint_violation</th>
      <td>30</td>
    </tr>
    <tr>
      <th>invalid_format</th>
      <td>5</td>
    </tr>
    <tr>
      <th>outside_answer_lexicon</th>
      <td>7</td>
    </tr>
    <tr>
      <th>repeated_history_guess</th>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">B</th>
      <th rowspan="5" valign="top">deployment</th>
      <th>constraint_violation</th>
      <td>20</td>
    </tr>
    <tr>
      <th>invalid_format</th>
      <td>2</td>
    </tr>
    <tr>
      <th>outside_answer_lexicon</th>
      <td>21</td>
    </tr>
    <tr>
      <th>repeated_history_guess</th>
      <td>3</td>
    </tr>
    <tr>
      <th>usable</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">training</th>
      <th>constraint_violation</th>
      <td>33</td>
    </tr>
    <tr>
      <th>invalid_format</th>
      <td>2</td>
    </tr>
    <tr>
      <th>outside_answer_lexicon</th>
      <td>8</td>
    </tr>
    <tr>
      <th>repeated_history_guess</th>
      <td>1</td>
    </tr>
    <tr>
      <th>usable</th>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">base</th>
      <th rowspan="4" valign="top">deployment</th>
      <th>constraint_violation</th>
      <td>25</td>
    </tr>
    <tr>
      <th>outside_answer_lexicon</th>
      <td>20</td>
    </tr>
    <tr>
      <th>repeated_history_guess</th>
      <td>1</td>
    </tr>
    <tr>
      <th>usable</th>
      <td>1</td>
    </tr>
    <tr>
      <th>training</th>
      <th>invalid_format</th>
      <td>47</td>
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
      <th></th>
      <th></th>
      <th>calls</th>
    </tr>
    <tr>
      <th>model</th>
      <th>interface</th>
      <th>violation_types</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="16" valign="top">A</th>
      <th rowspan="7" valign="top">deployment</th>
      <th>gray_or_excess_letter</th>
      <td>4</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved</th>
      <td>11</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_count_missing</th>
      <td>3</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_count_missing;yellow_reused_position</th>
      <td>2</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_count_missing</th>
      <td>3</td>
    </tr>
    <tr>
      <th>green_not_preserved;yellow_count_missing;yellow_reused_position</th>
      <td>1</td>
    </tr>
    <tr>
      <th>yellow_count_missing;yellow_reused_position</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">training</th>
      <th>gray_or_excess_letter</th>
      <td>4</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved</th>
      <td>9</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_count_missing</th>
      <td>1</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_count_missing;yellow_reused_position</th>
      <td>1</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_reused_position</th>
      <td>6</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_count_missing</th>
      <td>2</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_count_missing;yellow_reused_position</th>
      <td>2</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_reused_position</th>
      <td>2</td>
    </tr>
    <tr>
      <th>green_not_preserved</th>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="17" valign="top">B</th>
      <th rowspan="7" valign="top">deployment</th>
      <th>gray_or_excess_letter</th>
      <td>5</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved</th>
      <td>6</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_count_missing</th>
      <td>3</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_count_missing</th>
      <td>3</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_reused_position</th>
      <td>1</td>
    </tr>
    <tr>
      <th>green_not_preserved</th>
      <td>1</td>
    </tr>
    <tr>
      <th>yellow_count_missing</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">training</th>
      <th>gray_or_excess_letter</th>
      <td>3</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved</th>
      <td>9</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_count_missing</th>
      <td>2</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_count_missing;yellow_reused_position</th>
      <td>1</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_count_missing</th>
      <td>5</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_reused_position</th>
      <td>3</td>
    </tr>
    <tr>
      <th>green_not_preserved</th>
      <td>6</td>
    </tr>
    <tr>
      <th>green_not_preserved;yellow_count_missing</th>
      <td>2</td>
    </tr>
    <tr>
      <th>yellow_count_missing</th>
      <td>1</td>
    </tr>
    <tr>
      <th>yellow_reused_position</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">base</th>
      <th rowspan="7" valign="top">deployment</th>
      <th>gray_or_excess_letter</th>
      <td>4</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved</th>
      <td>9</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_count_missing</th>
      <td>3</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;green_not_preserved;yellow_count_missing;yellow_reused_position</th>
      <td>2</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_count_missing</th>
      <td>5</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_count_missing;yellow_reused_position</th>
      <td>1</td>
    </tr>
    <tr>
      <th>gray_or_excess_letter;yellow_reused_position</th>
      <td>1</td>
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
      <th></th>
      <th>output_repeat_rate</th>
      <th>any_constraint_violation</th>
    </tr>
    <tr>
      <th>model</th>
      <th>interface</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A</th>
      <th>deployment</th>
      <td>0.212766</td>
      <td>0.829787</td>
    </tr>
    <tr>
      <th>training</th>
      <td>0.106383</td>
      <td>0.893617</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>deployment</th>
      <td>0.063830</td>
      <td>0.914894</td>
    </tr>
    <tr>
      <th>training</th>
      <td>0.021277</td>
      <td>0.893617</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">base</th>
      <th>deployment</th>
      <td>0.021277</td>
      <td>0.957447</td>
    </tr>
    <tr>
      <th>training</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


## 16.4 Repair the gameplay repeat diagnostic

A parsed output now enters the diagnostic `seen` set even when it is outside the answer lexicon. This does not rewrite Lab 15 gameplay. It measures the collapse that the original repeat field missed.


```python
gameplay_failures = gameplay_calls.sort_values(
    ["model", "answer", "turn"]
).copy()
output_repeated = []
for (_, _), group in gameplay_failures.groupby(["model", "answer"], sort=False):
    seen_outputs = set()
    for row in group.itertuples():
        parsed = row.guess if isinstance(row.guess, str) else None
        output_repeated.append(parsed is not None and parsed in seen_outputs)
        if parsed is not None:
            seen_outputs.add(parsed)
gameplay_failures["output_repeated"] = output_repeated

repeat_comparison = gameplay_failures.groupby("model").agg(
    calls=("turn", "size"),
    lab15_repeat_rate=("repeated", "mean"),
    parsed_output_repeat_rate=("output_repeated", "mean"),
    answer_lexicon_rate=("in_answer_lexicon", "mean"),
    history_consistency_rate=("history_consistent", "mean"),
)
display(repeat_comparison)

attractors = (
    gameplay_failures.dropna(subset=["guess"])
    .groupby(["model", "guess"]).size()
    .rename("calls").reset_index()
    .sort_values(["model", "calls"], ascending=[True, False])
)
display(attractors.groupby("model").head(10))
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
      <th>calls</th>
      <th>lab15_repeat_rate</th>
      <th>parsed_output_repeat_rate</th>
      <th>answer_lexicon_rate</th>
      <th>history_consistency_rate</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>95</td>
      <td>0.705263</td>
      <td>0.705263</td>
      <td>0.905263</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B</th>
      <td>95</td>
      <td>0.115789</td>
      <td>0.715789</td>
      <td>0.200000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>base</th>
      <td>91</td>
      <td>0.000000</td>
      <td>0.615385</td>
      <td>0.186813</td>
      <td>0.010989</td>
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
      <th>guess</th>
      <th>calls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>BRAIN</td>
      <td>76</td>
    </tr>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>BAYOU</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>BRISE</td>
      <td>35</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>PETEL</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>B</td>
      <td>RASHY</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>BETEL</td>
      <td>19</td>
    </tr>
    <tr>
      <th>6</th>
      <td>base</td>
      <td>CRAKE</td>
      <td>74</td>
    </tr>
    <tr>
      <th>7</th>
      <td>base</td>
      <td>CRANE</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>


## 16.5 Locate the failure by state difficulty

The same taxonomy is crossed with candidate count and history depth. Aggregate validity can hide the broad-state failure that Dataset B was built to address.


```python
adapter_failures = fixed_failures.loc[fixed_failures["model"].isin(["A", "B"])]
display(adapter_failures.groupby(
    ["model", "interface", "difficulty"]
).agg(
    states=("state_key", "size"),
    format_valid_rate=("format_valid", "mean"),
    answer_lexicon_rate=("in_answer_lexicon", "mean"),
    output_repeat_rate=("output_repeated", "mean"),
    consistency_rate=("history_consistent", "mean"),
    usable_rate=("usable", "mean"),
))

display(adapter_failures.groupby(
    ["model", "interface", "history_depth"]
).agg(
    states=("state_key", "size"),
    consistency_rate=("history_consistent", "mean"),
    usable_rate=("usable", "mean"),
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
      <th></th>
      <th></th>
      <th>states</th>
      <th>format_valid_rate</th>
      <th>answer_lexicon_rate</th>
      <th>output_repeat_rate</th>
      <th>consistency_rate</th>
      <th>usable_rate</th>
    </tr>
    <tr>
      <th>model</th>
      <th>interface</th>
      <th>difficulty</th>
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
      <th rowspan="8" valign="top">A</th>
      <th rowspan="4" valign="top">deployment</th>
      <th>1-2</th>
      <td>16</td>
      <td>0.750000</td>
      <td>0.687500</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11-50</th>
      <td>8</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>17</td>
      <td>0.764706</td>
      <td>0.588235</td>
      <td>0.352941</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>6</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">training</th>
      <th>1-2</th>
      <td>16</td>
      <td>0.750000</td>
      <td>0.625000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11-50</th>
      <td>8</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>17</td>
      <td>0.941176</td>
      <td>0.647059</td>
      <td>0.176471</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>6</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">B</th>
      <th rowspan="4" valign="top">deployment</th>
      <th>1-2</th>
      <td>16</td>
      <td>0.937500</td>
      <td>0.625000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11-50</th>
      <td>8</td>
      <td>1.000000</td>
      <td>0.375000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>17</td>
      <td>0.941176</td>
      <td>0.470588</td>
      <td>0.058824</td>
      <td>0.058824</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>6</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">training</th>
      <th>1-2</th>
      <td>16</td>
      <td>0.875000</td>
      <td>0.625000</td>
      <td>0.000000</td>
      <td>0.187500</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>11-50</th>
      <td>8</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>17</td>
      <td>1.000000</td>
      <td>0.764706</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>6</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <th></th>
      <th></th>
      <th>states</th>
      <th>consistency_rate</th>
      <th>usable_rate</th>
    </tr>
    <tr>
      <th>model</th>
      <th>interface</th>
      <th>history_depth</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">A</th>
      <th rowspan="5" valign="top">deployment</th>
      <th>1</th>
      <td>17</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">training</th>
      <th>1</th>
      <td>17</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">B</th>
      <th rowspan="5" valign="top">deployment</th>
      <th>1</th>
      <td>17</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>0.1</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">training</th>
      <th>1</th>
      <td>17</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>0.2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


## 16.6 Check the recognition-selection split

Lab 15 suggested that Dataset B preserved constraint recognition better than candidate selection.


```python
display(auxiliary.groupby(["model", "task"]).agg(
    examples=("correct", "size"),
    accuracy=("correct", "mean"),
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
      <th></th>
      <th>examples</th>
      <th>accuracy</th>
    </tr>
    <tr>
      <th>model</th>
      <th>task</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A</th>
      <th>CHOOSE_VALID</th>
      <td>49</td>
      <td>0.836735</td>
    </tr>
    <tr>
      <th>VALID_CANDIDATE</th>
      <td>98</td>
      <td>0.744898</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>CHOOSE_VALID</th>
      <td>49</td>
      <td>0.653061</td>
    </tr>
    <tr>
      <th>VALID_CANDIDATE</th>
      <td>98</td>
      <td>0.765306</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">base</th>
      <th>CHOOSE_VALID</th>
      <td>49</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>VALID_CANDIDATE</th>
      <td>98</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


## 16.7 Build paired, reachable state perturbations

Each pair shares a reachable parent state and the same branch guess. The branch guess may be the fixed opening, the teacher action, or a controlled off-policy candidate. The two child histories differ only in the feedback produced by that guess. Both branches contain at least one possible answer, differ in one feedback position, and remain outside both training prompt sets.

This avoids arbitrary feedback edits that could describe impossible games.


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

def format_training_history(history: list[Turn]) -> str:
    return "\n".join(
        f"{' '.join(turn.guess)} -> {' '.join(turn.feedback)}"
        for turn in history
    )

def next_guess_prompt(history: list[Turn]) -> str:
    return (
        "Task: NEXT_GUESS\n"
        "You are playing Wordle.\n"
        "Use the game history to choose the next guess.\n"
        "Return exactly one uppercase five-letter word.\n\n"
        "History:\n"
        f"{format_training_history(history)}"
    )

def difficulty(candidate_count: int) -> str:
    if candidate_count <= 2:
        return "1-2"
    if candidate_count <= 10:
        return "3-10"
    if candidate_count <= 50:
        return "11-50"
    if candidate_count <= 200:
        return "51-200"
    return "201+"

train_prompts = set()
for path in [
    GENERATED_DIR / "wordle-sft-train.jsonl",
    GENERATED_DIR / "wordle-part2-policy-train.jsonl",
]:
    with path.open() as handle:
        train_prompts.update(json.loads(line)["prompt"] for line in handle)

parents = {
    "": {
        "history": [],
        "candidate_indices": expert.all_indices.copy(),
        "turn": 1,
    }
}
for answer in DEFAULT_EVAL_ANSWERS:
    opening = Turn("RAISE", score_string(answer, "RAISE"))
    history = [opening]
    candidates = expert.update(
        expert.all_indices, WORD_TO_INDEX["RAISE"], opening.feedback
    )
    for turn_number in range(2, 6):
        parent_key = format_training_history(history)
        parents.setdefault(parent_key, {
            "history": list(history),
            "candidate_indices": candidates.copy(),
            "turn": turn_number,
        })
        guess_index = expert.choose(candidates)
        feedback = score_string(answer, ANSWERS[guess_index])
        if feedback == "GGGGG":
            break
        history.append(Turn(ANSWERS[guess_index], feedback))
        candidates = expert.update(candidates, guess_index, feedback)

pair_candidates = []
for parent_key, parent in parents.items():
    candidates = parent["candidate_indices"]
    if len(candidates) < 2:
        continue
    teacher_guess_index = expert.choose(candidates)
    sample_positions = np.linspace(
        0, len(candidates) - 1, num=min(12, len(candidates)), dtype=int
    )
    branch_guess_indices = sorted({
        teacher_guess_index,
        *( [WORD_TO_INDEX["RAISE"]] if not parent["history"] else [] ),
        *(int(candidates[position]) for position in sample_positions),
    })
    for guess_index in branch_guess_indices:
        guess = ANSWERS[guess_index]
        branches = defaultdict(list)
        for answer_index, pattern_id in zip(
            candidates, PATTERNS[guess_index, candidates]
        ):
            feedback = decode_feedback(int(pattern_id))
            if feedback != "GGGGG":
                branches[feedback].append(int(answer_index))
        for feedback_a, feedback_b in combinations(sorted(branches), 2):
            if sum(a != b for a, b in zip(feedback_a, feedback_b)) != 1:
                continue
            history_a = parent["history"] + [Turn(guess, feedback_a)]
            history_b = parent["history"] + [Turn(guess, feedback_b)]
            prompt_a = next_guess_prompt(history_a)
            prompt_b = next_guess_prompt(history_b)
            if prompt_a in train_prompts or prompt_b in train_prompts:
                continue
            changed_position = next(
                index for index, (a, b) in enumerate(zip(feedback_a, feedback_b))
                if a != b
            )
            mark_order = {"B": 0, "Y": 1, "G": 2}
            feedback_change_type = "/".join(sorted(
                [feedback_a[changed_position], feedback_b[changed_position]],
                key=mark_order.get,
            ))
            child_count_a = len(branches[feedback_a])
            child_count_b = len(branches[feedback_b])
            if child_count_a >= 11 and child_count_b >= 11:
                pair_scope = "broad"
            elif child_count_a <= 10 and child_count_b <= 10:
                pair_scope = "narrow"
            else:
                pair_scope = "mixed"
            if not parent["history"] and guess == "RAISE":
                branch_source = "fixed_opening"
            elif guess_index == teacher_guess_index:
                branch_source = "teacher"
            else:
                branch_source = "controlled_off_policy"
            pair_candidates.append({
                "parent_key": parent_key,
                "parent_turn": parent["turn"],
                "parent_candidates": len(candidates),
                "parent_difficulty": difficulty(len(candidates)),
                "branch_guess": guess,
                "branch_source": branch_source,
                "feedback_a": feedback_a,
                "feedback_b": feedback_b,
                "feedback_change_position": changed_position + 1,
                "feedback_change_type": feedback_change_type,
                "history_a": history_a,
                "history_b": history_b,
                "candidates_a": child_count_a,
                "candidates_b": child_count_b,
                "difficulty_a": difficulty(child_count_a),
                "difficulty_b": difficulty(child_count_b),
                "pair_scope": pair_scope,
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
            })

pair_frame = pd.DataFrame(pair_candidates)
assert not pair_frame.empty
reachable_pair_candidates = len(pair_frame)
rng = np.random.default_rng(SEED)
pair_frame = pair_frame.iloc[rng.permutation(len(pair_frame))].copy()
pair_frame = pair_frame.groupby(
    ["parent_key", "branch_guess"], sort=False
).head(2)
group_capped_candidates = len(pair_frame)
perturbation_pairs = pair_frame.groupby(
    ["pair_scope", "branch_source"], sort=True
).head(PAIR_LIMIT_PER_STRATUM).reset_index(drop=True)
for row_index in perturbation_pairs.index:
    if not bool(rng.integers(0, 2)):
        continue
    for column in [
        "feedback", "history", "candidates", "difficulty", "prompt"
    ]:
        column_a, column_b = f"{column}_a", f"{column}_b"
        value_a = perturbation_pairs.at[row_index, column_a]
        perturbation_pairs.at[row_index, column_a] = perturbation_pairs.at[
            row_index, column_b
        ]
        perturbation_pairs.at[row_index, column_b] = value_a
perturbation_pairs["pair_id"] = [f"pair-{i:03d}" for i in range(len(perturbation_pairs))]

print("reachable pair candidates:", reachable_pair_candidates)
print("candidates after per-parent-guess cap:", group_capped_candidates)
print("selected perturbation pairs:", len(perturbation_pairs))
assert {"broad", "narrow"} <= set(perturbation_pairs["pair_scope"])
display(perturbation_pairs.groupby(["pair_scope", "branch_source"]).agg(
    pairs=("pair_id", "size"),
    mean_parent_candidates=("parent_candidates", "mean"),
    mean_branch_a_candidates=("candidates_a", "mean"),
    mean_branch_b_candidates=("candidates_b", "mean"),
))
```

    reachable pair candidates: 7796
    candidates after per-parent-guess cap: 442
    selected perturbation pairs: 34



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
      <th>mean_parent_candidates</th>
      <th>mean_branch_a_candidates</th>
      <th>mean_branch_b_candidates</th>
    </tr>
    <tr>
      <th>pair_scope</th>
      <th>branch_source</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">broad</th>
      <th>controlled_off_policy</th>
      <td>10</td>
      <td>1877.9</td>
      <td>80.6</td>
      <td>62.5</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>2315.0</td>
      <td>168.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">mixed</th>
      <th>controlled_off_policy</th>
      <td>10</td>
      <td>977.3</td>
      <td>11.4</td>
      <td>12.7</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>2315.0</td>
      <td>26.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">narrow</th>
      <th>controlled_off_policy</th>
      <td>10</td>
      <td>525.1</td>
      <td>2.5</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>teacher</th>
      <td>2</td>
      <td>46.0</td>
      <td>2.5</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>


## 16.8 Measure state perturbation sensitivity

Sensitivity is the fraction of paired states whose generated action changes. It is not a correctness metric. The diagnosis also checks whether each action is consistent with its corresponding child state.


```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def render_prompt(prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

@torch.no_grad()
def generate_training_prompt(model, prompt: str) -> str:
    batch = tokenizer(render_prompt(prompt), return_tensors="pt").to(device)
    output = model.generate(**batch, max_new_tokens=16, do_sample=False)
    new_tokens = output[0, batch["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def load_model(label: str):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)
    if label == "base":
        return base
    checkpoint = CHECKPOINTS[label]
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing adapter {checkpoint}; run Lab 15")
    return PeftModel.from_pretrained(base, checkpoint).to(device)

def release_model(model):
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

def normalized_action(raw: str) -> tuple[str, str | None]:
    parsed = parse_guess(raw)
    return (parsed or raw.strip().upper()), parsed

def diagnose_pair(
    parsed_a: bool,
    parsed_b: bool,
    changed: bool,
    consistent_a: bool,
    consistent_b: bool,
    cross_consistent_a: bool,
    cross_consistent_b: bool,
) -> str:
    if not parsed_a or not parsed_b:
        return "unparseable"
    if not changed and not consistent_a and not consistent_b:
        return "state_insensitive_invalid"
    if not changed:
        return "state_insensitive_partial"
    if consistent_a and consistent_b:
        return "state_conditioned"
    if cross_consistent_a and cross_consistent_b:
        return "wrong_branch_swap"
    return "state_misinterpretation"

def evaluate_perturbations(model, label: str, interface: str) -> pd.DataFrame:
    rows = []
    model.eval()
    for pair in perturbation_pairs.itertuples():
        side_results = {}
        for side in ["a", "b"]:
            history = getattr(pair, f"history_{side}")
            prompt = getattr(pair, f"prompt_{side}")
            if interface == "training":
                raw = generate_training_prompt(model, prompt)
            elif interface == "deployment":
                raw = generate_raw_guess(
                    history, tokenizer=tokenizer, model=model, device=device
                )
            else:
                raise ValueError(interface)
            action, parsed = normalized_action(raw)
            side_results[side] = {
                "raw": raw,
                "action": action,
                "parsed": parsed,
                "in_answer_lexicon": parsed in ANSWER_SET if parsed else False,
                "consistent": is_consistent(parsed, history) if parsed else False,
            }
        changed = side_results["a"]["action"] != side_results["b"]["action"]
        both_parse = bool(side_results["a"]["parsed"] and side_results["b"]["parsed"])
        cross_consistent_a = (
            is_consistent(side_results["a"]["parsed"], pair.history_b)
            if side_results["a"]["parsed"] else False
        )
        cross_consistent_b = (
            is_consistent(side_results["b"]["parsed"], pair.history_a)
            if side_results["b"]["parsed"] else False
        )
        diagnosis = diagnose_pair(
            side_results["a"]["parsed"] is not None,
            side_results["b"]["parsed"] is not None,
            changed,
            side_results["a"]["consistent"],
            side_results["b"]["consistent"],
            cross_consistent_a,
            cross_consistent_b,
        )
        rows.append({
            "model": label,
            "interface": interface,
            "pair_id": pair.pair_id,
            "parent_difficulty": pair.parent_difficulty,
            "parent_candidates": pair.parent_candidates,
            "pair_scope": pair.pair_scope,
            "branch_source": pair.branch_source,
            "branch_guess": pair.branch_guess,
            "feedback_a": pair.feedback_a,
            "feedback_b": pair.feedback_b,
            "feedback_change_position": pair.feedback_change_position,
            "feedback_change_type": pair.feedback_change_type,
            "candidates_a": pair.candidates_a,
            "candidates_b": pair.candidates_b,
            "difficulty_a": pair.difficulty_a,
            "difficulty_b": pair.difficulty_b,
            "action_a": side_results["a"]["action"],
            "action_b": side_results["b"]["action"],
            "action_changed": changed,
            "both_parse": both_parse,
            "consistent_a": side_results["a"]["consistent"],
            "consistent_b": side_results["b"]["consistent"],
            "both_consistent": (
                side_results["a"]["consistent"] and side_results["b"]["consistent"]
            ),
            "either_consistent": (
                side_results["a"]["consistent"] or side_results["b"]["consistent"]
            ),
            "action_a_consistent_with_b": cross_consistent_a,
            "action_b_consistent_with_a": cross_consistent_b,
            "both_cross_consistent": cross_consistent_a and cross_consistent_b,
            "in_answer_lexicon_a": side_results["a"]["in_answer_lexicon"],
            "in_answer_lexicon_b": side_results["b"]["in_answer_lexicon"],
            "diagnosis": diagnosis,
        })
    return pd.DataFrame(rows)

perturbation_results = pd.DataFrame()
if RUN_PERTURBATION_EVALUATION:
    frames = []
    labels = (["base"] if EVALUATE_BASE_MODEL else []) + ["A", "B"]
    for label in labels:
        model = load_model(label)
        for interface in ["training", "deployment"]:
            frames.append(evaluate_perturbations(model, label, interface))
        release_model(model)
    perturbation_results = pd.concat(frames, ignore_index=True)
else:
    print("Perturbation evaluation skipped. Set RUN_PERTURBATION_EVALUATION=True to run it.")
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


## 16.9 Diagnose sensitivity versus correctness

A low sensitivity rate with invalid shared actions supports state insensitivity. A high sensitivity rate dominated by inconsistent changed actions supports state misinterpretation.


```python
if not perturbation_results.empty:
    parse_summary = perturbation_results.groupby(["model", "interface"]).agg(
        pairs=("pair_id", "size"),
        both_parse_rate=("both_parse", "mean"),
    )
    parsed_pairs = perturbation_results.loc[perturbation_results["both_parse"]]
    parsed_pairs = parsed_pairs.copy()
    parsed_pairs["state_misinterpretation"] = parsed_pairs["diagnosis"].isin([
        "state_misinterpretation", "wrong_branch_swap"
    ])
    parsed_summary = parsed_pairs.groupby(["model", "interface"]).agg(
        parsed_pairs=("pair_id", "size"),
        state_perturbation_sensitivity=("action_changed", "mean"),
        branch_a_consistency=("consistent_a", "mean"),
        branch_b_consistency=("consistent_b", "mean"),
        either_consistent_rate=("either_consistent", "mean"),
        both_consistent_rate=("both_consistent", "mean"),
        both_cross_consistent_rate=("both_cross_consistent", "mean"),
        state_misinterpretation_rate=("state_misinterpretation", "mean"),
    )
    display(parse_summary.join(parsed_summary))
    display(perturbation_results.groupby(
        ["model", "interface", "diagnosis"]
    ).size().rename("pairs").to_frame())
    display(parsed_pairs.groupby(
        ["model", "interface", "pair_scope", "branch_source"]
    ).agg(
        pairs=("pair_id", "size"),
        sensitivity=("action_changed", "mean"),
        both_consistent=(
            "diagnosis", lambda values: (values == "state_conditioned").mean()
        ),
    ))
    display(parsed_pairs.groupby(
        ["model", "interface", "feedback_change_type"]
    ).agg(
        pairs=("pair_id", "size"),
        sensitivity=("action_changed", "mean"),
        either_consistent_rate=("either_consistent", "mean"),
        both_consistent_rate=("both_consistent", "mean"),
        both_cross_consistent_rate=("both_cross_consistent", "mean"),
    ))
else:
    print("No perturbation results yet.")
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
      <th>both_parse_rate</th>
      <th>parsed_pairs</th>
      <th>state_perturbation_sensitivity</th>
      <th>branch_a_consistency</th>
      <th>branch_b_consistency</th>
      <th>either_consistent_rate</th>
      <th>both_consistent_rate</th>
      <th>both_cross_consistent_rate</th>
      <th>state_misinterpretation_rate</th>
    </tr>
    <tr>
      <th>model</th>
      <th>interface</th>
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
      <th rowspan="2" valign="top">A</th>
      <th>deployment</th>
      <td>34</td>
      <td>0.852941</td>
      <td>29.0</td>
      <td>0.137931</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.137931</td>
    </tr>
    <tr>
      <th>training</th>
      <td>34</td>
      <td>0.911765</td>
      <td>31.0</td>
      <td>0.032258</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>deployment</th>
      <td>34</td>
      <td>0.823529</td>
      <td>28.0</td>
      <td>0.107143</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.107143</td>
    </tr>
    <tr>
      <th>training</th>
      <td>34</td>
      <td>0.911765</td>
      <td>31.0</td>
      <td>0.161290</td>
      <td>0.032258</td>
      <td>0.096774</td>
      <td>0.129032</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.161290</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">base</th>
      <th>deployment</th>
      <td>34</td>
      <td>0.882353</td>
      <td>30.0</td>
      <td>0.100000</td>
      <td>0.033333</td>
      <td>0.033333</td>
      <td>0.066667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>training</th>
      <td>34</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th></th>
      <th></th>
      <th>pairs</th>
    </tr>
    <tr>
      <th>model</th>
      <th>interface</th>
      <th>diagnosis</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">A</th>
      <th rowspan="3" valign="top">deployment</th>
      <th>state_insensitive_invalid</th>
      <td>25</td>
    </tr>
    <tr>
      <th>state_misinterpretation</th>
      <td>4</td>
    </tr>
    <tr>
      <th>unparseable</th>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">training</th>
      <th>state_insensitive_invalid</th>
      <td>30</td>
    </tr>
    <tr>
      <th>state_misinterpretation</th>
      <td>1</td>
    </tr>
    <tr>
      <th>unparseable</th>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">B</th>
      <th rowspan="4" valign="top">deployment</th>
      <th>state_insensitive_invalid</th>
      <td>24</td>
    </tr>
    <tr>
      <th>state_insensitive_partial</th>
      <td>1</td>
    </tr>
    <tr>
      <th>state_misinterpretation</th>
      <td>3</td>
    </tr>
    <tr>
      <th>unparseable</th>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">training</th>
      <th>state_insensitive_invalid</th>
      <td>23</td>
    </tr>
    <tr>
      <th>state_insensitive_partial</th>
      <td>3</td>
    </tr>
    <tr>
      <th>state_misinterpretation</th>
      <td>5</td>
    </tr>
    <tr>
      <th>unparseable</th>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">base</th>
      <th rowspan="4" valign="top">deployment</th>
      <th>state_insensitive_invalid</th>
      <td>25</td>
    </tr>
    <tr>
      <th>state_insensitive_partial</th>
      <td>2</td>
    </tr>
    <tr>
      <th>state_misinterpretation</th>
      <td>3</td>
    </tr>
    <tr>
      <th>unparseable</th>
      <td>4</td>
    </tr>
    <tr>
      <th>training</th>
      <th>unparseable</th>
      <td>34</td>
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
      <th></th>
      <th></th>
      <th></th>
      <th>pairs</th>
      <th>sensitivity</th>
      <th>both_consistent</th>
    </tr>
    <tr>
      <th>model</th>
      <th>interface</th>
      <th>pair_scope</th>
      <th>branch_source</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="12" valign="top">A</th>
      <th rowspan="6" valign="top">deployment</th>
      <th rowspan="2" valign="top">broad</th>
      <th>controlled_off_policy</th>
      <td>9</td>
      <td>0.222222</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">mixed</th>
      <th>controlled_off_policy</th>
      <td>8</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">narrow</th>
      <th>controlled_off_policy</th>
      <td>9</td>
      <td>0.222222</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>teacher</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">training</th>
      <th rowspan="2" valign="top">broad</th>
      <th>controlled_off_policy</th>
      <td>9</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">mixed</th>
      <th>controlled_off_policy</th>
      <td>10</td>
      <td>0.100000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">narrow</th>
      <th>controlled_off_policy</th>
      <td>9</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>teacher</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">B</th>
      <th rowspan="6" valign="top">deployment</th>
      <th rowspan="2" valign="top">broad</th>
      <th>controlled_off_policy</th>
      <td>7</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">mixed</th>
      <th>controlled_off_policy</th>
      <td>10</td>
      <td>0.200000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">narrow</th>
      <th>controlled_off_policy</th>
      <td>7</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>teacher</th>
      <td>2</td>
      <td>0.500000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">training</th>
      <th rowspan="2" valign="top">broad</th>
      <th>controlled_off_policy</th>
      <td>10</td>
      <td>0.200000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">mixed</th>
      <th>controlled_off_policy</th>
      <td>10</td>
      <td>0.300000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">narrow</th>
      <th>controlled_off_policy</th>
      <td>8</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>teacher</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">base</th>
      <th rowspan="6" valign="top">deployment</th>
      <th rowspan="2" valign="top">broad</th>
      <th>controlled_off_policy</th>
      <td>8</td>
      <td>0.125000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">mixed</th>
      <th>controlled_off_policy</th>
      <td>9</td>
      <td>0.111111</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fixed_opening</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">narrow</th>
      <th>controlled_off_policy</th>
      <td>9</td>
      <td>0.111111</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>teacher</th>
      <td>2</td>
      <td>0.000000</td>
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
      <th></th>
      <th></th>
      <th>pairs</th>
      <th>sensitivity</th>
      <th>either_consistent_rate</th>
      <th>both_consistent_rate</th>
      <th>both_cross_consistent_rate</th>
    </tr>
    <tr>
      <th>model</th>
      <th>interface</th>
      <th>feedback_change_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">A</th>
      <th rowspan="3" valign="top">deployment</th>
      <th>B/G</th>
      <td>10</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>12</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>7</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">training</th>
      <th>B/G</th>
      <td>11</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>13</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>7</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">B</th>
      <th rowspan="3" valign="top">deployment</th>
      <th>B/G</th>
      <td>11</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>13</td>
      <td>0.230769</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>4</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">training</th>
      <th>B/G</th>
      <td>12</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>12</td>
      <td>0.333333</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>7</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">base</th>
      <th rowspan="3" valign="top">deployment</th>
      <th>B/G</th>
      <td>10</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>13</td>
      <td>0.076923</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>7</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


## 16.10 Persist the diagnostic evidence

Lab 17 needs the individual pairs and actions, not only the aggregate diagnosis.


```python
LAB16_RESULTS.mkdir(parents=True, exist_ok=True)
fixed_failures.drop(columns=["history"]).to_csv(
    LAB16_RESULTS / "fixed-state-failures.csv", index=False
)
gameplay_failures.to_csv(
    LAB16_RESULTS / "gameplay-failures.csv", index=False
)
perturbation_pairs.drop(columns=[
    "history_a", "history_b"
]).to_csv(LAB16_RESULTS / "perturbation-pairs.csv", index=False)
if RUN_PERTURBATION_EVALUATION and not perturbation_results.empty:
    perturbation_results.to_csv(
        LAB16_RESULTS / "perturbation-results.csv", index=False
    )
print("saved available Lab 16 results to", LAB16_RESULTS)
```

    saved available Lab 16 results to ../results/lab16


## 16.11 Interpretation rules

| Result | Interpretation | Next intervention |
| --- | --- | --- |
| Low parsed-pair sensitivity, mostly shared invalid actions | The model largely ignores state changes | Make state features more explicit in Lab 17 |
| High sensitivity, mostly inconsistent changed actions | The model reads state but misinterprets constraints | Add targeted constraint supervision before changing the interface |
| Cross-branch consistency exceeds direct consistency | The model reacts to feedback but may bind branch semantics backward | Make feedback semantics explicit and retest |
| Training sensitivity exceeds deployment sensitivity | Prompt transfer suppresses learned state use | Test structured deployment representations in Lab 17 |
| Both sensitivity and paired consistency improve for B | Dataset B taught partial state conditioning | Preserve Dataset B and isolate the remaining failure buckets |
| Low both-parse rate | Output formatting prevents a state-sensitivity conclusion | Diagnose generation format separately |

Do not change Dataset B in this notebook. Diagnose the failure before choosing the next intervention.

# Lab 16 checkpoint

Record:

1. fixed-state failure modes by model, interface, and difficulty;
2. Lab 15 repeat rate versus parsed-output repeat rate;
3. dominant generated-action attractors;
4. constraint violation types;
5. paired perturbation count and difficulty coverage;
6. state perturbation sensitivity by model and interface;
7. state insensitivity versus state misinterpretation rates;
8. direct paired consistency and wrong-branch consistency;
9. sensitivity by feedback-change type;
10. the interpretation rule supported by the evidence.
