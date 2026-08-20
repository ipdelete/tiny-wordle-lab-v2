# Lab 18 - Learn from complete game trajectories

**Goal:** test whether Dataset G's broader turn-2 and alternative-opening state-visitation distribution improves state-conditioned policy behavior over Dataset B under a matched training budget.

Dataset B already contains complete canonical teacher trajectories and visitation weighting. Dataset G changes the leakage-exclusion granularity from individual states to whole games. The resulting clean RAISE deficit shifts policy exposure toward turn-2 states and the four existing alternative openings.

```text
B-structured  = frozen Lab 17 adapter + Dataset B mixed policy allocation
G-structured  = fresh identical LoRA adapter + whole-game-filtered policy allocation
```

## 18.1 Pre-registered experiment

Dataset G replaces Dataset B's 5,669 training policy rows with 5,669 whole-game-filtered visits. It copies Dataset B's 3,099 auxiliary training rows unchanged except for the established structured rendering. The generator selects or rejects a whole game, never an individual state.

B-structured and G-structured use the same base model, structured fields, LoRA configuration, optimizer, schedule, seed, effective batch size, 1,029 optimizer steps, compact response-only logits, and final-step checkpoint rule. Row and task budgets match by split. Input-token exposure is measured rather than assumed equal.

Rows are shuffled and optimized independently with per-row cross-entropy. Trajectory membership therefore changes only the multiset of state-action pairs and their multiplicities; it adds no sequential credit assignment. No result in this lab can establish sequential learning. G broadens opening and turn-2 coverage while narrowing unique-state, unique-target, and later-turn exposure.

The primary readout is a 620-state held-out battery formed from the union of B and G validation policy states, all disjoint from both training sets. The exact 34 Lab 16 pairs, 47 fixed states, 19 fixed-opening games, and paired turn-2 gameplay calls are guardrails. Unparseable outputs count as failures.

## 18.2 Run controls

Preflight generates and validates Dataset G without loading a model. Training and evaluation remain explicit expensive actions.


```python
RUN_TRAINING = True
RUN_EVALUATION = True

print("RUN_TRAINING:", RUN_TRAINING)
print("RUN_EVALUATION:", RUN_EVALUATION)
```

    RUN_TRAINING: True
    RUN_EVALUATION: True



```python
from collections import Counter, defaultdict
from dataclasses import dataclass
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
from tiny_wordle.expert import EntropyExpert
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
RESULTS_DIR = Path("../results/lab18")
B_STRUCTURED_CHECKPOINT = CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-b-structured"
G_STRUCTURED_CHECKPOINT = CHECKPOINT_ROOT / "qwen3-0.6b-wordle-lora-dataset-g-structured"
LAB15_RESULTS = Path("../results/lab15")
LAB16_RESULTS = Path("../results/lab16")
LAB17_RESULTS = Path("../results/lab17")

device = preferred_device()
torch.set_float32_matmul_precision("high")
print("device:", device)
```

    device: mps


## 18.3 Reuse the Lab 17 structured representation

The representation is computed from feedback, not from the hidden answer. For each letter, matched yellow and green occurrences establish a minimum count. A black duplicate beyond those matches establishes a maximum count. Every non-green occurrence excludes that position.


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
ALL_INDICES = expert.all_indices
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

## 18.4 Generate Dataset G from intact teacher games

Every policy row belongs to a solved game retained in full. The exclusion set contains all fixed states, perturbation parents and branches, and complete six-turn reserved-answer paths, including unsolved paths. Games producing a reserved answer as a teacher target are also excluded. The generator retains every clean RAISE game that fits the split budget, then fills the exact remaining rows with alternative-opening games.


```python
B_FILES = {
    "train": GENERATED_DIR / "wordle-part2-policy-train.jsonl",
    "validation": GENERATED_DIR / "wordle-part2-policy-dev.jsonl",
    "test": GENERATED_DIR / "wordle-part2-policy-test.jsonl",
}
G_FILES = {
    "train": GENERATED_DIR / "wordle-part2-game-train.jsonl",
    "validation": GENERATED_DIR / "wordle-part2-game-dev.jsonl",
    "test": GENERATED_DIR / "wordle-part2-game-test.jsonl",
}
RESERVED_ANSWERS = set(DEFAULT_EVAL_ANSWERS)
OPENINGS = ["RAISE", "FJORD", "STARE", "MOUND", "GLYPH"]
POLICY_ROW_TARGETS = {"train": 5669, "validation": 669, "test": 13}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def stable_bucket(value: str) -> int:
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:8], "big") % 1000

def trajectory_split(history: tuple[Turn, ...]) -> str:
    first_branch = format_history(history).splitlines()[0]
    bucket = stable_bucket("dataset-b-branch:" + first_branch)
    if bucket < 10:
        return "test"
    if bucket < 120:
        return "validation"
    return "train"

def format_history(history: tuple[Turn, ...] | list[Turn]) -> str:
    return "\n".join(
        f"{' '.join(turn.guess)} -> {' '.join(turn.feedback)}" for turn in history
    )

def raw_policy_prompt(state_key: str) -> str:
    return (
        "Task: NEXT_GUESS\nYou are playing Wordle.\n"
        "Use the game history to choose the next guess.\n"
        "Return exactly one uppercase five-letter word.\n\n"
        f"History:\n{state_key}"
    )

@dataclass(frozen=True)
class TrajectoryState:
    turn: int
    history: tuple[Turn, ...]
    candidate_indices: tuple[int, ...]
    candidate_count_after: int
    expert_guess: str
    teacher_entropy: float

@dataclass(frozen=True)
class TeacherGame:
    game_id: str
    answer: str
    opening: str
    split: str
    states: tuple[TrajectoryState, ...]
    solved_turn: int

choice_cache = {}

def choose_cached(candidate_indices: np.ndarray) -> int:
    key = tuple(int(index) for index in candidate_indices)
    if key not in choice_cache:
        choice_cache[key] = expert.choose(candidate_indices)
    return choice_cache[key]

def play_teacher_game(answer: str, opening: str) -> TeacherGame | None:
    candidates = ALL_INDICES.copy()
    opening_index = WORD_TO_INDEX[opening]
    feedback = score_string(answer, opening)
    if feedback == "GGGGG":
        return None
    history = [Turn(opening, feedback)]
    candidates = expert.update(candidates, opening_index, feedback)
    states = []
    for turn in range(2, 7):
        guess_index = choose_cached(candidates)
        guess = ANSWERS[guess_index]
        entropy = expert.entropy(guess_index, candidates)
        feedback = score_string(answer, guess)
        candidates_after = expert.update(candidates, guess_index, feedback)
        states.append(TrajectoryState(
            turn=turn,
            history=tuple(history),
            candidate_indices=tuple(int(index) for index in candidates),
            candidate_count_after=len(candidates_after),
            expert_guess=guess,
            teacher_entropy=entropy,
        ))
        history.append(Turn(guess, feedback))
        if feedback == "GGGGG":
            return TeacherGame(
                game_id=f"g-{opening.lower()}-{answer.lower()}",
                answer=answer,
                opening=opening,
                split=trajectory_split(states[0].history),
                states=tuple(states),
                solved_turn=turn,
            )
        candidates = candidates_after
    return None

def walk_teacher_histories(answer: str, opening: str) -> list[tuple[Turn, ...]]:
    candidates = ALL_INDICES.copy()
    feedback = score_string(answer, opening)
    if feedback == "GGGGG":
        return []
    history = [Turn(opening, feedback)]
    candidates = expert.update(candidates, WORD_TO_INDEX[opening], feedback)
    histories = []
    for _ in range(2, 7):
        histories.append(tuple(history))
        guess_index = choose_cached(candidates)
        guess = ANSWERS[guess_index]
        feedback = score_string(answer, guess)
        history.append(Turn(guess, feedback))
        if feedback == "GGGGG":
            break
        candidates = expert.update(candidates, guess_index, feedback)
    return histories

fixed_eval_keys = set(pd.read_csv(LAB15_RESULTS / "policy-results.csv").query(
    "model == 'B' and interface == 'training'"
)["state_key"])
pair_design_for_exclusion = pd.read_csv(LAB16_RESULTS / "perturbation-pairs.csv")
pair_eval_keys = set(pair_design_for_exclusion["parent_key"])
for side in ["a", "b"]:
    pair_eval_keys.update(
        prompt.split("\n\nHistory:\n", 1)[1]
        for prompt in pair_design_for_exclusion[f"prompt_{side}"]
    )
reserved_path_keys = {
    format_history(history)
    for answer in RESERVED_ANSWERS
    for history in walk_teacher_histories(answer, "RAISE")
}
EVAL_STATE_KEYS = fixed_eval_keys | pair_eval_keys | reserved_path_keys
assert len(fixed_eval_keys) == 47
assert len(pair_design_for_exclusion) == 34
assert reserved_path_keys

game_pool = defaultdict(list)
pool_failures = []
pool_leaks = []
for answer in ANSWERS:
    if answer in RESERVED_ANSWERS:
        continue
    for opening in OPENINGS:
        game = play_teacher_game(answer, opening)
        if game is None:
            pool_failures.append((answer, opening))
            continue
        state_keys = {format_history(state.history) for state in game.states}
        if state_keys & EVAL_STATE_KEYS:
            pool_leaks.append(game.game_id)
            continue
        if any(state.expert_guess in RESERVED_ANSWERS for state in game.states):
            pool_leaks.append(game.game_id + ":reserved-target")
            continue
        game_pool[(game.split, opening)].append(game)

def select_games_exact(games: list[TeacherGame], target_rows: int, salt: str) -> list[TeacherGame]:
    if target_rows == 0:
        return []
    ordered = sorted(games, key=lambda game: stable_bucket(salt + game.game_id))
    reachable = [False] * (target_rows + 1)
    predecessor = [None] * (target_rows + 1)
    reachable[0] = True
    for index, game in enumerate(ordered):
        length = len(game.states)
        for total in range(target_rows - length, -1, -1):
            new_total = total + length
            if reachable[total] and not reachable[new_total]:
                reachable[new_total] = True
                predecessor[new_total] = (total, index)
        if reachable[target_rows]:
            break
    if not reachable[target_rows]:
        available = sum(len(game.states) for game in games)
        raise ValueError(
            f"cannot select {target_rows} complete rows from {salt}; available={available}"
        )
    selected_indices = []
    total = target_rows
    while total:
        total, index = predecessor[total]
        selected_indices.append(index)
    return [ordered[index] for index in reversed(selected_indices)]

selected_games = []
for split, target_rows in POLICY_ROW_TARGETS.items():
    raise_games = game_pool[(split, "RAISE")]
    raise_capacity = sum(len(game.states) for game in raise_games)
    if raise_capacity <= target_rows:
        split_games = list(raise_games)
    else:
        split_games = select_games_exact(
            raise_games, target_rows, f"dataset-g:{split}:RAISE:"
        )
    retained_rows = sum(len(game.states) for game in split_games)
    if retained_rows < target_rows:
        alternatives = [
            game for opening in OPENINGS if opening != "RAISE"
            for game in game_pool[(split, opening)]
        ]
        split_games.extend(select_games_exact(
            alternatives,
            target_rows - retained_rows,
            f"dataset-g:{split}:alternatives:",
        ))
    assert sum(len(game.states) for game in split_games) == target_rows
    selected_games.extend(split_games)

policy_rows = defaultdict(list)
for game in selected_games:
    trajectory_length = len(game.states)
    for position, state in enumerate(game.states, 1):
        state_key = format_history(state.history)
        candidate_count = len(state.candidate_indices)
        policy_rows[game.split].append({
            "dataset": "G",
            "task": "NEXT_GUESS",
            "split": game.split,
            "source": "complete_game_trajectory",
            "game_id": game.game_id,
            "answer": game.answer,
            "opening": game.opening,
            "turn": state.turn,
            "trajectory_position": position,
            "trajectory_length": trajectory_length,
            "solved_turn": game.solved_turn,
            "candidate_count": candidate_count,
            "candidate_count_after": state.candidate_count_after,
            "teacher_entropy": state.teacher_entropy,
            "state_key": state_key,
            "prompt": transform_prompt(
                raw_policy_prompt(state_key), state_key, candidate_count
            ),
            "response": state.expert_guess,
            "representation": "derived_state_v1",
        })

b_hashes_before = {split: sha256_file(path) for split, path in B_FILES.items()}
b_rows = {
    split: [json.loads(line) for line in path.read_text().splitlines()]
    for split, path in B_FILES.items()
}
g_rows = {}
for split, rows in b_rows.items():
    auxiliary = []
    for row in rows:
        if row["task"] == "NEXT_GUESS":
            continue
        updated = dict(row)
        updated["dataset"] = "G"
        updated["prompt"] = transform_prompt(
            row["prompt"], row["state_key"], int(row["candidate_count"])
        )
        updated["representation"] = "derived_state_v1"
        auxiliary.append(updated)
    g_rows[split] = policy_rows[split] + auxiliary
    assert Counter(row["task"] for row in g_rows[split]) == Counter(
        row["task"] for row in rows
    )

for split, path in G_FILES.items():
    with path.open("w") as handle:
        for row in g_rows[split]:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
assert b_hashes_before == {split: sha256_file(path) for split, path in B_FILES.items()}

prompt_targets = defaultdict(set)
for rows in g_rows.values():
    for row in rows:
        prompt_targets[row["prompt"]].add(row["response"])
assert not any(len(targets) > 1 for targets in prompt_targets.values())

g_manifest = {
    "version": 1,
    "dataset": "G-1x",
    "generator": "Lab 18 complete teacher games",
    "expert": "candidate-only maximum Shannon entropy",
    "representation": "derived_state_v1",
    "policy_row_targets": POLICY_ROW_TARGETS,
    "selected_games": len(selected_games),
    "reserved_answers": sorted(RESERVED_ANSWERS),
    "excluded_evaluation_states": len(EVAL_STATE_KEYS),
    "reserved_history_overlap": len({
        row["state_key"] for rows in policy_rows.values() for row in rows
    } & EVAL_STATE_KEYS),
    "pool_failures": len(pool_failures),
    "excluded_leaking_games": len(pool_leaks),
    "dataset_b_sha256": b_hashes_before,
    "dataset_g_sha256": {split: sha256_file(path) for split, path in G_FILES.items()},
    "counts": {split: len(rows) for split, rows in g_rows.items()},
}
(GENERATED_DIR / "wordle-part2-game-manifest.json").write_text(
    json.dumps(g_manifest, indent=2)
)
display(pd.DataFrame({
    split: {"rows": len(rows), "tasks": dict(Counter(row["task"] for row in rows))}
    for split, rows in g_rows.items()
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


## 18.5 Verify trajectory integrity and measure coverage

The checks distinguish games, visits, and unique states. Every selected game must be solved, complete, confined to one split, and disjoint from the fixed evaluation histories.


```python
policy_df = pd.DataFrame(row for rows in policy_rows.values() for row in rows)
game_df = pd.DataFrame({
    "game_id": game.game_id,
    "answer": game.answer,
    "opening": game.opening,
    "split": game.split,
    "policy_rows": len(game.states),
    "solved_turn": game.solved_turn,
} for game in selected_games)

assert not set(policy_df["answer"]) & RESERVED_ANSWERS
assert not set(policy_df["state_key"]) & EVAL_STATE_KEYS
assert not set(policy_df["response"]) & RESERVED_ANSWERS
assert policy_df.groupby("game_id")["split"].nunique().max() == 1
assert policy_df.groupby("game_id").size().equals(
    game_df.set_index("game_id")["policy_rows"].sort_index()
)
for game_id, rows in policy_df.sort_values("trajectory_position").groupby("game_id"):
    assert rows["trajectory_position"].tolist() == list(range(1, len(rows) + 1))
    assert rows["turn"].tolist() == list(range(2, int(rows["solved_turn"].iloc[0]) + 1))
    reconstructed = [Turn(rows.iloc[0].opening, parse_state_key(rows.iloc[0].state_key)[0].feedback)]
    for row in rows.itertuples():
        assert parse_state_key(row.state_key) == reconstructed
        reconstructed.append(Turn(row.response, score_string(row.answer, row.response)))

unique_states = policy_df.groupby("state_key")["candidate_count"].first().to_dict()
for state_key, expected_count in unique_states.items():
    history = parse_state_key(state_key)
    candidates = filter_candidates(ANSWERS, history)
    assert len(candidates) == expected_count
    constraints = derive_constraints(history)
    for candidate in candidates:
        counts = Counter(candidate)
        assert all(counts[letter] >= low for letter, low in constraints["minimum"].items())
        assert all(counts[letter] <= high for letter, high in constraints["maximum"].items())

policy_df["candidate_bucket"] = pd.cut(
    policy_df["candidate_count"], [0, 2, 10, 50, 200, float("inf")],
    labels=["1-2", "3-10", "11-50", "51-200", "201+"],
)
policy_df["candidate_reduction"] = 1 - (
    policy_df["candidate_count_after"] / policy_df["candidate_count"]
)

print("selected games:", len(game_df))
print("policy visits:", len(policy_df))
print("unique policy states:", policy_df["state_key"].nunique())
print("repeat visits:", len(policy_df) - policy_df["state_key"].nunique())
display(game_df.groupby(["split", "opening"]).agg(
    games=("game_id", "size"), policy_rows=("policy_rows", "sum"),
))
display(policy_df.groupby(["split", "turn"]).agg(
    rows=("game_id", "size"),
    games=("game_id", "nunique"),
    unique_states=("state_key", "nunique"),
    median_candidates=("candidate_count", "median"),
    median_reduction=("candidate_reduction", "median"),
))
display(pd.crosstab(policy_df["turn"], policy_df["candidate_bucket"]))
b_policy_df = pd.DataFrame(
    row for row in b_rows["train"] if row["task"] == "NEXT_GUESS"
)
g_train_df = policy_df.loc[policy_df["split"] == "train"]
distribution_comparison = pd.DataFrame([
    {
        "dataset": "B",
        "rows": len(b_policy_df),
        "unique_states": b_policy_df["state_key"].nunique(),
        "repeat_visits": len(b_policy_df) - b_policy_df["state_key"].nunique(),
        "alternative_opening_share": (b_policy_df["opening"] != "EXPERT").mean(),
        "turn_2_share": (b_policy_df["turn"] == 2).mean(),
        "turn_3_6_rows": int((b_policy_df["turn"] >= 3).sum()),
        "mean_candidates": b_policy_df["candidate_count"].mean(),
        "unique_targets": b_policy_df["response"].nunique(),
        "reserved_target_rows": int(b_policy_df["response"].isin(RESERVED_ANSWERS).sum()),
    },
    {
        "dataset": "G",
        "rows": len(g_train_df),
        "unique_states": g_train_df["state_key"].nunique(),
        "repeat_visits": len(g_train_df) - g_train_df["state_key"].nunique(),
        "alternative_opening_share": (g_train_df["opening"] != "RAISE").mean(),
        "turn_2_share": (g_train_df["turn"] == 2).mean(),
        "turn_3_6_rows": int((g_train_df["turn"] >= 3).sum()),
        "mean_candidates": g_train_df["candidate_count"].mean(),
        "unique_targets": g_train_df["response"].nunique(),
        "reserved_target_rows": int(g_train_df["response"].isin(RESERVED_ANSWERS).sum()),
    },
])
display(distribution_comparison)

train_states = set(g_train_df["state_key"])
validation_states = set(policy_df.loc[policy_df["split"] == "validation", "state_key"])
test_states = set(policy_df.loc[policy_df["split"] == "test", "state_key"])
assert not train_states & validation_states
assert not train_states & test_states
assert not validation_states & test_states

```

    selected games: 2475
    policy visits: 6351
    unique policy states: 3106
    repeat visits: 3245



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
      <th>games</th>
      <th>policy_rows</th>
    </tr>
    <tr>
      <th>split</th>
      <th>opening</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">test</th>
      <th>GLYPH</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>RAISE</th>
      <td>6</td>
      <td>11</td>
    </tr>
    <tr>
      <th>STARE</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>FJORD</th>
      <td>211</td>
      <td>581</td>
    </tr>
    <tr>
      <th>GLYPH</th>
      <td>163</td>
      <td>434</td>
    </tr>
    <tr>
      <th>MOUND</th>
      <td>225</td>
      <td>587</td>
    </tr>
    <tr>
      <th>RAISE</th>
      <td>1338</td>
      <td>3376</td>
    </tr>
    <tr>
      <th>STARE</th>
      <td>278</td>
      <td>691</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">validation</th>
      <th>FJORD</th>
      <td>7</td>
      <td>22</td>
    </tr>
    <tr>
      <th>GLYPH</th>
      <td>109</td>
      <td>323</td>
    </tr>
    <tr>
      <th>MOUND</th>
      <td>10</td>
      <td>25</td>
    </tr>
    <tr>
      <th>RAISE</th>
      <td>103</td>
      <td>245</td>
    </tr>
    <tr>
      <th>STARE</th>
      <td>23</td>
      <td>54</td>
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
      <th>rows</th>
      <th>games</th>
      <th>unique_states</th>
      <th>median_candidates</th>
      <th>median_reduction</th>
    </tr>
    <tr>
      <th>split</th>
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
      <th rowspan="3" valign="top">test</th>
      <th>2</th>
      <td>8</td>
      <td>8</td>
      <td>4</td>
      <td>5.0</td>
      <td>0.675000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1.5</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>2</th>
      <td>2215</td>
      <td>2215</td>
      <td>273</td>
      <td>47.0</td>
      <td>0.930233</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2071</td>
      <td>2071</td>
      <td>1266</td>
      <td>3.0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1122</td>
      <td>1122</td>
      <td>969</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>230</td>
      <td>230</td>
      <td>210</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">validation</th>
      <th>2</th>
      <td>252</td>
      <td>252</td>
      <td>32</td>
      <td>58.5</td>
      <td>0.928571</td>
    </tr>
    <tr>
      <th>3</th>
      <td>235</td>
      <td>235</td>
      <td>148</td>
      <td>4.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>150</td>
      <td>150</td>
      <td>138</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>30</td>
      <td>29</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1.0</td>
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
      <th>candidate_bucket</th>
      <th>1-2</th>
      <th>3-10</th>
      <th>11-50</th>
      <th>51-200</th>
      <th>201+</th>
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
      <td>71</td>
      <td>287</td>
      <td>894</td>
      <td>1019</td>
      <td>204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>984</td>
      <td>1061</td>
      <td>257</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1135</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>248</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <th>dataset</th>
      <th>rows</th>
      <th>unique_states</th>
      <th>repeat_visits</th>
      <th>alternative_opening_share</th>
      <th>turn_2_share</th>
      <th>turn_3_6_rows</th>
      <th>mean_candidates</th>
      <th>unique_targets</th>
      <th>reserved_target_rows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>5669</td>
      <td>3099</td>
      <td>2570</td>
      <td>0.179573</td>
      <td>0.255071</td>
      <td>4223</td>
      <td>15.857294</td>
      <td>2160</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>G</td>
      <td>5669</td>
      <td>2749</td>
      <td>2920</td>
      <td>0.404481</td>
      <td>0.390721</td>
      <td>3454</td>
      <td>30.174987</td>
      <td>1780</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


## 18.6 Tokenize and measure exposure

Only assistant response tokens contribute to loss. Dataset G matches Dataset B's row and task counts, but different trajectory histories can change the realized input-token exposure over 1,029 shuffled updates.


```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
PAD_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

g_dataset = DatasetDict({
    split: Dataset.from_list(rows) for split, rows in g_rows.items()
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
    for split, dataset in g_dataset.items()
}
assert max(max(values) for values in lengths.values()) < MAX_LENGTH

stream = batch_stream(g_dataset["train"], SEED)
planned_g_tokens = 0
for _ in range(COMMON_STEPS):
    _, batch = next(stream)
    planned_g_tokens += int(batch["attention_mask"].sum())

b_manifest = json.loads((B_STRUCTURED_CHECKPOINT / "lab17-run.json").read_text())
b_tokens = int(b_manifest["processed_input_tokens"])
print("B-structured input tokens:", b_tokens)
print("G-structured planned input tokens:", planned_g_tokens)
print("G/B input-token ratio:", f"{planned_g_tokens / b_tokens:.3f}")
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

    B-structured input tokens: 2456934
    G-structured planned input tokens: 2414866
    G/B input-token ratio: 0.983



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
      <td>146.681455</td>
      <td>214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>validation</td>
      <td>1135</td>
      <td>148.687225</td>
      <td>205</td>
    </tr>
    <tr>
      <th>2</th>
      <td>test</td>
      <td>21</td>
      <td>131.714286</td>
      <td>161</td>
    </tr>
  </tbody>
</table>
</div>


## 18.7 Train G-structured

This is the only new model. B-structured remains the frozen Lab 17 checkpoint.


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
    in_progress = G_STRUCTURED_CHECKPOINT.with_name(
        G_STRUCTURED_CHECKPOINT.name + "-in-progress"
    )
    collisions = [path for path in [G_STRUCTURED_CHECKPOINT, in_progress] if path.exists()]
    if collisions:
        raise FileExistsError(f"existing Lab 18 paths: {collisions}")

    model = build_lora_model()
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_multiplier
    )
    stream = batch_stream(g_dataset["train"], SEED)
    baseline_val_loss = evaluate_loss(model, g_dataset["validation"])
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
                model, g_dataset["validation"]
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
        "dataset": "G-1x",
        "representation": "derived_state_v1",
        "base_model": MODEL_ID,
        "seed": SEED,
        "optimizer_steps": COMMON_STEPS,
        "effective_batch_size": BATCH_SIZE,
        "train_microbatch_size": TRAIN_MICROBATCH_SIZE,
        "processed_input_tokens": processed_input_tokens,
        "processed_supervised_tokens": processed_supervised_tokens,
        "b_structured_input_tokens": b_tokens,
        "input_token_ratio": processed_input_tokens / b_tokens,
        "baseline_val_loss": baseline_val_loss,
        "final_val_loss": next(
            row["val_loss"] for row in reversed(records) if row["val_loss"] is not None
        ),
        "elapsed_seconds": time.perf_counter() - start,
        "dataset_g_sha256": g_manifest["dataset_g_sha256"],
    }
    (in_progress / "lab18-run.json").write_text(json.dumps(run_manifest, indent=2))
    in_progress.rename(G_STRUCTURED_CHECKPOINT)
    release_model(model)
else:
    print("Training skipped. Set RUN_TRAINING=True to create G-structured.")
```


    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]


    trainable parameters: 2,293,760
    trainable share: 0.383%


    step    1/1029 loss=6.9128 lr=1.96e-06 epoch=1 mps_driver_gib=5.47


    step   25/1029 loss=2.2998 lr=4.90e-05 epoch=1 mps_driver_gib=10.30


    step   50/1029 loss=1.8116 lr=9.80e-05 epoch=1 mps_driver_gib=10.30


    step   75/1029 loss=1.7806 lr=9.99e-05 epoch=1 mps_driver_gib=11.30


    step  100/1029 loss=1.5739 lr=9.94e-05 epoch=1 mps_driver_gib=11.30


    step  125/1029 loss=1.0827 lr=9.86e-05 epoch=1 mps_driver_gib=11.30


    step  150/1029 loss=1.4058 lr=9.75e-05 epoch=1 mps_driver_gib=11.30


    step  175/1029 loss=1.8354 lr=9.61e-05 epoch=1 mps_driver_gib=11.30


    step  200/1029 loss=1.0436 lr=9.45e-05 epoch=1 mps_driver_gib=11.30
      validation loss=1.5089


    step  225/1029 loss=0.9528 lr=9.25e-05 epoch=1 mps_driver_gib=11.30


    step  250/1029 loss=1.0563 lr=9.02e-05 epoch=1 mps_driver_gib=11.30


    step  275/1029 loss=1.2365 lr=8.77e-05 epoch=1 mps_driver_gib=11.30


    step  300/1029 loss=1.1912 lr=8.50e-05 epoch=1 mps_driver_gib=11.30


    step  325/1029 loss=1.3235 lr=8.20e-05 epoch=1 mps_driver_gib=11.30


    step  350/1029 loss=1.3257 lr=7.88e-05 epoch=1 mps_driver_gib=11.30


    step  375/1029 loss=1.1430 lr=7.54e-05 epoch=1 mps_driver_gib=11.30


    step  400/1029 loss=0.8052 lr=7.19e-05 epoch=1 mps_driver_gib=11.30
      validation loss=1.4333


    step  425/1029 loss=0.8960 lr=6.82e-05 epoch=1 mps_driver_gib=11.30


    step  450/1029 loss=1.0514 lr=6.44e-05 epoch=1 mps_driver_gib=11.30


    step  475/1029 loss=1.0612 lr=6.05e-05 epoch=1 mps_driver_gib=11.30


    step  500/1029 loss=1.0358 lr=5.66e-05 epoch=1 mps_driver_gib=11.30


    step  525/1029 loss=1.5188 lr=5.26e-05 epoch=1 mps_driver_gib=11.30


    step  550/1029 loss=1.1191 lr=4.86e-05 epoch=2 mps_driver_gib=11.30


    step  575/1029 loss=0.7149 lr=4.46e-05 epoch=2 mps_driver_gib=11.30


    step  600/1029 loss=0.6879 lr=4.06e-05 epoch=2 mps_driver_gib=11.30
      validation loss=1.3681


    step  625/1029 loss=0.8349 lr=3.67e-05 epoch=2 mps_driver_gib=11.30


    step  650/1029 loss=0.7205 lr=3.28e-05 epoch=2 mps_driver_gib=11.31


    step  675/1029 loss=0.7702 lr=2.91e-05 epoch=2 mps_driver_gib=11.30


    step  700/1029 loss=0.8733 lr=2.56e-05 epoch=2 mps_driver_gib=11.30


    step  725/1029 loss=0.5316 lr=2.21e-05 epoch=2 mps_driver_gib=11.30


    step  750/1029 loss=0.5887 lr=1.89e-05 epoch=2 mps_driver_gib=11.30


    step  775/1029 loss=0.4326 lr=1.59e-05 epoch=2 mps_driver_gib=11.30


    step  800/1029 loss=0.5446 lr=1.30e-05 epoch=2 mps_driver_gib=11.30
      validation loss=1.3668


    step  825/1029 loss=1.0109 lr=1.05e-05 epoch=2 mps_driver_gib=11.30


    step  850/1029 loss=0.9492 lr=8.13e-06 epoch=2 mps_driver_gib=11.30


    step  875/1029 loss=0.8248 lr=6.07e-06 epoch=2 mps_driver_gib=11.30


    step  900/1029 loss=0.6190 lr=4.30e-06 epoch=2 mps_driver_gib=11.30


    step  925/1029 loss=0.9656 lr=2.82e-06 epoch=2 mps_driver_gib=11.30


    step  950/1029 loss=0.6471 lr=1.64e-06 epoch=2 mps_driver_gib=11.30


    step  975/1029 loss=0.7657 lr=7.78e-07 epoch=2 mps_driver_gib=11.30


    step 1000/1029 loss=0.5761 lr=2.32e-07 epoch=2 mps_driver_gib=11.30
      validation loss=1.3647


    step 1025/1029 loss=0.6563 lr=6.45e-09 epoch=2 mps_driver_gib=11.30


      validation loss=1.3649


## 18.8 Reuse the exact Lab 16 perturbation pairs

B-structured baselines come from the frozen Lab 17 result. G-structured receives the same 34 child-state pairs rendered through `derived_state_v1`.


```python
pair_design = pd.read_csv(LAB16_RESULTS / "perturbation-pairs.csv")
b_pair_baseline = pd.read_csv(LAB17_RESULTS / "pair-results.csv")
assert len(pair_design) == len(b_pair_baseline) == 34

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


## 18.9 Evaluate paired and branch consistency

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

g_pair_results = pd.DataFrame()
if RUN_EVALUATION:
    model = load_adapter(G_STRUCTURED_CHECKPOINT)
    g_pair_results = evaluate_structured_pairs(model, "G-structured")
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
        pair_summary(b_pair_baseline, "B-structured"),
        pair_summary(g_pair_results, "G-structured"),
    ])
    display(summaries)
    paired_comparison = b_pair_baseline[["pair_id", "both_consistent"]].merge(
        g_pair_results[["pair_id", "both_consistent"]],
        on="pair_id", suffixes=("_b", "_g"), validate="one_to_one",
    )
    print("exact paired-consistency p-value:", exact_paired_p_value(
        paired_comparison["both_consistent_b"],
        paired_comparison["both_consistent_g"],
    ))
    pair_breakdown = pd.concat([
        b_pair_baseline.assign(comparison_model="B-structured"),
        g_pair_results.assign(comparison_model="G-structured"),
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
      <td>B-structured</td>
      <td>34</td>
      <td>24</td>
      <td>0.705882</td>
      <td>0.750000</td>
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
    <tr>
      <th>1</th>
      <td>G-structured</td>
      <td>34</td>
      <td>27</td>
      <td>0.794118</td>
      <td>0.851852</td>
      <td>7</td>
      <td>0.205882</td>
      <td>0.103494</td>
      <td>0.367987</td>
      <td>26</td>
      <td>68</td>
      <td>0.382353</td>
      <td>0.264706</td>
      <td>0.514706</td>
      <td>0.259259</td>
      <td>0.462963</td>
    </tr>
  </tbody>
</table>
</div>


    exact paired-consistency p-value: 0.453125



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
      <th rowspan="9" valign="top">B-structured</th>
      <th rowspan="3" valign="top">broad</th>
      <th>B/G</th>
      <td>4</td>
      <td>1.000000</td>
      <td>0.250000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>1</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">mixed</th>
      <th>B/G</th>
      <td>3</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>2</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>5</td>
      <td>1.000000</td>
      <td>0.400000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">narrow</th>
      <th>B/G</th>
      <td>2</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>4</td>
      <td>0.750000</td>
      <td>0.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">G-structured</th>
      <th rowspan="3" valign="top">broad</th>
      <th>B/G</th>
      <td>6</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>7</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>1</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">mixed</th>
      <th>B/G</th>
      <td>2</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>5</td>
      <td>1.000000</td>
      <td>0.400000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">narrow</th>
      <th>B/G</th>
      <td>1</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>B/Y</th>
      <td>7</td>
      <td>0.571429</td>
      <td>0.142857</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Y/G</th>
      <td>2</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


## 18.10 Load frozen Lab 17 guardrails

The control is the executed B-structured model, not a newly trained replica. Pair, fixed-state, and auxiliary results remain frozen. Gameplay is re-evaluated for both adapters because Lab 18 terminates a game after a non-lexicon action instead of repeating an unchanged state.


```python
b_fixed_baseline = pd.read_csv(LAB17_RESULTS / "fixed-state-results.csv")
b_aux_baseline = pd.read_csv(LAB17_RESULTS / "auxiliary-results.csv")
b_gameplay_calls = pd.read_csv(LAB17_RESULTS / "gameplay-calls.csv")
b_gameplay_games = pd.read_csv(LAB17_RESULTS / "gameplay-games.csv")
print("B-structured fixed states:", len(b_fixed_baseline))
print("B-structured gameplay calls:", len(b_gameplay_calls))
```

    B-structured fixed states: 47
    B-structured gameplay calls: 84


## 18.11 Fixed-state, auxiliary, and gameplay evaluation

The same 47 fixed states and 19 RAISE-opening games compare Dataset B and G under the shared structured interface. By-turn tables report calls at risk and distinct decisions; they remain descriptive because solving changes which games reach later turns.


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

battery_records = {}
for source, rows in [
    ("B-validation", [row for row in b_rows["validation"] if row["task"] == "NEXT_GUESS"]),
    ("G-validation", policy_rows["validation"]),
]:
    for row in rows:
        record = battery_records.setdefault(row["state_key"], {
            "state_key": row["state_key"],
            "candidate_count": int(row["candidate_count"]),
            "expected": row["response"],
            "turn": int(row["turn"]),
            "sources": [],
        })
        assert record["expected"] == row["response"]
        assert record["candidate_count"] == int(row["candidate_count"])
        record["sources"].append(source)

state_battery = pd.DataFrame(battery_records.values())
state_battery["source"] = state_battery["sources"].map(
    lambda values: "+".join(sorted(values))
)
state_battery["history"] = state_battery["state_key"].map(parse_state_key)
state_battery["candidate_bucket"] = pd.cut(
    state_battery["candidate_count"], [0, 2, 10, 50, 200, float("inf")],
    labels=["1-2", "3-10", "11-50", "51-200", "201+"],
)
state_battery["structured_prompt"] = [
    transform_prompt(
        raw_policy_prompt(row.state_key), row.state_key, int(row.candidate_count)
    )
    for row in state_battery.itertuples()
]
b_train_states = {
    row["state_key"] for row in b_rows["train"] if row["task"] == "NEXT_GUESS"
}
assert len(state_battery) == 620
assert not set(state_battery["state_key"]) & b_train_states
assert not set(state_battery["state_key"]) & train_states
print("held-out state battery:", len(state_battery))
display(pd.crosstab(state_battery["turn"], state_battery["candidate_bucket"]))

def evaluate_state_battery(model, label: str) -> pd.DataFrame:
    rows = []
    for state in state_battery.itertuples():
        raw = generate_prompt(model, state.structured_prompt)
        guess = parse_guess(raw)
        in_lexicon = bool(guess and guess in ANSWER_SET)
        consistent = bool(in_lexicon and is_consistent(guess, state.history))
        repeated = bool(guess and guess in {turn.guess for turn in state.history})
        rows.append({
            "model": label,
            "state_key": state.state_key,
            "source": state.source,
            "turn": state.turn,
            "candidate_bucket": state.candidate_bucket,
            "expected": state.expected,
            "actual": guess or raw.strip().upper(),
            "format_valid": guess is not None,
            "in_answer_lexicon": in_lexicon,
            "history_consistent": consistent,
            "repeated": repeated,
            "usable": bool(in_lexicon and consistent and not repeated),
            "teacher_match": (guess or raw.strip().upper()) == state.expected,
        })
    return pd.DataFrame(rows)

def paired_battery_metric(
    b_results: pd.DataFrame,
    g_results: pd.DataFrame,
    metric: str,
    bootstrap_samples: int = 10_000,
) -> dict:
    paired = b_results[["state_key", metric]].merge(
        g_results[["state_key", metric]],
        on="state_key",
        suffixes=("_b", "_g"),
        validate="one_to_one",
    )
    b_values = paired[f"{metric}_b"].astype(bool)
    g_values = paired[f"{metric}_g"].astype(bool)
    differences = g_values.astype(float).to_numpy() - b_values.astype(float).to_numpy()
    rng = np.random.default_rng(SEED + (0 if metric == "usable" else 1))
    sampled_indices = rng.integers(
        0, len(differences), size=(bootstrap_samples, len(differences))
    )
    sampled_deltas = differences[sampled_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(sampled_deltas, [0.025, 0.975])
    return {
        "metric": metric,
        "states": len(paired),
        "b_rate": b_values.mean(),
        "g_rate": g_values.mean(),
        "delta": differences.mean(),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "b_only": int((b_values & ~g_values).sum()),
        "g_only": int((~b_values & g_values).sum()),
        "both": int((b_values & g_values).sum()),
        "neither": int((~b_values & ~g_values).sum()),
        "exact_p_value": exact_paired_p_value(b_values, g_values),
    }

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
            "model": "G-structured",
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

b_battery_results = pd.DataFrame()
g_battery_results = pd.DataFrame()
g_fixed_results = pd.DataFrame()
if RUN_EVALUATION:
    b_model = load_adapter(B_STRUCTURED_CHECKPOINT)
    b_battery_results = evaluate_state_battery(b_model, "B-structured")
    release_model(b_model)
    g_model = load_adapter(G_STRUCTURED_CHECKPOINT)
    g_battery_results = evaluate_state_battery(g_model, "G-structured")
    g_fixed_results = evaluate_fixed_states(g_model)
    release_model(g_model)
    battery_results = pd.concat([b_battery_results, g_battery_results])
    battery_paired_summary = pd.DataFrame([
        paired_battery_metric(b_battery_results, g_battery_results, "usable"),
        paired_battery_metric(
            b_battery_results, g_battery_results, "history_consistent"
        ),
    ])
    display(battery_results.groupby("model").agg(
        states=("state_key", "size"),
        format_valid_rate=("format_valid", "mean"),
        history_consistency_rate=("history_consistent", "mean"),
        usable_rate=("usable", "mean"),
        teacher_match_rate=("teacher_match", "mean"),
    ))
    display(battery_paired_summary)
    display(battery_results.groupby(["model", "turn", "candidate_bucket"], observed=True).agg(
        states=("state_key", "size"),
        usable_rate=("usable", "mean"),
        teacher_match_rate=("teacher_match", "mean"),
    ))
    display(pd.DataFrame([
        {"model": "B-structured", "usable_rate": b_fixed_baseline["usable"].mean()},
        {"model": "G-structured", "usable_rate": g_fixed_results["usable"].mean()},
    ]))
```

    held-out state battery: 620



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
      <th>3-10</th>
      <th>11-50</th>
      <th>51-200</th>
      <th>201+</th>
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
      <td>10</td>
      <td>12</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>96</td>
      <td>206</td>
      <td>39</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>53</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>states</th>
      <th>format_valid_rate</th>
      <th>history_consistency_rate</th>
      <th>usable_rate</th>
      <th>teacher_match_rate</th>
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
      <th>B-structured</th>
      <td>620</td>
      <td>0.927419</td>
      <td>0.145161</td>
      <td>0.145161</td>
      <td>0.074194</td>
    </tr>
    <tr>
      <th>G-structured</th>
      <td>620</td>
      <td>0.885484</td>
      <td>0.159677</td>
      <td>0.159677</td>
      <td>0.077419</td>
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
      <th>states</th>
      <th>b_rate</th>
      <th>g_rate</th>
      <th>delta</th>
      <th>delta_ci_low</th>
      <th>delta_ci_high</th>
      <th>b_only</th>
      <th>g_only</th>
      <th>both</th>
      <th>neither</th>
      <th>exact_p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>usable</td>
      <td>620</td>
      <td>0.145161</td>
      <td>0.159677</td>
      <td>0.014516</td>
      <td>-0.014516</td>
      <td>0.041935</td>
      <td>36</td>
      <td>45</td>
      <td>54</td>
      <td>485</td>
      <td>0.374174</td>
    </tr>
    <tr>
      <th>1</th>
      <td>history_consistent</td>
      <td>620</td>
      <td>0.145161</td>
      <td>0.159677</td>
      <td>0.014516</td>
      <td>-0.012903</td>
      <td>0.043548</td>
      <td>36</td>
      <td>45</td>
      <td>54</td>
      <td>485</td>
      <td>0.374174</td>
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
      <th>usable_rate</th>
      <th>teacher_match_rate</th>
    </tr>
    <tr>
      <th>model</th>
      <th>turn</th>
      <th>candidate_bucket</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="15" valign="top">B-structured</th>
      <th rowspan="5" valign="top">2</th>
      <th>1-2</th>
      <td>6</td>
      <td>0.166667</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>11-50</th>
      <td>12</td>
      <td>0.166667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>201+</th>
      <td>2</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>10</td>
      <td>0.200000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>6</td>
      <td>0.166667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">3</th>
      <th>1-2</th>
      <td>96</td>
      <td>0.104167</td>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>11-50</th>
      <td>39</td>
      <td>0.179487</td>
      <td>0.025641</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>206</td>
      <td>0.179612</td>
      <td>0.058252</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4</th>
      <th>1-2</th>
      <td>149</td>
      <td>0.127517</td>
      <td>0.107383</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>53</td>
      <td>0.132075</td>
      <td>0.094340</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">5</th>
      <th>1-2</th>
      <td>33</td>
      <td>0.090909</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">6</th>
      <th>1-2</th>
      <td>4</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="15" valign="top">G-structured</th>
      <th rowspan="5" valign="top">2</th>
      <th>1-2</th>
      <td>6</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11-50</th>
      <td>12</td>
      <td>0.250000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>201+</th>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>10</td>
      <td>0.300000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>6</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">3</th>
      <th>1-2</th>
      <td>96</td>
      <td>0.125000</td>
      <td>0.104167</td>
    </tr>
    <tr>
      <th>11-50</th>
      <td>39</td>
      <td>0.205128</td>
      <td>0.025641</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>206</td>
      <td>0.194175</td>
      <td>0.048544</td>
    </tr>
    <tr>
      <th>51-200</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4</th>
      <th>1-2</th>
      <td>149</td>
      <td>0.127517</td>
      <td>0.100671</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>53</td>
      <td>0.169811</td>
      <td>0.113208</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">5</th>
      <th>1-2</th>
      <td>33</td>
      <td>0.121212</td>
      <td>0.121212</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>2</td>
      <td>0.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">6</th>
      <th>1-2</th>
      <td>4</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3-10</th>
      <td>1</td>
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
      <th>model</th>
      <th>usable_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B-structured</td>
      <td>0.234043</td>
    </tr>
    <tr>
      <th>1</th>
      <td>G-structured</td>
      <td>0.276596</td>
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
    assert state_key not in b_train_states
    assert state_key not in train_states
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
            "model": "G-structured",
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
        terminated_invalid = False
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
            if not in_answer_lexicon:
                terminated_invalid = True
                break
            if feedback == "GGGGG":
                solved_turn = turn_number
                break
        game_rows.append({
            "model": label,
            "answer": answer,
            "solved": solved_turn is not None,
            "solved_turn": solved_turn,
            "terminated_invalid": terminated_invalid,
        })
    return pd.DataFrame(call_rows), pd.DataFrame(game_rows)

g_aux_results = pd.DataFrame()
g_gameplay_calls = pd.DataFrame()
g_gameplay_games = pd.DataFrame()
if RUN_EVALUATION:
    b_model = load_adapter(B_STRUCTURED_CHECKPOINT)
    b_gameplay_calls, b_gameplay_games = evaluate_gameplay(
        b_model, "B-structured", structured_next_guess_prompt
    )
    release_model(b_model)
    model = load_adapter(G_STRUCTURED_CHECKPOINT)
    g_aux_results = evaluate_structured_auxiliary(model)
    g_gameplay_calls, g_gameplay_games = (
        evaluate_gameplay(model, "G-structured", structured_next_guess_prompt)
    )
    release_model(model)
    display(pd.concat([
        b_aux_baseline.groupby("task")["correct"].mean().rename("B-structured"),
        g_aux_results.groupby("task")["correct"].mean().rename(
            "G-structured"
        ),
    ], axis=1))

    gameplay_summary = pd.DataFrame([
        {
            "model": "B-structured",
            "solve_rate": b_gameplay_games["solved"].mean(),
            "invalid_termination_rate": b_gameplay_games["terminated_invalid"].mean(),
        },
        {
            "model": "G-structured",
            "solve_rate": g_gameplay_games["solved"].mean(),
            "invalid_termination_rate": g_gameplay_games["terminated_invalid"].mean(),
        },
    ])
    display(gameplay_summary)
    turn_2_summary = pd.concat([b_gameplay_calls, g_gameplay_calls]).query(
        "turn == 2"
    ).groupby("model").agg(
        calls=("answer", "size"),
        format_valid_rate=("format_valid", "mean"),
        in_lexicon_rate=("in_answer_lexicon", "mean"),
        history_consistency_rate=("history_consistent", "mean"),
        usable_rate=("usable", "mean"),
    )
    assert (turn_2_summary["calls"] == len(DEFAULT_EVAL_ANSWERS)).all()
    display(turn_2_summary)
    by_turn = pd.concat([b_gameplay_calls, g_gameplay_calls]).groupby(
        ["model", "turn"]
    ).agg(
        calls=("answer", "size"),
        distinct_outputs=("guess", "nunique"),
        format_valid_rate=("format_valid", "mean"),
        history_consistency_rate=("history_consistent", "mean"),
        usable_rate=("usable", "mean"),
        repeat_rate=("output_repeated", "mean"),
    )
    display(by_turn)
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
      <th>B-structured</th>
      <th>G-structured</th>
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
      <td>0.959184</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>VALID_CANDIDATE</th>
      <td>0.918367</td>
      <td>0.948980</td>
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
      <th>invalid_termination_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B-structured</td>
      <td>0.263158</td>
      <td>0.736842</td>
    </tr>
    <tr>
      <th>1</th>
      <td>G-structured</td>
      <td>0.210526</td>
      <td>0.789474</td>
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
      <th>calls</th>
      <th>format_valid_rate</th>
      <th>in_lexicon_rate</th>
      <th>history_consistency_rate</th>
      <th>usable_rate</th>
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
      <th>B-structured</th>
      <td>19</td>
      <td>0.631579</td>
      <td>0.421053</td>
      <td>0.157895</td>
      <td>0.157895</td>
    </tr>
    <tr>
      <th>G-structured</th>
      <td>19</td>
      <td>0.842105</td>
      <td>0.684211</td>
      <td>0.368421</td>
      <td>0.368421</td>
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
      <th>calls</th>
      <th>distinct_outputs</th>
      <th>format_valid_rate</th>
      <th>history_consistency_rate</th>
      <th>usable_rate</th>
      <th>repeat_rate</th>
    </tr>
    <tr>
      <th>model</th>
      <th>turn</th>
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
      <th rowspan="5" valign="top">B-structured</th>
      <th>2</th>
      <td>19</td>
      <td>12</td>
      <td>0.631579</td>
      <td>0.157895</td>
      <td>0.157895</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>7</td>
      <td>1.000000</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>2</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">G-structured</th>
      <th>2</th>
      <td>19</td>
      <td>16</td>
      <td>0.842105</td>
      <td>0.368421</td>
      <td>0.368421</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>11</td>
      <td>1.000000</td>
      <td>0.272727</td>
      <td>0.272727</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


## 18.12 Persist results and interpret without moving the goalposts

| Result | Interpretation |
| --- | --- |
| Gameplay rises while pair/fixed-state guardrails hold | Broader turn-2 and alternative-opening coverage improves state-conditioned policy behavior; this is a distribution effect, not sequential learning |
| G validation loss improves but gameplay does not | The model fits G without improving deployed policy; G and B validation losses are not directly comparable |
| Gameplay rises while pair or broad-state consistency falls | Gains are narrow trajectory-pattern transfer, not stronger state binding |
| Nothing improves | This distribution shift does not address the dominant failure under this budget |

Do not increase Dataset G beyond the matched G-1x row budget in this notebook.


```python
if RUN_EVALUATION:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    g_pair_results.to_csv(RESULTS_DIR / "pair-results.csv", index=False)
    g_fixed_results.to_csv(RESULTS_DIR / "fixed-state-results.csv", index=False)
    b_battery_results.to_csv(RESULTS_DIR / "b-state-battery-results.csv", index=False)
    g_battery_results.to_csv(RESULTS_DIR / "state-battery-results.csv", index=False)
    battery_paired_summary.to_csv(
        RESULTS_DIR / "state-battery-paired-summary.csv", index=False
    )
    g_aux_results.to_csv(RESULTS_DIR / "auxiliary-results.csv", index=False)
    g_gameplay_calls.to_csv(RESULTS_DIR / "gameplay-calls.csv", index=False)
    g_gameplay_games.to_csv(RESULTS_DIR / "gameplay-games.csv", index=False)
    b_gameplay_calls.to_csv(RESULTS_DIR / "b-gameplay-calls.csv", index=False)
    b_gameplay_games.to_csv(RESULTS_DIR / "b-gameplay-games.csv", index=False)
    summaries.to_csv(RESULTS_DIR / "pair-summary.csv", index=False)
    gameplay_summary.to_csv(RESULTS_DIR / "gameplay-summary.csv", index=False)
    turn_2_summary.to_csv(RESULTS_DIR / "gameplay-turn-2-summary.csv")
    by_turn.to_csv(RESULTS_DIR / "gameplay-by-turn.csv")
    print("saved Lab 18 results to", RESULTS_DIR)
```

    saved Lab 18 results to ../results/lab18


# Lab 18 checkpoint

Record:

1. Dataset B and G row and task budgets by split;
2. selected games, solved-turn distribution, and rows per game;
3. opening, turn, candidate-bucket, unique-state, and repeat-visit coverage;
4. proof that every trajectory is complete and reserved-path overlap is zero;
5. G/B input-token exposure ratio;
6. usable, consistent, and teacher-match rates on the 620-state held-out battery;
7. paired and branch consistency for B-structured and G-structured;
8. fixed-state usable rate and auxiliary accuracy;
9. turn-2 paired gameplay metrics, solve rate, and descriptive later-turn calls at risk;
10. the pre-registered interpretation supported by the result.
