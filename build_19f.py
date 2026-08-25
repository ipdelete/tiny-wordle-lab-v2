"""Generate notebooks/19f_symbolic_teacher_benchmark.ipynb."""

import json
import textwrap
from pathlib import Path

cells: list[dict] = []


def md(text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
        }
    )


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
        }
    )


md(
    """
    # Lab 19f - Is one-ply entropy worth distilling?

    Lab 19e has completed Gate B but has not started Gate C. Before spending a
    model run on that target, this benchmark tests the symbolic teacher itself.
    It does not train, load, or score a learned model.

    The benchmark compares two deterministic one-ply entropy policies after the
    fixed `RAISE` opening:

    1. `candidate-only-entropy` chooses among remaining answers.
    2. `open-action-entropy` chooses among all 2,315 answer-list actions.

    Both policies close singleton states immediately. They run on the 19 Lab 18d
    reserved answers and on the complete 2,315-answer universe. The full
    universe determines policy quality; the reserved battery determines whether
    either symbolic policy improves on an exact singleton-closure counterfactual
    for the existing seed-45 Lab 18d trajectories.
    """
)

md(
    """
    ## 19f.1 Preregistration

    This first executable section hashes the source data and frozen Lab 18d
    artifacts, writes the policy definitions and decision rule, and seals that
    JSON before any counterfactual or symbolic policy result is calculated.

    A failed game has no solve turn and contributes seven turns to the
    lexicographic comparison. Symbolic policies rank by solved count first, then
    by lower penalized total turns.
    """
)

code(
    """
    from __future__ import annotations

    import ast
    import hashlib
    import json
    import math
    import os
    import platform
    import sys
    import time
    from collections import Counter
    from datetime import datetime, timezone
    from pathlib import Path

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    import pandas as pd
    from IPython.display import display

    from tiny_wordle.expert import (
        EntropyExpert,
        decode_feedback,
        encode_feedback,
    )

    NOTEBOOK_STARTED = time.perf_counter()
    ROOT = Path.cwd()
    if not (ROOT / "data").exists():
        ROOT = ROOT.parent
    DATA_DIR = ROOT / "data"
    BENCHMARK_PATH = ROOT / "src" / "tiny_wordle" / "benchmark.py"
    LAB18D_DIR = ROOT / "results" / "lab18d"
    RESULTS_DIR = ROOT / "results" / "lab19f"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ANSWER_PATH = DATA_DIR / "wordle-answers-original.txt"
    PATTERN_PATH = DATA_DIR / "wordle-patterns-original-2315.npy"
    LAB18D_CALLS_PATH = (
        LAB18D_DIR / "seed45-answer-constrained-calls.csv"
    )
    LAB18D_GAMES_PATH = (
        LAB18D_DIR / "seed45-answer-constrained-games.csv"
    )
    OPENING = "RAISE"
    MAX_TURNS = 6
    FAILURE_PENALTY = 7
    ENTROPY_TIE_ATOL = 1e-12
    SYMBOLIC_POLICIES = (
        "candidate-only-entropy",
        "open-action-entropy",
    )
    INCUMBENT_POLICY = "incumbent-singleton-closure"


    def read_default_eval_answers(path: Path) -> tuple[str, ...]:
        module = ast.parse(path.read_text())
        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            ]
            if "DEFAULT_EVAL_ANSWERS" in names:
                value = ast.literal_eval(node.value)
                return tuple(value)
        raise AssertionError("DEFAULT_EVAL_ANSWERS assignment not found")


    RESERVED_ANSWERS = read_default_eval_answers(BENCHMARK_PATH)

    ANSWERS = tuple(
        line.strip().upper()
        for line in ANSWER_PATH.read_text().splitlines()
        if line.strip()
    )
    ANSWER_ARRAY = np.asarray(ANSWERS)
    WORD_TO_INDEX = {
        word: index for index, word in enumerate(ANSWERS)
    }
    PATTERNS = np.load(PATTERN_PATH)
    EXPERT = EntropyExpert(list(ANSWERS), PATTERNS)
    ALL_INDICES = np.arange(len(ANSWERS), dtype=np.int32)
    OPENING_INDEX = WORD_TO_INDEX[OPENING]
    SOLVED_PATTERN = encode_feedback("GGGGG")

    assert len(ANSWERS) == 2315
    assert len(set(ANSWERS)) == len(ANSWERS)
    assert PATTERNS.shape == (2315, 2315)
    assert PATTERNS.dtype == np.uint8
    assert len(RESERVED_ANSWERS) == 19
    assert sys.modules.get("torch") is None

    print("processor:", platform.processor() or platform.machine())
    print("logical CPUs:", os.cpu_count())
    print("torch imported:", "torch" in sys.modules)
    print("answers:", len(ANSWERS))
    print("reserved answers:", len(RESERVED_ANSWERS))
    """
)

code(
    """
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def sha256_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()


    def atomic_text(text: str, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


    def atomic_json(value: dict, path: Path) -> None:
        text = json.dumps(value, indent=2, sort_keys=True) + "\\n"
        atomic_text(text, path)


    def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)


    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            ANSWER_PATH,
            PATTERN_PATH,
            BENCHMARK_PATH,
            LAB18D_CALLS_PATH,
            LAB18D_GAMES_PATH,
        )
    }

    PREREGISTRATION = {
        "experiment": "Lab 19f symbolic teacher benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Is one-ply entropy worth distilling, and is open-action "
            "entropy better than candidate-only entropy?"
        ),
        "compute": {
            "device": "CPU only",
            "libraries": ["Python standard library", "NumPy", "pandas"],
            "torch_or_model_load_permitted": False,
        },
        "sources_sha256": source_sha256,
        "reserved_answers": list(RESERVED_ANSWERS),
        "opening": OPENING,
        "max_turns": MAX_TURNS,
        "failure_penalty_turns": FAILURE_PENALTY,
        "entropy_tie_atol": ENTROPY_TIE_ATOL,
        "policies": {
            INCUMBENT_POLICY: {
                "scope": "the 19 reserved Lab 18d seed-45 trajectories only",
                "rule": (
                    "At the first recorded call with candidate_count_before "
                    "equal to one, choose the sole remaining answer and solve "
                    "on that turn. Otherwise preserve every recorded action "
                    "and outcome."
                ),
                "exactness": (
                    "Singleton closure terminates immediately, so it creates "
                    "no unseen future model state and requires no new model "
                    "scoring."
                ),
            },
            "candidate-only-entropy": {
                "rule": (
                    "After RAISE, close singleton states; otherwise maximize "
                    "one-ply feedback entropy among remaining candidates."
                ),
                "tie_break": "lexical action order",
            },
            "open-action-entropy": {
                "rule": (
                    "After RAISE, close singleton states; otherwise maximize "
                    "one-ply feedback entropy among all 2,315 answers."
                ),
                "tie_break": (
                    "prefer a remaining candidate on entropy ties, then "
                    "lexical action order"
                ),
            },
        },
        "execution": {
            "all_action_entropy": (
                "vectorized 2,315 x 243 feedback-count matrix per state"
            ),
            "tree_key": "tuple of candidate indices",
            "shared_state_rule": (
                "all hidden answers at one candidate state reuse one action"
            ),
        },
        "quality_order": [
            "solved_count higher",
            "penalized_total_turns lower",
        ],
        "decision_rule_precedence": [
            (
                "If neither symbolic policy beats incumbent singleton "
                "closure on reserved solved count: "
                "one_ply_entropy_not_justified."
            ),
            (
                "Else if open-action beats candidate-only on full-universe "
                "quality and beats incumbent reserved solved count: "
                "open_teacher_worth_testing."
            ),
            (
                "Else use_candidate_only_teacher. This includes a "
                "candidate-only full-universe win or exact tie."
            ),
        ],
    }

    preregistration_text = json.dumps(
        PREREGISTRATION, indent=2, sort_keys=True
    )
    PREREGISTRATION_SHA256 = sha256_text(preregistration_text)
    atomic_text(
        preregistration_text + "\\n",
        RESULTS_DIR / "lab19f-preregistration.json",
    )
    atomic_text(
        PREREGISTRATION_SHA256 + "\\n",
        RESULTS_DIR / "lab19f-preregistration.sha256",
    )

    assert sha256_text(
        (RESULTS_DIR / "lab19f-preregistration.json")
        .read_text()
        .rstrip("\\n")
    ) == PREREGISTRATION_SHA256
    print("preregistration sha256:", PREREGISTRATION_SHA256)
    print(json.dumps(PREREGISTRATION, indent=2, sort_keys=True))
    """
)

md(
    """
    ## 19f.2 Audit the exact incumbent counterfactual

    The seed-45 Lab 18d constrained decoder is the incumbent. Its recorded
    calls are replayed without model access. Candidate sets are reconstructed
    from the pattern matrix and checked against every recorded before/after
    count.

    If a recorded call starts with one candidate, that candidate must be the
    hidden answer. Choosing it produces `GGGGG` and ends the game at that turn.
    No later prompt or model score is needed, which is why this counterfactual
    is exact. When no such call exists, the recorded actions and outcome remain
    unchanged.
    """
)

code(
    """
    lab18d_calls = pd.read_csv(LAB18D_CALLS_PATH)
    lab18d_games = pd.read_csv(LAB18D_GAMES_PATH)

    assert lab18d_games["answer"].tolist() == list(RESERVED_ANSWERS)
    assert set(lab18d_calls["answer"]) == set(RESERVED_ANSWERS)
    assert set(lab18d_calls["seed"]) == {45}
    assert set(lab18d_calls["decoder"]) == {"answer-constrained"}
    assert int(lab18d_games["solved"].sum()) == 10
    assert not lab18d_calls.duplicated(["answer", "turn"]).any()
    assert not lab18d_games["answer"].duplicated().any()

    incumbent_rows = []
    incumbent_audit_rows = []
    for answer in RESERVED_ANSWERS:
        answer_index = WORD_TO_INDEX[answer]
        answer_calls = (
            lab18d_calls.loc[lab18d_calls["answer"].eq(answer)]
            .sort_values("turn", kind="stable")
        )
        original = lab18d_games.loc[
            lab18d_games["answer"].eq(answer)
        ].iloc[0]

        opening_pattern = int(PATTERNS[OPENING_INDEX, answer_index])
        candidates = ALL_INDICES[
            PATTERNS[OPENING_INDEX] == opening_pattern
        ]
        assert answer_index in candidates

        closure_turn = None
        sole_answer = None
        for row in answer_calls.itertuples(index=False):
            assert len(candidates) == int(row.candidate_count_before)
            assert answer_index in candidates
            if len(candidates) == 1:
                sole_answer = ANSWERS[int(candidates[0])]
                assert sole_answer == answer
                closure_turn = int(row.turn)
                break

            guess_index = WORD_TO_INDEX[row.guess]
            pattern = int(PATTERNS[guess_index, answer_index])
            assert decode_feedback(pattern) == row.feedback
            candidates = candidates[
                PATTERNS[guess_index, candidates] == pattern
            ]
            assert answer_index in candidates
            assert len(candidates) == int(row.candidate_count_after)

        original_solved = bool(original.solved)
        original_turn = (
            None
            if pd.isna(original.solved_turn)
            else int(original.solved_turn)
        )
        counterfactual_solved = original_solved or closure_turn is not None
        if closure_turn is None:
            counterfactual_turn = original_turn
        elif original_turn is None:
            counterfactual_turn = closure_turn
        else:
            counterfactual_turn = min(original_turn, closure_turn)

        incumbent_rows.append(
            {
                "policy": INCUMBENT_POLICY,
                "answer": answer,
                "solved": counterfactual_solved,
                "solved_turn": counterfactual_turn,
                "penalized_turns": (
                    counterfactual_turn
                    if counterfactual_solved
                    else FAILURE_PENALTY
                ),
                "original_solved": original_solved,
                "original_solved_turn": original_turn,
                "singleton_closure_applied": (
                    closure_turn is not None
                    and (
                        not original_solved
                        or closure_turn < original_turn
                    )
                ),
                "first_singleton_turn": closure_turn,
            }
        )
        incumbent_audit_rows.append(
            {
                "answer": answer,
                "recorded_calls": len(answer_calls),
                "first_singleton_turn": closure_turn,
                "sole_remaining_answer": sole_answer,
                "original_solved": original_solved,
                "counterfactual_solved": counterfactual_solved,
                "counterfactual_solved_turn": counterfactual_turn,
            }
        )

    incumbent_games = pd.DataFrame(incumbent_rows)
    incumbent_audit = pd.DataFrame(incumbent_audit_rows)
    INCUMBENT_RESERVED_SOLVES = int(incumbent_games["solved"].sum())

    assert len(incumbent_games) == 19
    assert INCUMBENT_RESERVED_SOLVES == 15
    assert (
        incumbent_games.loc[
            ~incumbent_games["singleton_closure_applied"],
            ["original_solved", "solved"],
        ]
        .nunique(axis=1)
        .eq(1)
        .all()
    )
    print("historical seed-45 baseline: 10/19")
    print(
        "exact singleton-closure counterfactual:",
        f"{INCUMBENT_RESERVED_SOLVES}/19",
    )
    display(incumbent_audit)
    """
)

md(
    """
    ## 19f.3 Vectorized entropy and deterministic policy trees

    `all_action_entropies` forms one 2,315 by 243 count matrix from
    `PATTERNS[:, candidates]`. It replaces the roughly 25 million Python
    entropy calls that a game-by-game implementation would make.

    The policy tree key is the tuple of remaining answer indices. Every answer
    sharing that state reuses one entropy calculation and one action. Each
    feedback branch is the exact subset selected by the pattern matrix.
    """
)

code(
    """
    ACTION_OFFSETS = (
        np.arange(len(ANSWERS), dtype=np.int64)[:, None] * 243
    )
    X_LOG2_X = np.zeros(len(ANSWERS) + 1, dtype=np.float64)
    positive_counts = np.arange(1, len(ANSWERS) + 1)
    X_LOG2_X[1:] = positive_counts * np.log2(positive_counts)


    def all_action_entropies(candidates: np.ndarray) -> np.ndarray:
        patterns = PATTERNS[:, candidates].astype(np.int64, copy=False)
        encoded = patterns + ACTION_OFFSETS
        counts = np.bincount(
            encoded.ravel(),
            minlength=len(ANSWERS) * 243,
        ).reshape(len(ANSWERS), 243)
        candidate_count = len(candidates)
        return (
            math.log2(candidate_count)
            - X_LOG2_X[counts].sum(axis=1) / candidate_count
        )


    entropy_cache: dict[tuple[int, ...], np.ndarray] = {}


    def state_entropies(candidates: np.ndarray) -> np.ndarray:
        key = tuple(int(index) for index in candidates)
        if key not in entropy_cache:
            entropy_cache[key] = all_action_entropies(candidates)
        return entropy_cache[key]


    def tied_max_indices(
        indices: np.ndarray, entropies: np.ndarray
    ) -> np.ndarray:
        values = entropies[indices]
        best = float(values.max())
        return indices[
            np.isclose(
                values,
                best,
                rtol=0.0,
                atol=ENTROPY_TIE_ATOL,
            )
        ]


    def choose_action(
        policy: str, candidates: np.ndarray
    ) -> dict:
        if len(candidates) == 1:
            action_index = int(candidates[0])
            return {
                "action_index": action_index,
                "action": ANSWERS[action_index],
                "entropy_bits": 0.0,
                "tie_size": 1,
                "candidate_ties_at_open_max": 1,
                "singleton_closure": True,
                "chosen_is_candidate": True,
            }

        entropies = state_entropies(candidates)
        if policy == "candidate-only-entropy":
            tied = tied_max_indices(candidates, entropies)
            action_index = min(
                (int(index) for index in tied),
                key=lambda index: ANSWERS[index],
            )
            candidate_ties = len(tied)
        elif policy == "open-action-entropy":
            tied = tied_max_indices(ALL_INDICES, entropies)
            candidate_tied = np.intersect1d(
                tied, candidates, assume_unique=True
            )
            eligible = candidate_tied if len(candidate_tied) else tied
            action_index = min(
                (int(index) for index in eligible),
                key=lambda index: ANSWERS[index],
            )
            candidate_ties = len(candidate_tied)
        else:
            raise ValueError(f"unknown policy: {policy}")

        return {
            "action_index": action_index,
            "action": ANSWERS[action_index],
            "entropy_bits": float(entropies[action_index]),
            "tie_size": int(len(tied)),
            "candidate_ties_at_open_max": int(candidate_ties),
            "singleton_closure": False,
            "chosen_is_candidate": bool(
                np.any(candidates == action_index)
            ),
        }


    rng = np.random.default_rng(19)
    entropy_probe_errors = []
    for size in (2, 7, 49, 2315):
        candidates = np.sort(
            rng.choice(len(ANSWERS), size=size, replace=False)
        ).astype(np.int32)
        vectorized = all_action_entropies(candidates)
        probe = rng.choice(len(ANSWERS), size=20, replace=False)
        reference = np.asarray(
            [
                EXPERT.entropy(int(index), candidates)
                for index in probe
            ]
        )
        entropy_probe_errors.append(
            float(np.max(np.abs(vectorized[probe] - reference)))
        )

    MAX_ENTROPY_PROBE_ERROR = max(entropy_probe_errors)
    assert MAX_ENTROPY_PROBE_ERROR < 1e-12
    print(
        "vectorized entropy max absolute error:",
        f"{MAX_ENTROPY_PROBE_ERROR:.3e}",
    )
    """
)

code(
    """
    opening_patterns = PATTERNS[OPENING_INDEX]
    opening_roots = []
    opening_covered = []
    for pattern in np.unique(opening_patterns):
        members = ALL_INDICES[opening_patterns == pattern]
        if int(pattern) == SOLVED_PATTERN:
            assert members.tolist() == [OPENING_INDEX]
            opening_covered.extend(int(index) for index in members)
            continue
        opening_roots.append(members)
        opening_covered.extend(int(index) for index in members)

    assert sorted(opening_covered) == list(range(len(ANSWERS)))


    def build_policy_tree(policy: str) -> tuple[dict, float]:
        choices: dict[tuple[int, ...], dict] = {}
        terminal_answers: list[int] = [OPENING_INDEX]
        started = time.perf_counter()

        def visit(candidates: np.ndarray, turn: int) -> None:
            key = tuple(int(index) for index in candidates)
            assert key not in choices
            choice = choose_action(policy, candidates)
            choice = {
                **choice,
                "turn": turn,
                "candidate_count": len(candidates),
                "candidate_indices": key,
            }
            choices[key] = choice

            action_index = int(choice["action_index"])
            patterns = PATTERNS[action_index, candidates]
            branch_members = []
            for pattern in np.unique(patterns):
                child = candidates[patterns == pattern]
                assert len(child) > 0
                assert all(
                    PATTERNS[action_index, int(answer_index)]
                    == int(pattern)
                    for answer_index in child
                )
                branch_members.extend(int(index) for index in child)

                if int(pattern) == SOLVED_PATTERN:
                    assert child.tolist() == [action_index]
                    terminal_answers.append(action_index)
                elif turn == MAX_TURNS:
                    terminal_answers.extend(int(index) for index in child)
                else:
                    assert len(child) < len(candidates)
                    visit(child, turn + 1)

            assert sorted(branch_members) == sorted(key)

        for root in opening_roots:
            visit(root, 2)

        assert len(terminal_answers) == len(ANSWERS)
        assert sorted(terminal_answers) == list(range(len(ANSWERS)))
        assert Counter(terminal_answers).most_common(1)[0][1] == 1
        return choices, time.perf_counter() - started


    policy_choices = {}
    policy_tree_seconds = {}
    benchmark_started = time.perf_counter()
    for policy in SYMBOLIC_POLICIES:
        choices, elapsed = build_policy_tree(policy)
        policy_choices[policy] = choices
        policy_tree_seconds[policy] = elapsed
        print(
            policy,
            "tree nodes:",
            len(choices),
            "seconds:",
            f"{elapsed:.3f}",
        )
    TREE_RUNTIME_SECONDS = time.perf_counter() - benchmark_started
    print("shared entropy states:", len(entropy_cache))
    print("tree runtime seconds:", f"{TREE_RUNTIME_SECONDS:.3f}")
    """
)

md(
    """
    ## 19f.4 Execute and audit every universe game

    Execution follows the frozen tree rather than recalculating decisions per
    answer. The audit checks every transition against the hidden answer's
    feedback, checks that the hidden answer remains in the child candidate set,
    and checks that each policy produces exactly one terminal game row for each
    of the 2,315 answers.
    """
)

code(
    """
    def execute_policy(
        policy: str, choices: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        game_rows = []
        call_rows = []
        for answer_index, answer in enumerate(ANSWERS):
            opening_pattern = int(
                PATTERNS[OPENING_INDEX, answer_index]
            )
            opening_after = ALL_INDICES[
                opening_patterns == opening_pattern
            ]
            assert answer_index in opening_after
            call_rows.append(
                {
                    "policy": policy,
                    "answer": answer,
                    "turn": 1,
                    "state_key": "opening",
                    "guess": OPENING,
                    "feedback": decode_feedback(opening_pattern),
                    "candidate_count_before": len(ANSWERS),
                    "candidate_count_after": len(opening_after),
                    "action_entropy_bits": float(
                        EXPERT.entropy(OPENING_INDEX, ALL_INDICES)
                    ),
                    "chosen_is_candidate": True,
                    "singleton_closure": False,
                    "entropy_tie_size": np.nan,
                    "candidate_ties_at_open_max": np.nan,
                }
            )

            solved_turn = 1 if opening_pattern == SOLVED_PATTERN else None
            candidates = opening_after
            if solved_turn is None:
                for turn in range(2, MAX_TURNS + 1):
                    key = tuple(int(index) for index in candidates)
                    assert key in choices
                    choice = choices[key]
                    assert int(choice["turn"]) == turn
                    action_index = int(choice["action_index"])
                    pattern = int(
                        PATTERNS[action_index, answer_index]
                    )
                    after = candidates[
                        PATTERNS[action_index, candidates] == pattern
                    ]
                    assert answer_index in after
                    assert len(after) > 0
                    call_rows.append(
                        {
                            "policy": policy,
                            "answer": answer,
                            "turn": turn,
                            "state_key": sha256_text(
                                ",".join(str(index) for index in key)
                            ),
                            "guess": choice["action"],
                            "feedback": decode_feedback(pattern),
                            "candidate_count_before": len(candidates),
                            "candidate_count_after": len(after),
                            "action_entropy_bits": choice[
                                "entropy_bits"
                            ],
                            "chosen_is_candidate": choice[
                                "chosen_is_candidate"
                            ],
                            "singleton_closure": choice[
                                "singleton_closure"
                            ],
                            "entropy_tie_size": choice["tie_size"],
                            "candidate_ties_at_open_max": choice[
                                "candidate_ties_at_open_max"
                            ],
                        }
                    )
                    if pattern == SOLVED_PATTERN:
                        assert answer_index == action_index
                        candidates = after
                        solved_turn = turn
                        break
                    candidates = after

            game_rows.append(
                {
                    "policy": policy,
                    "answer": answer,
                    "solved": solved_turn is not None,
                    "solved_turn": solved_turn,
                    "penalized_turns": (
                        solved_turn
                        if solved_turn is not None
                        else FAILURE_PENALTY
                    ),
                    "calls": sum(
                        row["policy"] == policy
                        and row["answer"] == answer
                        for row in call_rows
                    ),
                    "final_candidate_count": len(candidates),
                }
            )

        games = pd.DataFrame(game_rows)
        calls = pd.DataFrame(call_rows)
        assert games["answer"].tolist() == list(ANSWERS)
        assert not games["answer"].duplicated().any()
        assert set(calls["answer"]) == set(ANSWERS)
        assert not calls.duplicated(["answer", "turn"]).any()
        assert int(games["solved"].sum()) + int(
            (~games["solved"]).sum()
        ) == len(ANSWERS)
        return games, calls


    universe_games = {}
    universe_calls = {}
    execution_seconds = {}
    for policy in SYMBOLIC_POLICIES:
        started = time.perf_counter()
        games, calls = execute_policy(
            policy, policy_choices[policy]
        )
        execution_seconds[policy] = time.perf_counter() - started
        universe_games[policy] = games
        universe_calls[policy] = calls
        stem = policy
        atomic_csv(
            games,
            RESULTS_DIR / f"{stem}-universe-games.csv",
        )
        atomic_csv(
            calls,
            RESULTS_DIR / f"{stem}-universe-calls.csv",
        )
        print(
            policy,
            "games:",
            len(games),
            "calls:",
            len(calls),
            "execution seconds:",
            f"{execution_seconds[policy]:.3f}",
        )

    atomic_csv(
        incumbent_games,
        RESULTS_DIR
        / "incumbent-singleton-closure-reserved-games.csv",
    )
    atomic_csv(
        incumbent_audit,
        RESULTS_DIR
        / "incumbent-singleton-closure-audit.csv",
    )
    """
)

md(
    """
    ## 19f.5 Reserved and universe results

    The reserved symbolic rows are exact subsets of the universe execution.
    This keeps the candidate prior and action space fixed at all 2,315 answers
    while allowing direct answer-paired comparison with Lab 18d.
    """
)

code(
    """
    def turn_distribution(games: pd.DataFrame) -> dict[str, int]:
        distribution = {
            f"turn_{turn}": int(
                games["solved_turn"].eq(turn).sum()
            )
            for turn in range(1, MAX_TURNS + 1)
        }
        distribution["failures"] = int((~games["solved"]).sum())
        return distribution


    def summarize_policy(
        policy: str,
        scope: str,
        games: pd.DataFrame,
        calls: pd.DataFrame | None,
        tree_choices: dict | None,
    ) -> dict:
        solved = games.loc[games["solved"]]
        summary = {
            "scope": scope,
            "policy": policy,
            "answers": len(games),
            "solved_count": int(games["solved"].sum()),
            "failure_count": int((~games["solved"]).sum()),
            "solve_rate": float(games["solved"].mean()),
            "mean_solved_turns": float(
                solved["solved_turn"].mean()
            ),
            "penalized_total_turns": int(
                games["penalized_turns"].sum()
            ),
            **turn_distribution(games),
        }
        if calls is None:
            summary.update(
                {
                    "singleton_closures": int(
                        games["singleton_closure_applied"].sum()
                    ),
                    "exploratory_noncandidate_states": np.nan,
                    "exploratory_noncandidate_state_rate": np.nan,
                    "entropy_tied_states": np.nan,
                    "mean_entropy_tie_size": np.nan,
                    "max_entropy_tie_size": np.nan,
                    "mean_candidate_count_decision_states": np.nan,
                    "median_candidate_count_decision_states": np.nan,
                    "max_candidate_count_decision_states": np.nan,
                    "tree_nodes": np.nan,
                    "tree_runtime_seconds": np.nan,
                    "execution_runtime_seconds": np.nan,
                }
            )
            return summary

        decision_calls = calls.loc[calls["turn"].ge(2)]
        if scope == "reserved":
            visited_state_keys = set(decision_calls["state_key"])
            state_values = [
                choice
                for key, choice in tree_choices.items()
                if sha256_text(
                    ",".join(str(index) for index in key)
                )
                in visited_state_keys
            ]
        else:
            state_values = list(tree_choices.values())
        state_rows = pd.DataFrame(state_values)
        summary.update(
            {
                "singleton_closures": int(
                    decision_calls["singleton_closure"].sum()
                ),
                "exploratory_noncandidate_states": int(
                    (~state_rows["chosen_is_candidate"]).sum()
                ),
                "exploratory_noncandidate_state_rate": float(
                    (~state_rows["chosen_is_candidate"]).mean()
                ),
                "entropy_tied_states": int(
                    state_rows["tie_size"].gt(1).sum()
                ),
                "mean_entropy_tie_size": float(
                    state_rows["tie_size"].mean()
                ),
                "max_entropy_tie_size": int(
                    state_rows["tie_size"].max()
                ),
                "mean_candidate_count_decision_states": float(
                    state_rows["candidate_count"].mean()
                ),
                "median_candidate_count_decision_states": float(
                    state_rows["candidate_count"].median()
                ),
                "max_candidate_count_decision_states": int(
                    state_rows["candidate_count"].max()
                ),
                "tree_nodes": len(state_rows),
                "tree_runtime_seconds": policy_tree_seconds[policy],
                "execution_runtime_seconds": execution_seconds[
                    policy
                ],
            }
        )
        return summary


    reserved_summary_rows = [
        summarize_policy(
            INCUMBENT_POLICY,
            "reserved",
            incumbent_games,
            None,
            None,
        )
    ]
    universe_summary_rows = []
    reserved_games = {}
    for policy in SYMBOLIC_POLICIES:
        games = universe_games[policy]
        calls = universe_calls[policy]
        reserved = games.loc[
            games["answer"].isin(RESERVED_ANSWERS)
        ].copy()
        reserved = (
            reserved.set_index("answer")
            .loc[list(RESERVED_ANSWERS)]
            .reset_index()
        )
        reserved_call_rows = calls.loc[
            calls["answer"].isin(RESERVED_ANSWERS)
        ].copy()
        assert reserved["answer"].tolist() == list(RESERVED_ANSWERS)
        reserved_games[policy] = reserved
        reserved_summary_rows.append(
            summarize_policy(
                policy,
                "reserved",
                reserved,
                reserved_call_rows,
                policy_choices[policy],
            )
        )
        universe_summary_rows.append(
            summarize_policy(
                policy,
                "universe",
                games,
                calls,
                policy_choices[policy],
            )
        )

    reserved_summary = pd.DataFrame(reserved_summary_rows)
    universe_summary = pd.DataFrame(universe_summary_rows)
    atomic_csv(
        reserved_summary,
        RESULTS_DIR / "reserved-summary.csv",
    )
    atomic_csv(
        universe_summary,
        RESULTS_DIR / "universe-summary.csv",
    )

    display(reserved_summary)
    display(universe_summary)
    """
)

md(
    """
    ## 19f.6 Shared-state policy disagreement

    A disagreement is counted only when both policy trees visit the exact same
    candidate-index tuple. The persisted row carries both actions and their
    entropy values on that state. Open-action exploratory choices are also
    counted directly from its state table.
    """
)

code(
    """
    candidate_choices = policy_choices["candidate-only-entropy"]
    open_choices = policy_choices["open-action-entropy"]
    shared_keys = sorted(
        set(candidate_choices) & set(open_choices),
        key=lambda key: (len(key), key),
    )
    disagreement_rows = []
    for key in shared_keys:
        candidate_choice = candidate_choices[key]
        open_choice = open_choices[key]
        entropies = state_entropies(np.asarray(key, dtype=np.int32))
        disagreement_rows.append(
            {
                "state_key": sha256_text(
                    ",".join(str(index) for index in key)
                ),
                "candidate_indices": " ".join(
                    str(index) for index in key
                ),
                "candidate_words": " ".join(
                    ANSWERS[index] for index in key
                ),
                "candidate_count": len(key),
                "candidate_policy_turn": candidate_choice["turn"],
                "open_policy_turn": open_choice["turn"],
                "candidate_action": candidate_choice["action"],
                "open_action": open_choice["action"],
                "candidate_action_entropy_bits": float(
                    entropies[candidate_choice["action_index"]]
                ),
                "open_action_entropy_bits": float(
                    entropies[open_choice["action_index"]]
                ),
                "open_action_is_candidate": open_choice[
                    "chosen_is_candidate"
                ],
                "candidate_tie_size": candidate_choice["tie_size"],
                "open_tie_size": open_choice["tie_size"],
                "actions_differ": (
                    candidate_choice["action"]
                    != open_choice["action"]
                ),
            }
        )

    paired_states = pd.DataFrame(disagreement_rows)
    non_singleton_shared = paired_states.loc[
        paired_states["candidate_count"].gt(1)
    ]
    disagreement_summary = pd.DataFrame(
        [
            {
                "shared_states": len(paired_states),
                "shared_non_singleton_states": len(
                    non_singleton_shared
                ),
                "differing_states": int(
                    paired_states["actions_differ"].sum()
                ),
                "differing_state_rate": float(
                    paired_states["actions_differ"].mean()
                ),
                "differing_non_singleton_states": int(
                    non_singleton_shared["actions_differ"].sum()
                ),
                "differing_non_singleton_state_rate": float(
                    non_singleton_shared["actions_differ"].mean()
                ),
                "open_noncandidate_shared_states": int(
                    (~paired_states["open_action_is_candidate"]).sum()
                ),
                "open_noncandidate_shared_state_rate": float(
                    (~paired_states["open_action_is_candidate"]).mean()
                ),
            }
        ]
    )

    atomic_csv(
        paired_states,
        RESULTS_DIR / "disagreement-states.csv",
    )
    atomic_csv(
        disagreement_summary,
        RESULTS_DIR / "disagreement-summary.csv",
    )
    display(disagreement_summary)
    display(
        paired_states.loc[paired_states["actions_differ"]].head(20)
    )
    """
)

md(
    """
    ## 19f.7 Frozen verdict

    The full-universe ordering is lexicographic: more solves wins; with equal
    solves, fewer penalized turns wins. The 19-answer incumbent comparison is a
    separate gate. The no-justification outcome has first precedence because a
    relative win between two teachers is not useful if neither improves on the
    exact singleton-closure incumbent.
    """
)

code(
    """
    summary_by_policy = universe_summary.set_index("policy")
    candidate_summary = summary_by_policy.loc[
        "candidate-only-entropy"
    ]
    open_summary = summary_by_policy.loc["open-action-entropy"]


    def quality_beats(left: pd.Series, right: pd.Series) -> bool:
        left_solved = int(left["solved_count"])
        right_solved = int(right["solved_count"])
        if left_solved != right_solved:
            return left_solved > right_solved
        return int(left["penalized_total_turns"]) < int(
            right["penalized_total_turns"]
        )


    open_beats_candidate = quality_beats(
        open_summary, candidate_summary
    )
    candidate_beats_open = quality_beats(
        candidate_summary, open_summary
    )
    full_universe_exact_tie = bool(
        int(open_summary["solved_count"])
        == int(candidate_summary["solved_count"])
        and int(open_summary["penalized_total_turns"])
        == int(candidate_summary["penalized_total_turns"])
    )
    candidate_better_or_tied = (
        candidate_beats_open or full_universe_exact_tie
    )

    reserved_solved = {
        policy: int(games["solved"].sum())
        for policy, games in reserved_games.items()
    }
    open_beats_incumbent_reserved = (
        reserved_solved["open-action-entropy"]
        > INCUMBENT_RESERVED_SOLVES
    )
    candidate_beats_incumbent_reserved = (
        reserved_solved["candidate-only-entropy"]
        > INCUMBENT_RESERVED_SOLVES
    )
    neither_symbolic_beats_incumbent = not (
        open_beats_incumbent_reserved
        or candidate_beats_incumbent_reserved
    )

    one_ply_rule_trigger = neither_symbolic_beats_incumbent
    open_rule_trigger = bool(
        not one_ply_rule_trigger
        and open_beats_candidate
        and open_beats_incumbent_reserved
    )
    candidate_rule_trigger = bool(
        not one_ply_rule_trigger
        and not open_rule_trigger
        and candidate_better_or_tied
    )
    fallback_candidate_rule_trigger = bool(
        not one_ply_rule_trigger
        and not open_rule_trigger
        and not candidate_rule_trigger
    )

    if one_ply_rule_trigger:
        OUTCOME = "one_ply_entropy_not_justified"
        GATE_C_ACTION = "do_not_run_gate_c"
    elif open_rule_trigger:
        OUTCOME = "open_teacher_worth_testing"
        GATE_C_ACTION = "lab19e_gate_c_may_proceed_unchanged"
    else:
        OUTCOME = "use_candidate_only_teacher"
        GATE_C_ACTION = (
            "do_not_run_current_gate_c_until_target_is_changed"
        )

    decision = {
        "outcome": OUTCOME,
        "gate_c_action": GATE_C_ACTION,
        "full_universe": {
            policy: {
                "solved_count": int(
                    summary_by_policy.loc[policy, "solved_count"]
                ),
                "penalized_total_turns": int(
                    summary_by_policy.loc[
                        policy, "penalized_total_turns"
                    ]
                ),
            }
            for policy in SYMBOLIC_POLICIES
        },
        "reserved_solved_count": {
            INCUMBENT_POLICY: INCUMBENT_RESERVED_SOLVES,
            **reserved_solved,
        },
        "booleans": {
            "open_beats_candidate_full_universe": (
                open_beats_candidate
            ),
            "candidate_beats_open_full_universe": (
                candidate_beats_open
            ),
            "full_universe_exact_tie": full_universe_exact_tie,
            "candidate_better_or_tied_full_universe": (
                candidate_better_or_tied
            ),
            "open_beats_incumbent_reserved": (
                open_beats_incumbent_reserved
            ),
            "candidate_beats_incumbent_reserved": (
                candidate_beats_incumbent_reserved
            ),
            "neither_symbolic_beats_incumbent_reserved": (
                neither_symbolic_beats_incumbent
            ),
            "one_ply_rule_trigger": one_ply_rule_trigger,
            "open_rule_trigger": open_rule_trigger,
            "candidate_rule_trigger": candidate_rule_trigger,
            "fallback_candidate_rule_trigger": (
                fallback_candidate_rule_trigger
            ),
        },
    }

    assert sum(
        [
            one_ply_rule_trigger,
            open_rule_trigger,
            candidate_rule_trigger,
            fallback_candidate_rule_trigger,
        ]
    ) == 1
    print(json.dumps(decision, indent=2))
    """
)

code(
    """
    failure_frames = []
    for policy in SYMBOLIC_POLICIES:
        universe_failures = universe_games[policy].loc[
            ~universe_games[policy]["solved"]
        ].copy()
        universe_failures.insert(0, "scope", "universe")
        failure_frames.append(universe_failures)

        reserved_failures = reserved_games[policy].loc[
            ~reserved_games[policy]["solved"]
        ].copy()
        reserved_failures.insert(0, "scope", "reserved")
        failure_frames.append(reserved_failures)

    incumbent_failures = incumbent_games.loc[
        ~incumbent_games["solved"]
    ].copy()
    incumbent_failures.insert(0, "scope", "reserved")
    failure_frames.append(incumbent_failures)
    failures = pd.concat(failure_frames, ignore_index=True, sort=False)
    atomic_csv(failures, RESULTS_DIR / "failures.csv")

    TOTAL_RUNTIME_SECONDS = time.perf_counter() - benchmark_started
    NOTEBOOK_RUNTIME_SECONDS = time.perf_counter() - NOTEBOOK_STARTED
    run = {
        "experiment": "Lab 19f symbolic teacher benchmark",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_sha256": source_sha256,
        "cpu_only": True,
        "torch_imported": "torch" in sys.modules,
        "answers": len(ANSWERS),
        "reserved_answers": list(RESERVED_ANSWERS),
        "opening": OPENING,
        "max_turns": MAX_TURNS,
        "failure_penalty_turns": FAILURE_PENALTY,
        "entropy_tie_atol": ENTROPY_TIE_ATOL,
        "max_vectorized_entropy_error": (
            MAX_ENTROPY_PROBE_ERROR
        ),
        "cached_entropy_states_after_disagreement": len(entropy_cache),
        "tree_nodes": {
            policy: len(policy_choices[policy])
            for policy in SYMBOLIC_POLICIES
        },
        "runtime_seconds": {
            "policy_tree": policy_tree_seconds,
            "tree_total": TREE_RUNTIME_SECONDS,
            "game_execution": execution_seconds,
            "benchmark_total_from_tree_start": TOTAL_RUNTIME_SECONDS,
            "notebook_total": NOTEBOOK_RUNTIME_SECONDS,
        },
        "decision": decision,
        "artifacts": sorted(
            {
                path.name
                for path in RESULTS_DIR.iterdir()
                if path.is_file()
            }
            | {"run.json"}
        ),
    }
    atomic_json(run, RESULTS_DIR / "run.json")

    assert run["torch_imported"] is False
    assert sha256_text(
        (RESULTS_DIR / "lab19f-preregistration.json")
        .read_text()
        .rstrip("\\n")
    ) == PREREGISTRATION_SHA256
    assert all(
        len(universe_games[policy]) == len(ANSWERS)
        for policy in SYMBOLIC_POLICIES
    )

    print("failures:", len(failures))
    print("total benchmark seconds:", f"{TOTAL_RUNTIME_SECONDS:.3f}")
    print("total notebook seconds:", f"{NOTEBOOK_RUNTIME_SECONDS:.3f}")
    print("verdict:", OUTCOME)
    print("Gate C:", GATE_C_ACTION)
    print("written to:", RESULTS_DIR)
    """
)

md(
    """
    ## Lab 19f checkpoint

    Read the result in this order:

    1. Did either symbolic policy beat the 15/19 exact incumbent closure
       counterfactual?
    2. Which symbolic policy won the full-universe lexicographic comparison?
    3. How often did open-action entropy leave the candidate set, and did that
       freedom improve solves or only change tie-equivalent actions?
    4. Apply the frozen verdict before changing or running Lab 19e Gate C.

    These results test a deterministic one-ply teacher under a uniform
    candidate prior. They do not show that a learned model can imitate it, that
    the teacher remains optimal under model error, or that the 2,315-answer
    action list contains every useful exploratory Wordle guess.
    """
)


for index, cell in enumerate(cells):
    cell["id"] = f"lab19f-{index:02d}-{cell['cell_type']}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = Path("notebooks/19f_symbolic_teacher_benchmark.ipynb")
path.write_text(json.dumps(notebook, indent=1))
print(f"wrote {path} with {len(cells)} cells")
