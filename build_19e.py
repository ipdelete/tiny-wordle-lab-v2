"""Generate notebooks/19e_simulator_ranked_policy.ipynb."""

import json
import textwrap
from pathlib import Path

cells = []


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
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
        }
    )


# The process cap is deliberately the first notebook cell.
code(
    """
    MEMORY_CAP_GIB = 128.0

    import os
    import torch

    if torch.backends.mps.is_available():
        total_gib = torch.mps.recommended_max_memory() / 1024**3
        torch.mps.set_per_process_memory_fraction(MEMORY_CAP_GIB / total_gib)
        print(f"MPS cap: {MEMORY_CAP_GIB:.0f} GiB of {total_gib:.0f} GiB")

    RUN_MODEL = os.environ.get("LAB19E_RUN_MODEL", "0") == "1"
    RUN_TRAINING = (
        RUN_MODEL
        and os.environ.get("LAB19E_RUN_TRAINING", "1") == "1"
    )
    REUSE_GATE_B = (
        RUN_MODEL
        and os.environ.get("LAB19E_REUSE_GATE_B", "0") == "1"
    )
    print("LAB19E_RUN_MODEL:", RUN_MODEL)
    print("LAB19E_RUN_TRAINING:", RUN_TRAINING)
    print("LAB19E_REUSE_GATE_B:", REUSE_GATE_B)
    """
)

md(
    """
    # Lab 19e - Simulator-ranked answer-constrained policy

    Lab 18b found useful candidate ranking in the seed-45 B-structured adapter,
    but its ranking did not follow the entropy teacher. Lab 19 then learned its
    fixed twelve-action objectives while losing the full 2,315-answer ranking.
    Lab 19d identified the missing comparison: a fixed local support cannot
    constrain generic winners that emerge later.

    This lab asks one narrow question. Can one-ply simulator labels improve
    ranking across the answer-constrained 2,315-action policy without
    destroying the incumbent's broader answer ranking?

    This is supervised ranking, not GRPO. The treatment learns pairwise order
    from frozen one-ply entropy utilities. Open-teacher actions represent
    valuable exploratory guesses, online full-list actions track the moving
    model, and a truncated-support KL constrains movement on the same support.
    A matched preservation-only control isolates the optimizer, weight decay,
    support construction, and truncated-support KL mechanics.
    """
)

md(
    """
    ## Gate A: preregistration and data audit

    Gate A loads no language model. It reconstructs remaining candidates from
    the exact Wordle pattern matrix, selects the fixed train/dev states, computes
    all 2,315 action entropies for each state, audits pair coverage and teacher
    diversity, and writes the preregistration under `results/lab19e/`.

    Later gates run only with `LAB19E_RUN_MODEL=1`. A full run must use:

    ```bash
    scripts/memguard.py --min-free 64 -- uv run jupyter nbconvert \
        --to notebook --execute --inplace \
        notebooks/19e_simulator_ranked_policy.ipynb
    ```
    """
)

code(
    """
    import gc
    import hashlib
    import json
    import math
    import random
    import time
    from collections import Counter
    from contextlib import contextmanager
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from IPython.display import display

    from tiny_wordle.benchmark import DEFAULT_EVAL_ANSWERS
    from tiny_wordle.expert import EntropyExpert
    from tiny_wordle.game import Turn, score_string
    from tiny_wordle.representation import (
        candidate_indices_from_history,
        parse_state_key,
        structured_next_guess_prompt,
    )

    ROOT = Path.cwd()
    if not (ROOT / "data").exists():
        ROOT = ROOT.parent
    DATA_DIR = ROOT / "data"
    GENERATED_DIR = DATA_DIR / "generated"
    RESULTS_DIR = ROOT / "results" / "lab19e"
    CHECKPOINT_ROOT = ROOT / "checkpoints"
    LAB18B_DIR = ROOT / "results" / "lab18b"
    LAB18D_DIR = ROOT / "results" / "lab18d"
    LAB20_DIR = ROOT / "results" / "lab20"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_ID = "Qwen/Qwen3-0.6B"
    SEED = 45
    OPENING = "RAISE"
    MAX_TURNS = 6
    INCUMBENT = (
        CHECKPOINT_ROOT
        / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed45"
    )
    ARMS = ("preservation-control", "entropy-ranking")
    TRAINED_CHECKPOINTS = {
        arm: CHECKPOINT_ROOT / f"qwen3-0.6b-wordle-lab19e-{arm}-seed45"
        for arm in ARMS
    }

    STRUCTURED_FILES = {
        "train": GENERATED_DIR / "wordle-part2-structured-train.jsonl",
        "dev": GENERATED_DIR / "wordle-part2-structured-dev.jsonl",
    }
    TRAIN_STATES = 128
    DEV_STATES = 64
    MIN_CANDIDATES = 3
    MAX_CANDIDATES = 16
    CANDIDATE_BUCKETS = ("3-4", "5-8", "9-16")
    BUCKET_QUOTAS = {
        "train": {"3-4": 64, "5-8": 40, "9-16": 24},
        "dev": {"3-4": 32, "5-8": 20, "9-16": 12},
    }
    SELECTION_SEED = 1905
    ORDER_SEED = 19051

    UTILITY_DECIMALS = 9
    OPEN_TEACHER_TOP_K = 16
    INCUMBENT_TOP_K = 16
    CURRENT_TOP_K = 16
    MAX_SUPPORT_SIZE = (
        MAX_CANDIDATES
        + OPEN_TEACHER_TOP_K
        + INCUMBENT_TOP_K
        + CURRENT_TOP_K
    )
    REFRESH_CADENCE = 16
    EPOCHS = 2
    UPDATES = TRAIN_STATES * EPOCHS
    DRIFT_CHECK_EVERY = 32

    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_FRACTION = 0.05
    GRAD_CLIP = 1.0
    TRUNCATED_SUPPORT_KL_WEIGHT = 0.25
    TRUNCATED_SUPPORT_KL_TEMPERATURE = 1.0

    CHUNK_SIZE = 256
    SOAK_ITERATIONS = 40
    MEMORY_ABORT_GIB = 96.0

    CANDIDATE_MASS_RATIO_FLOOR = 0.85
    BEST_RANK_MULTIPLIER = 4.0
    BEST_RANK_FLOOR = 10.0
    WINNER_SHARE_FLOOR = 0.50
    WINNER_SHARE_MARGIN = 0.25
    MIN_DEV_REGRET_IMPROVEMENT_BITS = 0.10
    MIN_CONTROL_ADJUSTED_REGRET_IMPROVEMENT_BITS = 0.10
    MIN_ADAPTER_RELATIVE_DELTA = 0.005

    RESERVED_ANSWERS = tuple(DEFAULT_EVAL_ANSWERS)
    assert len(RESERVED_ANSWERS) == 19
    assert sum(BUCKET_QUOTAS["train"].values()) == TRAIN_STATES
    assert sum(BUCKET_QUOTAS["dev"].values()) == DEV_STATES
    assert UPDATES % REFRESH_CADENCE == 0
    assert UPDATES % DRIFT_CHECK_EVERY == 0
    assert set(ARMS) == {"preservation-control", "entropy-ranking"}


    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def sha256_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()


    def atomic_write(text: str, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


    def atomic_json(value: dict, path: Path) -> None:
        atomic_write(
            json.dumps(value, indent=2, sort_keys=True, default=str),
            path,
        )


    def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)


    def jsonl_text(records: list[dict]) -> str:
        return "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\\n"
            for record in records
        )
    """
)

code(
    """
    ANSWERS = tuple(
        line.strip().upper()
        for line in (DATA_DIR / "wordle-answers-original.txt").read_text().splitlines()
        if line.strip()
    )
    PATTERNS = np.load(DATA_DIR / "wordle-patterns-original-2315.npy")
    EXPERT = EntropyExpert(list(ANSWERS), PATTERNS)
    WORD_TO_INDEX = EXPERT.word_to_index
    ALL_INDICES = EXPERT.all_indices
    RESERVED_INDICES = {WORD_TO_INDEX[word] for word in RESERVED_ANSWERS}
    assert len(ANSWERS) == 2315
    assert PATTERNS.shape == (2315, 2315)

    structured_manifest = json.loads(
        (GENERATED_DIR / "wordle-part2-structured-manifest.json").read_text()
    )
    source_hashes = {
        split: sha256_file(path) for split, path in STRUCTURED_FILES.items()
    }
    assert source_hashes["train"] == structured_manifest["structured_sha256"]["train"]
    assert source_hashes["dev"] == structured_manifest["structured_sha256"]["validation"]

    anchor_file = LAB20_DIR / "anchor-states.csv"
    if not anchor_file.exists():
        raise FileNotFoundError(
            "Lab 20 anchor-states.csv is required before state selection"
        )
    anchor_source = pd.read_csv(anchor_file)
    ANCHOR_STATE_KEYS = frozenset(anchor_source["state_key"])
    assert len(anchor_source) == 20
    assert len(ANCHOR_STATE_KEYS) == 20

    lab18d_manifest = json.loads((LAB18D_DIR / "lab18d-run.json").read_text())
    incumbent_file = INCUMBENT / "adapter_model.safetensors"
    baseline_game_file = LAB18D_DIR / "seed45-answer-constrained-games.csv"
    if not incumbent_file.exists():
        raise FileNotFoundError(f"missing seed-45 incumbent: {incumbent_file}")
    if not baseline_game_file.exists():
        raise FileNotFoundError(
            f"missing Lab 18d seed-45 baseline: {baseline_game_file}"
        )
    incumbent_sha256 = sha256_file(incumbent_file)
    assert incumbent_sha256 == lab18d_manifest["checkpoint_sha256"]["45"]
    lab18d_games = pd.read_csv(baseline_game_file)
    assert len(lab18d_games) == 19
    assert int(lab18d_games["solved"].sum()) == 10


    def candidate_bucket(count: int) -> str:
        if 3 <= count <= 4:
            return "3-4"
        if 5 <= count <= 8:
            return "5-8"
        if 9 <= count <= 16:
            return "9-16"
        raise ValueError(f"candidate count outside the frozen bound: {count}")


    def selection_digest(split: str, state_key: str) -> str:
        return sha256_text(f"lab19e|{SELECTION_SEED}|{split}|{state_key}")


    def order_digest(split: str, state_key: str) -> str:
        return sha256_text(f"lab19e-order|{ORDER_SEED}|{split}|{state_key}")


    def reachable_by_reserved_answer(candidate_indices: np.ndarray) -> bool:
        return bool(RESERVED_INDICES.intersection(map(int, candidate_indices)))


    def load_state_pool(split: str) -> pd.DataFrame:
        rows = [
            json.loads(line)
            for line in STRUCTURED_FILES[split].read_text().splitlines()
        ]
        unique = {}
        for row in rows:
            if row["task"] == "NEXT_GUESS":
                unique.setdefault(row["state_key"], row)
        records = []
        for state_key, row in unique.items():
            history = parse_state_key(state_key)
            candidates = candidate_indices_from_history(
                history,
                ANSWERS,
                PATTERNS,
                expert=EXPERT,
            )
            assert len(candidates) == int(row["candidate_count"])
            assert structured_next_guess_prompt(
                history, len(candidates)
            ) == row["prompt"]
            records.append(
                {
                    "split": split,
                    "state_key": state_key,
                    "prompt": row["prompt"],
                    "turn": int(row["turn"]),
                    "candidate_count": int(len(candidates)),
                    "candidate_indices": [int(index) for index in candidates],
                    "reserved_reachable": reachable_by_reserved_answer(candidates),
                }
            )
        return pd.DataFrame(records)


    source_frames = {
        split: load_state_pool(split) for split in ("train", "dev")
    }
    assert not (
        set(source_frames["train"]["state_key"])
        & set(source_frames["dev"]["state_key"])
    )

    distribution_rows = []
    eligible_frames = {}
    for split, frame in source_frames.items():
        broad = frame.loc[frame["candidate_count"] >= MIN_CANDIDATES].copy()
        bounded = broad.loc[
            broad["candidate_count"].between(
                MIN_CANDIDATES, MAX_CANDIDATES
            )
        ].copy()
        without_reserved = bounded.loc[~bounded["reserved_reachable"]].copy()
        eligible = without_reserved.loc[
            ~without_reserved["state_key"].isin(ANCHOR_STATE_KEYS)
        ].copy()
        eligible["candidate_bucket"] = eligible["candidate_count"].map(
            candidate_bucket
        )
        eligible_frames[split] = eligible.reset_index(drop=True)
        distribution_rows.append(
            {
                "split": split,
                "all_unique_next_guess": len(frame),
                "broad_states": len(broad),
                "bounded_broad_states": len(bounded),
                "bounded_broad_fraction": len(bounded) / len(broad),
                "eligible_after_all_exclusions": len(eligible),
                "eligible_before_anchor_exclusion": len(without_reserved),
                "lab20_anchors_excluded": int(
                    without_reserved["state_key"].isin(ANCHOR_STATE_KEYS).sum()
                ),
                "max_unbounded_candidate_count": int(
                    broad["candidate_count"].max()
                ),
                "frozen_candidate_bound": MAX_CANDIDATES,
            }
        )

    source_distribution = pd.DataFrame(distribution_rows)
    assert (
        source_distribution["bounded_broad_fraction"] >= 0.90
    ).all(), "the frozen bound must retain at least 90% of broad states"


    def select_states(split: str) -> pd.DataFrame:
        eligible = eligible_frames[split]
        selected_parts = []
        for bucket in CANDIDATE_BUCKETS:
            pool = eligible.loc[
                eligible["candidate_bucket"] == bucket
            ].copy()
            quota = BUCKET_QUOTAS[split][bucket]
            turns = sorted(pool["turn"].unique())
            assert quota >= len(turns)
            first_per_turn = []
            for turn in turns:
                turn_pool = pool.loc[pool["turn"] == turn].copy()
                turn_pool["selection_digest"] = turn_pool["state_key"].map(
                    lambda key: selection_digest(
                        f"{split}|{bucket}|turn{turn}", key
                    )
                )
                first_per_turn.append(
                    turn_pool.sort_values(
                        "selection_digest", kind="stable"
                    ).iloc[0]
                )
            first = pd.DataFrame(first_per_turn)
            remaining = pool.loc[
                ~pool["state_key"].isin(first["state_key"])
            ].copy()
            remaining["selection_digest"] = remaining["state_key"].map(
                lambda key: selection_digest(f"{split}|{bucket}|fill", key)
            )
            fill = remaining.sort_values(
                "selection_digest", kind="stable"
            ).head(quota - len(first))
            chosen = pd.concat(
                [first.drop(columns=["selection_digest"]), fill.drop(
                    columns=["selection_digest"]
                )],
                ignore_index=True,
            )
            assert len(chosen) == quota
            assert set(turns) <= set(chosen["turn"])
            selected_parts.append(chosen)
        selected = pd.concat(selected_parts, ignore_index=True)
        selected["order_digest"] = selected["state_key"].map(
            lambda key: order_digest(split, key)
        )
        selected = selected.sort_values(
            "order_digest", kind="stable"
        ).drop(columns=["order_digest"]).reset_index(drop=True)
        expected = TRAIN_STATES if split == "train" else DEV_STATES
        assert len(selected) == expected
        assert selected["state_key"].is_unique
        return selected


    selected_frames = {
        split: select_states(split) for split in ("train", "dev")
    }
    assert not (
        set(selected_frames["train"]["state_key"])
        & set(selected_frames["dev"]["state_key"])
    )
    assert ANCHOR_STATE_KEYS.isdisjoint(
        set(selected_frames["train"]["state_key"])
    )
    assert ANCHOR_STATE_KEYS.isdisjoint(
        set(selected_frames["dev"]["state_key"])
    )
    """
)

code(
    """
    def all_action_entropies(candidate_indices: np.ndarray) -> np.ndarray:
        values = np.array(
            [
                EXPERT.entropy(int(index), candidate_indices)
                for index in ALL_INDICES
            ],
            dtype=np.float64,
        )
        return np.round(values, UTILITY_DECIMALS)


    def entropy_order(
        entropies: np.ndarray, state_key: str
    ) -> np.ndarray:
        def state_tie_break(index: int) -> str:
            return sha256_text(
                f"{state_key}\\0{ANSWERS[index]}"
            )

        return np.array(
            sorted(
                range(len(ANSWERS)),
                key=lambda index: (
                    -entropies[index],
                    state_tie_break(index),
                ),
            ),
            dtype=np.int64,
        )


    def deduplicated_union(*groups) -> list[int]:
        support = []
        seen = set()
        for group in groups:
            for value in group:
                index = int(value)
                if index not in seen:
                    support.append(index)
                    seen.add(index)
        return support


    def pair_audit(
        support: list[int], all_action_entropy_bits: np.ndarray
    ) -> dict:
        utilities = all_action_entropy_bits[np.asarray(support, dtype=np.int64)]
        valid_pair_count = 0
        tie_pair_count = 0
        utility_gaps = []
        for left in range(len(support)):
            for right in range(left + 1, len(support)):
                gap = abs(float(utilities[left] - utilities[right]))
                if gap == 0.0:
                    tie_pair_count += 1
                else:
                    valid_pair_count += 1
                    utility_gaps.append(gap)
        assert valid_pair_count + tie_pair_count == (
            len(support) * (len(support) - 1) // 2
        )
        return {
            "valid_pair_count": valid_pair_count,
            "tie_pair_count": tie_pair_count,
            "mean_utility_gap_bits": (
                float(np.mean(utility_gaps)) if utility_gaps else 0.0
            ),
        }


    def target_record(row) -> dict:
        candidates = np.array(row.candidate_indices, dtype=np.int64)
        candidate_set = set(map(int, candidates))
        entropies = all_action_entropies(candidates)
        order = entropy_order(entropies, row.state_key)
        open_teacher_top = order[:OPEN_TEACHER_TOP_K]
        open_teacher_index = int(open_teacher_top[0])
        global_maximum = float(entropies[open_teacher_index])
        global_tie_count = int(
            np.count_nonzero(entropies == global_maximum)
        )
        candidate_entropies = entropies[candidates]
        candidate_maximum = float(candidate_entropies.max())
        candidate_tie_count = int(
            np.count_nonzero(candidate_entropies == candidate_maximum)
        )
        candidate_teacher_index = min(
            (
                int(index)
                for index in candidates
                if entropies[int(index)] == candidate_maximum
            ),
            key=lambda index: ANSWERS[index],
        )
        teacher_support = deduplicated_union(
            candidates, open_teacher_top
        )
        support_entropies = entropies[teacher_support]
        support_pair_audit = pair_audit(teacher_support, entropies)
        return {
            "split": row.split,
            "state_key": row.state_key,
            "prompt": row.prompt,
            "turn": int(row.turn),
            "candidate_count": int(row.candidate_count),
            "candidate_bucket": row.candidate_bucket,
            "candidate_indices": [int(index) for index in candidates],
            "candidate_words": [ANSWERS[int(index)] for index in candidates],
            "all_action_entropy_bits": [
                float(value) for value in entropies
            ],
            "open_teacher_top_indices": [
                int(index) for index in open_teacher_top
            ],
            "open_teacher_top_words": [
                ANSWERS[int(index)] for index in open_teacher_top
            ],
            "open_teacher_top_entropy_bits": [
                float(entropies[int(index)])
                for index in open_teacher_top
            ],
            "open_teacher_index": open_teacher_index,
            "open_teacher_word": ANSWERS[open_teacher_index],
            "open_teacher_top1_is_candidate": (
                open_teacher_index in candidate_set
            ),
            "global_top_entropy_tie_count": global_tie_count,
            "candidate_teacher_index": candidate_teacher_index,
            "candidate_teacher_word": ANSWERS[candidate_teacher_index],
            "candidate_top_entropy_tie_count": candidate_tie_count,
            "teacher_support_indices": teacher_support,
            "teacher_support_words": [
                ANSWERS[index] for index in teacher_support
            ],
            "teacher_support_entropy_bits": [
                float(value) for value in support_entropies
            ],
            "teacher_support_size": len(teacher_support),
            "teacher_support_valid_pair_count": support_pair_audit[
                "valid_pair_count"
            ],
            "teacher_support_tie_pair_count": support_pair_audit[
                "tie_pair_count"
            ],
            "teacher_support_mean_utility_gap_bits": support_pair_audit[
                "mean_utility_gap_bits"
            ],
        }


    target_records = {
        split: [
            target_record(row)
            for row in frame.itertuples(index=False)
        ]
        for split, frame in selected_frames.items()
    }
    for split, records in target_records.items():
        for record in records:
            assert len(record["candidate_indices"]) == record["candidate_count"]
            assert len(record["all_action_entropy_bits"]) == len(ANSWERS)
            assert len(record["open_teacher_top_indices"]) == OPEN_TEACHER_TOP_K
            assert set(record["candidate_indices"]) <= set(
                record["teacher_support_indices"]
            )
            assert set(record["open_teacher_top_indices"]) <= set(
                record["teacher_support_indices"]
            )
            assert record["teacher_support_valid_pair_count"] >= 0
            assert record["teacher_support_mean_utility_gap_bits"] >= 0.0
            assert record["candidate_count"] >= MIN_CANDIDATES

    target_paths = {}
    target_hashes = {}
    state_order_hashes = {}
    audit_rows = []
    open_teacher_rows = []
    for split, records in target_records.items():
        path = RESULTS_DIR / f"{split}-targets.jsonl"
        payload = jsonl_text(records)
        atomic_write(payload, path)
        target_paths[split] = path
        target_hashes[split] = sha256_text(payload)
        state_order_hashes[split] = sha256_text(
            "\\n".join(record["state_key"] for record in records)
        )
        for record in records:
            candidate_set = set(record["candidate_indices"])
            audit_rows.append(
                {
                    key: record[key]
                    for key in (
                        "split",
                        "state_key",
                        "turn",
                        "candidate_count",
                        "candidate_bucket",
                        "open_teacher_word",
                        "open_teacher_top1_is_candidate",
                        "global_top_entropy_tie_count",
                        "candidate_teacher_word",
                        "candidate_top_entropy_tie_count",
                        "teacher_support_size",
                        "teacher_support_valid_pair_count",
                        "teacher_support_tie_pair_count",
                        "teacher_support_mean_utility_gap_bits",
                    )
                }
                | {
                    "open_teacher_noncandidate_actions": sum(
                        index not in candidate_set
                        for index in record["open_teacher_top_indices"]
                    ),
                    "candidate_support_coverage": (
                        len(
                            candidate_set
                            & set(record["teacher_support_indices"])
                        )
                        / len(candidate_set)
                    ),
                    "open_teacher_support_coverage": (
                        len(
                            set(record["open_teacher_top_indices"])
                            & set(record["teacher_support_indices"])
                        )
                        / OPEN_TEACHER_TOP_K
                    ),
                }
            )
            for rank, (index, entropy) in enumerate(
                zip(
                    record["open_teacher_top_indices"],
                    record["open_teacher_top_entropy_bits"],
                ),
                start=1,
            ):
                open_teacher_rows.append(
                    {
                        "split": split,
                        "state_key": record["state_key"],
                        "candidate_count": record["candidate_count"],
                        "rank": rank,
                        "action_index": index,
                        "action_word": ANSWERS[index],
                        "entropy_bits": entropy,
                        "is_current_candidate": index in candidate_set,
                        "tied_for_global_best": (
                            entropy
                            == record["open_teacher_top_entropy_bits"][0]
                        ),
                    }
                )

    target_audit = pd.DataFrame(audit_rows)
    atomic_csv(source_distribution, RESULTS_DIR / "source-distribution.csv")
    atomic_csv(target_audit, RESULTS_DIR / "target-audit.csv")
    atomic_csv(
        pd.DataFrame(open_teacher_rows),
        RESULTS_DIR / "open-teacher-top16.csv",
    )
    teacher_diversity = pd.DataFrame(
        [
            {
                "split": split,
                "states": len(records),
                "unique_top1_words": len(
                    {record["open_teacher_word"] for record in records}
                ),
                "unique_top16_words": len(
                    {
                        word
                        for record in records
                        for word in record["open_teacher_top_words"]
                    }
                ),
                "mean_noncandidate_actions_in_top16": float(
                    np.mean(
                        [
                            sum(
                                index
                                not in set(record["candidate_indices"])
                                for index in record[
                                    "open_teacher_top_indices"
                                ]
                            )
                            for record in records
                        ]
                    )
                ),
            }
            for split, records in target_records.items()
        ]
    )
    atomic_csv(
        teacher_diversity, RESULTS_DIR / "teacher-diversity.csv"
    )

    selection_audit = target_audit.groupby(
        ["split", "candidate_bucket", "turn"], sort=True
    ).agg(
        states=("state_key", "size"),
        min_candidates=("candidate_count", "min"),
        max_candidates=("candidate_count", "max"),
        min_valid_pair_count=("teacher_support_valid_pair_count", "min"),
        mean_valid_pair_count=("teacher_support_valid_pair_count", "mean"),
        mean_tie_pair_count=("teacher_support_tie_pair_count", "mean"),
        mean_utility_gap_bits=(
            "teacher_support_mean_utility_gap_bits",
            "mean",
        ),
        states_without_valid_pairs=(
            "teacher_support_valid_pair_count",
            lambda values: int((values == 0).sum()),
        ),
        globally_tied_teacher_states=(
            "global_top_entropy_tie_count",
            lambda values: int((values > 1).sum()),
        ),
        open_teacher_top1_candidate_rate=(
            "open_teacher_top1_is_candidate",
            "mean",
        ),
        mean_open_teacher_noncandidate_actions=(
            "open_teacher_noncandidate_actions",
            "mean",
        ),
        min_candidate_support_coverage=(
            "candidate_support_coverage",
            "min",
        ),
        min_open_teacher_support_coverage=(
            "open_teacher_support_coverage",
            "min",
        ),
    ).reset_index()
    atomic_csv(selection_audit, RESULTS_DIR / "selection-audit.csv")

    PREREGISTRATION = {
        "experiment": "Lab 19e simulator-ranked answer-constrained policy",
        "scientific_question": (
            "Can simulator-derived entropy labels improve ranking across the "
            "answer-constrained 2,315-action policy without destroying the "
            "seed-45 incumbent ranking?"
        ),
        "method": (
            "supervised 2,315-action policy ranking with open-teacher support, "
            "online full-list actions, matched control, and truncated-support KL"
        ),
        "not_grpo": True,
        "seed": SEED,
        "incumbent": INCUMBENT.name,
        "incumbent_sha256": incumbent_sha256,
        "representation": "derived_state_v1 NEXT_GUESS",
        "source_sha256": source_hashes,
        "answers_sha256": sha256_file(
            DATA_DIR / "wordle-answers-original.txt"
        ),
        "patterns_sha256": sha256_file(
            DATA_DIR / "wordle-patterns-original-2315.npy"
        ),
        "state_selection": {
            "train_states": TRAIN_STATES,
            "dev_states": DEV_STATES,
            "minimum_candidates": MIN_CANDIDATES,
            "maximum_candidates": MAX_CANDIDATES,
            "bound_rationale": (
                "candidate_count <= 16 retains at least 90% of broad states "
                "in both source splits and bounds one category of task support"
            ),
            "exclude_singletons": True,
            "exclude_reserved_reachable_states": True,
            "exclude_lab20_anchor_states": True,
            "lab20_anchor_state_count": len(ANCHOR_STATE_KEYS),
            "candidate_buckets": CANDIDATE_BUCKETS,
            "bucket_quotas": BUCKET_QUOTAS,
            "selection_seed": SELECTION_SEED,
            "order_seed": ORDER_SEED,
            "turn_coverage": "at least one state from every available turn in each bucket",
            "state_order_sha256": state_order_hashes,
            "target_jsonl_sha256": target_hashes,
        },
        "target": {
            "action_space": "all 2,315 answer words",
            "support": (
                "deduplicated union of every current remaining candidate, "
                "open-teacher entropy top 16, incumbent full-list top 16, "
                "and current-policy full-list top 16"
            ),
            "open_teacher_top_k": OPEN_TEACHER_TOP_K,
            "utility": "one-ply EntropyExpert entropy in bits",
            "utility_round_decimals": UTILITY_DECIMALS,
            "equal_utility_tie_break": (
                "sha256(state_key + NUL + action_word), ascending"
            ),
            "pairwise_target": (
                "all unordered support pairs with unequal utility after "
                "nine-decimal rounding; higher entropy must score higher"
            ),
        },
        "training": {
            "epochs": EPOCHS,
            "updates": UPDATES,
            "optimizer_seed_schedule": (
                "seed + zero-based update, reset identically before each "
                "arm's train forward"
            ),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION,
            "gradient_clip": GRAD_CLIP,
            "support": (
                "all current candidates union open-teacher entropy top-16 "
                "union incumbent full-list top-16 union current full-list top-16"
            ),
            "maximum_support_before_deduplication": MAX_SUPPORT_SIZE,
            "open_teacher_top_k": OPEN_TEACHER_TOP_K,
            "incumbent_top_k": INCUMBENT_TOP_K,
            "current_top_k": CURRENT_TOP_K,
            "online_refresh_cadence_updates": REFRESH_CADENCE,
            "arms": ARMS,
            "task_loss": {
                "preservation-control": "exactly zero",
                "entropy-ranking": (
                    "unweighted mean softplus(-(score_hi-score_lo)) over "
                    "all valid unordered support pairs"
                ),
            },
            "truncated_support_kl": (
                "truncated_support_kl: KL(frozen incumbent || current) on "
                "the current deduplicated support only"
            ),
            "truncated_support_kl_temperature": (
                TRUNCATED_SUPPORT_KL_TEMPERATURE
            ),
            "truncated_support_kl_weight": TRUNCATED_SUPPORT_KL_WEIGHT,
            "matched_fields": [
                "independent seed-45 incumbent reload",
                "state order",
                "epochs",
                "optimizer seed schedule",
                "learning rate",
                "weight decay",
                "warmup",
                "update count",
                "refresh cadence",
                "support construction",
                "truncated_support_kl",
                "drift suite",
                "checkpoints",
                "evaluation",
            ],
        },
        "drift": {
            "anchor_source": "frozen Lab 20 anchor-states.csv",
            "anchor_sha256": sha256_file(anchor_file),
            "checkpoint_updates": list(
                range(0, UPDATES + 1, DRIFT_CHECK_EVERY)
            ),
            "candidate_mass_ratio_floor": CANDIDATE_MASS_RATIO_FLOOR,
            "best_candidate_rank_ceiling": (
                "per regime: max(10, 4x incumbent median best-candidate rank)"
            ),
            "winner_share_ceiling": (
                "per regime: max(0.50, incumbent largest-winner-share + 0.25)"
            ),
            "singleton_raw_rank_ceiling": (
                "regime 1: max(10, 4x incumbent raw singleton-answer rank)"
            ),
            "reported_only": ["top1_churn", "adapter_relative_delta"],
            "adapter_relative_delta": (
                "L2(policy - initial incumbent) / L2(initial incumbent)"
            ),
            "stop_before_further_training": True,
        },
        "evaluation": {
            "dev": (
                "frozen 64 broad states, deterministic argmax over all 2,315 "
                "actions, and global one-ply entropy regret"
            ),
            "gameplay": (
                "Lab 18d 19-answer battery, deterministic 2,315-answer scorer, "
                "symmetric deterministic singleton closure with raw singleton "
                "model rank reported before closure"
            ),
            "stochastic_evaluation": False,
            "decoder_shopping": False,
            "historical_lab18d_seed45_solves_without_singleton_closure": 10,
        },
        "advance": {
            "no_hard_stop": True,
            "minimum_dev_entropy_regret_improvement_bits": (
                MIN_DEV_REGRET_IMPROVEMENT_BITS
            ),
            "minimum_control_adjusted_regret_improvement_bits": (
                MIN_CONTROL_ADJUSTED_REGRET_IMPROVEMENT_BITS
            ),
            "minimum_final_candidate_mass_ratio": (
                CANDIDATE_MASS_RATIO_FLOOR
            ),
            "treatment_solve_count_not_below_incumbent_or_control": True,
            "minimum_adapter_relative_delta_for_movement": (
                MIN_ADAPTER_RELATIVE_DELTA
            ),
            "no_movement_verdict": "inconclusive",
        },
        "memory": {
            "mps_cap_gib": MEMORY_CAP_GIB,
            "abort_gib": MEMORY_ABORT_GIB,
            "soak_iterations": SOAK_ITERATIONS,
            "required_soaks": [
                "fixed-shape full-list refresh",
                "fixed-shape 64-action worst-case support training",
            ],
            "watchdog": "scripts/memguard.py --min-free 64",
        },
    }
    preregistration_text = json.dumps(
        PREREGISTRATION, indent=2, sort_keys=True
    )
    PREREGISTRATION_SHA256 = sha256_text(preregistration_text)
    atomic_write(
        preregistration_text + "\\n",
        RESULTS_DIR / "lab19e-preregistration.json",
    )
    atomic_write(
        PREREGISTRATION_SHA256 + "\\n",
        RESULTS_DIR / "lab19e-preregistration.sha256",
    )

    print("preregistration sha256:", PREREGISTRATION_SHA256)
    print("incumbent sha256:", incumbent_sha256)
    display(source_distribution)
    display(teacher_diversity)
    display(selection_audit)
    display(
        target_audit.groupby("split").agg(
            states=("state_key", "size"),
            candidates=("candidate_count", "sum"),
            globally_tied_teacher_states=(
                "global_top_entropy_tie_count",
                lambda values: int((values > 1).sum()),
            ),
            open_teacher_top1_candidate_rate=(
                "open_teacher_top1_is_candidate",
                "mean",
            ),
            min_teacher_support_size=("teacher_support_size", "min"),
            max_teacher_support_size=("teacher_support_size", "max"),
            min_valid_pair_count=(
                "teacher_support_valid_pair_count",
                "min",
            ),
            mean_tie_pair_count=(
                "teacher_support_tie_pair_count",
                "mean",
            ),
            mean_utility_gap_bits=(
                "teacher_support_mean_utility_gap_bits",
                "mean",
            ),
        )
    )
    """
)

md(
    """
    ### Gate A checks

    The candidate-count cap is a compute bound, not a result-driven cutoff. It
    retains at least 90% of broad states in each source split. Fixed bucket
    quotas prevent the 3-4-candidate majority from swallowing the sample, and
    each bucket includes every turn available after the held-out-answer
    and Lab 20 anchor exclusions.

    The JSONL target files persist all 2,315 one-ply action entropies for every
    selected state. They also list the open teacher's top 16 actions and audit
    valid pair counts, tie pair counts, utility gaps, and teacher diversity
    before model-ranked actions are added. No model score enters state selection
    or open-teacher construction.
    """
)

code(
    """
    assert (target_audit["candidate_count"] >= MIN_CANDIDATES).all()
    assert (target_audit["candidate_count"] <= MAX_CANDIDATES).all()
    assert len(target_records["train"]) == TRAIN_STATES
    assert len(target_records["dev"]) == DEV_STATES
    assert all(
        not record["candidate_count"] == 1
        for records in target_records.values()
        for record in records
    )
    assert all(
        set(record["candidate_indices"]).isdisjoint(RESERVED_INDICES)
        for records in target_records.values()
        for record in records
    )
    assert all(
        len(record["teacher_support_indices"])
        <= MAX_CANDIDATES + OPEN_TEACHER_TOP_K
        for records in target_records.values()
        for record in records
    )
    assert target_audit["teacher_support_valid_pair_count"].min() >= 0
    assert target_audit["teacher_support_mean_utility_gap_bits"].min() >= 0.0
    teacher_diversity_by_split = teacher_diversity.set_index("split")
    assert (
        int(teacher_diversity_by_split.loc["train", "unique_top1_words"])
        >= 100
    )
    assert (
        int(teacher_diversity_by_split.loc["dev", "unique_top1_words"])
        >= 50
    )
    assert ANCHOR_STATE_KEYS.isdisjoint(set(target_audit["state_key"]))
    print("Gate A passed: preregistration and data audit are on disk")
    """
)

md(
    """
    ## Gate B: scorer identity and bounded GPU soaks

    Gate B loads the seed-45 checkpoint twice in one PEFT model. `policy` is
    trainable. `incumbent` remains frozen and supplies the reference scores for
    `truncated_support_kl`.

    Before training:

    1. the differentiable support scorer must match the full-list scorer at the
       same action indices across prompts and every action-token-length bucket;
    2. the full-list scorer must reproduce a persisted Lab 18d seed-45 vector;
    3. the fixed-shape full-list refresh must plateau for 40 iterations;
    4. the worst-case 64-action training support must plateau for 40 updates.

    Full-list prefill requests only the last-position logits. Training requests
    only response-predicting positions with `use_cache=False`. Both paths use
    target gather minus `logsumexp`; neither materializes `log_softmax` over the
    vocabulary.
    """
)

code(
    """
    if RUN_MODEL:
        from peft import PeftModel
        from torch.optim import AdamW
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tiny_wordle.hardware import preferred_device

        device = preferred_device()
        torch.set_float32_matmul_precision("high")


        def driver_memory_gib() -> float:
            if device.type == "mps":
                return torch.mps.driver_allocated_memory() / 1024**3
            if device.type == "cuda":
                return torch.cuda.memory_allocated() / 1024**3
            return float("nan")


        def clear_device_cache() -> None:
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()


        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        PAD_ID = tokenizer.pad_token_id or tokenizer.eos_token_id


        def render_prompt(prompt: str) -> str:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )


        ACTION_TOKENS = [
            tokenizer.encode(word, add_special_tokens=False)
            + [tokenizer.eos_token_id]
            for word in ANSWERS
        ]
        ACTION_WIDTH = max(map(len, ACTION_TOKENS))
        probe_prompt_ids = tokenizer(
            render_prompt(target_records["train"][0]["prompt"]),
            add_special_tokens=False,
        )["input_ids"]
        for index in range(len(ANSWERS)):
            contextual = tokenizer(
                render_prompt(target_records["train"][0]["prompt"])
                + ANSWERS[index]
                + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            assert contextual[: len(probe_prompt_ids)] == probe_prompt_ids
            assert contextual[len(probe_prompt_ids) :] == ACTION_TOKENS[index]

        SCORER_SHA256 = sha256_text(
            "|".join(
                [
                    MODEL_ID,
                    str(ACTION_WIDTH),
                    json.dumps(ACTION_TOKENS),
                    json.dumps(ANSWERS),
                ]
            )
        )

        LENGTH_BUCKETS = {}
        for length in sorted({len(tokens) for tokens in ACTION_TOKENS}):
            indices = [
                index
                for index, tokens in enumerate(ACTION_TOKENS)
                if len(tokens) == length
            ]
            padding = (-len(indices)) % CHUNK_SIZE
            padded = indices + [indices[-1]] * padding
            LENGTH_BUCKETS[length] = (
                torch.tensor(padded, dtype=torch.long),
                torch.tensor(
                    [ACTION_TOKENS[index] for index in padded],
                    dtype=torch.long,
                    device=device,
                ),
            )


        def encode_actions(
            prompt_text: str, action_indices: list[int]
        ) -> dict[str, torch.Tensor | int]:
            prompt_ids = tokenizer(
                render_prompt(prompt_text), add_special_tokens=False
            )["input_ids"]
            rows = [ACTION_TOKENS[int(index)] for index in action_indices]
            width = max(map(len, rows))
            input_ids = []
            attention_mask = []
            targets = []
            target_mask = []
            for tokens in rows:
                body = prompt_ids + tokens[:-1]
                padding = width - len(tokens)
                input_ids.append(body + [PAD_ID] * padding)
                attention_mask.append([1] * len(body) + [0] * padding)
                targets.append(tokens + [PAD_ID] * padding)
                target_mask.append(
                    [1.0] * len(tokens) + [0.0] * padding
                )
            return {
                "input_ids": torch.tensor(
                    input_ids, dtype=torch.long, device=device
                ),
                "attention_mask": torch.tensor(
                    attention_mask, dtype=torch.long, device=device
                ),
                "targets": torch.tensor(
                    targets, dtype=torch.long, device=device
                ),
                "target_mask": torch.tensor(
                    target_mask, dtype=torch.float32, device=device
                ),
                "positions": torch.arange(
                    len(prompt_ids) - 1,
                    len(prompt_ids) - 1 + width,
                    dtype=torch.long,
                    device=device,
                ),
            }


        def score_encoded_actions(model, encoded: dict) -> torch.Tensor:
            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                logits_to_keep=encoded["positions"],
                use_cache=False,
            ).logits.float()
            gathered = logits.gather(
                2, encoded["targets"].unsqueeze(-1)
            ).squeeze(-1)
            token_log_probabilities = gathered - logits.logsumexp(-1)
            scores = (
                token_log_probabilities * encoded["target_mask"]
            ).sum(dim=1)
            return scores


        @torch.no_grad()
        def score_single_action(
            model, prompt_text: str, action_index: int
        ) -> float:
            encoded = encode_actions(prompt_text, [action_index])
            score = score_encoded_actions(model, encoded)
            value = float(score[0].cpu())
            del encoded, score
            clear_device_cache()
            return value


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
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=1,
            )
            final_logits = prefill.logits[0, -1].float()
            first_log_probabilities = (
                final_logits - final_logits.logsumexp(-1)
            )
            cache = prefill.past_key_values
            cache.batch_repeat_interleave(CHUNK_SIZE)
            peak = driver_memory_gib()
            scores = torch.zeros(len(ANSWERS), dtype=torch.float32)

            for length, (indices, tokens) in LENGTH_BUCKETS.items():
                for start in range(0, len(indices), CHUNK_SIZE):
                    chunk = tokens[start : start + CHUNK_SIZE]
                    total = first_log_probabilities[chunk[:, 0]].clone()
                    if length > 1:
                        steps = length - 1
                        output = model(
                            input_ids=chunk[:, :steps],
                            past_key_values=cache,
                            use_cache=True,
                            logits_to_keep=steps,
                        )
                        logits = output.logits.float()
                        gathered = logits.gather(
                            2, chunk[:, 1:].unsqueeze(-1)
                        ).squeeze(-1)
                        total = total + (
                            gathered - logits.logsumexp(-1)
                        ).sum(dim=1)
                        peak = max(peak, driver_memory_gib())
                        cache.crop(-steps)
                        del output, logits, gathered
                    scores[indices[start : start + CHUNK_SIZE]] = total.cpu()

            LAST_STATE_PEAK_GIB = peak
            values = scores.numpy()
            del (
                input_ids,
                cache,
                prefill,
                final_logits,
                first_log_probabilities,
                scores,
            )
            clear_device_cache()
            return values


        def reset_seeds(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)


        def load_dual_adapter():
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32
            ).to(device)
            base.config.use_cache = False
            model = PeftModel.from_pretrained(
                base,
                INCUMBENT,
                adapter_name="policy",
                is_trainable=True,
            ).to(device)
            model.load_adapter(
                INCUMBENT,
                adapter_name="incumbent",
                is_trainable=False,
            )
            model.set_adapter("policy")
            policy_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if "lora_" in name and ".policy." in name
            ]
            incumbent_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if "lora_" in name and ".incumbent." in name
            ]
            assert policy_parameters
            assert len(policy_parameters) == len(incumbent_parameters)
            assert all(parameter.requires_grad for parameter in policy_parameters)
            assert all(
                not parameter.requires_grad
                for parameter in incumbent_parameters
            )
            return model, policy_parameters, incumbent_parameters


        @contextmanager
        def active_adapter(model, policy_parameters, name: str, *, train: bool):
            previous_adapter = model.active_adapter
            previous_training = model.training
            previous_requires_grad = [
                parameter.requires_grad for parameter in policy_parameters
            ]
            model.set_adapter(name)
            model.train(train)
            try:
                yield
            finally:
                model.set_adapter(previous_adapter)
                model.train(previous_training)
                for parameter, requires_grad in zip(
                    policy_parameters, previous_requires_grad
                ):
                    parameter.requires_grad_(requires_grad)


        def release_model(model) -> None:
            model.to("cpu")
            del model
            clear_device_cache()


        def policy_digest(policy_parameters) -> str:
            digest = hashlib.sha256()
            for parameter in policy_parameters:
                digest.update(
                    parameter.detach().to("cpu", torch.float32).numpy().tobytes()
                )
            return digest.hexdigest()


        def parameter_snapshot(policy_parameters) -> list[torch.Tensor]:
            return [
                parameter.detach().to("cpu", torch.float32).clone()
                for parameter in policy_parameters
            ]


        def relative_parameter_delta(
            policy_parameters, initial_parameters
        ) -> float:
            squared_delta = 0.0
            squared_initial = 0.0
            for parameter, initial in zip(
                policy_parameters, initial_parameters
            ):
                current = parameter.detach().to("cpu", torch.float32)
                squared_delta += float(((current - initial) ** 2).sum())
                squared_initial += float((initial**2).sum())
            return math.sqrt(squared_delta) / max(
                math.sqrt(squared_initial), 1e-12
            )


        def model_top_k(scores: np.ndarray, count: int) -> list[int]:
            order = np.argsort(-scores, kind="stable")
            return [int(index) for index in order[:count]]


        def support_indices(
            candidates: list[int],
            open_teacher_top: list[int],
            incumbent_top: list[int],
            current_top: list[int],
        ) -> list[int]:
            support = deduplicated_union(
                candidates,
                open_teacher_top,
                incumbent_top,
                current_top,
            )
            seen = set(support)
            assert set(candidates) <= seen
            assert set(open_teacher_top) <= seen
            assert set(incumbent_top) <= seen
            assert set(current_top) <= seen
            assert len(support) <= MAX_SUPPORT_SIZE
            return support


        def pairwise_indices(
            support: list[int], record: dict
        ) -> tuple[list[int], list[int], int, float]:
            utilities = np.asarray(
                record["all_action_entropy_bits"], dtype=np.float64
            )[support]
            high = []
            low = []
            tie_pair_count = 0
            gaps = []
            for left in range(len(support)):
                for right in range(left + 1, len(support)):
                    gap = float(utilities[left] - utilities[right])
                    if gap == 0.0:
                        tie_pair_count += 1
                    elif gap > 0.0:
                        high.append(left)
                        low.append(right)
                        gaps.append(gap)
                    else:
                        high.append(right)
                        low.append(left)
                        gaps.append(-gap)
            assert len(high) + tie_pair_count == (
                len(support) * (len(support) - 1) // 2
            )
            return (
                high,
                low,
                tie_pair_count,
                float(np.mean(gaps)) if gaps else 0.0,
            )


        def loss_terms(
            policy_scores: torch.Tensor,
            incumbent_scores: torch.Tensor,
            support: list[int],
            record: dict,
            *,
            include_task_loss: bool,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            int,
            float,
        ]:
            high, low, tie_pair_count, mean_gap = pairwise_indices(
                support, record
            )
            if include_task_loss and high:
                high_tensor = torch.tensor(
                    high, dtype=torch.long, device=device
                )
                low_tensor = torch.tensor(
                    low, dtype=torch.long, device=device
                )
                pairwise_ranking_loss = torch.nn.functional.softplus(
                    -(policy_scores[high_tensor] - policy_scores[low_tensor])
                ).mean()
            else:
                pairwise_ranking_loss = policy_scores.sum() * 0.0
            assert float(pairwise_ranking_loss.detach().cpu()) == 0.0 or (
                include_task_loss
            )
            incumbent_probabilities = torch.softmax(
                incumbent_scores / TRUNCATED_SUPPORT_KL_TEMPERATURE,
                dim=-1,
            )
            incumbent_log_probabilities = torch.log_softmax(
                incumbent_scores / TRUNCATED_SUPPORT_KL_TEMPERATURE,
                dim=-1,
            )
            truncated_policy_log_probabilities = torch.log_softmax(
                policy_scores / TRUNCATED_SUPPORT_KL_TEMPERATURE,
                dim=-1,
            )
            truncated_support_kl = (
                incumbent_probabilities
                * (
                    incumbent_log_probabilities
                    - truncated_policy_log_probabilities
                )
            ).sum()
            total = (
                pairwise_ranking_loss
                + TRUNCATED_SUPPORT_KL_WEIGHT * truncated_support_kl
            )
            return (
                total,
                pairwise_ranking_loss,
                truncated_support_kl,
                len(high),
                tie_pair_count,
                mean_gap,
            )


        print("device:", device)
        print("scorer sha256:", SCORER_SHA256)
        print("action token widths:", sorted({len(tokens) for tokens in ACTION_TOKENS}))
    else:
        print("Gate B skipped; set LAB19E_RUN_MODEL=1 under memguard")
    """
)

code(
    """
    if RUN_MODEL and not REUSE_GATE_B:
        reset_seeds(SEED)
        soak_model, soak_parameters, soak_incumbent_parameters = (
            load_dual_adapter()
        )
        probe = max(
            target_records["train"],
            key=lambda record: len(
                tokenizer(
                    render_prompt(record["prompt"]),
                    add_special_tokens=False,
                )["input_ids"]
            ),
        )
        prompt_probes = [
            target_records["train"][0],
            probe,
            target_records["dev"][-1],
        ]
        length_representatives = [
            next(
                index
                for index, tokens in enumerate(ACTION_TOKENS)
                if len(tokens) == length
            )
            for length in sorted({len(tokens) for tokens in ACTION_TOKENS})
        ]
        scorer_differences = []
        with active_adapter(
            soak_model, soak_parameters, "policy", train=False
        ), torch.no_grad():
            for prompt_record in prompt_probes:
                full_scores = score_all_words(
                    soak_model, prompt_record["prompt"]
                )
                encoded = encode_actions(
                    prompt_record["prompt"], length_representatives
                )
                direct_scores = (
                    score_encoded_actions(soak_model, encoded)
                    .detach()
                    .cpu()
                    .numpy()
                )
                scorer_differences.extend(
                    np.abs(
                        direct_scores
                        - full_scores[np.asarray(length_representatives)]
                    )
                )
                del encoded, direct_scores, full_scores

        with active_adapter(
            soak_model, soak_parameters, "incumbent", train=False
        ):
            probe_incumbent_scores = score_all_words(
                soak_model, probe["prompt"]
            )
        with active_adapter(
            soak_model, soak_parameters, "policy", train=False
        ), torch.no_grad():
            probe_current_scores = score_all_words(
                soak_model, probe["prompt"]
            )
        candidates = list(map(int, probe["candidate_indices"]))
        open_teacher_top = list(
            map(int, probe["open_teacher_top_indices"])
        )
        incumbent_top = model_top_k(
            probe_incumbent_scores, INCUMBENT_TOP_K
        )
        current_top = model_top_k(
            probe_current_scores, CURRENT_TOP_K
        )
        actual_soak_support = support_indices(
            candidates,
            open_teacher_top,
            incumbent_top,
            current_top,
        )
        max_width_index = next(
            index
            for index, tokens in enumerate(ACTION_TOKENS)
            if len(tokens) == ACTION_WIDTH
        )
        category_representatives = [
            candidates[0],
            open_teacher_top[0],
            incumbent_top[0],
            current_top[0],
        ]
        filler_order = list(range(len(ANSWERS)))
        soak_support = deduplicated_union(
            category_representatives,
            [max_width_index],
            actual_soak_support,
            filler_order,
        )[:MAX_SUPPORT_SIZE]
        assert len(soak_support) == MAX_SUPPORT_SIZE
        assert set(candidates) & set(soak_support)
        assert set(open_teacher_top) & set(soak_support)
        assert set(incumbent_top) & set(soak_support)
        assert set(current_top) & set(soak_support)
        assert max_width_index in soak_support
        soak_encoded = encode_actions(probe["prompt"], soak_support)
        assert soak_encoded["input_ids"].shape[0] == MAX_SUPPORT_SIZE
        assert soak_encoded["targets"].shape == (
            MAX_SUPPORT_SIZE,
            ACTION_WIDTH,
        )
        assert soak_encoded["target_mask"].shape == (
            MAX_SUPPORT_SIZE,
            ACTION_WIDTH,
        )
        assert soak_encoded["positions"].shape == (ACTION_WIDTH,)

        with active_adapter(
            soak_model, soak_parameters, "policy", train=False
        ), torch.no_grad():
            full_scores = score_all_words(soak_model, probe["prompt"])
            direct_scores = (
                score_encoded_actions(soak_model, soak_encoded)
                .detach()
                .cpu()
                .numpy()
            )
        scorer_differences.extend(
            np.abs(direct_scores - full_scores[np.asarray(soak_support)])
        )
        action_scorer_max_abs_diff = float(max(scorer_differences))
        print(
            "support-vs-full-list scorer max abs diff:",
            f"{action_scorer_max_abs_diff:.3e}",
        )
        assert action_scorer_max_abs_diff < 1e-3
        del direct_scores, full_scores

        lab18d_keys = pd.read_csv(
            LAB18D_DIR / "seed45-answer-constrained-score-keys.csv"
        )
        first_key = lab18d_keys.iloc[0]
        regression_history = [
            Turn(OPENING, score_string(first_key["answer"], OPENING))
        ]
        regression_candidates = candidate_indices_from_history(
            regression_history,
            ANSWERS,
            PATTERNS,
            expert=EXPERT,
        )
        regression_prompt = structured_next_guess_prompt(
            regression_history, len(regression_candidates)
        )
        with active_adapter(
            soak_model, soak_parameters, "policy", train=False
        ):
            reproduced = score_all_words(soak_model, regression_prompt)
        persisted = np.load(
            LAB18D_DIR / "seed45-answer-constrained-scores.npy",
            mmap_mode="r",
        )[0]
        full_list_max_abs_diff = float(
            np.max(np.abs(reproduced - persisted))
        )
        print(
            "Lab 18d full-list max abs diff:",
            f"{full_list_max_abs_diff:.3e}",
        )
        assert full_list_max_abs_diff < 1e-3

        full_list_peaks = []
        with active_adapter(
            soak_model, soak_parameters, "policy", train=False
        ):
            for _ in range(SOAK_ITERATIONS):
                score_all_words(soak_model, probe["prompt"])
                full_list_peaks.append(LAST_STATE_PEAK_GIB)
        third = SOAK_ITERATIONS // 3
        full_list_creep = float(
            np.mean(full_list_peaks[-third:])
            - np.mean(full_list_peaks[third : 2 * third])
        )
        full_list_late_range = float(np.ptp(full_list_peaks[-third:]))
        assert full_list_creep < 0.5
        assert full_list_late_range < 0.5
        assert max(full_list_peaks) < MEMORY_ABORT_GIB
        atomic_csv(
            pd.DataFrame(
                {
                    "iteration": range(1, SOAK_ITERATIONS + 1),
                    "driver_peak_gib": full_list_peaks,
                }
            ),
            RESULTS_DIR / "full-list-soak.csv",
        )

        optimizer = AdamW(
            soak_parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        training_peaks = []
        for iteration in range(SOAK_ITERATIONS):
            with active_adapter(
                soak_model, soak_parameters, "policy", train=False
            ):
                score_all_words(soak_model, probe["prompt"])
            peak = LAST_STATE_PEAK_GIB
            optimizer.zero_grad(set_to_none=True)
            with active_adapter(
                soak_model, soak_parameters, "incumbent", train=False
            ), torch.no_grad():
                incumbent_scores = score_encoded_actions(
                    soak_model, soak_encoded
                ).detach()
            with active_adapter(
                soak_model, soak_parameters, "policy", train=True
            ):
                policy_scores = score_encoded_actions(
                    soak_model, soak_encoded
                )
                (
                    total_loss,
                    pairwise_ranking_loss,
                    truncated_support_kl,
                    valid_pair_count,
                    tie_pair_count,
                    mean_utility_gap_bits,
                ) = loss_terms(
                    policy_scores,
                    incumbent_scores,
                    soak_support,
                    probe,
                    include_task_loss=True,
                )
                peak = max(peak, driver_memory_gib())
                total_loss.backward()
                peak = max(peak, driver_memory_gib())
                torch.nn.utils.clip_grad_norm_(
                    soak_parameters, GRAD_CLIP
                )
                optimizer.step()
                peak = max(peak, driver_memory_gib())
            training_peaks.append(peak)
            del (
                incumbent_scores,
                policy_scores,
                total_loss,
                pairwise_ranking_loss,
                truncated_support_kl,
            )
            clear_device_cache()

        training_creep = float(
            np.mean(training_peaks[-third:])
            - np.mean(training_peaks[third : 2 * third])
        )
        training_late_range = float(np.ptp(training_peaks[-third:]))
        assert training_creep < 0.5
        assert training_late_range < 0.5
        assert max(training_peaks) < MEMORY_ABORT_GIB
        atomic_csv(
            pd.DataFrame(
                {
                    "iteration": range(1, SOAK_ITERATIONS + 1),
                    "driver_peak_gib": training_peaks,
                }
            ),
            RESULTS_DIR / "training-soak.csv",
        )
        gate_b = {
            "action_scorer_max_abs_diff": action_scorer_max_abs_diff,
            "full_list_max_abs_diff": full_list_max_abs_diff,
            "full_list_peak_gib": max(full_list_peaks),
            "full_list_creep_gib": full_list_creep,
            "full_list_late_range_gib": full_list_late_range,
            "training_peak_gib": max(training_peaks),
            "training_creep_gib": training_creep,
            "training_late_range_gib": training_late_range,
            "support_size": len(soak_support),
            "support_size_before_padding": len(actual_soak_support),
            "support_slots_before_deduplication": MAX_SUPPORT_SIZE,
            "support_category_sizes": {
                "candidates": len(candidates),
                "open_teacher": len(open_teacher_top),
                "incumbent": len(incumbent_top),
                "current": len(current_top),
            },
            "max_action_width": ACTION_WIDTH,
            "valid_pair_count": valid_pair_count,
            "tie_pair_count": tie_pair_count,
            "mean_utility_gap_bits": mean_utility_gap_bits,
            "equivalence_prompt_count": len(prompt_probes),
            "equivalence_action_length_buckets": sorted(
                {len(tokens) for tokens in ACTION_TOKENS}
            ),
            "passed": True,
        }
        atomic_json(gate_b, RESULTS_DIR / "gate-b.json")
        del optimizer, soak_encoded, soak_incumbent_parameters
        release_model(soak_model)
        del soak_model
        print("Gate B passed:", gate_b)
    elif RUN_MODEL:
        gate_b = json.loads((RESULTS_DIR / "gate-b.json").read_text())
        assert gate_b["passed"] is True
        assert gate_b["action_scorer_max_abs_diff"] < 1e-3
        assert gate_b["full_list_max_abs_diff"] < 1e-3
        assert gate_b["full_list_peak_gib"] < MEMORY_ABORT_GIB
        assert gate_b["training_peak_gib"] < MEMORY_ABORT_GIB
        assert gate_b["full_list_creep_gib"] < 0.5
        assert gate_b["training_creep_gib"] < 0.5
        print("Gate B reused:", gate_b)
    else:
        print("Gate B numerical and memory checks not run")
    """
)

md(
    """
    ## Gate C: matched-arm training with live drift stops

    The training stream repeats the 128 frozen states twice. At the start of
    every 16-update block, the current policy ranks all 2,315 answers for the
    next 16 states under `torch.no_grad()`. The refresh file records the model
    digest and top-16 actions for every state. Each optimizer support is:

    ```text
    all current remaining candidates
      union open-teacher entropy top 16 over all 2,315 actions
      union frozen-incumbent full-list top 16
      union current-policy full-list top 16
    ```

    The treatment averages a pairwise logistic loss over every unequal-utility
    action pair in the deduplicated support. The matched control sets that task
    loss to exactly zero. Both arms use
    `0.25 * truncated_support_kl`, the same optimizer schedule, state order,
    refreshes, supports, checkpoints, and evaluations.

    Every 32 updates, before another update can run, both models score the 20
    frozen Lab 20 anchors over all 2,315 answers. Candidate mass, best-candidate
    rank, raw singleton-answer rank, and winner concentration are checked
    separately in each Lab 20 regime. Any regime trip ends that arm at the
    checkpoint.
    """
)

code(
    """
    if RUN_TRAINING:
        anchor_records = []
        for row in anchor_source.itertuples(index=False):
            history = parse_state_key(row.state_key)
            candidates = candidate_indices_from_history(
                history,
                ANSWERS,
                PATTERNS,
                expert=EXPERT,
            )
            assert len(candidates) == int(row.candidate_count)
            anchor_records.append(
                {
                    "state_key": row.state_key,
                    "turn": int(row.turn),
                    "regime": str(row.regime),
                    "candidate_count": int(row.candidate_count),
                    "candidate_indices": [
                        int(index) for index in candidates
                    ],
                    "prompt": structured_next_guess_prompt(
                        history, len(candidates)
                    ),
                }
            )
        assert len(anchor_records) == 20


        def rank_vector(scores: np.ndarray) -> np.ndarray:
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(len(scores), dtype=np.int64)
            ranks[order] = np.arange(1, len(scores) + 1)
            return ranks


        def anchor_metrics(
            score_matrix: np.ndarray,
            incumbent_matrix: np.ndarray,
        ) -> pd.DataFrame:
            rows = []
            for record, scores, incumbent_scores in zip(
                anchor_records, score_matrix, incumbent_matrix
            ):
                candidates = np.array(
                    record["candidate_indices"], dtype=np.int64
                )
                ranks = rank_vector(scores)
                shifted = scores - scores.max()
                weights = np.exp(shifted)
                incumbent_shifted = incumbent_scores - incumbent_scores.max()
                incumbent_weights = np.exp(incumbent_shifted)
                winner_index = int(scores.argmax())
                incumbent_winner = int(incumbent_scores.argmax())
                rows.append(
                    {
                        "state_key": record["state_key"],
                        "regime": record["regime"],
                        "candidate_count": record["candidate_count"],
                        "candidate_mass": float(
                            weights[candidates].sum() / weights.sum()
                        ),
                        "incumbent_candidate_mass": float(
                            incumbent_weights[candidates].sum()
                            / incumbent_weights.sum()
                        ),
                        "best_candidate_rank": int(
                            ranks[candidates].min()
                        ),
                        "raw_singleton_answer_rank": (
                            int(ranks[candidates[0]])
                            if record["regime"] == "1"
                            else np.nan
                        ),
                        "winner_index": winner_index,
                        "winner_word": ANSWERS[winner_index],
                        "incumbent_winner_index": incumbent_winner,
                        "incumbent_winner_word": ANSWERS[incumbent_winner],
                        "top1_changed": winner_index != incumbent_winner,
                    }
                )
            return pd.DataFrame(rows)


        def anchor_summary(
            metrics: pd.DataFrame,
            adapter_relative_delta: float,
        ) -> dict:
            winner_counts = metrics["winner_word"].value_counts()
            incumbent_mass = float(
                metrics["incumbent_candidate_mass"].median()
            )
            candidate_mass = float(metrics["candidate_mass"].median())
            per_regime = {}
            for regime, regime_metrics in metrics.groupby(
                "regime", sort=True
            ):
                regime_winners = regime_metrics[
                    "winner_word"
                ].value_counts()
                regime_incumbent_mass = float(
                    regime_metrics["incumbent_candidate_mass"].median()
                )
                regime_candidate_mass = float(
                    regime_metrics["candidate_mass"].median()
                )
                per_regime[str(regime)] = {
                    "incumbent_median_candidate_mass": (
                        regime_incumbent_mass
                    ),
                    "current_median_candidate_mass": regime_candidate_mass,
                    "candidate_mass_ratio": (
                        regime_candidate_mass
                        / max(regime_incumbent_mass, 1e-12)
                    ),
                    "median_best_candidate_rank": float(
                        regime_metrics["best_candidate_rank"].median()
                    ),
                    "raw_singleton_answer_rank": (
                        float(
                            regime_metrics[
                                "raw_singleton_answer_rank"
                            ].median()
                        )
                        if str(regime) == "1"
                        else None
                    ),
                    "largest_winner_share": float(
                        regime_winners.iloc[0] / len(regime_metrics)
                    ),
                    "unique_winners": int(regime_winners.size),
                    "top1_churn": float(
                        regime_metrics["top1_changed"].mean()
                    ),
                }
            return {
                "incumbent_median_candidate_mass": incumbent_mass,
                "current_median_candidate_mass": candidate_mass,
                "candidate_mass_ratio": (
                    candidate_mass / max(incumbent_mass, 1e-12)
                ),
                "median_best_candidate_rank": float(
                    metrics["best_candidate_rank"].median()
                ),
                "top1_churn": float(metrics["top1_changed"].mean()),
                "unique_winners": int(winner_counts.size),
                "largest_winner_share": float(
                    winner_counts.iloc[0] / len(metrics)
                ),
                "adapter_relative_delta": adapter_relative_delta,
                "per_regime": per_regime,
            }


        def drift_rules(current: dict, baseline: dict) -> dict:
            rules = {}
            for regime in sorted(current["per_regime"]):
                current_regime = current["per_regime"][regime]
                baseline_regime = baseline["per_regime"][regime]
                rank_ceiling = max(
                    BEST_RANK_FLOOR,
                    BEST_RANK_MULTIPLIER
                    * baseline_regime["median_best_candidate_rank"],
                )
                winner_ceiling = max(
                    WINNER_SHARE_FLOOR,
                    baseline_regime["largest_winner_share"]
                    + WINNER_SHARE_MARGIN,
                )
                prefix = f"regime_{regime}"
                rules[f"{prefix}_candidate_mass_collapse"] = {
                    "regime": regime,
                    "kind": "candidate_mass",
                    "value": current_regime["candidate_mass_ratio"],
                    "threshold": CANDIDATE_MASS_RATIO_FLOOR,
                    "tripped": (
                        current_regime["candidate_mass_ratio"]
                        < CANDIDATE_MASS_RATIO_FLOOR
                    ),
                }
                rules[f"{prefix}_best_candidate_rank_collapse"] = {
                    "regime": regime,
                    "kind": "best_candidate_rank",
                    "value": current_regime[
                        "median_best_candidate_rank"
                    ],
                    "threshold": rank_ceiling,
                    "tripped": (
                        current_regime["median_best_candidate_rank"]
                        > rank_ceiling
                    ),
                }
                rules[f"{prefix}_winner_concentration"] = {
                    "regime": regime,
                    "kind": "winner_concentration",
                    "value": current_regime["largest_winner_share"],
                    "threshold": winner_ceiling,
                    "tripped": (
                        current_regime["largest_winner_share"]
                        > winner_ceiling
                    ),
                }
                if regime == "1":
                    singleton_ceiling = max(
                        BEST_RANK_FLOOR,
                        BEST_RANK_MULTIPLIER
                        * baseline_regime["raw_singleton_answer_rank"],
                    )
                    rules["regime_1_raw_singleton_rank_collapse"] = {
                        "regime": regime,
                        "kind": "raw_singleton_answer_rank",
                        "value": current_regime[
                            "raw_singleton_answer_rank"
                        ],
                        "threshold": singleton_ceiling,
                        "tripped": (
                            current_regime["raw_singleton_answer_rank"]
                            > singleton_ceiling
                        ),
                    }
            rules["any_tripped"] = any(
                rule["tripped"] for rule in rules.values()
            )
            return rules


        @torch.no_grad()
        def full_list_for_adapter(
            model,
            policy_parameters,
            adapter_name: str,
            prompt: str,
        ) -> np.ndarray:
            with active_adapter(
                model,
                policy_parameters,
                adapter_name,
                train=False,
            ):
                values = score_all_words(model, prompt)
            assert LAST_STATE_PEAK_GIB < MEMORY_ABORT_GIB
            return values


        def score_anchor_matrix(
            model, policy_parameters, adapter_name: str
        ) -> np.ndarray:
            matrix = np.zeros(
                (len(anchor_records), len(ANSWERS)),
                dtype=np.float32,
            )
            for position, record in enumerate(anchor_records):
                matrix[position] = full_list_for_adapter(
                    model,
                    policy_parameters,
                    adapter_name,
                    record["prompt"],
                )
            return matrix


        def save_policy_checkpoint(
            model, arm_dir: Path, step: int
        ) -> Path:
            checkpoint = arm_dir / "checkpoints" / f"step-{step:04d}"
            model.save_pretrained(
                str(checkpoint), selected_adapters=["policy"]
            )
            policy_path = checkpoint / "policy"
            assert (policy_path / "adapter_model.safetensors").exists()
            return policy_path


        def lr_multiplier(step: int) -> float:
            warmup_steps = max(1, int(UPDATES * WARMUP_FRACTION))
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(
                1, UPDATES - warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        print("Gate C helpers skipped")
    """
)

code(
    """
    if RUN_TRAINING:
        for checkpoint in TRAINED_CHECKPOINTS.values():
            if checkpoint.exists():
                raise FileExistsError(
                    "refusing to overwrite existing Lab 19e checkpoint: "
                    f"{checkpoint}"
                )

        def drift_row(
            step: int,
            summary: dict,
            rules: dict,
            checkpoint_path: Path | None,
        ) -> dict:
            tripped = [
                name
                for name, rule in rules.items()
                if name != "any_tripped" and rule["tripped"]
            ]
            return {
                "step": step,
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "per_regime"
                },
                "per_regime_json": json.dumps(
                    summary["per_regime"], sort_keys=True
                ),
                "hard_stop": rules["any_tripped"],
                "tripped_rules": ",".join(tripped),
                "checkpoint": (
                    str(checkpoint_path)
                    if checkpoint_path is not None
                    else ""
                ),
            }


        arm_states = {}
        incumbent_top_texts = {}
        incumbent_anchor_matrices = {}
        for arm in ARMS:
            arm_dir = RESULTS_DIR / "arms" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            reset_seeds(SEED)
            (
                arm_model,
                arm_policy_parameters,
                arm_incumbent_parameters,
            ) = load_dual_adapter()
            initial_policy_digest = policy_digest(arm_policy_parameters)
            initial_incumbent_digest = policy_digest(
                arm_incumbent_parameters
            )
            assert initial_policy_digest == initial_incumbent_digest

            incumbent_top_records = []
            incumbent_top_by_state = {}
            for position, record in enumerate(target_records["train"]):
                scores = full_list_for_adapter(
                    arm_model,
                    arm_policy_parameters,
                    "incumbent",
                    record["prompt"],
                )
                top = model_top_k(scores, INCUMBENT_TOP_K)
                incumbent_top_by_state[record["state_key"]] = top
                incumbent_top_records.append(
                    {
                        "position": position,
                        "state_key": record["state_key"],
                        "top_indices": top,
                        "top_words": [ANSWERS[index] for index in top],
                        "top_scores": [
                            float(scores[index]) for index in top
                        ],
                    }
                )
            incumbent_top_text = jsonl_text(incumbent_top_records)
            incumbent_top_texts[arm] = incumbent_top_text
            atomic_write(
                incumbent_top_text, arm_dir / "incumbent-top16.jsonl"
            )

            incumbent_anchor_matrix = score_anchor_matrix(
                arm_model, arm_policy_parameters, "incumbent"
            )
            incumbent_anchor_matrices[arm] = incumbent_anchor_matrix
            np.save(
                arm_dir / "anchor-scores-incumbent.npy",
                incumbent_anchor_matrix,
            )
            baseline_metrics = anchor_metrics(
                incumbent_anchor_matrix, incumbent_anchor_matrix
            )
            baseline_summary = anchor_summary(baseline_metrics, 0.0)
            baseline_rules = drift_rules(
                baseline_summary, baseline_summary
            )
            assert not baseline_rules["any_tripped"]
            atomic_csv(
                baseline_metrics, arm_dir / "anchor-step-0000.csv"
            )

            optimizer = AdamW(
                arm_policy_parameters,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_multiplier
            )
            arm_states[arm] = {
                "dir": arm_dir,
                "model": arm_model,
                "policy_parameters": arm_policy_parameters,
                "incumbent_parameters": arm_incumbent_parameters,
                "initial_parameters": parameter_snapshot(
                    arm_policy_parameters
                ),
                "initial_policy_digest": initial_policy_digest,
                "initial_incumbent_digest": initial_incumbent_digest,
                "incumbent_top_by_state": incumbent_top_by_state,
                "incumbent_anchor_matrix": incumbent_anchor_matrix,
                "baseline_summary": baseline_summary,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "refresh_top_by_step": {},
                "training_rows": [],
                "drift_rows": [
                    drift_row(0, baseline_summary, baseline_rules, None)
                ],
                "drift_rule_records": [
                    {
                        "step": 0,
                        "rules": baseline_rules,
                    }
                ],
                "hard_stop": False,
                "tripped_rules": [],
            }

        initial_digests = {
            state["initial_policy_digest"]
            for state in arm_states.values()
        }
        incumbent_digests = {
            state["initial_incumbent_digest"]
            for state in arm_states.values()
        }
        assert len(initial_digests) == 1
        assert initial_digests == incumbent_digests
        assert len(set(incumbent_top_texts.values())) == 1
        assert np.array_equal(
            incumbent_anchor_matrices[ARMS[0]],
            incumbent_anchor_matrices[ARMS[1]],
        )

        training_sequence = [
            position
            for _ in range(EPOCHS)
            for position in range(TRAIN_STATES)
        ]
        matched_hard_stop = False
        stop_step = None
        started = time.perf_counter()

        for zero_step, state_position in enumerate(training_sequence):
            if zero_step % REFRESH_CADENCE == 0:
                block_positions = training_sequence[
                    zero_step : zero_step + REFRESH_CADENCE
                ]
                for arm in ARMS:
                    state = arm_states[arm]
                    refresh_records = []
                    state["refresh_top_by_step"] = {}
                    refresh_digest = policy_digest(
                        state["policy_parameters"]
                    )
                    for block_offset, refresh_position in enumerate(
                        block_positions
                    ):
                        refresh_record = target_records["train"][
                            refresh_position
                        ]
                        scores = full_list_for_adapter(
                            state["model"],
                            state["policy_parameters"],
                            "policy",
                            refresh_record["prompt"],
                        )
                        top = model_top_k(scores, CURRENT_TOP_K)
                        state["refresh_top_by_step"][
                            zero_step + block_offset
                        ] = top
                        refresh_records.append(
                            {
                                "training_step": (
                                    zero_step + block_offset + 1
                                ),
                                "state_position": refresh_position,
                                "state_key": refresh_record["state_key"],
                                "policy_digest": refresh_digest,
                                "top_indices": top,
                                "top_words": [
                                    ANSWERS[index] for index in top
                                ],
                                "top_scores": [
                                    float(scores[index]) for index in top
                                ],
                            }
                        )
                    atomic_write(
                        jsonl_text(refresh_records),
                        state["dir"]
                        / f"online-refresh-step-{zero_step:04d}.jsonl",
                    )

            record = target_records["train"][state_position]
            for arm in ARMS:
                state = arm_states[arm]
                model = state["model"]
                policy_parameters = state["policy_parameters"]
                optimizer = state["optimizer"]
                scheduler = state["scheduler"]
                incumbent_top = state["incumbent_top_by_state"][
                    record["state_key"]
                ]
                current_top = state["refresh_top_by_step"][zero_step]
                support = support_indices(
                    record["candidate_indices"],
                    record["open_teacher_top_indices"],
                    incumbent_top,
                    current_top,
                )
                support_set = set(support)
                assert set(record["candidate_indices"]) <= support_set
                assert set(record["open_teacher_top_indices"]) <= support_set
                assert set(incumbent_top) <= support_set
                assert set(current_top) <= support_set
                encoded = encode_actions(record["prompt"], support)

                with active_adapter(
                    model,
                    policy_parameters,
                    "incumbent",
                    train=False,
                ), torch.no_grad():
                    incumbent_scores = score_encoded_actions(
                        model, encoded
                    ).detach()

                reset_seeds(SEED + zero_step)
                optimizer.zero_grad(set_to_none=True)
                with active_adapter(
                    model,
                    policy_parameters,
                    "policy",
                    train=True,
                ):
                    policy_scores = score_encoded_actions(model, encoded)
                    (
                        total_loss,
                        pairwise_ranking_loss,
                        truncated_support_kl,
                        valid_pair_count,
                        tie_pair_count,
                        mean_utility_gap_bits,
                    ) = loss_terms(
                        policy_scores,
                        incumbent_scores,
                        support,
                        record,
                        include_task_loss=(arm == "entropy-ranking"),
                    )
                    peak = driver_memory_gib()
                    total_loss.backward()
                    peak = max(peak, driver_memory_gib())
                    torch.nn.utils.clip_grad_norm_(
                        policy_parameters, GRAD_CLIP
                    )
                    optimizer.step()
                    peak = max(peak, driver_memory_gib())
                scheduler.step()

                step = zero_step + 1
                task_value = float(
                    pairwise_ranking_loss.detach().cpu()
                )
                if arm == "preservation-control":
                    assert task_value == 0.0
                state["training_rows"].append(
                    {
                        "arm": arm,
                        "step": step,
                        "epoch": zero_step // TRAIN_STATES + 1,
                        "state_position": state_position,
                        "state_key": record["state_key"],
                        "turn": record["turn"],
                        "candidate_count": record["candidate_count"],
                        "support_size": len(support),
                        "support_indices": json.dumps(support),
                        "support_words": ",".join(
                            ANSWERS[index] for index in support
                        ),
                        "noncandidate_support_size": len(
                            support_set
                            - set(record["candidate_indices"])
                        ),
                        "open_teacher_top_k": OPEN_TEACHER_TOP_K,
                        "incumbent_top_k": INCUMBENT_TOP_K,
                        "current_top_k": CURRENT_TOP_K,
                        "contains_all_candidates": True,
                        "contains_open_teacher_top16": True,
                        "contains_incumbent_top16": True,
                        "contains_current_top16": True,
                        "valid_pair_count": valid_pair_count,
                        "tie_pair_count": tie_pair_count,
                        "mean_utility_gap_bits": mean_utility_gap_bits,
                        "pairwise_ranking_loss": task_value,
                        "truncated_support_kl": float(
                            truncated_support_kl.detach().cpu()
                        ),
                        "total_loss": float(total_loss.detach().cpu()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "driver_peak_gib": peak,
                    }
                )
                assert peak < MEMORY_ABORT_GIB
                del (
                    encoded,
                    incumbent_scores,
                    policy_scores,
                    total_loss,
                    pairwise_ranking_loss,
                    truncated_support_kl,
                )
                clear_device_cache()

            step = zero_step + 1
            if step % DRIFT_CHECK_EVERY == 0:
                checkpoint_trips = []
                for arm in ARMS:
                    state = arm_states[arm]
                    current_anchor_matrix = score_anchor_matrix(
                        state["model"], state["policy_parameters"], "policy"
                    )
                    metrics = anchor_metrics(
                        current_anchor_matrix,
                        state["incumbent_anchor_matrix"],
                    )
                    delta = relative_parameter_delta(
                        state["policy_parameters"],
                        state["initial_parameters"],
                    )
                    summary = anchor_summary(metrics, delta)
                    rules = drift_rules(
                        summary, state["baseline_summary"]
                    )
                    tripped = [
                        name
                        for name, rule in rules.items()
                        if name != "any_tripped" and rule["tripped"]
                    ]
                    state["hard_stop"] = rules["any_tripped"]
                    state["tripped_rules"] = tripped
                    checkpoint_trips.extend(
                        f"{arm}:{name}" for name in tripped
                    )
                    atomic_csv(
                        metrics,
                        state["dir"] / f"anchor-step-{step:04d}.csv",
                    )
                    np.save(
                        state["dir"]
                        / f"anchor-scores-step-{step:04d}.npy",
                        current_anchor_matrix,
                    )
                    checkpoint_path = save_policy_checkpoint(
                        state["model"], state["dir"], step
                    )
                    state["drift_rows"].append(
                        drift_row(step, summary, rules, checkpoint_path)
                    )
                    state["drift_rule_records"].append(
                        {"step": step, "rules": rules}
                    )
                    atomic_csv(
                        pd.DataFrame(state["drift_rows"]),
                        state["dir"] / "drift-checks.csv",
                    )
                    atomic_write(
                        jsonl_text(state["drift_rule_records"]),
                        state["dir"] / "drift-rules.jsonl",
                    )
                if checkpoint_trips:
                    matched_hard_stop = True
                    stop_step = step
                    print(
                        f"matched hard stop at step {step}: "
                        + ", ".join(checkpoint_trips)
                    )
                    break

        completed_updates = min(
            len(state["training_rows"]) for state in arm_states.values()
        )
        assert all(
            len(state["training_rows"]) == completed_updates
            for state in arm_states.values()
        )
        training_manifests = {}
        for arm in ARMS:
            state = arm_states[arm]
            final_policy_path = (
                state["dir"]
                / "checkpoints"
                / f"step-{completed_updates:04d}"
                / "policy"
            )
            if not (
                final_policy_path / "adapter_model.safetensors"
            ).exists():
                final_policy_path = save_policy_checkpoint(
                    state["model"], state["dir"], completed_updates
                )
            atomic_csv(
                pd.DataFrame(state["training_rows"]),
                state["dir"] / "training-history.csv",
            )
            assert (
                policy_digest(state["incumbent_parameters"])
                == state["initial_incumbent_digest"]
            )

            trained_checkpoint = TRAINED_CHECKPOINTS[arm]
            trained_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            state["model"].save_pretrained(
                str(trained_checkpoint), selected_adapters=["policy"]
            )
            trained_policy_dir = trained_checkpoint / "policy"
            assert (
                trained_policy_dir / "adapter_model.safetensors"
            ).exists()
            training_manifest = {
                "experiment": (
                    "Lab 19e matched simulator-ranked "
                    "answer-constrained policy"
                ),
                "arm": arm,
                "task_loss": (
                    "zero"
                    if arm == "preservation-control"
                    else "pairwise_entropy_ranking"
                ),
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "scorer_sha256": SCORER_SHA256,
                "incumbent_sha256": incumbent_sha256,
                "initial_policy_digest": state[
                    "initial_policy_digest"
                ],
                "initial_incumbent_digest": state[
                    "initial_incumbent_digest"
                ],
                "frozen_incumbent_digest": policy_digest(
                    state["incumbent_parameters"]
                ),
                "final_policy_digest": policy_digest(
                    state["policy_parameters"]
                ),
                "completed_updates": completed_updates,
                "planned_updates": UPDATES,
                "matched_hard_stop": matched_hard_stop,
                "arm_hard_stop": state["hard_stop"],
                "tripped_rules": state["tripped_rules"],
                "stop_step": stop_step,
                "final_policy_path": str(trained_policy_dir),
                "last_checkpoint_path": str(final_policy_path),
                "incumbent_top16_sha256": sha256_text(
                    incumbent_top_texts[arm]
                ),
                "online_refresh_sha256": {
                    path.name: sha256_file(path)
                    for path in sorted(
                        state["dir"].glob(
                            "online-refresh-step-*.jsonl"
                        )
                    )
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
            training_manifests[arm] = training_manifest
            atomic_json(
                training_manifest,
                state["dir"] / "training-manifest.json",
            )

        assert len(
            {
                manifest["initial_policy_digest"]
                for manifest in training_manifests.values()
            }
        ) == 1
        assert len(
            {
                manifest["initial_incumbent_digest"]
                for manifest in training_manifests.values()
            }
        ) == 1
        assert len(
            {
                manifest["completed_updates"]
                for manifest in training_manifests.values()
            }
        ) == 1
        atomic_json(
            {
                "arms": training_manifests,
                "initial_adapter_digest_equal": True,
                "frozen_incumbent_identity_equal": True,
                "matched_completed_updates": completed_updates,
                "matched_hard_stop": matched_hard_stop,
                "stop_step": stop_step,
            },
            RESULTS_DIR / "matched-training-manifest.json",
        )
        for arm in ARMS:
            release_model(arm_states[arm]["model"])
        del arm_states
        print("matched training manifests:", training_manifests)
    else:
        print("training skipped")
    """
)

md(
    """
    ## Frozen outcome evaluation

    There is one outcome path. Dev ranking uses deterministic argmax over all
    2,315 answer-constrained actions and measures regret against the global
    one-ply entropy maximum. Gameplay uses the same deterministic full-list
    scorer for the incumbent and both arms. When exactly one candidate remains,
    the environment closes deterministically for every policy, but it also
    records the raw model rank of that sole answer before closure.

    Only `entropy-ranking` can advance. It must improve mean full-list dev
    entropy regret by at least 0.10 bits against both the incumbent and matched
    control, complete without a treatment hard stop, pass every treatment
    mass/rank guard, and solve at least as many games as both comparators.
    """
)

code(
    """
    if RUN_TRAINING:
        def load_evaluation_model(
            policy_path: Path | None,
        ):
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32
            ).to(device)
            base.config.use_cache = False
            source = INCUMBENT if policy_path is None else policy_path
            evaluation_model = PeftModel.from_pretrained(
                base,
                source,
                adapter_name="policy",
                is_trainable=False,
            ).to(device)
            evaluation_model.load_adapter(
                INCUMBENT,
                adapter_name="incumbent",
                is_trainable=False,
            )
            evaluation_parameters = [
                parameter
                for name, parameter in evaluation_model.named_parameters()
                if "lora_" in name and ".policy." in name
            ]
            assert evaluation_parameters
            return evaluation_model, evaluation_parameters


        @torch.no_grad()
        def evaluate_dev_adapter(
            model,
            policy_parameters,
            adapter_name: str,
            result_name: str,
        ) -> pd.DataFrame:
            rows = []
            with active_adapter(
                model,
                policy_parameters,
                adapter_name,
                train=False,
            ):
                for record in target_records["dev"]:
                    candidates = set(
                        map(int, record["candidate_indices"])
                    )
                    scores = score_all_words(
                        model, record["prompt"]
                    )
                    entropies = np.asarray(
                        record["all_action_entropy_bits"],
                        dtype=np.float64,
                    )
                    chosen_index = int(scores.argmax())
                    chosen_entropy = float(entropies[chosen_index])
                    best_entropy = float(entropies.max())
                    rows.append(
                        {
                            "adapter": result_name,
                            "state_key": record["state_key"],
                            "turn": record["turn"],
                            "candidate_count": record["candidate_count"],
                            "candidate_bucket": record[
                                "candidate_bucket"
                            ],
                            "chosen_index": chosen_index,
                            "chosen_word": ANSWERS[chosen_index],
                            "chosen_is_candidate": (
                                chosen_index in candidates
                            ),
                            "open_teacher_word": record[
                                "open_teacher_word"
                            ],
                            "open_teacher_exact_match": (
                                chosen_index
                                == record["open_teacher_index"]
                            ),
                            "chosen_is_global_teacher_tie": (
                                entropies[chosen_index] == best_entropy
                            ),
                            "candidate_teacher_word": record[
                                "candidate_teacher_word"
                            ],
                            "candidate_only_teacher_match": (
                                chosen_index
                                == record["candidate_teacher_index"]
                            ),
                            "entropy_regret_bits": float(
                                best_entropy - chosen_entropy
                            ),
                        }
                    )
                    del scores
                    clear_device_cache()
            return pd.DataFrame(rows)


        def play_game(
            model,
            policy_parameters,
            adapter_name: str,
            result_name: str,
            answer: str,
        ) -> tuple[list[dict], dict]:
            history = [Turn(OPENING, score_string(answer, OPENING))]
            calls = []
            solved_turn = None
            for turn_number in range(2, MAX_TURNS + 1):
                candidates = candidate_indices_from_history(
                    history,
                    ANSWERS,
                    PATTERNS,
                    expert=EXPERT,
                )
                prompt = structured_next_guess_prompt(
                    history, len(candidates)
                )
                scores = full_list_for_adapter(
                    model,
                    policy_parameters,
                    adapter_name,
                    prompt,
                )
                ranks = rank_vector(scores)
                shifted = scores - scores.max()
                weights = np.exp(shifted)
                candidate_mass = float(
                    weights[candidates].sum() / weights.sum()
                )
                best_candidate_rank = int(ranks[candidates].min())
                raw_winner_index = int(scores.argmax())
                raw_winner_word = ANSWERS[raw_winner_index]
                raw_singleton_answer_rank = (
                    int(ranks[int(candidates[0])])
                    if len(candidates) == 1
                    else np.nan
                )
                driver_peak = LAST_STATE_PEAK_GIB
                if len(candidates) == 1:
                    winner_index = int(candidates[0])
                    guess = ANSWERS[winner_index]
                    decision_source = "singleton_closure"
                else:
                    winner_index = raw_winner_index
                    guess = ANSWERS[winner_index]
                    decision_source = "model_full_list"

                entropies = all_action_entropies(candidates)
                best_entropy = float(entropies.max())
                chosen_entropy = float(
                    entropies[winner_index]
                )
                feedback = score_string(answer, guess)
                calls.append(
                    {
                        "adapter": result_name,
                        "answer": answer,
                        "turn": turn_number,
                        "candidate_count_before": len(candidates),
                        "guess": guess,
                        "decision_source": decision_source,
                        "raw_model_winner": raw_winner_word,
                        "raw_model_chosen_is_candidate": (
                            raw_winner_index in set(map(int, candidates))
                        ),
                        "raw_singleton_answer_rank": (
                            raw_singleton_answer_rank
                        ),
                        "chosen_is_candidate": (
                            winner_index in set(map(int, candidates))
                        ),
                        "candidate_mass": candidate_mass,
                        "best_candidate_rank": best_candidate_rank,
                        "entropy_regret_bits": float(
                            best_entropy - chosen_entropy
                        ),
                        "feedback": feedback,
                        "driver_peak_gib": driver_peak,
                    }
                )
                history.append(Turn(guess, feedback))
                if feedback == "GGGGG":
                    solved_turn = turn_number
                    break
            return calls, {
                "adapter": result_name,
                "answer": answer,
                "solved": solved_turn is not None,
                "solved_turn": solved_turn,
                "model_calls": sum(
                    row["decision_source"] == "model_full_list"
                    for row in calls
                ),
                "raw_scoring_calls": len(calls),
                "singleton_closures": sum(
                    row["decision_source"] == "singleton_closure"
                    for row in calls
                ),
            }


        dev_frames = []
        gameplay_calls = []
        gameplay_games = []
        evaluation_specs = [
            ("incumbent", None, "incumbent"),
            (
                "preservation-control",
                Path(
                    training_manifests["preservation-control"][
                        "final_policy_path"
                    ]
                ),
                "policy",
            ),
            (
                "entropy-ranking",
                Path(
                    training_manifests["entropy-ranking"][
                        "final_policy_path"
                    ]
                ),
                "policy",
            ),
        ]
        for result_name, policy_path, adapter_name in evaluation_specs:
            evaluation_model, evaluation_parameters = (
                load_evaluation_model(policy_path)
            )
            dev_frames.append(
                evaluate_dev_adapter(
                    evaluation_model,
                    evaluation_parameters,
                    adapter_name,
                    result_name,
                )
            )
            for answer in RESERVED_ANSWERS:
                calls, game = play_game(
                    evaluation_model,
                    evaluation_parameters,
                    adapter_name,
                    result_name,
                    answer,
                )
                gameplay_calls.extend(calls)
                gameplay_games.append(game)
            release_model(evaluation_model)
            del evaluation_model

        dev_results = pd.concat(dev_frames, ignore_index=True)
        atomic_csv(dev_results, RESULTS_DIR / "dev-results.csv")
        dev_summary = dev_results.groupby("adapter", sort=True).agg(
            states=("state_key", "size"),
            mean_entropy_regret_bits=("entropy_regret_bits", "mean"),
            median_entropy_regret_bits=("entropy_regret_bits", "median"),
            open_teacher_exact_match_rate=(
                "open_teacher_exact_match",
                "mean",
            ),
            global_teacher_tie_match_rate=(
                "chosen_is_global_teacher_tie",
                "mean",
            ),
            chosen_is_candidate_rate=("chosen_is_candidate", "mean"),
            candidate_only_teacher_match_rate=(
                "candidate_only_teacher_match",
                "mean",
            ),
        ).reset_index()
        atomic_csv(dev_summary, RESULTS_DIR / "dev-summary.csv")

        gameplay_calls = pd.DataFrame(gameplay_calls)
        gameplay_games = pd.DataFrame(gameplay_games)
        atomic_csv(
            gameplay_calls, RESULTS_DIR / "gameplay-calls.csv"
        )
        atomic_csv(
            gameplay_games, RESULTS_DIR / "gameplay-games.csv"
        )
        game_summary = gameplay_games.groupby(
            "adapter", sort=True
        ).agg(
            games=("answer", "size"),
            solved=("solved", "sum"),
            solve_rate=("solved", "mean"),
            model_calls=("model_calls", "sum"),
            raw_scoring_calls=("raw_scoring_calls", "sum"),
            singleton_closures=("singleton_closures", "sum"),
        ).reset_index()
        raw_singleton_summary = gameplay_calls.loc[
            gameplay_calls["candidate_count_before"] == 1
        ].groupby("adapter", sort=True).agg(
            raw_singleton_calls=("answer", "size"),
            raw_singleton_median_rank=(
                "raw_singleton_answer_rank",
                "median",
            ),
            raw_singleton_top1_rate=(
                "raw_model_chosen_is_candidate",
                "mean",
            ),
        ).reset_index()
        game_summary = game_summary.merge(
            raw_singleton_summary,
            on="adapter",
            how="left",
            validate="one_to_one",
        )
        atomic_csv(game_summary, RESULTS_DIR / "game-summary.csv")

        dev_by_adapter = dev_summary.set_index("adapter")
        incumbent_dev = dev_by_adapter.loc["incumbent"]
        control_dev = dev_by_adapter.loc["preservation-control"]
        treatment_dev = dev_by_adapter.loc["entropy-ranking"]
        game_by_adapter = game_summary.set_index("adapter")
        treatment_vs_incumbent = float(
            incumbent_dev["mean_entropy_regret_bits"]
            - treatment_dev["mean_entropy_regret_bits"]
        )
        treatment_vs_control = float(
            control_dev["mean_entropy_regret_bits"]
            - treatment_dev["mean_entropy_regret_bits"]
        )
        difference_in_change = -treatment_vs_control

        final_rules = {}
        final_drifts = {}
        for arm in ARMS:
            arm_dir = RESULTS_DIR / "arms" / arm
            final_drifts[arm] = pd.read_csv(
                arm_dir / "drift-checks.csv"
            ).iloc[-1].to_dict()
            final_rule_record = json.loads(
                (arm_dir / "drift-rules.jsonl")
                .read_text()
                .splitlines()[-1]
            )
            final_rules[arm] = final_rule_record["rules"]
        treatment_rule_values = [
            rule
            for name, rule in final_rules["entropy-ranking"].items()
            if name != "any_tripped"
        ]
        treatment_mass_rank_guards_pass = not any(
            rule["tripped"]
            for rule in treatment_rule_values
            if rule["kind"]
            in {
                "candidate_mass",
                "best_candidate_rank",
                "raw_singleton_answer_rank",
            }
        )
        treatment_no_hard_stop = not training_manifests[
            "entropy-ranking"
        ]["arm_hard_stop"]
        matched_run_completed = (
            completed_updates == UPDATES and not matched_hard_stop
        )
        regret_vs_incumbent_pass = (
            treatment_vs_incumbent
            >= MIN_DEV_REGRET_IMPROVEMENT_BITS
        )
        regret_vs_control_pass = (
            treatment_vs_control
            >= MIN_CONTROL_ADJUSTED_REGRET_IMPROVEMENT_BITS
        )
        solve_not_below_incumbent = (
            int(game_by_adapter.loc["entropy-ranking", "solved"])
            >= int(game_by_adapter.loc["incumbent", "solved"])
        )
        solve_not_below_control = (
            int(game_by_adapter.loc["entropy-ranking", "solved"])
            >= int(
                game_by_adapter.loc["preservation-control", "solved"]
            )
        )
        if (
            matched_run_completed
            and treatment_no_hard_stop
            and treatment_mass_rank_guards_pass
            and regret_vs_incumbent_pass
            and regret_vs_control_pass
            and solve_not_below_incumbent
            and solve_not_below_control
        ):
            verdict = "advance"
        else:
            verdict = "do_not_advance"

        outcome = {
            "experiment": (
                "Lab 19e matched simulator-ranked answer-constrained policy"
            ),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "scorer_sha256": SCORER_SHA256,
            "incumbent_sha256": incumbent_sha256,
            "arm_adapter_sha256": {
                arm: sha256_file(
                    Path(training_manifests[arm]["final_policy_path"])
                    / "adapter_model.safetensors"
                )
                for arm in ARMS
            },
            "completed_updates": completed_updates,
            "matched_hard_stop": matched_hard_stop,
            "stop_step": stop_step,
            "mean_dev_entropy_regret_bits": {
                adapter: float(
                    dev_by_adapter.loc[
                        adapter, "mean_entropy_regret_bits"
                    ]
                )
                for adapter in (
                    "incumbent",
                    "preservation-control",
                    "entropy-ranking",
                )
            },
            "treatment_improvement_vs_incumbent_bits": (
                treatment_vs_incumbent
            ),
            "treatment_improvement_vs_control_bits": (
                treatment_vs_control
            ),
            "treatment_minus_control_change_in_regret_bits": (
                difference_in_change
            ),
            "solve_counts": {
                adapter: int(game_by_adapter.loc[adapter, "solved"])
                for adapter in (
                    "incumbent",
                    "preservation-control",
                    "entropy-ranking",
                )
            },
            "raw_singleton_ranking": {
                adapter: {
                    "median_rank": (
                        None
                        if pd.isna(
                            game_by_adapter.loc[
                                adapter, "raw_singleton_median_rank"
                            ]
                        )
                        else float(
                            game_by_adapter.loc[
                                adapter, "raw_singleton_median_rank"
                            ]
                        )
                    ),
                    "top1_rate": (
                        None
                        if pd.isna(
                            game_by_adapter.loc[
                                adapter, "raw_singleton_top1_rate"
                            ]
                        )
                        else float(
                            game_by_adapter.loc[
                                adapter, "raw_singleton_top1_rate"
                            ]
                        )
                    ),
                }
                for adapter in (
                    "incumbent",
                    "preservation-control",
                    "entropy-ranking",
                )
            },
            "arm_outcomes": {
                arm: {
                    "hard_stop": training_manifests[arm][
                        "arm_hard_stop"
                    ],
                    "tripped_rules": training_manifests[arm][
                        "tripped_rules"
                    ],
                    "adapter_relative_delta": float(
                        final_drifts[arm]["adapter_relative_delta"]
                    ),
                    "per_regime": json.loads(
                        final_drifts[arm]["per_regime_json"]
                    ),
                }
                for arm in ARMS
            },
            "criteria": {
                "matched_run_completed": matched_run_completed,
                "treatment_no_hard_stop": treatment_no_hard_stop,
                "treatment_mass_rank_guards_pass": (
                    treatment_mass_rank_guards_pass
                ),
                "treatment_improves_vs_incumbent": (
                    regret_vs_incumbent_pass
                ),
                "treatment_improves_vs_control": regret_vs_control_pass,
                "treatment_solves_not_below_incumbent": (
                    solve_not_below_incumbent
                ),
                "treatment_solves_not_below_control": (
                    solve_not_below_control
                ),
            },
            "selected_trained_recipe": (
                "entropy-ranking" if verdict == "advance" else None
            ),
            "control_is_never_selectable": True,
            "verdict": verdict,
            "claims": (
                "This seed-45 result concerns ranking across the "
                "answer-constrained 2,315-action policy under the frozen state "
                "sample and deterministic 19-game battery only."
            ),
        }
        atomic_json(outcome, RESULTS_DIR / "lab19e-run.json")
        display(dev_summary)
        display(game_summary)
        display(pd.DataFrame([outcome]))
        print("verdict:", verdict)
    else:
        print("frozen outcome evaluation skipped")
    """
)

md(
    """
    ## Interpretation boundary

    An advance means the supervised ranker passed this seed-45 mechanism test.
    It does not establish seed replication or a population gameplay gain. A
    failure after measurable adapter movement rejects these frozen
    hyperparameters. A run below the preregistered movement floor is
    inconclusive rather than evidence against simulator-ranked supervision.
    """
)

for index, cell in enumerate(cells):
    cell["id"] = f"lab19e-{index:02d}-{cell['cell_type']}"

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
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path("notebooks/19e_simulator_ranked_policy.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {output}")
