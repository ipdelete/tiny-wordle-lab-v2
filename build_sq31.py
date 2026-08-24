"""Generate notebooks/sq31_wordle_environment.ipynb."""

import json
import textwrap
from pathlib import Path

cells = []


def md(text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": textwrap.dedent(text).strip("\n").splitlines(
                keepends=True
            ),
        }
    )


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": textwrap.dedent(text).strip("\n").splitlines(
                keepends=True
            ),
        }
    )


md(
    """
    # SQ31 - Wordle environment and stochastic policy

    This side quest pulls the environment and sampling-policy gate forward
    from Labs 31 and 34. It starts from the Lab 18d seed 45 incumbent, changes
    no model weights, and stops before simulator GRPO.

    Gate A is model-free. It checks the frozen representation, transitions,
    replay traces, and the future update-data contract. Gate B loads the
    checkpoint and checks the tokenizer, trie policy, memory behavior, and
    sparse-reward diversity pilot. If Gate B cannot produce mixed win/loss
    groups, SQ34 is blocked. The notebook records that null instead of adding
    shaped reward.
    """
)

md(
    """
    ## SQ31.1 Contract and preregistration

    The environment owns the hidden answer, candidate tracking, transitions,
    terminal truth, and protected diagnostics. The policy sees only the
    `derived_state_v1` observation. The action vocabulary is the 2,315 original
    answer words. The fixed `RAISE` opening is a real first transition and
    consumes one of six turns, including the valid answer `RAISE`.

    Training uses token-level trie sampling. The deterministic full-string
    argmax remains the evaluation decoder. SQ31 measures their disagreement
    rather than treating them as the same policy.
    """
)

code(
    """
    import hashlib
    import json
    import math
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from tiny_wordle.benchmark import DEFAULT_EVAL_ANSWERS, parse_guess
    from tiny_wordle.environment import (
        EnvironmentConfig,
        WordleEnvironment,
        replay_trace,
        serialize_trace,
    )
    from tiny_wordle.expert import EntropyExpert
    from tiny_wordle.game import score_string
    from tiny_wordle.representation import (
        candidate_indices_from_history,
        parse_state_key,
        structured_next_guess_prompt,
    )
    from tiny_wordle.rollout import PolicyDecision, collect_trajectory

    ROOT = Path.cwd()
    if not (ROOT / "data").exists():
        ROOT = ROOT.parent
    DATA_DIR = ROOT / "data"
    RESULTS_DIR = ROOT / "results" / "sq31"
    LAB18D_DIR = ROOT / "results" / "lab18d"
    ANSWERS = tuple(
        line.strip().upper()
        for line in (DATA_DIR / "wordle-answers-original.txt").read_text().splitlines()
        if line.strip()
    )
    PATTERNS = np.load(DATA_DIR / "wordle-patterns-original-2315.npy")
    EXPERT = EntropyExpert(list(ANSWERS), PATTERNS)
    RESERVED_ANSWERS = tuple(DEFAULT_EVAL_ANSWERS)
    RESERVED_SET = set(RESERVED_ANSWERS)
    CONFIG = EnvironmentConfig(
        answers=ANSWERS,
        opening="RAISE",
        max_turns=6,
    )
    RUN_MODEL = os.environ.get("SQ31_RUN_MODEL", "0") == "1"

    assert len(ANSWERS) == 2315
    assert PATTERNS.shape == (2315, 2315)
    assert len(set(ANSWERS)) == len(ANSWERS)
    assert CONFIG.answers == ANSWERS
    print("answers:", len(ANSWERS))
    print("run model gate:", RUN_MODEL)
    """
)

code(
    """
    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    manifest = json.loads((LAB18D_DIR / "lab18d-run.json").read_text())
    assert manifest["opening"] == "RAISE"
    assert manifest["max_turns"] == 6
    assert manifest["answers"] == list(RESERVED_ANSWERS)
    CHECKPOINT = (
        ROOT
        / "checkpoints"
        / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed45"
    )
    ADAPTER = CHECKPOINT / "adapter_model.safetensors"
    checkpoint_sha256 = sha256_file(ADAPTER) if ADAPTER.exists() else None
    if checkpoint_sha256 is not None:
        assert checkpoint_sha256 == manifest["checkpoint_sha256"]["45"]
    print("seed 45 adapter available:", ADAPTER.exists())
    print("seed 45 adapter sha256:", checkpoint_sha256)
    """
)

md(
    """
    ## SQ31.2 Gate A: representation fidelity

    Rebuild every unique `NEXT_GUESS` prompt in the stored structured train and
    validation corpora from its state key. This is the guard against drifting
    while `build_18d.py` through `build_20.py` retain their historical inline
    copies.
    """
)

code(
    """
    structured_files = [
        DATA_DIR / "generated/wordle-part2-structured-train.jsonl",
        DATA_DIR / "generated/wordle-part2-structured-dev.jsonl",
    ]
    seen_states = set()
    verified_states = 0
    for path in structured_files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["task"] != "NEXT_GUESS":
                continue
            key = (row["split"], row["state_key"])
            if key in seen_states:
                continue
            seen_states.add(key)
            history = parse_state_key(row["state_key"])
            candidates = candidate_indices_from_history(
                history,
                ANSWERS,
                PATTERNS,
                expert=EXPERT,
            )
            assert len(candidates) == row["candidate_count"]
            assert structured_next_guess_prompt(
                history,
                len(candidates),
            ) == row["prompt"]
            verified_states += 1
    print("stored structured states reproduced:", verified_states)
    """
)

md(
    """
    ## SQ31.3 Gate A: environment walkthrough and terminal rules

    The following examples exercise a valid wrong action, a repeat, a solved
    opening, an invalid action, and exhaustion without an opening. The hidden
    answer appears only in protected `info` after termination, never in the
    observation prompt.
    """
)

code(
    """
    def make_environment(*, opening="RAISE", max_turns=6):
        return WordleEnvironment(
            EnvironmentConfig(
                answers=ANSWERS,
                opening=opening,
                max_turns=max_turns,
            ),
            expert=EXPERT,
            patterns=PATTERNS,
        )


    environment = make_environment()
    start = environment.reset("SHORE", episode_id="walkthrough", seed=31)
    print(
        "after opening:",
        start.observation.turn,
        start.observation.candidate_count,
        start.opening_record.feedback,
    )
    _, _, _, repeat_info = environment.step("CRANE")
    _, _, _, repeat_info = environment.step("CRANE")
    print("repeat:", repeat_info["repeated"])
    environment.step("SHORE")
    print("solved:", environment.last_record.terminal_reason)

    solved_opening = make_environment().reset("RAISE")
    assert solved_opening.done
    assert solved_opening.reward == 1
    assert solved_opening.opening_record.terminal_reason == "solved"

    invalid_environment = make_environment()
    invalid_environment.reset("SHORE")
    _, _, invalid_done, invalid_info = invalid_environment.step("crane")
    assert invalid_done
    assert invalid_info["terminal_reason"] == "contract_violation"
    assert invalid_environment.last_record.feedback is None

    exhausted = make_environment(opening=None, max_turns=2)
    exhausted.reset("SHORE")
    exhausted.step("CRANE")
    _, reward, done, info = exhausted.step("APPLE")
    assert done and reward == 0
    assert info["terminal_reason"] == "exhausted"
    print("terminal rules: solved, contract_violation, exhausted")
    """
)

code(
    """
    assert "SHORE" not in start.observation.prompt
    assert "answer" not in repeat_info
    assert parse_guess("crane") == "CRANE"
    divergence = make_environment()
    divergence.reset("SHORE")
    _, _, done, info = divergence.step("crane")
    assert done and info["terminal_reason"] == "contract_violation"
    print("benchmark divergence documented: lowercase is normalized by benchmark")
    print("environment contract rejects lowercase as an out-of-vocabulary action")
    """
)

md(
    """
    ## SQ31.4 Gate A: replay the Lab 18d trajectories

    These are the real seed 42, 45, and 47 answer-constrained trajectories.
    Replay checks feedback, candidate counts, repeat flags, and terminal
    outcomes. The stored rows contain only shared valid actions, so malformed
    and out-of-vocabulary behavior is tested separately above.
    """
)

code(
    """
    def replay_seed(seed):
        calls_path = LAB18D_DIR / f"seed{seed}-answer-constrained-calls.csv"
        games_path = LAB18D_DIR / f"seed{seed}-answer-constrained-games.csv"
        calls = pd.read_csv(calls_path)
        games = pd.read_csv(games_path)
        checked = 0
        for answer, rows in calls.groupby("answer", sort=False):
            env = make_environment()
            env.reset(answer, episode_id=f"seed{seed}-{answer}", seed=seed)
            for row in rows.itertuples(index=False):
                if env.done:
                    break
                env.step(row.guess)
                record = env.last_record
                assert record.feedback == row.feedback
                assert record.candidate_count_before == row.candidate_count_before
                assert record.candidate_count_after == row.candidate_count_after
                assert record.repeated == bool(row.repeated)
                checked += 1
            game = games.loc[games["answer"] == answer].iloc[0]
            final = env.last_record
            assert bool(game.solved) == (final.terminal_reason == "solved")
            assert int(game.model_calls) == len(rows)
            if bool(game.solved):
                assert int(game.solved_turn) == final.turn
        return checked


    replay_counts = {seed: replay_seed(seed) for seed in (42, 45, 47)}
    print("replayed calls:", replay_counts)
    """
)

code(
    """
    trace_environment = make_environment()
    trace_environment.reset("SHORE", episode_id="trace", seed=9)
    trace_environment.step("CRANE")
    trace_environment.step("SHORE")
    trace_payload = serialize_trace(trace_environment.trace)
    assert replay_trace(
        trace_payload,
        config=CONFIG,
        patterns=PATTERNS,
    ).to_dict() == trace_payload
    tampered = json.loads(json.dumps(trace_payload))
    tampered["records"][1]["feedback"] = "GGGGG"
    try:
        replay_trace(tampered, config=CONFIG, patterns=PATTERNS)
    except ValueError:
        print("trace tamper detection: passed")
    else:
        raise AssertionError("tampered trace replayed")
    """
)

md(
    """
    ## SQ31.5 Gate A: SQ34 update-data contract

    SQ34 owns importance ratios, clipping, advantages, and reference KL. SQ31
    proves that one immutable trajectory contains every input those
    calculations need: observations, actions, behavior probabilities, policy
    identity, reference identity, and transition diagnostics. No optimizer
    objective lives in this notebook.
    """
)

code(
    """
    class ContractPolicy:
        action_vocabulary_digest = "contract-vocabulary"

        def sample(self, observation, *, temperature, seed):
            return PolicyDecision(
                word="SHORE",
                token_ids=(1, 2, 3),
                per_token_log_probabilities=(-0.2, -0.3, 0.0),
                action_log_probability=-0.5,
                checkpoint_digest="behavior-checkpoint",
                temperature=temperature,
                mask_version="answer-token-trie-v1",
                sampling_seed=seed,
                tokenizer_digest="contract-tokenizer",
                model_calls=1,
            )

        def log_probability(self, observation, word, *, temperature):
            return -0.5, (-0.2, -0.3, 0.0)


    contract_trajectory = collect_trajectory(
        make_environment(),
        ContractPolicy(),
        "SHORE",
        group_id="contract-group",
        answer_split="development",
        reference_checkpoint_digest="reference-checkpoint",
        temperature=1.0,
        sampling_seed=31,
        episode_id="contract-episode",
    )
    assert contract_trajectory.reference_checkpoint_digest
    assert contract_trajectory.policy_checkpoint_digest
    assert contract_trajectory.tokenizer_digest
    assert contract_trajectory.action_vocabulary_digest
    assert contract_trajectory.return_value == 1
    assert contract_trajectory.steps
    contract_step = contract_trajectory.steps[0]
    assert contract_step.observation == contract_step.transition.observation_before
    assert contract_step.decision.action_log_probability == -0.5
    assert contract_step.transition.teacher_diagnostics.action_entropy_bits is not None
    print("SQ34 trace inputs:", sorted(contract_trajectory.to_dict()))
    """
)

md(
    """
    ## SQ31.6 Gate B: freeze the seed 45 checkpoint and tokenizer

    The model gate is opt-in with `SQ31_RUN_MODEL=1`. Gate A remains executable
    on a machine without the local checkpoint.
    """
)

code(
    """
    if RUN_MODEL:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tiny_wordle.hardware import preferred_device
        from tiny_wordle.policy import (
            TokenTrie,
            TriePolicy,
            digest_action_vocabulary,
            digest_tokenizer,
        )

        MODEL_ID = "Qwen/Qwen3-0.6B"
        MEMORY_CAP_GIB = 128.0
        MEMORY_ABORT_GIB = 96.0
        if not ADAPTER.exists():
            raise FileNotFoundError(f"missing seed 45 adapter: {ADAPTER}")
        device = preferred_device()
        if device.type == "mps":
            total_gib = torch.mps.recommended_max_memory() / 1024**3
            torch.mps.set_per_process_memory_fraction(
                MEMORY_CAP_GIB / total_gib
            )
            print(f"MPS cap: {MEMORY_CAP_GIB:.0f} GiB of {total_gib:.0f} GiB")

        def driver_memory_gib():
            if device.type == "mps":
                return torch.mps.driver_allocated_memory() / 1024**3
            if device.type == "cuda":
                return torch.cuda.memory_allocated() / 1024**3
            return float("nan")


        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer_digest = digest_tokenizer(tokenizer)

        def render_prompt(prompt):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        probe = make_environment().reset("SHORE")
        rendered_probe = render_prompt(probe.observation.prompt)
        trie = TokenTrie.from_tokenizer(
            tokenizer,
            ANSWERS,
            rendered_prompt=rendered_probe,
        )
        trie_shape = trie.shape()
        print("trie shape before model load:", trie_shape)
        assert trie_shape["leaf_count"] == len(ANSWERS)
        assert trie_shape["branching_by_depth"] == {0: 1, 1: 235, 2: 113}
        assert trie_shape["internal_nodes"] == 3364
        assert trie_shape["maximum_depth"] == 4
        assert max(trie_shape["branching_by_depth"].values()) <= 256
        assert len(trie_shape["branching_by_depth"]) <= 3

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
        ).to(device)
        model = PeftModel.from_pretrained(base_model, CHECKPOINT).to(device).eval()
        policy = TriePolicy(
            model,
            tokenizer,
            trie,
            device=device,
            prompt_renderer=render_prompt,
            checkpoint_digest=checkpoint_sha256,
            tokenizer_digest=tokenizer_digest,
            memory_probe=driver_memory_gib,
        )
        policy.verify_prompt_contract(rendered_probe)
        print("device:", device)
        print("tokenizer contract: passed for", len(ANSWERS), "answers")
    else:
        print("model gate skipped; set SQ31_RUN_MODEL=1 for Gate B")
    """
)

md(
    """
    ## SQ31.7 Gate B: probabilities and memory

    Trie sampling normalizes only the allowed next tokens. Forced trie edges
    have probability one. The exact distribution batches the 349 branching
    nodes by depth, instead of treating a 2,315-word Monte Carlo sample as the
    reference.
    """
)

code(
    """
    if RUN_MODEL:
        from collections import Counter

        probe_environment = make_environment()
        probe_start = probe_environment.reset("SHORE")
        probe_observation = probe_start.observation
        exact_trace_start = len(policy.forward_memory_trace)
        exact = policy.exact_distribution(
            probe_observation,
            temperature=1.0,
            max_model_calls=3,
            max_batch_size=256,
        )
        exact_forward_peaks = policy.forward_memory_trace[exact_trace_start:]
        finite_exact_peaks = [
            value for value in exact_forward_peaks if math.isfinite(value)
        ]
        if finite_exact_peaks:
            assert max(finite_exact_peaks) < MEMORY_ABORT_GIB
        assert abs(sum(exact.values()) - 1.0) < 1e-6
        decision = policy.sample(
            probe_observation,
            temperature=1.0,
            seed=31,
        )
        total, per_token = policy.log_probability(
            probe_observation,
            decision.word,
            temperature=decision.temperature,
        )
        assert math.isclose(
            total,
            decision.action_log_probability,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        assert len(per_token) == len(decision.per_token_log_probabilities)
        assert all(
            math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
            for actual, expected in zip(
                per_token,
                decision.per_token_log_probabilities,
            )
        )
        assert decision.word in exact
        assert decision.token_ids == policy.trie.sequence_for_word(decision.word)
        print("exact distribution:", len(exact), "actions")
        print(
            "exact-walk live driver peaks GiB:",
            exact_forward_peaks,
        )
        print("sampled action:", decision.word, decision.action_log_probability)
    """
)

code(
    """
    if RUN_MODEL:
        def assert_memory_plateau(peaks, label):
            if not peaks:
                return
            third = len(peaks) // 3
            middle = peaks[third:2 * third]
            final = peaks[-third:]
            creep = sum(final) / len(final) - sum(middle) / len(middle)
            assert creep < 0.5, f"{label} still climbing {creep:+.2f} GiB"
            assert max(final) - min(final) < 0.5, (
                f"{label} working set has not plateaued"
            )
            assert max(peaks) < MEMORY_ABORT_GIB, (
                f"{label} peak {max(peaks):.1f} GiB exceeds abort threshold"
            )


        sampling_driver_peaks = []
        for iteration in range(40):
            trace_start = len(policy.forward_memory_trace)
            policy.sample(
                probe_observation,
                temperature=1.0,
                seed=1000 + iteration,
            )
            iteration_peaks = policy.forward_memory_trace[trace_start:]
            finite = [value for value in iteration_peaks if math.isfinite(value)]
            if finite:
                sampling_driver_peaks.append(max(finite))
            if device.type == "mps":
                torch.mps.empty_cache()
        assert_memory_plateau(sampling_driver_peaks, "sampling")
        print("sampling memory soak:", sampling_driver_peaks)

        exact_driver_peaks = []
        for iteration in range(40):
            trace_start = len(policy.forward_memory_trace)
            policy.exact_distribution(
                probe_observation,
                temperature=1.0,
                max_model_calls=3,
                max_batch_size=256,
            )
            iteration_peaks = policy.forward_memory_trace[trace_start:]
            finite = [value for value in iteration_peaks if math.isfinite(value)]
            if finite:
                exact_driver_peaks.append(max(finite))
            if device.type == "mps":
                torch.mps.empty_cache()
        assert_memory_plateau(exact_driver_peaks, "exact walk")
        print("exact-walk memory soak:", exact_driver_peaks)
    """
)

code(
    """
    if RUN_MODEL:
        frequency_sample_count = 512
        sampled_words = [
            policy.sample(
                probe_observation,
                temperature=1.0,
                seed=2000 + index,
            ).word
            for index in range(frequency_sample_count)
        ]
        observed = Counter(sampled_words)
        bins = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 + 1e-12)

        def probability_bin(value):
            for index in range(len(bins) - 1):
                if bins[index] <= value < bins[index + 1]:
                    return index
            return len(bins) - 2

        expected_bins = np.zeros(len(bins) - 1)
        observed_bins = np.zeros(len(bins) - 1)
        for word, probability in exact.items():
            index = probability_bin(probability)
            expected_bins[index] += probability
            observed_bins[index] += observed[word] / frequency_sample_count
        total_variation = float(
            0.5 * np.abs(expected_bins - observed_bins).sum()
        )
        assert total_variation < 0.15
        print(
            "binned frequency total variation:",
            total_variation,
            "expected:",
            expected_bins,
            "observed:",
            observed_bins,
        )
    """
)

md(
    """
    ## SQ31.8 Gate B: deterministic baseline demo and decoder gap

    The persisted Lab 18d score matrix is the frozen seed 45 full-string
    argmax evidence. Its row order is verified against the call log before
    comparing it with the trie decoder on the same observations.
    """
)

code(
    """
    seed45_calls = pd.read_csv(
        LAB18D_DIR / "seed45-answer-constrained-calls.csv"
    )
    seed45_keys = pd.read_csv(
        LAB18D_DIR / "seed45-answer-constrained-score-keys.csv"
    )
    seed45_scores = np.load(
        LAB18D_DIR / "seed45-answer-constrained-scores.npy"
    )
    answer_array = np.asarray(ANSWERS)
    assert len(seed45_calls) == len(seed45_keys) == len(seed45_scores)
    assert seed45_keys[["answer", "turn"]].astype(str).reset_index(drop=True).equals(
        seed45_calls[["answer", "turn"]].astype(str).reset_index(drop=True)
    )
    persisted_argmax = answer_array[seed45_scores.argmax(axis=1)]
    assert np.array_equal(persisted_argmax, seed45_calls["guess"].to_numpy())
    print("persisted full-string argmax matches all", len(seed45_calls), "calls")
    """
)

code(
    """
    if RUN_MODEL:
        trie_predictions = []
        row_index = 0
        for answer, rows in seed45_calls.groupby("answer", sort=False):
            env = make_environment()
            observation = env.reset(answer).observation
            for row in rows.itertuples(index=False):
                trie_predictions.append(policy.greedy_word(observation))
                observation, _, done, _ = env.step(row.guess)
                row_index += 1
                if done:
                    break
        agreement = float(
            np.mean(np.asarray(trie_predictions) == persisted_argmax)
        )
        print("greedy-trie/full-string agreement:", agreement)
    """
)

md(
    """
    ## SQ31.9 Gate B: reward-diversity pilot

    Choose temperature on a calibration half of development answers and report
    it on confirmation. This notebook freezes a temperature before training.
    It does not silently retune temperature if GRPO later sharpens the policy.
    Mean sampled action surprisal estimates policy entropy by turn without
    enumerating the complete distribution at every trajectory state.
    """
)

code(
    """
    if RUN_MODEL:
        from tiny_wordle.rollout import collect_trajectory

        dev_rows = [
            json.loads(line)
            for line in (
                DATA_DIR / "generated/wordle-part2-structured-dev.jsonl"
            ).read_text().splitlines()
        ]
        dev_answers = sorted(
            {
                row["answer"]
                for row in dev_rows
                if row["answer"] not in RESERVED_SET
            },
            key=lambda answer: hashlib.sha256(
                f"sq31-pilot|{answer}".encode()
            ).hexdigest(),
        )
        pilot_answers = dev_answers[:32]
        calibration_answers = pilot_answers[:16]
        confirmation_answers = pilot_answers[16:]
        assert set(calibration_answers).isdisjoint(confirmation_answers)
        assert not set(pilot_answers) & RESERVED_SET
        temperatures = (0.5, 0.75, 1.0, 1.25, 1.5)
        groups_per_split = 16
        group_size = 4
        minimum_mixed_fraction = 0.10
        minimum_mixed_lower_bound = 0.02
        assert groups_per_split == len(calibration_answers)
        assert groups_per_split == len(confirmation_answers)

        def wilson_lower(successes, trials, z=1.96):
            if trials == 0:
                return 0.0
            fraction = successes / trials
            denominator = 1 + z**2 / trials
            center = fraction + z**2 / (2 * trials)
            margin = z * math.sqrt(
                fraction * (1 - fraction) / trials
                + z**2 / (4 * trials**2)
            )
            return (center - margin) / denominator

        def run_pilot(temperatures, answers, label):
            group_rows = []
            episode_rows = []
            turn_rows = []
            for temperature in temperatures:
                for group_index in range(groups_per_split):
                    answer = answers[group_index % len(answers)]
                    returns = []
                    for member in range(group_size):
                        trajectory = collect_trajectory(
                            make_environment(),
                            policy,
                            answer,
                            group_id=f"{label}-{temperature}-{group_index}",
                            answer_split="development",
                            reference_checkpoint_digest=checkpoint_sha256,
                            temperature=temperature,
                            sampling_seed=31000 + group_index * 100 + member,
                            episode_id=f"{label}-{group_index}-{member}",
                        )
                        returns.append(trajectory.return_value)
                        episode_rows.append(
                            {
                                "temperature": temperature,
                                "episode_length": len(trajectory.records),
                                "model_calls": sum(
                                    step.decision.model_calls
                                    for step in trajectory.steps
                                ),
                                "repeat_rate": (
                                    sum(
                                        step.transition.repeated
                                        for step in trajectory.steps
                                    )
                                    / len(trajectory.steps)
                                    if trajectory.steps
                                    else 0.0
                                ),
                            }
                        )
                        turn_rows.extend(
                            {
                                "temperature": temperature,
                                "turn": step.transition.turn,
                                "action_surprisal_nats": (
                                    -step.decision.action_log_probability
                                ),
                            }
                            for step in trajectory.steps
                        )
                    group_rows.append(
                        {
                            "temperature": temperature,
                            "mixed": int(0 < sum(returns) < group_size),
                            "all_zero": int(sum(returns) == 0),
                            "all_one": int(sum(returns) == group_size),
                            "solve_rate": float(np.mean(returns)),
                        }
                    )
            group_frame = pd.DataFrame(group_rows)
            episode_frame = pd.DataFrame(episode_rows)
            turn_frame = pd.DataFrame(turn_rows)
            summary = group_frame.groupby("temperature", as_index=False).agg(
                groups=("mixed", "size"),
                mixed_groups=("mixed", "sum"),
                mixed_fraction=("mixed", "mean"),
                all_zero_fraction=("all_zero", "mean"),
                all_one_fraction=("all_one", "mean"),
                solve_rate=("solve_rate", "mean"),
            )
            episode_summary = episode_frame.groupby(
                "temperature",
                as_index=False,
            ).agg(
                mean_episode_length=("episode_length", "mean"),
                mean_model_calls=("model_calls", "mean"),
                repeat_rate=("repeat_rate", "mean"),
            )
            summary = summary.merge(
                episode_summary,
                on="temperature",
                validate="one_to_one",
            )
            summary["mixed_lower_95"] = [
                wilson_lower(int(row.mixed_groups), int(row.groups))
                for row in summary.itertuples()
            ]
            turn_summary = turn_frame.groupby(
                ["temperature", "turn"],
                as_index=False,
            ).agg(
                action_entropy_nats=(
                    "action_surprisal_nats",
                    "mean",
                ),
                sampled_actions=("action_surprisal_nats", "size"),
            )
            return summary, turn_summary

        calibration, calibration_turns = run_pilot(
            temperatures,
            calibration_answers,
            "calibration",
        )
        best_temperature = (
            calibration.sort_values(
                ["mixed_fraction", "temperature"],
                ascending=[False, True],
            ).iloc[0]["temperature"]
        )
        confirmation, confirmation_turns = run_pilot(
            (best_temperature,),
            confirmation_answers,
            "confirmation",
        )
        selected_confirmation = confirmation.iloc[0]
        sampling_gate_passed = bool(
            selected_confirmation.mixed_fraction >= minimum_mixed_fraction
            and selected_confirmation.mixed_lower_95 >= minimum_mixed_lower_bound
        )
        print("calibration:")
        display(calibration)
        display(calibration_turns)
        print("confirmation:")
        display(confirmation)
        display(confirmation_turns)
        print("frozen temperature:", best_temperature)
        print("sampling gate passed:", sampling_gate_passed)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        run_manifest = {
            "experiment": "SQ31 Wordle environment and stochastic policy",
            "representation": "derived_state_v1",
            "checkpoint_sha256": checkpoint_sha256,
            "action_vocabulary_sha256": digest_action_vocabulary(ANSWERS),
            "tokenizer_sha256": tokenizer_digest,
            "opening": "RAISE",
            "max_turns": 6,
            "reserved_answers": list(RESERVED_ANSWERS),
            "temperatures": list(temperatures),
            "frozen_temperature": float(best_temperature),
            "minimum_confirmation_mixed_fraction": minimum_mixed_fraction,
            "minimum_confirmation_mixed_lower_bound": minimum_mixed_lower_bound,
            "sampling_gate_passed": sampling_gate_passed,
            "calibration": calibration.to_dict(orient="records"),
            "calibration_action_entropy_by_turn": calibration_turns.to_dict(
                orient="records"
            ),
            "confirmation": confirmation.to_dict(orient="records"),
            "confirmation_action_entropy_by_turn": (
                confirmation_turns.to_dict(orient="records")
            ),
            "trie_shape": policy.trie.shape(),
        }
        (RESULTS_DIR / "sq31-run.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True)
        )
        print("wrote", RESULTS_DIR / "sq31-run.json")
    """
)

md(
    """
    ## SQ31 checkpoint

    Gate A must pass before the model gate is trusted. SQ31 is complete only
    when the model gate either freezes a development temperature with enough
    mixed groups or records the preregistered sparse-reward null. SQ34 does not
    start from an all-zero or all-one pilot.
    """
)


for index, cell in enumerate(cells):
    cell["id"] = f"sq31-{index:02d}-{cell['cell_type']}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = Path("notebooks/sq31_wordle_environment.ipynb")
path.write_text(json.dumps(notebook, indent=1))
print(f"wrote {path} with {len(cells)} cells")
