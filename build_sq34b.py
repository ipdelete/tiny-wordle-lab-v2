"""Generate notebooks/sq34b_optimization_study.ipynb."""

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
    # SQ34b - Optimization study

    SQ34 moved the LoRA, failed to improve either decoder, and tripped the
    candidate-mass floor. That rejects `5e-5 / 0.02`. It does not reject
    simulator GRPO.

    This notebook asks a smaller question. Can the same sparse reward carry
    this adapter through SQ34's failure horizon without wrecking candidate
    mass, if the step is smaller or the reference is held harder?

    Three contemporaneous arms, all from seed 45, warmup 8, at most six
    rounds. The old SQ34 path stays on the plot as a prior run. It is not
    the control.
    """
)

md(
    """
    ## SQ34b.1 Preregistration

    The knobs, the horizon, and the advance rules go to disk before any
    sampling. A rule invented after the numbers is not a rule.
    """
)

code(
    """
    import hashlib
    import json
    import math
    import os
    import time
    from collections import Counter
    from typing import NamedTuple
    from contextlib import contextmanager
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from tiny_wordle.benchmark import DEFAULT_EVAL_ANSWERS
    from tiny_wordle.environment import EnvironmentConfig, WordleEnvironment
    from tiny_wordle.expert import EntropyExpert
    from tiny_wordle.grpo import (
        GroupBatch,
        GrpoHyperparameters,
        episode_objective,
        group_advantages,
        group_loss,
    )
    from tiny_wordle.representation import (
        candidate_indices_from_history,
        parse_state_key,
        structured_next_guess_prompt,
    )
    from tiny_wordle.rollout import collect_trajectory

    ROOT = Path.cwd()
    if not (ROOT / "data").exists():
        ROOT = ROOT.parent
    DATA_DIR = ROOT / "data"
    RESULTS_DIR = ROOT / "results" / "sq34b"
    SQ31_DIR = ROOT / "results" / "sq31"
    SQ34_DIR = ROOT / "results" / "sq34"
    LAB18D_DIR = ROOT / "results" / "lab18d"
    LAB20_DIR = ROOT / "results" / "lab20"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ANSWERS = tuple(
        line.strip().upper()
        for line in (DATA_DIR / "wordle-answers-original.txt").read_text().splitlines()
        if line.strip()
    )
    PATTERNS = np.load(DATA_DIR / "wordle-patterns-original-2315.npy")
    EXPERT = EntropyExpert(list(ANSWERS), PATTERNS)
    RESERVED_ANSWERS = tuple(DEFAULT_EVAL_ANSWERS)
    RESERVED_SET = set(RESERVED_ANSWERS)
    OPENING = "RAISE"
    MAX_TURNS = 6

    RUN_MODEL = os.environ.get("SQ34B_RUN_MODEL", "0") == "1"

    assert len(ANSWERS) == 2315
    assert PATTERNS.shape == (2315, 2315)
    assert len(RESERVED_ANSWERS) == 19
    print("answers:", len(ANSWERS))
    print("reserved answers:", len(RESERVED_ANSWERS))
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


    def sha256_text(text):
        return hashlib.sha256(text.encode()).hexdigest()


    def atomic_json(value, path):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
        temporary.replace(path)


    sq31 = json.loads((SQ31_DIR / "sq31-run.json").read_text())
    sq34_run = json.loads((SQ34_DIR / "sq34-run.json").read_text())
    sq34_rounds = pd.read_csv(SQ34_DIR / "sq34-rounds.csv")
    sq34_prereg = json.loads((SQ34_DIR / "sq34-preregistration.json").read_text())
    assert sq31["sampling_gate_passed"] is True
    assert sq31["opening"] == OPENING
    assert sq31["max_turns"] == MAX_TURNS
    assert tuple(sq31["reserved_answers"]) == RESERVED_ANSWERS
    assert sq34_run["pass_criterion_met"] is False
    TEMPERATURE = float(sq31["frozen_temperature"])

    CHECKPOINT = (
        ROOT
        / "checkpoints"
        / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed45"
    )
    ADAPTER = CHECKPOINT / "adapter_model.safetensors"
    checkpoint_sha256 = sha256_file(ADAPTER) if ADAPTER.exists() else None
    EXPECTED_FILE_HASH = (
        "a3b849ac3cbc57c085ec4f1f7697d113f13e87168377420662baaba3b75d614c"
    )
    EXPECTED_LIVE_DIGEST = (
        "e3f95f58c0639e068300fd02af8d533c797800b6c70d95bb7fc16e81e1923b6c"
    )
    if checkpoint_sha256 is not None:
        assert checkpoint_sha256 == sq31["checkpoint_sha256"]
        assert checkpoint_sha256 == EXPECTED_FILE_HASH
        assert checkpoint_sha256 == sq34_run["entry_checkpoint_sha256"]
    assert sq34_run["initial_lora_digest"] == EXPECTED_LIVE_DIGEST
    print("frozen temperature:", TEMPERATURE)
    print("seed 45 adapter available:", ADAPTER.exists())
    print("seed 45 adapter sha256:", checkpoint_sha256)
    print("SQ34 stop reason:", sq34_run["stop_reason"])
    """
)

md(
    """
    ### The three arms

    `baseline-recipe` is the SQ34 schedule run again, live. I want a
    contemporaneous control, not the old CSV. `lower-step` cuts the learning
    rate by five. `stronger-kl` leaves the step alone and multiplies the
    reference term by five. Warmup stays at 8 so the named knob is the
    intended difference.
    """
)

code(
    """
    ARMS = (
        {
            "name": "baseline-recipe",
            "learning_rate": 5e-5,
            "kl_coefficient": 0.02,
        },
        {
            "name": "lower-step",
            "learning_rate": 1e-5,
            "kl_coefficient": 0.02,
        },
        {
            "name": "stronger-kl",
            "learning_rate": 5e-5,
            "kl_coefficient": 0.10,
        },
    )

    GROUP_SIZE = 4
    GROUPS_PER_ROUND = 16
    ROUNDS = 6
    TOTAL_GROUPS = GROUPS_PER_ROUND * ROUNDS
    TOTAL_EPISODES = TOTAL_GROUPS * GROUP_SIZE
    WARMUP_STEPS = 8
    GRADIENT_CLIP_NORM = 1.0
    OPTIMIZER_SEED = 34
    POOL_SEED = 3401
    SAMPLING_SEED_BASE = 340000
    EVAL_SEED_BASE = 900000
    FINAL_EVAL_SEED_BASE = 1100000
    STOCHASTIC_EVAL_EPISODES = 8

    MEMORY_CAP_GIB = 128.0
    MEMORY_ABORT_GIB = 96.0
    SOAK_ITERATIONS = 40

    TRIE_FORWARD_BUDGET = 180000
    FULL_LIST_SWEEP_BUDGET = 1200

    DRIFT_CANDIDATE_MASS_RATIO = 0.70
    DRIFT_RANK_MULTIPLIER = 4
    DRIFT_RANK_FLOOR = 10
    DRIFT_WINNER_SHARE_FLOOR = 0.50
    DRIFT_WINNER_SHARE_MARGIN = 0.25

    MIN_MIXED_FRACTION = 0.15
    MAX_REPEAT_RATE = 0.25
    MIN_SURPRISAL_RATIO = 0.30

    MASS_ADVANCE_RATIO = 0.85
    STOCHASTIC_VETO_DROP = 8
    MIN_UPDATES_FOR_EXPOSURE = 15
    MIN_RELATIVE_DELTA = 0.005
    MIN_FINAL_KL = 0.02
    RATIO_IDENTITY_TOLERANCE = 1e-4

    assert TOTAL_GROUPS == 96 and TOTAL_EPISODES == 384
    assert WARMUP_STEPS == 8
    assert tuple(arm["name"] for arm in ARMS) == (
        "baseline-recipe",
        "lower-step",
        "stronger-kl",
    )

    PREREGISTRATION = {
        "experiment": "SQ34b optimization study",
        "parent_experiment": "SQ34 simulator GRPO",
        "parent_preregistration_sha256": sq34_run["preregistration_sha256"],
        "entry_checkpoint": "lab18d seed 45 structured LoRA",
        "entry_checkpoint_sha256": checkpoint_sha256,
        "expected_live_lora_digest": EXPECTED_LIVE_DIGEST,
        "sq31_tokenizer_sha256": sq31["tokenizer_sha256"],
        "sq31_action_vocabulary_sha256": sq31["action_vocabulary_sha256"],
        "temperature": TEMPERATURE,
        "opening": OPENING,
        "max_turns": MAX_TURNS,
        "group_size": GROUP_SIZE,
        "groups_per_round": GROUPS_PER_ROUND,
        "rounds": ROUNDS,
        "total_groups_per_arm": TOTAL_GROUPS,
        "total_episodes_per_arm": TOTAL_EPISODES,
        "warmup_steps": WARMUP_STEPS,
        "schedule": "linear warmup then constant",
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "gradient_accumulation": "one optimizer step per group",
        "optimizer": "AdamW, LoRA parameters only, weight decay 0",
        "optimizer_seed": OPTIMIZER_SEED,
        "answer_pool_seed": POOL_SEED,
        "sampling_seed_base": SAMPLING_SEED_BASE,
        "final_eval_seed_base": FINAL_EVAL_SEED_BASE,
        "checkpoint_cadence": "every round",
        "held_out_decoder_evaluations": ["sq34 baseline from disk", "per-arm final"],
        "stochastic_eval_episodes_per_answer": STOCHASTIC_EVAL_EPISODES,
        "dropout": "disabled for sampling, scoring, and training",
        "reference_policy": "frozen seed 45 adapter, loaded as a second LoRA",
        "sampling": "on-policy per arm, do not replay SQ34 traces",
        "arms": list(ARMS),
        "clip_bounds": {"lower": 0.8, "upper": 1.2},
        "advantage_epsilon": 1e-4,
        "trie_forward_budget": TRIE_FORWARD_BUDGET,
        "full_list_sweep_budget": FULL_LIST_SWEEP_BUDGET,
        "stop_rules": {
            "anchor_drift": "Lab 20 thresholds, unchanged",
            "minimum_mixed_group_fraction": MIN_MIXED_FRACTION,
            "maximum_repeat_rate": MAX_REPEAT_RATE,
            "minimum_surprisal_ratio_to_round_zero": MIN_SURPRISAL_RATIO,
        },
        "exposure_bar": {
            "min_optimizer_steps": MIN_UPDATES_FOR_EXPOSURE,
            "min_relative_delta": MIN_RELATIVE_DELTA,
            "min_final_round_mean_kl": MIN_FINAL_KL,
        },
        "advance_rules": {
            "mass_ratio": MASS_ADVANCE_RATIO,
            "stochastic_veto_drop": STOCHASTIC_VETO_DROP,
            "deterministic_selects": False,
            "greedy_selects": False,
            "wilson_decides": False,
        },
        "memory": "one full-step soak after an exercised arm reset; SQ34 soaks not rerun",
    }

    preregistration_text = json.dumps(PREREGISTRATION, indent=2, sort_keys=True)
    PREREGISTRATION_SHA256 = sha256_text(preregistration_text)
    atomic_json(PREREGISTRATION, RESULTS_DIR / "sq34b-preregistration.json")
    print("preregistration sha256:", PREREGISTRATION_SHA256)
    print(preregistration_text)
    """
)

md(
    """
    ## SQ34b.2 Gate A: the inherited contracts

    Gate A in SQ34 proved the objective. Those tests still pass and this
    notebook does not re-derive them. What it does freeze is the 96-answer
    prefix of SQ34's pool, so the tasks match and the traces do not.
    """
)

code(
    """
    pool_candidates = tuple(
        answer
        for answer in ANSWERS
        if answer not in RESERVED_SET and answer != OPENING
    )
    pool_rng = np.random.default_rng(POOL_SEED)
    permutation = pool_rng.permutation(len(pool_candidates))
    SQ34_POOL = tuple(
        pool_candidates[int(index)] for index in permutation[:128]
    )
    ANSWER_POOL = SQ34_POOL[:TOTAL_GROUPS]
    assert sha256_text("\\n".join(SQ34_POOL)) == sq34_run["answer_pool_sha256"]
    assert len(ANSWER_POOL) == TOTAL_GROUPS
    assert len(set(ANSWER_POOL)) == TOTAL_GROUPS
    assert not (set(ANSWER_POOL) & RESERVED_SET)
    assert OPENING not in ANSWER_POOL
    ANSWER_POOL_SHA256 = sha256_text("\\n".join(ANSWER_POOL))
    print("training answers:", len(ANSWER_POOL))
    print("pool sha256:", ANSWER_POOL_SHA256)
    print("first eight:", ANSWER_POOL[:8])
    print("matches SQ34 prefix:", ANSWER_POOL == SQ34_POOL[:96])
    """
)

code(
    """
    class PromptOnly(NamedTuple):
        prompt: str


    def make_environment(*, opening=OPENING, max_turns=MAX_TURNS):
        return WordleEnvironment(
            EnvironmentConfig(
                answers=ANSWERS,
                opening=opening,
                max_turns=max_turns,
            ),
            expert=EXPERT,
            patterns=PATTERNS,
        )


    def wilson_interval(successes, trials, z=1.96):
        if trials == 0:
            return (float("nan"), float("nan"))
        proportion = successes / trials
        denominator = 1.0 + z * z / trials
        centre = proportion + z * z / (2 * trials)
        spread = z * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        return (
            max(0.0, (centre - spread) / denominator),
            min(1.0, (centre + spread) / denominator),
        )


    def mcnemar_exact(only_before, only_after):
        discordant = only_before + only_after
        if discordant == 0:
            return 1.0
        from math import comb

        tail = sum(
            comb(discordant, k)
            for k in range(min(only_before, only_after) + 1)
        )
        return min(1.0, 2.0 * tail / (2**discordant))


    print("helpers ready")
    """
)

md(
    """
    ## SQ34b.3 Gate A: the frozen anchor suite

    The 20 Lab 20 states, unchanged. They never pick a checkpoint.
    """
)

code(
    """
    anchor_states = pd.read_csv(LAB20_DIR / "anchor-states.csv")
    anchor_manifest = json.loads((LAB20_DIR / "anchor-manifest.json").read_text())
    ANCHOR_COUNT = len(anchor_states)
    assert ANCHOR_COUNT == anchor_manifest["states"] == 20

    ANCHOR_PROMPTS = []
    ANCHOR_CANDIDATES = []
    ANCHOR_TEACHER_INDEX = []
    ANCHOR_REGIME = []
    for row in anchor_states.itertuples(index=False):
        history = parse_state_key(row.state_key)
        candidates = candidate_indices_from_history(
            history, ANSWERS, PATTERNS, expert=EXPERT
        )
        assert len(candidates) == row.candidate_count
        ANCHOR_PROMPTS.append(
            structured_next_guess_prompt(history, len(candidates))
        )
        ANCHOR_CANDIDATES.append(np.asarray(candidates))
        ANCHOR_TEACHER_INDEX.append(int(row.teacher_index))
        ANCHOR_REGIME.append(str(row.regime))

    reserved_indices = {ANSWERS.index(answer) for answer in RESERVED_ANSWERS}
    leaking = [
        position
        for position, candidates in enumerate(ANCHOR_CANDIDATES)
        if set(int(index) for index in candidates) <= reserved_indices
    ]
    assert not leaking, f"anchor states resolve only to reserved answers: {leaking}"
    ANCHOR_SHA256 = sha256_text("\\n".join(ANCHOR_PROMPTS))
    assert ANCHOR_SHA256 == sq34_run["anchor_prompt_sha256"]
    print("anchor states:", ANCHOR_COUNT)
    print("anchor prompt sha256:", ANCHOR_SHA256)
    """
)

code(
    """
    def rank_vector(scores):
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks


    def anchor_metrics(score_matrix):
        rows = []
        for position in range(ANCHOR_COUNT):
            scores = score_matrix[position]
            candidates = ANCHOR_CANDIDATES[position]
            ranks = rank_vector(scores)
            shifted = scores - scores.max()
            weights = np.exp(shifted)
            regime = ANCHOR_REGIME[position]
            rows.append({
                "regime": regime,
                "winner_word": ANSWERS[int(scores.argmax())],
                "candidate_mass": float(weights[candidates].sum() / weights.sum()),
                "best_candidate_rank": int(ranks[candidates].min()),
                "candidate_teacher_rank": int(ranks[ANCHOR_TEACHER_INDEX[position]]),
                "singleton_candidate_rank": (
                    int(ranks[candidates].min()) if regime == "1" else np.nan
                ),
            })
        return pd.DataFrame(rows)


    def anchor_summary(metrics):
        winner_counts = metrics["winner_word"].value_counts()
        singleton = metrics.loc[metrics["regime"] == "1", "singleton_candidate_rank"]
        return {
            "median_candidate_mass": float(metrics["candidate_mass"].median()),
            "median_best_candidate_rank": float(
                metrics["best_candidate_rank"].median()
            ),
            "median_candidate_teacher_rank": float(
                metrics["candidate_teacher_rank"].median()
            ),
            "singleton_median_rank": float(singleton.median()),
            "unique_winners": int(winner_counts.size),
            "largest_winner_share": float(winner_counts.iloc[0] / len(metrics)),
        }


    def drift_check(current, baseline):
        mass_floor = DRIFT_CANDIDATE_MASS_RATIO * baseline["median_candidate_mass"]
        rank_ceiling = max(
            DRIFT_RANK_FLOOR,
            DRIFT_RANK_MULTIPLIER * baseline["median_best_candidate_rank"],
        )
        singleton_ceiling = max(
            DRIFT_RANK_FLOOR,
            DRIFT_RANK_MULTIPLIER * baseline["singleton_median_rank"],
        )
        share_ceiling = max(
            DRIFT_WINNER_SHARE_FLOOR,
            baseline["largest_winner_share"] + DRIFT_WINNER_SHARE_MARGIN,
        )
        failures = []
        if current["median_candidate_mass"] < mass_floor:
            failures.append("candidate mass collapsed")
        if current["median_best_candidate_rank"] > rank_ceiling:
            failures.append("best-candidate rank regressed")
        if current["singleton_median_rank"] > singleton_ceiling:
            failures.append("singleton closure regressed")
        if current["largest_winner_share"] > share_ceiling:
            failures.append("full-list winner concentrated")
        return {
            "failures": failures,
            "tripped": bool(failures),
            "candidate_mass_floor": mass_floor,
            "best_rank_ceiling": rank_ceiling,
            "singleton_rank_ceiling": singleton_ceiling,
            "winner_share_ceiling": share_ceiling,
        }


    BASELINE_ANCHOR_SUMMARY = dict(sq34_run["baseline_anchor_summary"])
    BASELINE_AGREEMENT = float(sq34_run["baseline_decoder_agreement"])
    MASS_FLOOR = (
        DRIFT_CANDIDATE_MASS_RATIO * BASELINE_ANCHOR_SUMMARY["median_candidate_mass"]
    )
    MASS_ADVANCE_FLOOR = (
        MASS_ADVANCE_RATIO * BASELINE_ANCHOR_SUMMARY["median_candidate_mass"]
    )
    print("incumbent median candidate mass:", BASELINE_ANCHOR_SUMMARY["median_candidate_mass"])
    print("hard mass floor:", MASS_FLOOR)
    print("advance mass floor:", MASS_ADVANCE_FLOOR)
    print("SQ34 prior mass path:")
    print(
        sq34_rounds[
            ["round", "optimizer_steps", "mean_kl", "parameter_delta_norm",
             "anchor_median_candidate_mass"]
        ].to_string(index=False)
    )
    """
)
md(
    """
    ## SQ34b.4 Gate B: load the checkpoint twice

    Same dual-adapter layout as SQ34. `policy` trains. `reference` stays the
    frozen seed 45 adapter. `set_adapter` flips `requires_grad`, so every
    switch restores the policy adapter on the way out.

    Set `SQ34B_RUN_MODEL=1` for this gate.
    """
)

code(
    """
    if RUN_MODEL:
        import gc

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tiny_wordle.hardware import preferred_device
        from tiny_wordle.policy import TokenTrie, TriePolicy, digest_tokenizer

        MODEL_ID = "Qwen/Qwen3-0.6B"
        CHUNK_SIZE = 256
        if not ADAPTER.exists():
            raise FileNotFoundError(f"missing seed 45 adapter: {ADAPTER}")

        device = preferred_device()
        if device.type == "mps":
            total_gib = torch.mps.recommended_max_memory() / 1024**3
            torch.mps.set_per_process_memory_fraction(MEMORY_CAP_GIB / total_gib)
            print(f"MPS cap: {MEMORY_CAP_GIB:.0f} GiB of {total_gib:.0f} GiB")

        def driver_memory_gib():
            if device.type == "mps":
                return torch.mps.driver_allocated_memory() / 1024**3
            if device.type == "cuda":
                return torch.cuda.memory_allocated() / 1024**3
            return float("nan")


        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer_digest = digest_tokenizer(tokenizer)
        assert tokenizer_digest == sq31["tokenizer_sha256"]

        def render_prompt(prompt):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        probe_environment = make_environment()
        probe_reset = probe_environment.reset("SHORE")
        probe_observation = probe_reset.observation
        rendered_probe = render_prompt(probe_observation.prompt)
        trie = TokenTrie.from_tokenizer(
            tokenizer, ANSWERS, rendered_prompt=rendered_probe
        )
        print("trie shape:", trie.shape())
    else:
        print("model gate skipped; set SQ34B_RUN_MODEL=1 for Gate B")
    """
)

code(
    """
    if RUN_MODEL:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to(device)
        model = PeftModel.from_pretrained(
            base_model, CHECKPOINT, adapter_name="policy", is_trainable=True
        ).to(device)
        model.load_adapter(CHECKPOINT, adapter_name="reference")
        model.set_adapter("policy")
        model.eval()

        def recapture_policy_parameters():
            parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if "lora_" in name and ".policy." in name
            ]
            assert parameters, "no trainable policy LoRA tensors were found"
            assert all(parameter.requires_grad for parameter in parameters), (
                "the policy adapter loaded frozen; is_trainable=True is required"
            )
            trainable = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            assert all("lora_" in name and ".policy." in name for name in trainable), (
                f"unexpected trainable tensors: {trainable[:4]}"
            )
            return parameters


        POLICY_PARAMETERS = recapture_policy_parameters()
        REFERENCE_PARAMETER_NAMES = [
            name
            for name, _ in model.named_parameters()
            if "lora_" in name and ".reference." in name
        ]
        assert REFERENCE_PARAMETER_NAMES
        assert len(POLICY_PARAMETERS) == len(REFERENCE_PARAMETER_NAMES)
        print("trainable policy LoRA tensors:", len(POLICY_PARAMETERS))
        print(
            "trainable parameters:",
            sum(parameter.numel() for parameter in POLICY_PARAMETERS),
        )
    """
)

code(
    """
    if RUN_MODEL:
        @contextmanager
        def using_adapter(name):
            previous = model.active_adapter
            model.set_adapter(name)
            try:
                yield
            finally:
                model.set_adapter(previous)
                for parameter in POLICY_PARAMETERS:
                    parameter.requires_grad_(True)


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
        assert policy.action_vocabulary_digest == sq31["action_vocabulary_sha256"]

        def lora_digest():
            digest = hashlib.sha256()
            for parameter in POLICY_PARAMETERS:
                digest.update(
                    parameter.detach().to("cpu", torch.float32).numpy().tobytes()
                )
            return digest.hexdigest()


        def reference_lora_digest():
            digest = hashlib.sha256()
            for name, parameter in model.named_parameters():
                if "lora_" in name and ".reference." in name:
                    digest.update(
                        parameter.detach().to("cpu", torch.float32).numpy().tobytes()
                    )
            return digest.hexdigest()


        def assert_reference_frozen():
            references = [
                parameter
                for name, parameter in model.named_parameters()
                if "lora_" in name and ".reference." in name
            ]
            assert references, "the reference adapter is missing"
            assert all(not parameter.requires_grad for parameter in references), (
                "a reference tensor is trainable"
            )
            assert reference_lora_digest() == INITIAL_REFERENCE_DIGEST, (
                "the reference adapter weights moved"
            )


        INITIAL_LORA_DIGEST = lora_digest()
        INITIAL_REFERENCE_DIGEST = reference_lora_digest()
        INITIAL_LORA_STATE = [
            parameter.detach().to("cpu", torch.float32).clone()
            for parameter in POLICY_PARAMETERS
        ]
        INITIAL_NORM = math.sqrt(
            sum(float((tensor * tensor).sum()) for tensor in INITIAL_LORA_STATE)
        )
        assert INITIAL_LORA_DIGEST == EXPECTED_LIVE_DIGEST, (
            f"live digest {INITIAL_LORA_DIGEST} != {EXPECTED_LIVE_DIGEST}"
        )
        print("device:", device)
        print("initial policy LoRA digest:", INITIAL_LORA_DIGEST)
        print("initial reference LoRA digest:", INITIAL_REFERENCE_DIGEST)
        print("initial LoRA norm:", INITIAL_NORM)
    """
)

md(
    """
    ### Reference identity

    Both adapters came from the same file. Before any update their scores
    must match exactly. After the first real step they must not.
    """
)

code(
    """
    if RUN_MODEL:
        IDENTITY_WORDS = ("SHORE", "CRANE", "APPLE", "MOUNT", "ZEBRA")
        with torch.no_grad():
            policy_scores = [
                float(
                    policy.score_action(
                        probe_observation,
                        word,
                        temperature=TEMPERATURE,
                        requires_grad=False,
                    )[0]
                )
                for word in IDENTITY_WORDS
            ]
            with using_adapter("reference"):
                reference_scores_before = [
                    float(
                        policy.score_action(
                            probe_observation,
                            word,
                            temperature=TEMPERATURE,
                            requires_grad=False,
                        )[0]
                    )
                    for word in IDENTITY_WORDS
                ]

        for word, left, right in zip(
            IDENTITY_WORDS, policy_scores, reference_scores_before
        ):
            print(f"{word}: policy {left:.6f}  reference {right:.6f}")
        assert policy_scores == reference_scores_before
        assert model.active_adapter == "policy"
        assert all(parameter.requires_grad for parameter in POLICY_PARAMETERS)
        print("reference identity before training: exact match")
    """
)

md(
    """
    ### Scoring paths still agree

    One-state teacher-forced versus sequential. If this fails, stop. The
    objective is then scoring a different function than the sampler.
    """
)

code(
    """
    if RUN_MODEL:
        environment = make_environment()
        observation = environment.reset("SHORE").observation
        if not environment.done:
            observation, _, done, _ = environment.step("CLOUT")
        with torch.no_grad():
            batched_total, batched_values = policy.score_action(
                observation, "APPLE", temperature=TEMPERATURE, requires_grad=False
            )
            walk_total, walk_values = policy.log_probability_tensor(
                observation, "APPLE", temperature=TEMPERATURE
            )
        worst = max(
            abs(float(left) - float(right))
            for left, right in zip(batched_values, walk_values)
        )
        print("worst per-token difference:", worst)
        assert worst < 1e-4, "the scoring paths disagree; do not train"

        out_of_trie = None
        try:
            policy.score_action(probe_observation, "QQQQQ", temperature=TEMPERATURE)
        except Exception as error:  # noqa: BLE001
            out_of_trie = type(error).__name__
        assert out_of_trie is not None

        model.train()
        dropout_guard = None
        try:
            policy.score_action(
                probe_observation, "SHORE", temperature=TEMPERATURE
            )
        except RuntimeError as error:
            dropout_guard = str(error)
        model.eval()
        assert dropout_guard is not None
        print("dropout guard still raises in train mode")
    """
)

code(
    """
    if RUN_MODEL:
        def freeze_behavior_digest():
            digest = lora_digest()
            policy.checkpoint_digest = digest
            return digest


        def sample_group(answer, *, group_index, round_index, behavior_digest, arm_name):
            assert policy.checkpoint_digest == behavior_digest
            trajectories = []
            for episode_index in range(GROUP_SIZE):
                environment = make_environment()
                trajectory = collect_trajectory(
                    environment,
                    policy,
                    answer,
                    group_id=f"{arm_name}-r{round_index:02d}-g{group_index:02d}",
                    answer_split="train",
                    temperature=TEMPERATURE,
                    sampling_seed=(
                        SAMPLING_SEED_BASE
                        + round_index * 10000
                        + group_index * 100
                        + episode_index
                    ),
                    episode_id=(
                        f"{arm_name}-r{round_index:02d}-g{group_index:02d}"
                        f"-e{episode_index}"
                    ),
                    policy_checkpoint_digest=behavior_digest,
                    reference_checkpoint_digest=INITIAL_LORA_DIGEST,
                )
                trajectories.append(trajectory)
                if device.type == "mps":
                    torch.mps.empty_cache()
            return trajectories


        def episode_reward(trajectory):
            return 1.0 if trajectory.terminal_reason == "solved" else 0.0


        def behavior_log_probabilities(trajectory):
            return tuple(
                float(step.decision.action_log_probability)
                for step in trajectory.steps
            )


        def episode_actions(trajectory):
            return tuple(
                (step.observation, step.decision.word) for step in trajectory.steps
            )


        def score_episode(trajectory, *, adapter, requires_grad):
            values = []
            with using_adapter(adapter):
                context = (
                    torch.enable_grad() if requires_grad else torch.no_grad()
                )
                with context:
                    for observation, word in episode_actions(trajectory):
                        total, _ = policy.score_action(
                            observation,
                            word,
                            temperature=TEMPERATURE,
                            requires_grad=requires_grad,
                        )
                        values.append(total)
            return values


        print("sampling and scoring helpers ready")
    """
)
md(
    """
    ## SQ34b.5 Gate B: the graded decoder, and SQ34's baseline from disk

    The curriculum still grades unmasked argmax over the 2,315 answers. GRPO
    still does not train that function. I am not replaying the 19-answer
    baseline. SQ34 already measured 10/19, 7/19 greedy, and 56/152 stochastic
    on this exact checkpoint.
    """
)

code(
    """
    if RUN_MODEL:
        WORD_TOKENS = [
            tokenizer.encode(word, add_special_tokens=False)
            + [tokenizer.eos_token_id]
            for word in ANSWERS
        ]
        LENGTH_BUCKETS = {}
        for length in sorted({len(tokens) for tokens in WORD_TOKENS}):
            indices = [
                index
                for index, tokens in enumerate(WORD_TOKENS)
                if len(tokens) == length
            ]
            padding = (-len(indices)) % CHUNK_SIZE
            padded = indices + [indices[-1]] * padding
            LENGTH_BUCKETS[length] = (
                torch.tensor(padded),
                torch.tensor([WORD_TOKENS[index] for index in padded], device=device),
            )

        BUDGET = Counter()
        LAST_STATE_PEAK_GIB = 0.0

        @torch.no_grad()
        def score_all_words(prompt_text):
            global LAST_STATE_PEAK_GIB
            BUDGET["full_list_sweeps"] += 1
            assert BUDGET["full_list_sweeps"] <= FULL_LIST_SWEEP_BUDGET, (
                "full-list sweep budget exhausted"
            )
            input_ids = tokenizer(
                render_prompt(prompt_text),
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(device)
            prefill = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
            final_logits = prefill.logits[0, -1].float()
            first_logprobs = final_logits - final_logits.logsumexp(-1)
            cache = prefill.past_key_values
            cache.batch_repeat_interleave(CHUNK_SIZE)
            peak = 0.0
            scores = torch.zeros(len(ANSWERS), dtype=torch.float32)

            for length, (indices, tokens) in LENGTH_BUCKETS.items():
                for start in range(0, len(indices), CHUNK_SIZE):
                    chunk = tokens[start:start + CHUNK_SIZE]
                    total = first_logprobs[chunk[:, 0]].clone()
                    if length > 1:
                        step = length - 1
                        output = model(
                            input_ids=chunk[:, :step],
                            past_key_values=cache,
                            use_cache=True,
                        )
                        logits = output.logits.float()
                        targets = logits.gather(
                            2, chunk[:, 1:].unsqueeze(-1)
                        ).squeeze(-1)
                        total = total + (targets - logits.logsumexp(-1)).sum(dim=1)
                        peak = max(peak, driver_memory_gib())
                        cache.crop(-step)
                        del output, logits, targets
                    scores[indices[start:start + CHUNK_SIZE]] = total.cpu()

            LAST_STATE_PEAK_GIB = peak
            del cache, prefill, final_logits, first_logprobs
            if device.type == "mps":
                torch.mps.empty_cache()
            return scores.numpy()


        def deterministic_game(answer):
            environment = make_environment()
            reset = environment.reset(answer)
            observation, done = reset.observation, reset.done
            guesses = [OPENING]
            noncandidate_guesses = 0
            while not done:
                candidates = set(
                    int(index)
                    for index in candidate_indices_from_history(
                        observation.history, ANSWERS, PATTERNS, expert=EXPERT
                    )
                )
                scores = score_all_words(observation.prompt)
                winner = int(scores.argmax())
                guess = ANSWERS[winner]
                noncandidate_guesses += int(winner not in candidates)
                guesses.append(guess)
                observation, _, done, info = environment.step(guess)
            return {
                "answer": answer,
                "solved": environment.last_record.terminal_reason == "solved",
                "turns": len(guesses),
                "guesses": guesses,
                "noncandidate_guesses": noncandidate_guesses,
            }


        def greedy_trie_game(answer):
            environment = make_environment()
            reset = environment.reset(answer)
            observation, done = reset.observation, reset.done
            guesses = [OPENING]
            while not done:
                word = policy.greedy_word(observation)
                BUDGET["trie_forwards"] += 1
                guesses.append(word)
                observation, _, done, info = environment.step(word)
            return {
                "answer": answer,
                "solved": environment.last_record.terminal_reason == "solved",
                "turns": len(guesses),
                "guesses": guesses,
            }


        def stochastic_play(answer, *, seed_base):
            solved = 0
            for episode in range(STOCHASTIC_EVAL_EPISODES):
                environment = make_environment()
                trajectory = collect_trajectory(
                    environment,
                    policy,
                    answer,
                    group_id="eval",
                    answer_split="reserved",
                    protected_answer_id=answer,
                    temperature=TEMPERATURE,
                    sampling_seed=seed_base + episode,
                    episode_id=f"eval-{answer}-{episode}",
                    policy_checkpoint_digest=policy.checkpoint_digest,
                    reference_checkpoint_digest=INITIAL_LORA_DIGEST,
                )
                solved += int(trajectory.terminal_reason == "solved")
                if device.type == "mps":
                    torch.mps.empty_cache()
            return solved


        def evaluate_reserved(label, *, seed_base=FINAL_EVAL_SEED_BASE):
            rows = []
            for position, answer in enumerate(RESERVED_ANSWERS):
                game = deterministic_game(answer)
                greedy = greedy_trie_game(answer)
                rows.append({
                    "answer": answer,
                    "label": label,
                    "deterministic_solved": bool(game["solved"]),
                    "deterministic_turns": game["turns"],
                    "deterministic_noncandidates": game["noncandidate_guesses"],
                    "deterministic_guesses": " ".join(game["guesses"]),
                    "greedy_solved": bool(greedy["solved"]),
                    "greedy_turns": greedy["turns"],
                    "stochastic_solved": stochastic_play(
                        answer, seed_base=seed_base + position * 1000
                    ),
                })
            return pd.DataFrame(rows)


        def anchor_score_matrix():
            return np.stack([score_all_words(prompt) for prompt in ANCHOR_PROMPTS])


        BASELINE_EVAL = pd.read_csv(SQ34_DIR / "sq34-baseline-eval.csv")
        BASELINE_DETERMINISTIC = int(BASELINE_EVAL["deterministic_solved"].sum())
        BASELINE_GREEDY = int(BASELINE_EVAL["greedy_solved"].sum())
        BASELINE_STOCHASTIC = int(BASELINE_EVAL["stochastic_solved"].sum())
        BASELINE_STOCHASTIC_TRIALS = 19 * STOCHASTIC_EVAL_EPISODES
        assert BASELINE_DETERMINISTIC == 10
        assert BASELINE_GREEDY == int(sq34_run["greedy_baseline"])
        assert BASELINE_STOCHASTIC == 56
        assert BASELINE_STOCHASTIC_TRIALS == 152
        print("loaded SQ34 baseline eval from disk")
        print("deterministic:", BASELINE_DETERMINISTIC, "/ 19")
        print("greedy trie:", BASELINE_GREEDY, "/ 19")
        print("stochastic trie:", BASELINE_STOCHASTIC, "/", BASELINE_STOCHASTIC_TRIALS)
    """
)

md(
    """
    ### One soak, after an arm reset

    SQ34 already plateaued sampling, reference scoring, retained-graph
    scoring, and a training step. I am not repeating those four. I am
    repeating the one new lifecycle. Step the policy, reload seed 45, then
    soak a training step on the reloaded tensors.
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


        def peak_since(start):
            finite = [
                value
                for value in policy.forward_memory_trace[start:]
                if math.isfinite(value)
            ]
            return max(finite) if finite else float("nan")


        def reload_seed45_policy():
            global POLICY_PARAMETERS
            if "policy" in model.peft_config:
                model.set_adapter("reference")
                model.delete_adapter("policy")
            model.load_adapter(
                str(CHECKPOINT), adapter_name="policy", is_trainable=True
            )
            model.set_adapter("policy")
            POLICY_PARAMETERS = recapture_policy_parameters()
            for parameter in POLICY_PARAMETERS:
                parameter.requires_grad_(True)
            model.eval()
            digest = lora_digest()
            assert sha256_file(ADAPTER) == EXPECTED_FILE_HASH
            assert digest == EXPECTED_LIVE_DIGEST, (
                f"reload produced {digest}, expected {EXPECTED_LIVE_DIGEST}"
            )
            assert model.active_adapter == "policy"
            policy.model = model
            policy.checkpoint_digest = digest
            return digest


        def assert_arm_start():
            digest = lora_digest()
            assert digest == EXPECTED_LIVE_DIGEST, (
                f"arm start digest {digest} != {EXPECTED_LIVE_DIGEST}"
            )
            assert sha256_file(ADAPTER) == EXPECTED_FILE_HASH
            assert model.active_adapter == "policy"
            assert all(parameter.requires_grad for parameter in POLICY_PARAMETERS)
            assert_reference_frozen()
            with torch.no_grad(), using_adapter("reference"):
                reference_now = [
                    float(
                        policy.score_action(
                            probe_observation,
                            word,
                            temperature=TEMPERATURE,
                            requires_grad=False,
                        )[0]
                    )
                    for word in IDENTITY_WORDS
                ]
            assert reference_now == reference_scores_before, (
                "the reference adapter moved"
            )
            assert_reference_frozen()
            return digest


        soak_environment = make_environment()
        soak_environment.reset("PIQUE")
        soak_observation = soak_environment.observation
        for filler in ("CLOUT", "BRAND", "MERGE", "SHYLY"):
            if soak_environment.done:
                break
            soak_observation, _, _, _ = soak_environment.step(filler)
        SOAK_WORD = max(ANSWERS, key=lambda word: len(trie.sequence_for_word(word)))

        dummy = torch.optim.AdamW(POLICY_PARAMETERS, lr=1e-4, weight_decay=0.0)
        dummy.zero_grad(set_to_none=True)
        retained = [
            policy.score_action(
                soak_observation,
                SOAK_WORD,
                temperature=TEMPERATURE,
                requires_grad=True,
            )[0]
            for _ in range(2)
        ]
        torch.stack(retained).sum().backward()
        dummy.step()
        dummy.zero_grad(set_to_none=True)
        del retained, dummy
        disturbed = lora_digest()
        assert disturbed != EXPECTED_LIVE_DIGEST, (
            "the dummy step did not move the policy; the reset soak is fake"
        )
        print("digest after dummy step:", disturbed[:16])
        print("reloading seed 45 policy adapter")
        reload_seed45_policy()
        assert_arm_start()

        rehearsal = torch.optim.AdamW(POLICY_PARAMETERS, lr=0.0, weight_decay=0.0)
        step_peaks = []
        for iteration in range(SOAK_ITERATIONS):
            start = len(policy.forward_memory_trace)
            rehearsal.zero_grad(set_to_none=True)
            for _ in range(GROUP_SIZE):
                values = [
                    policy.score_action(
                        soak_observation,
                        SOAK_WORD,
                        temperature=TEMPERATURE,
                        requires_grad=True,
                    )[0]
                    for _ in range(MAX_TURNS - 1)
                ]
                stacked = torch.stack(values)
                objective = episode_objective(
                    stacked,
                    stacked.detach().new_full((len(values),), -3.0),
                    stacked.detach().new_full((len(values),), -3.0),
                    advantage=1.0,
                    hyperparameters=GrpoHyperparameters(),
                )
                (-objective / GROUP_SIZE).backward()
                del values, stacked, objective
            torch.nn.utils.clip_grad_norm_(POLICY_PARAMETERS, GRADIENT_CLIP_NORM)
            rehearsal.step()
            step_peaks.append(peak_since(start))
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        assert_memory_plateau(step_peaks, "reset training step")
        rehearsal.zero_grad(set_to_none=True)
        assert lora_digest() == EXPECTED_LIVE_DIGEST
        atomic_json(
            {
                "reset_training_step": step_peaks,
                "abort_threshold_gib": MEMORY_ABORT_GIB,
                "soak_word": SOAK_WORD,
            },
            RESULTS_DIR / "sq34b-reset-soak.json",
        )
        print(
            f"reset training-step soak: peak {max(step_peaks):.2f} GiB, "
            f"final {step_peaks[-1]:.2f} GiB"
        )
    """
)
md(
    """
    ## SQ34b.6 Gate C: three arms, six rounds

    Each arm reloads seed 45, takes one optimizer step per mixed group, and
    stops on the same guards SQ34 used. Gameplay happens once, at the end of
    the arm. I will not shop the mid-run decoder.
    """
)

code(
    """
    if RUN_MODEL:
        def parameter_delta_norm():
            total = 0.0
            for parameter, initial in zip(POLICY_PARAMETERS, INITIAL_LORA_STATE):
                delta = parameter.detach().to("cpu", torch.float32) - initial
                total += float((delta * delta).sum())
            return math.sqrt(total)


        def diversity_metrics(trajectories):
            words = [
                step.decision.word
                for trajectory in trajectories
                for step in trajectory.steps
            ]
            surprisal = [
                -float(step.decision.action_log_probability)
                for trajectory in trajectories
                for step in trajectory.steps
            ]
            repeats = 0
            for trajectory in trajectories:
                guessed = [OPENING]
                for step in trajectory.steps:
                    if step.decision.word in guessed:
                        repeats += 1
                    guessed.append(step.decision.word)
            return {
                "actions": len(words),
                "unique_actions": len(set(words)),
                "mean_surprisal": (
                    float(np.mean(surprisal)) if surprisal else float("nan")
                ),
                "repeat_rate": repeats / len(words) if words else float("nan"),
            }


        def effective_sample_size(ratios):
            if not ratios:
                return float("nan")
            weights = np.asarray(ratios, dtype=np.float64)
            return float(weights.sum() ** 2 / (weights**2).sum())


        def persist_group(
            arm_dir, round_index, group_index, answer, digest, trajectories
        ):
            payload = {
                "round": round_index,
                "group": group_index,
                "answer": answer,
                "behavior_checkpoint_digest": digest,
                "reference_checkpoint_digest": INITIAL_LORA_DIGEST,
                "temperature": TEMPERATURE,
                "sampling_seeds": [t.sampling_seed for t in trajectories],
                "trajectory_sha256": [
                    sha256_text(t.to_json()) for t in trajectories
                ],
                "trajectories": [t.to_dict() for t in trajectories],
            }
            text = json.dumps(payload, indent=2, sort_keys=True, default=str)
            digest_of_text = sha256_text(text)
            stem = f"group-{round_index:02d}-{group_index:03d}"
            trace_dir = arm_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            path = trace_dir / f"{stem}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(text)
            temporary.replace(path)
            sidecar = trace_dir / f"{stem}.sha256"
            sidecar_temporary = sidecar.with_suffix(".tmp")
            sidecar_temporary.write_text(digest_of_text)
            sidecar_temporary.replace(sidecar)
            return digest_of_text


        def verify_persisted_traces(arm_dir, expected):
            files = sorted((arm_dir / "traces").glob("group-*.json"))
            if len(files) != expected:
                print(f"trace files: {len(files)} present, {expected} expected")
                return False
            for path in files:
                sidecar = path.with_suffix(".sha256")
                if not sidecar.exists():
                    return False
                if sha256_text(path.read_text()) != sidecar.read_text().strip():
                    return False
            return True


        def classify_arm(
            movement,
            final_mass,
            drift_tripped,
            stochastic_final,
            *,
            ratio_held,
            evaluated,
        ):
            exposed = movement["optimizer_steps"] >= MIN_UPDATES_FOR_EXPOSURE and (
                movement["relative_delta"] >= MIN_RELATIVE_DELTA
                or movement["final_round_mean_kl"] >= MIN_FINAL_KL
            )
            mass_ok = (
                not drift_tripped
                and math.isfinite(final_mass)
                and final_mass >= MASS_ADVANCE_FLOOR
            )
            stochastic_ok = (
                evaluated
                and stochastic_final
                >= BASELINE_STOCHASTIC - STOCHASTIC_VETO_DROP
            )
            if not ratio_held:
                status = "reject"
            elif not exposed:
                status = "inconclusive"
            elif mass_ok and stochastic_ok:
                status = "advance"
            else:
                status = "reject"
            return {
                "exposed": exposed,
                "mass_ok": mass_ok,
                "stochastic_ok": stochastic_ok,
                "ratio_held": ratio_held,
                "evaluated": evaluated,
                "status": status,
            }


        print("arm helpers ready")
    """
)

code(
    """
    if RUN_MODEL:
        ARM_RESULTS = []
        ALL_ROUNDS = []
        ALL_GROUPS = []
        ALL_EPISODES = []

        GROUP_COLUMNS = [
            "arm", "round", "group", "answer", "rewards", "degenerate",
            "updated", "gate_failed", "trace_sha256", "loss", "mean_kl",
            "clipped_fraction", "ratio_spread", "gradient_norm", "learning_rate",
        ]
        ROUND_COLUMNS = [
            "arm", "round", "behavior_checkpoint_digest", "groups",
            "groups_updated", "groups_degenerate", "mixed_fraction",
            "solve_rate", "mean_objective", "mean_kl", "mean_clipped_fraction",
            "worst_ratio_gap", "effective_sample_size", "parameter_delta_norm",
            "optimizer_steps", "drift_tripped",
        ]

        for arm in ARMS:
            arm_name = arm["name"]
            arm_dir = RESULTS_DIR / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            print("=" * 72)
            print("starting arm", arm_name, "lr", arm["learning_rate"], "kl", arm["kl_coefficient"])
            reload_seed45_policy()
            start_digest = assert_arm_start()
            print("arm start digest:", start_digest)

            hyper = GrpoHyperparameters(kl_coefficient=arm["kl_coefficient"])
            torch.manual_seed(OPTIMIZER_SEED)
            optimizer = torch.optim.AdamW(
                POLICY_PARAMETERS, lr=arm["learning_rate"], weight_decay=0.0
            )
            optimizer_steps = 0

            def learning_rate_for(step, peak=arm["learning_rate"]):
                if step < WARMUP_STEPS:
                    return peak * (step + 1) / WARMUP_STEPS
                return peak

            round_records = []
            group_records = []
            episode_records = []
            stop_reason = None
            rounds_completed = 0
            training_start = time.perf_counter()

            for round_index in range(ROUNDS):
                if stop_reason is not None:
                    break
                round_start_digest = lora_digest()
                round_trajectories = []
                round_ratio_gap = 0.0
                round_ratios = []
                groups_updated = 0
                groups_degenerate = 0
                round_objective = 0.0
                round_kl = []
                round_clipped = []

                for slot in range(GROUPS_PER_ROUND):
                    group_index = round_index * GROUPS_PER_ROUND + slot
                    answer = ANSWER_POOL[group_index]
                    behavior_digest = freeze_behavior_digest()
                    trajectories = sample_group(
                        answer,
                        group_index=group_index,
                        round_index=round_index,
                        behavior_digest=behavior_digest,
                        arm_name=arm_name,
                    )
                    round_trajectories.extend(trajectories)
                    rewards = tuple(episode_reward(t) for t in trajectories)
                    behavior = tuple(
                        behavior_log_probabilities(t) for t in trajectories
                    )
                    for trajectory, reward in zip(trajectories, rewards):
                        episode_records.append({
                            "arm": arm_name,
                            "round": round_index,
                            "group": group_index,
                            "episode_id": trajectory.episode_id,
                            "answer": answer,
                            "reward": reward,
                            "actions": len(trajectory.steps),
                            "terminal_reason": trajectory.terminal_reason,
                            "behavior_checkpoint_digest": behavior_digest,
                            "guesses": " ".join(
                                step.decision.word for step in trajectory.steps
                            ),
                        })
                    trace_sha256 = persist_group(
                        arm_dir,
                        round_index,
                        group_index,
                        answer,
                        behavior_digest,
                        trajectories,
                    )
                    actionless = any(not t.steps for t in trajectories)
                    if len(set(rewards)) == 1 or actionless:
                        groups_degenerate += 1
                        group_records.append({
                            "arm": arm_name,
                            "round": round_index,
                            "group": group_index,
                            "answer": answer,
                            "rewards": sum(rewards),
                            "degenerate": True,
                            "updated": False,
                            "gate_failed": False,
                            "trace_sha256": trace_sha256,
                        })
                        continue

                    reference = []
                    with torch.no_grad():
                        for trajectory in trajectories:
                            reference.append(
                                tuple(
                                    float(value)
                                    for value in score_episode(
                                        trajectory,
                                        adapter="reference",
                                        requires_grad=False,
                                    )
                                )
                            )
                    batch = GroupBatch(
                        group_id=f"{arm_name}-r{round_index:02d}-g{group_index:02d}",
                        answer=answer,
                        behavior_checkpoint_digest=behavior_digest,
                        reference_checkpoint_digest=INITIAL_LORA_DIGEST,
                        rewards=rewards,
                        behavior_log_probabilities=behavior,
                        reference_log_probabilities=tuple(reference),
                    )
                    advantages = group_advantages(
                        torch.tensor(rewards), epsilon=hyper.advantage_epsilon
                    )
                    optimizer.zero_grad(set_to_none=True)
                    detached = []
                    accumulated = 0.0
                    gate_failed = False
                    for position, trajectory in enumerate(trajectories):
                        current = score_episode(
                            trajectory, adapter="policy", requires_grad=True
                        )
                        current_tensor = torch.stack(current)
                        behavior_cpu = torch.tensor(
                            behavior[position], dtype=torch.float32
                        )
                        behavior_tensor = current_tensor.detach().new_tensor(
                            behavior[position]
                        )
                        reference_tensor = current_tensor.detach().new_tensor(
                            reference[position]
                        )
                        gaps = (
                            (current_tensor.detach().cpu() - behavior_cpu)
                            .exp()
                            .sub(1.0)
                            .abs()
                        )
                        gap = float(gaps.max()) if len(gaps) else 0.0
                        round_ratio_gap = max(round_ratio_gap, gap)
                        if gap >= RATIO_IDENTITY_TOLERANCE:
                            gate_failed = True
                            del current, current_tensor
                            del behavior_tensor, reference_tensor, behavior_cpu
                            gc.collect()
                            if device.type == "mps":
                                torch.mps.empty_cache()
                            break
                        round_ratios.extend(
                            (current_tensor.detach().cpu() - behavior_cpu)
                            .exp()
                            .tolist()
                        )
                        objective = episode_objective(
                            current_tensor,
                            behavior_tensor,
                            reference_tensor,
                            advantage=float(advantages[position]),
                            hyperparameters=hyper,
                        )
                        (-objective / GROUP_SIZE).backward()
                        accumulated += float(objective.detach()) / GROUP_SIZE
                        detached.append(current_tensor.detach().cpu())
                        del current, current_tensor, objective
                        del behavior_tensor, reference_tensor, behavior_cpu
                        gc.collect()
                        if device.type == "mps":
                            torch.mps.empty_cache()

                    if gate_failed:
                        optimizer.zero_grad(set_to_none=True)
                        group_records.append({
                            "arm": arm_name,
                            "round": round_index,
                            "group": group_index,
                            "answer": answer,
                            "rewards": sum(rewards),
                            "degenerate": False,
                            "updated": False,
                            "gate_failed": True,
                            "trace_sha256": trace_sha256,
                        })
                        stop_reason = (
                            f"ratio identity failed in {arm_name} round "
                            f"{round_index}, group {group_index}, "
                            f"gap {round_ratio_gap:.2e}"
                        )
                        break

                    diagnostic_loss, diagnostics = group_loss(
                        batch, detached, hyperparameters=hyper
                    )
                    assert math.isclose(
                        float(diagnostic_loss),
                        -accumulated,
                        rel_tol=1e-4,
                        abs_tol=1e-6,
                    )
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = learning_rate_for(optimizer_steps)
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            POLICY_PARAMETERS, GRADIENT_CLIP_NORM
                        )
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    groups_updated += 1
                    round_objective += accumulated
                    round_kl.append(diagnostics.mean_kl)
                    round_clipped.append(diagnostics.clipped_fraction)
                    group_records.append({
                        "arm": arm_name,
                        "round": round_index,
                        "group": group_index,
                        "answer": answer,
                        "rewards": sum(rewards),
                        "degenerate": False,
                        "updated": True,
                        "gate_failed": False,
                        "trace_sha256": trace_sha256,
                        "loss": float(diagnostic_loss),
                        "mean_kl": diagnostics.mean_kl,
                        "clipped_fraction": diagnostics.clipped_fraction,
                        "ratio_spread": diagnostics.ratio_spread,
                        "gradient_norm": grad_norm,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    })

                if stop_reason is not None and "ratio identity" in stop_reason:
                    break

                diversity = diversity_metrics(round_trajectories)
                mixed_fraction = 1.0 - groups_degenerate / GROUPS_PER_ROUND
                if round_index == 0:
                    round_zero_surprisal = diversity["mean_surprisal"]
                checkpoint_scores = anchor_score_matrix()
                checkpoint_metrics = anchor_metrics(checkpoint_scores)
                checkpoint_summary = anchor_summary(checkpoint_metrics)
                drift = drift_check(checkpoint_summary, BASELINE_ANCHOR_SUMMARY)
                round_record = {
                    "arm": arm_name,
                    "round": round_index,
                    "behavior_checkpoint_digest": round_start_digest,
                    "groups": GROUPS_PER_ROUND,
                    "groups_updated": groups_updated,
                    "groups_degenerate": groups_degenerate,
                    "mixed_fraction": mixed_fraction,
                    "solve_rate": float(
                        np.mean([episode_reward(t) for t in round_trajectories])
                    ),
                    "mean_objective": (
                        round_objective / groups_updated if groups_updated else 0.0
                    ),
                    "mean_kl": float(np.mean(round_kl)) if round_kl else 0.0,
                    "mean_clipped_fraction": (
                        float(np.mean(round_clipped)) if round_clipped else 0.0
                    ),
                    "worst_ratio_gap": round_ratio_gap,
                    "effective_sample_size": effective_sample_size(round_ratios),
                    "parameter_delta_norm": parameter_delta_norm(),
                    "optimizer_steps": optimizer_steps,
                    "trie_forwards": policy.forward_call_count,
                    "full_list_sweeps": BUDGET["full_list_sweeps"],
                    **{f"diversity_{key}": value for key, value in diversity.items()},
                    **{
                        f"anchor_{key}": value
                        for key, value in checkpoint_summary.items()
                    },
                    "drift_tripped": drift["tripped"],
                    "drift_failures": "; ".join(drift["failures"]),
                }
                round_records.append(round_record)
                rounds_completed = round_index + 1
                checkpoint_directory = (
                    arm_dir / f"checkpoint-round-{round_index:02d}"
                )
                model.save_pretrained(
                    str(checkpoint_directory), selected_adapters=["policy"]
                )
                assert (checkpoint_directory / "policy").exists()
                print(
                    f"{arm_name} round {round_index}: "
                    f"updated {groups_updated}/{GROUPS_PER_ROUND}"
                    f"  KL {round_record['mean_kl']:.2e}"
                    f"  |delta| {round_record['parameter_delta_norm']:.4f}"
                    f"  mass {checkpoint_summary['median_candidate_mass']:.4f}"
                )
                if drift["tripped"]:
                    stop_reason = f"anchor drift: {'; '.join(drift['failures'])}"
                elif mixed_fraction < MIN_MIXED_FRACTION:
                    stop_reason = (
                        f"mixed-group fraction collapsed to {mixed_fraction:.3f}"
                    )
                elif diversity["repeat_rate"] > MAX_REPEAT_RATE:
                    stop_reason = (
                        f"repeat rate rose to {diversity['repeat_rate']:.3f}"
                    )
                elif (
                    diversity["mean_surprisal"]
                    < MIN_SURPRISAL_RATIO * round_zero_surprisal
                ):
                    stop_reason = (
                        f"action surprisal fell to {diversity['mean_surprisal']:.3f}"
                    )
                elif policy.forward_call_count > TRIE_FORWARD_BUDGET:
                    stop_reason = "trie forward budget exhausted"
                elif BUDGET["full_list_sweeps"] > FULL_LIST_SWEEP_BUDGET:
                    stop_reason = "full-list sweep budget exhausted"
                if stop_reason is not None:
                    print("stopping", arm_name, ":", stop_reason)

            training_minutes = (time.perf_counter() - training_start) / 60.0
            rounds_table = pd.DataFrame(round_records)
            groups_table = pd.DataFrame(group_records)
            episodes_table = pd.DataFrame(episode_records)
            for column in GROUP_COLUMNS:
                if column not in groups_table.columns:
                    groups_table[column] = np.nan
            if "updated" in groups_table.columns:
                groups_table["updated"] = (
                    groups_table["updated"].fillna(False).astype(bool)
                )
            else:
                groups_table["updated"] = False

            ratio_held = stop_reason is None or "ratio identity" not in stop_reason
            if ratio_held:
                freeze_behavior_digest()
                final_eval = evaluate_reserved(
                    f"{arm_name}-final", seed_base=FINAL_EVAL_SEED_BASE
                )
                comparison = BASELINE_EVAL.merge(
                    final_eval, on="answer", suffixes=("_baseline", "_final")
                )
                deterministic_gained = int(
                    (
                        ~comparison["deterministic_solved_baseline"]
                        & comparison["deterministic_solved_final"]
                    ).sum()
                )
                deterministic_lost = int(
                    (
                        comparison["deterministic_solved_baseline"]
                        & ~comparison["deterministic_solved_final"]
                    ).sum()
                )
                greedy_gained = int(
                    (
                        ~comparison["greedy_solved_baseline"]
                        & comparison["greedy_solved_final"]
                    ).sum()
                )
                greedy_lost = int(
                    (
                        comparison["greedy_solved_baseline"]
                        & ~comparison["greedy_solved_final"]
                    ).sum()
                )
                final_deterministic = int(final_eval["deterministic_solved"].sum())
                final_greedy = int(final_eval["greedy_solved"].sum())
                final_stochastic = int(final_eval["stochastic_solved"].sum())
            else:
                print(
                    "skipping gameplay eval for",
                    arm_name,
                    "; ratio identity failed mid-round",
                )
                final_eval = None
                deterministic_gained = 0
                deterministic_lost = 0
                greedy_gained = 0
                greedy_lost = 0
                final_deterministic = None
                final_greedy = None
                final_stochastic = None
            if len(rounds_table):
                final_anchor = {
                    key.removeprefix("anchor_"): rounds_table.iloc[-1][key]
                    for key in rounds_table.columns
                    if key.startswith("anchor_")
                }
                final_drift = drift_check(final_anchor, BASELINE_ANCHOR_SUMMARY)
                final_mass = float(final_anchor["median_candidate_mass"])
            else:
                final_anchor = dict(BASELINE_ANCHOR_SUMMARY)
                final_drift = {
                    "failures": ["no rounds completed"],
                    "tripped": True,
                }
                final_mass = float("nan")

            movement = {
                "weights_changed": lora_digest() != EXPECTED_LIVE_DIGEST,
                "parameter_delta_norm": parameter_delta_norm(),
                "initial_parameter_norm": INITIAL_NORM,
                "relative_delta": parameter_delta_norm() / INITIAL_NORM,
                "optimizer_steps": optimizer_steps,
                "groups_seen": len(groups_table),
                "groups_updated": int(groups_table["updated"].sum()),
                "fraction_of_groups_updating": (
                    float(groups_table["updated"].mean())
                    if len(groups_table)
                    else 0.0
                ),
                "final_round_mean_kl": (
                    float(rounds_table["mean_kl"].iloc[-1])
                    if len(rounds_table)
                    else 0.0
                ),
            }
            decision = classify_arm(
                movement,
                final_mass,
                final_drift["tripped"],
                final_stochastic if final_stochastic is not None else -1,
                ratio_held=ratio_held,
                evaluated=ratio_held,
            )
            arm_result = {
                "arm": arm_name,
                "learning_rate": arm["learning_rate"],
                "kl_coefficient": arm["kl_coefficient"],
                "rounds_completed": rounds_completed,
                "stop_reason": stop_reason or "ran to completion",
                "training_minutes": training_minutes,
                "movement": movement,
                "final_mass": final_mass,
                "mass_advance_floor": MASS_ADVANCE_FLOOR,
                "mass_hard_floor": MASS_FLOOR,
                "deterministic_final": final_deterministic,
                "deterministic_gained": deterministic_gained,
                "deterministic_lost": deterministic_lost,
                "greedy_final": final_greedy,
                "greedy_gained": greedy_gained,
                "greedy_lost": greedy_lost,
                "stochastic_final": final_stochastic,
                "stochastic_trials": BASELINE_STOCHASTIC_TRIALS,
                "final_drift": final_drift,
                "decision": decision,
                "worst_ratio_gap": (
                    float(rounds_table["worst_ratio_gap"].max())
                    if len(rounds_table)
                    else float("nan")
                ),
                "mean_clipped_fraction": (
                    float(rounds_table["mean_clipped_fraction"].mean())
                    if len(rounds_table)
                    else float("nan")
                ),
                "traces_persisted": verify_persisted_traces(
                    arm_dir, len(groups_table)
                ),
                "final_lora_digest": lora_digest(),
            }
            rounds_table.to_csv(arm_dir / "rounds.csv", index=False)
            groups_table.to_csv(arm_dir / "groups.csv", index=False)
            episodes_table.to_csv(arm_dir / "episodes.csv", index=False)
            if final_eval is not None:
                final_eval.to_csv(arm_dir / "final-eval.csv", index=False)
            atomic_json(arm_result, arm_dir / "run.json")
            ARM_RESULTS.append(arm_result)
            ALL_ROUNDS.append(rounds_table)
            ALL_GROUPS.append(groups_table)
            ALL_EPISODES.append(episodes_table)
            print(
                arm_name,
                "status",
                decision["status"],
                "steps",
                optimizer_steps,
                "mass",
                final_mass,
                "stoch",
                final_stochastic,
                "/",
                BASELINE_STOCHASTIC_TRIALS,
            )
            del optimizer
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
    """
)
md(
    """
    ## SQ34b.7 Movement, then play, then the advance table

    Read movement first. An arm that did not clear the exposure bar cannot
    advance, even if mass looks pretty. Deterministic 10/19 and greedy 7/19
    stay in the table. They do not pick a winner.
    """
)

code(
    """
    if RUN_MODEL:
        ROUNDS_TABLE = (
            pd.concat(ALL_ROUNDS, ignore_index=True) if ALL_ROUNDS else pd.DataFrame()
        )
        GROUPS_TABLE = (
            pd.concat(ALL_GROUPS, ignore_index=True) if ALL_GROUPS else pd.DataFrame()
        )
        EPISODES_TABLE = (
            pd.concat(ALL_EPISODES, ignore_index=True)
            if ALL_EPISODES
            else pd.DataFrame()
        )
        ROUNDS_TABLE.to_csv(RESULTS_DIR / "sq34b-rounds.csv", index=False)
        GROUPS_TABLE.to_csv(RESULTS_DIR / "sq34b-groups.csv", index=False)
        EPISODES_TABLE.to_csv(RESULTS_DIR / "sq34b-episodes.csv", index=False)

        rows = [
            {
                "arm": "sq34-prior",
                "learning_rate": 5e-5,
                "kl_coefficient": 0.02,
                "source": "historical",
                "rounds": int(sq34_run["rounds_completed"]),
                "optimizer_steps": int(sq34_run["optimizer_steps"]),
                "relative_delta": sq34_run["movement"]["relative_delta"],
                "final_kl": sq34_run["movement"]["final_round_mean_kl"],
                "final_mass": sq34_run["final_anchor_summary"]["median_candidate_mass"],
                "deterministic": (
                    f"{sq34_run['deterministic_baseline']}->"
                    f"{sq34_run['deterministic_final']}"
                ),
                "greedy": f"{sq34_run['greedy_baseline']}->{sq34_run['greedy_final']}",
                "stochastic": (
                    f"{sq34_run['stochastic_baseline']}->"
                    f"{sq34_run['stochastic_final']}"
                ),
                "clip_mean": float(sq34_rounds["mean_clipped_fraction"].mean()),
                "status": "prior-reject",
                "stop_reason": sq34_run["stop_reason"],
            }
        ]
        for result in ARM_RESULTS:
            movement = result["movement"]
            rows.append({
                "arm": result["arm"],
                "learning_rate": result["learning_rate"],
                "kl_coefficient": result["kl_coefficient"],
                "source": "contemporaneous",
                "rounds": result["rounds_completed"],
                "optimizer_steps": movement["optimizer_steps"],
                "relative_delta": movement["relative_delta"],
                "final_kl": movement["final_round_mean_kl"],
                "final_mass": result["final_mass"],
                "deterministic": f"{BASELINE_DETERMINISTIC}->{result['deterministic_final']}",
                "greedy": f"{BASELINE_GREEDY}->{result['greedy_final']}",
                "stochastic": f"{BASELINE_STOCHASTIC}->{result['stochastic_final']}",
                "clip_mean": result["mean_clipped_fraction"],
                "status": result["decision"]["status"],
                "stop_reason": result["stop_reason"],
            })
        ADVANCE = pd.DataFrame(rows)
        print(ADVANCE.to_string(index=False))
        ADVANCE.to_csv(RESULTS_DIR / "sq34b-advance.csv", index=False)

        statuses = {result["arm"]: result["decision"]["status"] for result in ARM_RESULTS}
        advancing = [name for name, status in statuses.items() if status == "advance"]
        mass_rejects = [
            result
            for result in ARM_RESULTS
            if result["decision"]["exposed"]
            and not result["decision"]["mass_ok"]
            and result["rounds_completed"] >= 6
        ]
        if advancing:
            next_step = (
                "advance " + ", ".join(advancing) + "; SQ34c may use that parent"
            )
        elif len(mass_rejects) == len(ARMS):
            next_step = (
                "all three arms moved through the horizon and wrecked mass; "
                "stop tuning this schedule and change the objective"
            )
        else:
            next_step = (
                "no arm advanced; do not abandon the loop yet. "
                "read the inconclusive versus reject split before designing SQ34c"
            )

        path_rows = []
        for _, row in sq34_rounds.iterrows():
            path_rows.append({
                "arm": "sq34-prior",
                "round": int(row["round"]),
                "optimizer_steps": int(row["optimizer_steps"]),
                "parameter_delta_norm": float(row["parameter_delta_norm"]),
                "mean_kl": float(row["mean_kl"]),
                "anchor_median_candidate_mass": float(
                    row["anchor_median_candidate_mass"]
                ),
                "censored": False,
            })
        if len(ROUNDS_TABLE):
            for _, row in ROUNDS_TABLE.iterrows():
                path_rows.append({
                    "arm": row["arm"],
                    "round": int(row["round"]),
                    "optimizer_steps": int(row["optimizer_steps"]),
                    "parameter_delta_norm": float(row["parameter_delta_norm"]),
                    "mean_kl": float(row["mean_kl"]),
                    "anchor_median_candidate_mass": float(
                        row["anchor_median_candidate_mass"]
                    ),
                    "censored": bool(row["drift_tripped"]),
                })
        PATHS = pd.DataFrame(path_rows)
        print("mass / movement path by update count:")
        print(PATHS.to_string(index=False))
        PATHS.to_csv(RESULTS_DIR / "sq34b-trajectories.csv", index=False)
        print("next step:", next_step)

        RUN = {
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "answer_pool_sha256": ANSWER_POOL_SHA256,
            "anchor_prompt_sha256": ANCHOR_SHA256,
            "entry_checkpoint_sha256": checkpoint_sha256,
            "initial_lora_digest": INITIAL_LORA_DIGEST,
            "temperature": TEMPERATURE,
            "baseline_deterministic": BASELINE_DETERMINISTIC,
            "baseline_greedy": BASELINE_GREEDY,
            "baseline_stochastic": BASELINE_STOCHASTIC,
            "baseline_stochastic_trials": BASELINE_STOCHASTIC_TRIALS,
            "mass_hard_floor": MASS_FLOOR,
            "mass_advance_floor": MASS_ADVANCE_FLOOR,
            "arms": ARM_RESULTS,
            "statuses": statuses,
            "next_step": next_step,
            "trie_forwards": policy.forward_call_count,
            "full_list_sweeps": BUDGET["full_list_sweeps"],
            "sq34_prior_stop": sq34_run["stop_reason"],
        }
        atomic_json(RUN, RESULTS_DIR / "sq34b-run.json")
        print(json.dumps({"statuses": statuses, "next_step": next_step}, indent=2))
    """
)

md(
    """
    ## SQ34b.8 What this can settle

    I care about movement first, then candidate mass through six rounds, then
    whether stochastic play fell by more than one reserved answer's battery.
    The graded decoder is a footnote on 19 games.

    If clip is still zero, this still was not a test of GRPO clipping. A safe
    arm is a parent for SQ34c, not a Lab 34 result.

    If `lower-step` barely twitches, that is inconclusive. I will not pretend
    preserved mass is a finding.

    Lab 20 stays paused.
    """
)

for index, cell in enumerate(cells):
    cell["id"] = f"sq34b-{index:02d}-{cell['cell_type'][:2]}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).parent / "notebooks" / "sq34b_optimization_study.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {target} with {len(cells)} cells")
