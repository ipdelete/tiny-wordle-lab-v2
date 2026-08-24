"""Generate notebooks/sq34_simulator_grpo.ipynb."""

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
    # SQ34 - Simulator GRPO

    SQ31 built the environment, the trie policy, and the immutable trace, and
    froze temperature 1.0 with 56.2% mixed-outcome groups. Nothing has been
    trained. This notebook asks whether sparse win/lose reward over complete
    episodes improves the Lab 18d seed 45 policy.

    It asks that question twice, because the answer can differ.

    GRPO trains the *trie policy*, the distribution defined by masked
    token-level sampling over the 2,315-word answer vocabulary. The curriculum
    grades the *deterministic answer-constrained full-string decoder*. SQ31
    measured those two decoders agreeing on only 45.5% of Lab 18d states, so
    they are different functions of the same weights. A gain in one does not
    imply a gain in the other.

    So the run reports three things separately, and in this order:

    1. whether training moved the weights at all;
    2. whether it improved the policy that was trained;
    3. whether it improved the policy that gets graded.

    Reporting only the third would hide a real result. Reporting only the
    second would claim one that the curriculum cannot use.

    Three gates, each blocking the next. Gate A is model-free and checks the
    objective. Gate B loads the checkpoint and checks that the machinery is
    faithful and bounded, without touching a weight. Gate C trains.
    """
)

md(
    """
    ## SQ34.1 Preregistration

    Everything the PRD requires frozen is frozen here, written to disk, and
    hashed before a single episode is sampled. The stop rules are part of it.
    A rule invented after seeing the numbers is not a stop rule.
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
        mean_only_advantages,
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
    RESULTS_DIR = ROOT / "results" / "sq34"
    SQ31_DIR = ROOT / "results" / "sq31"
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

    RUN_MODEL = os.environ.get("SQ34_RUN_MODEL", "0") == "1"

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
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
        temporary.replace(path)


    sq31 = json.loads((SQ31_DIR / "sq31-run.json").read_text())
    assert sq31["sampling_gate_passed"] is True, (
        "SQ34 does not start from a failed sampler gate"
    )
    assert sq31["opening"] == OPENING
    assert sq31["max_turns"] == MAX_TURNS
    assert tuple(sq31["reserved_answers"]) == RESERVED_ANSWERS
    TEMPERATURE = float(sq31["frozen_temperature"])

    CHECKPOINT = (
        ROOT
        / "checkpoints"
        / "qwen3-0.6b-wordle-lora-dataset-b-structured-seed45"
    )
    ADAPTER = CHECKPOINT / "adapter_model.safetensors"
    checkpoint_sha256 = sha256_file(ADAPTER) if ADAPTER.exists() else None
    if checkpoint_sha256 is not None:
        assert checkpoint_sha256 == sq31["checkpoint_sha256"], (
            "the seed 45 adapter differs from the one SQ31 froze"
        )
    print("frozen temperature:", TEMPERATURE)
    print("seed 45 adapter available:", ADAPTER.exists())
    print("seed 45 adapter sha256:", checkpoint_sha256)
    """
)

md(
    """
    ### Why 5e-5

    512 episodes is a small run. Roughly half the groups will be degenerate, so
    about 128 optimizer steps carry the whole experiment.

    Lab 20 trained supervised at 1e-5. Reusing that here would probably return a
    null caused by the step count rather than by the method, and that is the one
    outcome nothing can be learned from.

    So 5e-5, preregistered. High enough to move a LoRA in 128 steps, and an
    order of magnitude below where LoRA training usually falls apart. The anchor
    drift guard and the KL term are what catch an overshoot, and a drift stop is
    a recorded outcome, not a broken notebook.
    """
)

code(
    """
    HYPER = GrpoHyperparameters(
        clip_lower=0.8,
        clip_upper=1.2,
        advantage_epsilon=1e-4,
        kl_coefficient=0.02,
    )

    GROUP_SIZE = 4
    GROUPS_PER_ROUND = 16
    ROUNDS = 8
    TOTAL_GROUPS = GROUPS_PER_ROUND * ROUNDS
    TOTAL_EPISODES = TOTAL_GROUPS * GROUP_SIZE
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 8
    GRADIENT_CLIP_NORM = 1.0
    OPTIMIZER_SEED = 34
    POOL_SEED = 3401
    SAMPLING_SEED_BASE = 340000
    EVAL_SEED_BASE = 900000
    STOCHASTIC_EVAL_EPISODES = 8

    MEMORY_CAP_GIB = 128.0
    MEMORY_ABORT_GIB = 96.0
    SOAK_ITERATIONS = 40

    # Model-call budget. Trie forwards and full-list sweeps are counted
    # separately because a full-list sweep costs roughly 3.46 s and a trie
    # forward costs a small fraction of that. Comparing methods on episode
    # counts when their call counts differ is exactly what the PRD forbids.
    TRIE_FORWARD_BUDGET = 60000
    FULL_LIST_SWEEP_BUDGET = 900

    # Drift thresholds, taken unchanged from Lab 20.
    DRIFT_CANDIDATE_MASS_RATIO = 0.70
    DRIFT_RANK_MULTIPLIER = 4
    DRIFT_RANK_FLOOR = 10
    DRIFT_WINNER_SHARE_FLOOR = 0.50
    DRIFT_WINNER_SHARE_MARGIN = 0.25

    # Diversity-collapse thresholds. SQ31 measured 56.2% mixed groups and a
    # 3.1% repeat rate on the confirmation half at this temperature, so these
    # are floors well below the observed starting point rather than targets.
    MIN_MIXED_FRACTION = 0.15
    MAX_REPEAT_RATE = 0.25
    MIN_SURPRISAL_RATIO = 0.30

    TOTAL_ASSERTED = TOTAL_GROUPS == 128 and TOTAL_EPISODES == 512
    assert TOTAL_ASSERTED, "the PRD caps 128 groups and 512 episodes"

    PREREGISTRATION = {
        "experiment": "SQ34 simulator GRPO",
        "entry_checkpoint": "lab18d seed 45 structured LoRA",
        "entry_checkpoint_sha256": checkpoint_sha256,
        "sq31_tokenizer_sha256": sq31["tokenizer_sha256"],
        "sq31_action_vocabulary_sha256": sq31["action_vocabulary_sha256"],
        "temperature": TEMPERATURE,
        "opening": OPENING,
        "max_turns": MAX_TURNS,
        "group_size": GROUP_SIZE,
        "groups_per_round": GROUPS_PER_ROUND,
        "rounds": ROUNDS,
        "total_groups": TOTAL_GROUPS,
        "total_episodes": TOTAL_EPISODES,
        "learning_rate": LEARNING_RATE,
        "schedule": "linear warmup then constant",
        "warmup_steps": WARMUP_STEPS,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "gradient_accumulation": "one optimizer step per group",
        "optimizer": "AdamW, LoRA parameters only, weight decay 0",
        "optimizer_seed": OPTIMIZER_SEED,
        "answer_pool_seed": POOL_SEED,
        "sampling_seed_base": SAMPLING_SEED_BASE,
        "checkpoint_cadence": "every round",
        "held_out_decoder_evaluations": ["baseline", "round 3", "final"],
        "stochastic_eval_episodes_per_answer": STOCHASTIC_EVAL_EPISODES,
        "dropout": "disabled for sampling, scoring, and training",
        "reference_policy": "frozen seed 45 adapter, loaded as a second LoRA",
        "trie_forward_budget": TRIE_FORWARD_BUDGET,
        "full_list_sweep_budget": FULL_LIST_SWEEP_BUDGET,
        "stop_rules": {
            "anchor_drift": "Lab 20 thresholds, unchanged",
            "minimum_mixed_group_fraction": MIN_MIXED_FRACTION,
            "maximum_repeat_rate": MAX_REPEAT_RATE,
            "minimum_surprisal_ratio_to_round_zero": MIN_SURPRISAL_RATIO,
            "trie_forward_budget": TRIE_FORWARD_BUDGET,
            "full_list_sweep_budget": FULL_LIST_SWEEP_BUDGET,
        },
        "primary_outcome": (
            "deterministic answer-constrained solve rate on the 19 reserved "
            "answers, against the seed 45 baseline of 10/19, paired McNemar"
        ),
        "secondary_outcome": (
            "stochastic trie-policy solve rate on the same 19 answers"
        ),
        "movement_check": (
            "reported before any gameplay claim: reference KL, LoRA parameter "
            "delta norm, and the fraction of groups producing a nonzero update"
        ),
    }
    PREREGISTRATION["objective"] = HYPER.as_manifest()

    preregistration_text = json.dumps(PREREGISTRATION, indent=2, sort_keys=True)
    PREREGISTRATION_SHA256 = sha256_text(preregistration_text)
    atomic_json(PREREGISTRATION, RESULTS_DIR / "sq34-preregistration.json")
    print("preregistration sha256:", PREREGISTRATION_SHA256)
    print(preregistration_text)
    """
)

md(
    """
    ## SQ34.2 Gate A: the objective, on a hand-built group

    Two properties of GRPO are easy to state and easy to forget once a run is
    printing numbers. This cell shows both on hand-built tensors, so nothing the
    checkpoint does can reach the result.

    The first is that **the loss value is zero at collection while the gradient
    is not**. Group-relative advantages are centered by construction, so at
    ratio one every group's loss is zero. A training curve of this loss would
    look flat no matter what was happening. That is why the movement check
    exists and why the loss is not the headline number.

    The second is that **per-episode normalization is what makes a six-turn
    loss comparable to a two-turn win**. Without it, an episode's influence
    would scale with its length, and length is correlated with losing.
    """
)

code(
    """
    import torch

    demo_batch = GroupBatch(
        group_id="demo",
        answer="SHORE",
        behavior_checkpoint_digest="demo",
        reference_checkpoint_digest="demo",
        rewards=(1.0, 1.0, 0.0, 0.0),
        behavior_log_probabilities=(
            (-3.0, -3.0),
            (-3.0, -3.0),
            (-3.0,) * 5,
            (-3.0,) * 5,
        ),
        reference_log_probabilities=(
            (-3.0, -3.0),
            (-3.0, -3.0),
            (-3.0,) * 5,
            (-3.0,) * 5,
        ),
    )
    demo_current = [
        torch.tensor([-3.0] * count, requires_grad=True)
        for count in demo_batch.action_counts
    ]
    demo_loss, demo_diagnostics = group_loss(
        demo_batch, demo_current, hyperparameters=HYPER
    )
    demo_loss.backward()

    print("group rewards:", demo_batch.rewards)
    print("action counts:", demo_batch.action_counts)
    print("loss value:", float(demo_loss.detach()))
    print("winning-episode action gradient:", float(demo_current[0].grad[0]))
    print("losing-episode action gradient:", float(demo_current[2].grad[0]))
    print("mean ratio:", demo_diagnostics.mean_ratio)
    print("clipped fraction:", demo_diagnostics.clipped_fraction)

    assert abs(float(demo_loss.detach())) < 1e-5, "loss is centered by design"
    assert float(demo_current[0].grad[0]) < 0.0
    assert float(demo_current[2].grad[0]) > 0.0
    # A two-action win and a five-action loss carry equal and opposite total
    # influence, which is the per-episode normalization doing its job.
    assert math.isclose(
        abs(float(demo_current[0].grad.sum())),
        abs(float(demo_current[2].grad.sum())),
        rel_tol=1e-5,
    )
    """
)

code(
    """
    degenerate_batch = GroupBatch(
        group_id="degenerate",
        answer="SHORE",
        behavior_checkpoint_digest="demo",
        reference_checkpoint_digest="demo",
        rewards=(0.0, 0.0, 0.0, 0.0),
        behavior_log_probabilities=((-3.0, -3.0),) * 4,
        reference_log_probabilities=((-3.0, -3.0),) * 4,
    )
    degenerate_current = [
        torch.tensor([-3.0, -3.0], requires_grad=True) for _ in range(4)
    ]
    degenerate_loss, degenerate_diagnostics = group_loss(
        degenerate_batch, degenerate_current, hyperparameters=HYPER
    )
    degenerate_loss.backward()

    assert degenerate_diagnostics.degenerate is True
    assert all(
        float(tensor.grad.abs().sum()) == 0.0 for tensor in degenerate_current
    )
    print("all-zero group advantage:", degenerate_diagnostics.max_absolute_advantage)
    print("all-zero group gradient: none, as required")

    # The recorded caveat about standard-deviation normalization, as a number.
    lopsided = group_advantages(
        torch.tensor([1.0, 0.0, 0.0, 0.0]), epsilon=HYPER.advantage_epsilon
    )
    balanced = group_advantages(
        torch.tensor([1.0, 1.0, 0.0, 0.0]), epsilon=HYPER.advantage_epsilon
    )
    print("one-of-four advantages:", [round(v, 4) for v in lopsided.tolist()])
    print("two-of-four advantages:", [round(v, 4) for v in balanced.tolist()])
    print(
        "lopsided winner push relative to balanced:",
        round(float(lopsided.max()) / float(balanced.max()), 4),
    )
    print(
        "mean-only variant, recorded but not trained:",
        [round(v, 4) for v in mean_only_advantages(
            torch.tensor([1.0, 0.0, 0.0, 0.0])
        ).tolist()],
    )
    """
)

md(
    """
    ## SQ34.3 Gate A: the training answer pool

    128 answers, one per group, drawn by a seeded permutation and disjoint from
    the 19 reserved evaluation answers. `RAISE` is excluded. It is a legal
    answer, but the fixed opening solves it before the policy acts, so its group
    would be four zero-action episodes and no gradient.
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
    ANSWER_POOL = tuple(
        pool_candidates[int(index)] for index in permutation[:TOTAL_GROUPS]
    )

    assert len(ANSWER_POOL) == TOTAL_GROUPS
    assert len(set(ANSWER_POOL)) == TOTAL_GROUPS, "no answer trains twice"
    assert not (set(ANSWER_POOL) & RESERVED_SET), "training leaked into held out"
    assert OPENING not in ANSWER_POOL

    ANSWER_POOL_SHA256 = sha256_text("\\n".join(ANSWER_POOL))
    print("training answers:", len(ANSWER_POOL))
    print("pool sha256:", ANSWER_POOL_SHA256)
    print("first eight:", ANSWER_POOL[:8])
    """
)

code(
    """
    class PromptOnly(NamedTuple):
        \"\"\"The only field the policy reads off an observation.\"\"\"

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
        \"\"\"Two-sided exact McNemar test on discordant pairs.

        The same 19 answers are played before and after, so the comparison is
        paired. Treating the two solve rates as independent samples would
        throw away that pairing and badly understate the evidence.
        \"\"\"
        discordant = only_before + only_after
        if discordant == 0:
            return 1.0
        tail = sum(
            math.comb(discordant, k) for k in range(min(only_before, only_after) + 1)
        )
        return min(1.0, 2.0 * tail / (2 ** discordant))


    def smallest_detectable_discordance(alpha=0.05, limit=19):
        \"\"\"How lopsided must the discordant pairs be to reach significance.

        Reported before the run so an 11/19 result is read for what it is.
        \"\"\"
        for discordant in range(1, limit + 1):
            if mcnemar_exact(0, discordant) <= alpha:
                return discordant
        return None


    DETECTABLE = smallest_detectable_discordance()
    print("environment ready")
    print(
        "smallest all-one-way discordant count reaching p<=0.05:",
        DETECTABLE,
    )
    print(
        "so fewer than", DETECTABLE,
        "newly solved answers cannot be called significant, however they split",
    )
    """
)

md(
    """
    ## SQ34.4 Gate A: the frozen anchor suite

    The 20 Lab 20 full-list anchor states, reused unchanged along with their
    metric definitions and thresholds. Anchors are a safety signal. They never
    select the reported checkpoint.

    The leakage check matters. An anchor state reachable only by a reserved
    answer would feed held-out information into a guard that runs every round.
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
    print("anchor states:", ANCHOR_COUNT)
    print("regimes:", Counter(ANCHOR_REGIME))
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


    print("anchor metric and drift definitions loaded from the Lab 20 contract")
    """
)


md(
    """
    ## SQ34.5 Gate B: load the checkpoint twice

    The reference policy is the frozen seed 45 adapter, not the base model, so
    `disable_adapter()` would compare against the wrong thing. Loading a whole
    second model would duplicate 0.6B base weights to hold one extra LoRA.

    So the same adapter is loaded twice onto one base model, under the names
    `policy` and `reference`, and `set_adapter` switches between them. Only
    `policy` is trainable.

    One PEFT detail decides whether this works. `PeftModel.from_pretrained`
    defaults to inference mode, which freezes the adapter it loads, and
    `set_adapter` flips `requires_grad` on whichever adapter it activates. So
    `policy` is loaded with `is_trainable=True`, the optimizer's parameter list
    is captured once while `policy` is active, and reference scoring always
    runs under `no_grad`. All three are asserted rather than assumed.

    The model gate is opt-in with `SQ34_RUN_MODEL=1`, which keeps Gate A
    runnable on a machine without the checkpoint.
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
        trie_shape = trie.shape()
        assert trie_shape == sq31["trie_shape"] or (
            trie_shape["branching_by_depth"] == {0: 1, 1: 235, 2: 113}
        )
        print("trie shape:", trie_shape)
    else:
        print("model gate skipped; set SQ34_RUN_MODEL=1 for Gate B")
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

        POLICY_PARAMETERS = [
            parameter
            for name, parameter in model.named_parameters()
            if "lora_" in name and ".policy." in name
        ]
        REFERENCE_PARAMETER_NAMES = [
            name
            for name, _ in model.named_parameters()
            if "lora_" in name and ".reference." in name
        ]
        assert POLICY_PARAMETERS, "no trainable policy LoRA tensors were found"
        assert REFERENCE_PARAMETER_NAMES, "the reference adapter did not load"
        assert len(POLICY_PARAMETERS) == len(REFERENCE_PARAMETER_NAMES)
        assert all(
            parameter.requires_grad for parameter in POLICY_PARAMETERS
        ), "the policy adapter loaded frozen; is_trainable=True is required"

        trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        assert all("lora_" in name and ".policy." in name for name in trainable), (
            f"unexpected trainable tensors: {trainable[:4]}"
        )
        POLICY_PARAMETER_COUNT = sum(
            parameter.numel() for parameter in POLICY_PARAMETERS
        )
        print("trainable policy LoRA tensors:", len(POLICY_PARAMETERS))
        print("trainable parameters:", POLICY_PARAMETER_COUNT)
    """
)

code(
    """
    if RUN_MODEL:
        @contextmanager
        def using_adapter(name):
            \"\"\"Activate an adapter and restore the policy afterwards.

            set_adapter flips requires_grad on the adapter it activates, so
            every exit path has to restore the policy adapter or the optimizer
            silently stops receiving gradients.
            \"\"\"
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


        INITIAL_LORA_DIGEST = lora_digest()
        INITIAL_LORA_STATE = [
            parameter.detach().to("cpu", torch.float32).clone()
            for parameter in POLICY_PARAMETERS
        ]
        print("device:", device)
        print("initial policy LoRA digest:", INITIAL_LORA_DIGEST)
    """
)

md(
    """
    ### Verification anchor 1: the reference adapter switches something

    Both adapters were loaded from the same file, so before any update their
    scores must agree exactly. Bit-exact is the right bar here rather than
    approximate, because identical weights through identical code produce
    identical floats. After the first optimizer step they must disagree.

    A reference that silently tracked the policy would drive the KL term to
    zero and remove the only thing holding the update near the incumbent, and
    the run would look healthy the whole way down.
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
        assert policy_scores == reference_scores_before, (
            "the two adapters disagree before training; the load is wrong"
        )
        assert model.active_adapter == "policy"
        assert all(parameter.requires_grad for parameter in POLICY_PARAMETERS)
        print("reference identity before training: exact match")
    """
)

md(
    """
    ### Verification anchor 2: the cheap scoring path is the same function

    `score_action` reads every position from one teacher-forced forward.
    `log_probability_tensor` walks the action one forward at a time. They
    compute the same masked quantity, so their per-token values must agree, and
    the forward counter must show the saving is real.

    The comparison uses a float tolerance rather than equality, because the
    two paths reduce in different orders and run through different kernel
    shapes.
    """
)

code(
    """
    if RUN_MODEL:
        equivalence_states = []
        for answer in ("SHORE", "MOUNT", "PIQUE"):
            environment = make_environment()
            reset = environment.reset(answer)
            equivalence_states.append(reset.observation)
            if not reset.done:
                observation, _, done, _ = environment.step("CLOUT")
                equivalence_states.append(observation)
                if not done:
                    observation, _, done, _ = environment.step("BRINE")
                    if not done:
                        equivalence_states.append(observation)

        equivalence_rows = []
        with torch.no_grad():
            for state_index, observation in enumerate(equivalence_states):
                for word in ("SHORE", "APPLE", "AMPLE", "ZEBRA", "MOUNT"):
                    before_batched = policy.forward_call_count
                    batched_total, batched_values = policy.score_action(
                        observation,
                        word,
                        temperature=TEMPERATURE,
                        requires_grad=False,
                    )
                    batched_calls = policy.forward_call_count - before_batched
                    before_walk = policy.forward_call_count
                    walk_total, walk_values = policy.log_probability_tensor(
                        observation, word, temperature=TEMPERATURE
                    )
                    walk_calls = policy.forward_call_count - before_walk
                    difference = max(
                        abs(float(left) - float(right))
                        for left, right in zip(batched_values, walk_values)
                    )
                    equivalence_rows.append({
                        "state": state_index,
                        "word": word,
                        "batched_total": float(batched_total),
                        "walk_total": float(walk_total),
                        "max_token_difference": difference,
                        "batched_forwards": batched_calls,
                        "walk_forwards": walk_calls,
                    })

        equivalence = pd.DataFrame(equivalence_rows)
        worst = float(equivalence["max_token_difference"].max())
        print(equivalence.head(10).to_string(index=False))
        print("states:", len(equivalence_states), " actions:", len(equivalence))
        print("worst per-token difference:", worst)
        print(
            "forwards, batched vs sequential:",
            int(equivalence["batched_forwards"].sum()),
            "vs",
            int(equivalence["walk_forwards"].sum()),
        )
        assert worst < 1e-4, "the scoring paths disagree; do not train"
        assert equivalence["batched_forwards"].max() == 1
        assert (
            equivalence["batched_forwards"].sum()
            < equivalence["walk_forwards"].sum()
        )
        equivalence.to_csv(RESULTS_DIR / "sq34-scoring-equivalence.csv", index=False)
    """
)

code(
    """
    if RUN_MODEL:
        out_of_trie = None
        try:
            policy.score_action(
                probe_observation, "QQQQQ", temperature=TEMPERATURE
            )
        except Exception as error:  # noqa: BLE001
            out_of_trie = type(error).__name__
        print("scoring a word outside the action vocabulary raises:", out_of_trie)
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
        print("scoring in train mode raises:", dropout_guard)
        assert dropout_guard is not None, (
            "dropout during scoring would corrupt the ratio silently"
        )
    """
)


md(
    """
    ## SQ34.6 Gate B: sampling a group, and the ratio identity

    One group is four episodes on the same answer, sampled from a frozen
    behavior checkpoint. The reward is sparse and binary. One for solving
    within six turns, zero otherwise. Nothing shapes it.

    Then verification anchor 3, the strongest check in the notebook. At the
    moment a group is collected, the current policy *is* the behavior policy,
    so every word-level ratio must be one. That single comparison catches mask
    drift, tokenizer drift, a failed adapter switch, a prompt rendered
    differently at scoring time than at sampling time, and dropout leaking into
    the forward. A round that fails it does not run its optimizer step.

    The tolerance is not decorative. Sampling sums per-token values with
    `math.fsum`; scoring sums them with `torch.stack().sum()`. Those disagree in
    the last bits by construction, so exact equality would fail on a correct
    implementation.
    """
)

code(
    """
    if RUN_MODEL:
        RATIO_IDENTITY_TOLERANCE = 1e-4

        def freeze_behavior_digest():
            \"\"\"Stamp the live weights as the behavior checkpoint.

            TriePolicy stamps every decision with its own checkpoint_digest and
            collect_trajectory refuses a trajectory whose collector metadata
            disagrees. The digest loaded at construction is the hash of the
            safetensors file, which stops being true the moment the optimizer
            steps, so it is replaced here by a hash of the live parameter bytes
            and both sides are handed the same value.
            \"\"\"
            digest = lora_digest()
            policy.checkpoint_digest = digest
            return digest


        def sample_group(answer, *, group_index, round_index, behavior_digest):
            assert policy.checkpoint_digest == behavior_digest, (
                "sample_group was called without freezing the behavior digest"
            )
            trajectories = []
            for episode_index in range(GROUP_SIZE):
                environment = make_environment()
                trajectory = collect_trajectory(
                    environment,
                    policy,
                    answer,
                    group_id=f"r{round_index:02d}-g{group_index:02d}",
                    answer_split="train",
                    temperature=TEMPERATURE,
                    sampling_seed=(
                        SAMPLING_SEED_BASE
                        + round_index * 10000
                        + group_index * 100
                        + episode_index
                    ),
                    episode_id=f"r{round_index:02d}-g{group_index:02d}-e{episode_index}",
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


        probe_digest = freeze_behavior_digest()
        probe_group = sample_group(
            ANSWER_POOL[0],
            group_index=0,
            round_index=0,
            behavior_digest=probe_digest,
        )
        print("probe answer:", ANSWER_POOL[0])
        for trajectory in probe_group:
            print(
                " ",
                [step.decision.word for step in trajectory.steps],
                trajectory.terminal_reason,
                "reward",
                episode_reward(trajectory),
            )
    """
)

code(
    """
    if RUN_MODEL:
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


        def ratio_identity_report(trajectories):
            rows = []
            for trajectory in trajectories:
                behavior = behavior_log_probabilities(trajectory)
                with torch.no_grad():
                    current = score_episode(
                        trajectory, adapter="policy", requires_grad=False
                    )
                for index, (left, right) in enumerate(zip(behavior, current)):
                    rows.append({
                        "episode": trajectory.episode_id,
                        "action": index,
                        "behavior": left,
                        "current": float(right),
                        "ratio": math.exp(float(right) - left),
                    })
            return pd.DataFrame(rows)


        identity = ratio_identity_report(probe_group)
        worst_ratio_gap = float((identity["ratio"] - 1.0).abs().max())
        print(identity.head(8).to_string(index=False))
        print("actions compared:", len(identity))
        print("worst |ratio - 1|:", worst_ratio_gap)
        assert worst_ratio_gap < RATIO_IDENTITY_TOLERANCE, (
            "sampling and scoring disagree on the same policy; do not train"
        )
        identity.to_csv(RESULTS_DIR / "sq34-ratio-identity-gate.csv", index=False)
    """
)

md(
    """
    ## SQ34.7 Gate B: four memory soaks

    SQ31 established that a plateau is the only acceptable memory shape, and
    that the peak has to be read from inside the forward while tensors are
    live, because a reading taken after the call has already missed it.

    Four fixed-shape soaks, 40 iterations each, run on the worst case rather
    than the average. The longest prompt the environment produces, a four-token
    action, and a full accumulation group.

    1. sampling
    2. reference scoring under `no_grad`
    3. current-policy scoring with the graph retained
    4. a full training step, backward and optimizer update included

    The third and fourth are the ones that matter. They are the only ones that
    hold an activation graph, and they are the reason the update accumulates
    per episode rather than building all four graphs and then calling backward
    once.
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


        soak_environment = make_environment()
        soak_environment.reset("PIQUE")
        soak_observation = soak_environment.observation
        for filler in ("CLOUT", "BRAND", "MERGE", "SHYLY"):
            if soak_environment.done:
                break
            soak_observation, _, _, _ = soak_environment.step(filler)
        soak_prompt_length = len(
            tokenizer(render_prompt(soak_observation.prompt)).input_ids
        )
        SOAK_WORD = max(
            ANSWERS, key=lambda word: len(trie.sequence_for_word(word))
        )
        print("soak turn:", soak_observation.turn)
        print("soak prompt tokens:", soak_prompt_length)
        print(
            "soak action:",
            SOAK_WORD,
            "tokens:",
            len(trie.sequence_for_word(SOAK_WORD)),
        )
    """
)

code(
    """
    if RUN_MODEL:
        sampling_peaks = []
        for iteration in range(SOAK_ITERATIONS):
            start = len(policy.forward_memory_trace)
            policy.sample(
                soak_observation, temperature=TEMPERATURE, seed=7000 + iteration
            )
            sampling_peaks.append(peak_since(start))
            if device.type == "mps":
                torch.mps.empty_cache()
        assert_memory_plateau(sampling_peaks, "sampling")
        print(
            f"sampling soak: peak {max(sampling_peaks):.2f} GiB, "
            f"final {sampling_peaks[-1]:.2f} GiB"
        )

        reference_peaks = []
        for iteration in range(SOAK_ITERATIONS):
            start = len(policy.forward_memory_trace)
            with torch.no_grad(), using_adapter("reference"):
                policy.score_action(
                    soak_observation,
                    SOAK_WORD,
                    temperature=TEMPERATURE,
                    requires_grad=False,
                )
            reference_peaks.append(peak_since(start))
            if device.type == "mps":
                torch.mps.empty_cache()
        assert_memory_plateau(reference_peaks, "reference scoring")
        print(
            f"reference scoring soak: peak {max(reference_peaks):.2f} GiB, "
            f"final {reference_peaks[-1]:.2f} GiB"
        )
    """
)

code(
    """
    if RUN_MODEL:
        graph_peaks = []
        for iteration in range(SOAK_ITERATIONS):
            start = len(policy.forward_memory_trace)
            retained = [
                policy.score_action(
                    soak_observation,
                    SOAK_WORD,
                    temperature=TEMPERATURE,
                    requires_grad=True,
                )[0]
                for _ in range(MAX_TURNS - 1)
            ]
            graph_peaks.append(peak_since(start))
            total = torch.stack(retained).sum()
            total.backward()
            model.zero_grad(set_to_none=True)
            del retained, total
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        assert_memory_plateau(graph_peaks, "current-policy scoring with graph")
        print(
            f"retained-graph soak: peak {max(graph_peaks):.2f} GiB, "
            f"final {graph_peaks[-1]:.2f} GiB"
        )
    """
)

md(
    """
    ### Why the training step accumulates per episode

    `group_loss(...).backward()` is the obvious way to write the update, and it
    holds all four episodes' activation graphs alive at once, roughly sixteen
    forwards, until backward runs.

    Calling backward once per episode on `-objective / episode_count` frees
    each graph immediately and gives the same update, since the sum of
    per-episode gradients is the gradient of the mean. Peak memory becomes one
    episode instead of four.

    The group loss is still computed for diagnostics, on **detached** values, and
    asserted against the accumulated objective. That costs no forwards and keeps
    the reported number honest.
    """
)

code(
    """
    if RUN_MODEL:
        rehearsal_optimizer = torch.optim.AdamW(
            POLICY_PARAMETERS, lr=0.0, weight_decay=0.0
        )
        step_peaks = []
        for iteration in range(SOAK_ITERATIONS):
            start = len(policy.forward_memory_trace)
            rehearsal_optimizer.zero_grad(set_to_none=True)
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
                    hyperparameters=HYPER,
                )
                (-objective / GROUP_SIZE).backward()
                del values, stacked, objective
            torch.nn.utils.clip_grad_norm_(POLICY_PARAMETERS, GRADIENT_CLIP_NORM)
            rehearsal_optimizer.step()
            step_peaks.append(peak_since(start))
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        assert_memory_plateau(step_peaks, "full training step")
        rehearsal_optimizer.zero_grad(set_to_none=True)
        assert lora_digest() == INITIAL_LORA_DIGEST, (
            "the zero-learning-rate rehearsal changed the weights"
        )
        print(
            f"training-step soak: peak {max(step_peaks):.2f} GiB, "
            f"final {step_peaks[-1]:.2f} GiB"
        )
        print("weights unchanged after rehearsal:", lora_digest()[:16])

        atomic_json(
            {
                "sampling": sampling_peaks,
                "reference_scoring": reference_peaks,
                "retained_graph_scoring": graph_peaks,
                "training_step": step_peaks,
                "abort_threshold_gib": MEMORY_ABORT_GIB,
                "soak_prompt_tokens": soak_prompt_length,
                "soak_word": SOAK_WORD,
            },
            RESULTS_DIR / "sq34-memory-soaks.json",
        )
    """
)


md(
    """
    ## SQ34.8 Gate B: the deterministic decoder the curriculum grades with

    A frozen local copy of the Lab 18d chunked KV-cache kernel, per the
    repository convention that each notebook carries its own scoring code
    rather than importing a shared one that can change underneath a result.

    The PRD measures its pass criterion on this function, and GRPO does not
    train it. Running both in one notebook is the only way to tell the two
    results apart.

    Answer-constrained means constrained to the 2,315-word answer list, not
    constrained to the words still consistent with the feedback. Lab 18d takes
    the unmasked argmax over the whole list, and the seed 45 baseline picks a
    word that cannot be the answer on most of its turns. Adding a consistency
    mask here would build a different and much easier decoder, and 10/19 would
    stop being the number it is compared against. The count of inconsistent
    winners is printed below as evidence that the mask would not have been
    cosmetic.
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


        sweep_start = time.perf_counter()
        sweep_probe = score_all_words(probe_observation.prompt)
        sweep_seconds = time.perf_counter() - sweep_start
        print(f"one full-list sweep: {sweep_seconds:.2f} s")
        print(f"sweep peak: {LAST_STATE_PEAK_GIB:.2f} GiB")
        print("argmax word:", ANSWERS[int(sweep_probe.argmax())])
        assert LAST_STATE_PEAK_GIB < MEMORY_ABORT_GIB
    """
)

code(
    """
    if RUN_MODEL:
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
                )
                solved += int(trajectory.terminal_reason == "solved")
                if device.type == "mps":
                    torch.mps.empty_cache()
            return solved


        def evaluate_reserved(label, *, deterministic=True, seed_base=EVAL_SEED_BASE):
            rows = []
            for position, answer in enumerate(RESERVED_ANSWERS):
                record = {"answer": answer, "label": label}
                if deterministic:
                    game = deterministic_game(answer)
                    record["deterministic_solved"] = bool(game["solved"])
                    record["deterministic_turns"] = game["turns"]
                    record["deterministic_noncandidates"] = game[
                        "noncandidate_guesses"
                    ]
                    record["deterministic_guesses"] = " ".join(game["guesses"])
                greedy = greedy_trie_game(answer)
                record["greedy_solved"] = bool(greedy["solved"])
                record["greedy_turns"] = greedy["turns"]
                record["stochastic_solved"] = stochastic_play(
                    answer, seed_base=seed_base + position * 1000
                )
                rows.append(record)
            return pd.DataFrame(rows)


        print("evaluation helpers ready")
    """
)

code(
    """
    if RUN_MODEL:
        baseline_start = time.perf_counter()
        BASELINE_EVAL = evaluate_reserved("baseline")
        baseline_minutes = (time.perf_counter() - baseline_start) / 60.0

        BASELINE_DETERMINISTIC = int(BASELINE_EVAL["deterministic_solved"].sum())
        BASELINE_GREEDY = int(BASELINE_EVAL["greedy_solved"].sum())
        BASELINE_STOCHASTIC = int(BASELINE_EVAL["stochastic_solved"].sum())
        BASELINE_STOCHASTIC_TRIALS = 19 * STOCHASTIC_EVAL_EPISODES

        print(BASELINE_EVAL.drop(columns=["deterministic_guesses"]).to_string(index=False))
        print(f"baseline evaluation took {baseline_minutes:.1f} min")
        print("deterministic solved:", BASELINE_DETERMINISTIC, "/ 19")
        print("greedy trie solved:", BASELINE_GREEDY, "/ 19")
        print(
            "stochastic trie solved:",
            BASELINE_STOCHASTIC,
            "/",
            BASELINE_STOCHASTIC_TRIALS,
            "interval",
            tuple(
                round(value, 4)
                for value in wilson_interval(
                    BASELINE_STOCHASTIC, BASELINE_STOCHASTIC_TRIALS
                )
            ),
        )
        print(
            "turns whose winner was not feedback-consistent:",
            int(BASELINE_EVAL["deterministic_noncandidates"].sum()),
            "of",
            int(BASELINE_EVAL["deterministic_turns"].sum() - 19),
        )
        assert BASELINE_DETERMINISTIC == 10, (
            f"expected the Lab 18d seed 45 baseline of 10/19, got "
            f"{BASELINE_DETERMINISTIC}/19"
        )
        BASELINE_EVAL.to_csv(RESULTS_DIR / "sq34-baseline-eval.csv", index=False)
    """
)

md(
    """
    ### The decoder gap, restated as a number before training

    SQ31 measured greedy trie decoding and full-string argmax picking the same
    word on 45.5% of Lab 18d states. That figure is recomputed here on the
    anchor states, before any weight changes, so the after-training figure has
    something to move against.

    If training improves the trie policy and this agreement rate falls, the two
    decoders have diverged further and the curriculum's training interface and
    its evaluation interface are drifting apart. That is a publishable result
    on its own, and it is the outcome this notebook is built to be able to see.
    """
)

code(
    """
    if RUN_MODEL:
        def anchor_score_matrix():
            return np.stack(
                [score_all_words(prompt) for prompt in ANCHOR_PROMPTS]
            )


        def anchor_agreement(score_matrix):
            agreements = []
            for position, prompt in enumerate(ANCHOR_PROMPTS):
                full_string = ANSWERS[int(score_matrix[position].argmax())]
                greedy = policy.greedy_word(PromptOnly(prompt))
                BUDGET["trie_forwards"] += 1
                agreements.append(full_string == greedy)
            return float(np.mean(agreements)), agreements


        BASELINE_ANCHOR_SCORES = anchor_score_matrix()
        BASELINE_ANCHOR_METRICS = anchor_metrics(BASELINE_ANCHOR_SCORES)
        BASELINE_ANCHOR_SUMMARY = anchor_summary(BASELINE_ANCHOR_METRICS)
        BASELINE_AGREEMENT, _ = anchor_agreement(BASELINE_ANCHOR_SCORES)

        print(BASELINE_ANCHOR_METRICS.to_string(index=False))
        print("baseline anchor summary:", BASELINE_ANCHOR_SUMMARY)
        print("baseline greedy-trie vs full-string agreement:", BASELINE_AGREEMENT)
        print("SQ31 recorded 0.4545 on the Lab 18d state suite")
        print("full-list sweeps used so far:", BUDGET["full_list_sweeps"])
    """
)


md(
    """
    ## SQ34.9 Gate C: eight rounds

    The PRD requires a frozen behavior checkpoint per group and one optimizer
    epoch per behavior batch. Sampling all 512 episodes up front would satisfy
    the letter of that and violate its point. Every round after the first would
    run off-policy against a staler behavior policy, and the ratios would drift
    out of the near-one regime the clipping assumes.

    So the run is eight rounds of 16 groups of 4 episodes. 128 groups, 512
    episodes, exactly the PRD caps.

    A round freezes the behavior digest, samples, computes rewards and
    group-relative advantages, drops degenerate groups, gates on the ratio
    identity, and takes one optimizer step per surviving group. Then it
    checkpoints, scores the anchors, measures diversity, and checks every stop
    rule.

    Expect roughly 56% of groups to survive. SQ31 measured that mixed-outcome
    rate at this temperature, so it is preregistered rather than explained
    afterwards.
    """
)

code(
    """
    if RUN_MODEL:
        torch.manual_seed(OPTIMIZER_SEED)
        optimizer = torch.optim.AdamW(
            POLICY_PARAMETERS, lr=LEARNING_RATE, weight_decay=0.0
        )
        optimizer_steps = 0

        def learning_rate_for(step):
            if step < WARMUP_STEPS:
                return LEARNING_RATE * (step + 1) / WARMUP_STEPS
            return LEARNING_RATE


        def parameter_delta_norm():
            total = 0.0
            for parameter, initial in zip(POLICY_PARAMETERS, INITIAL_LORA_STATE):
                delta = parameter.detach().to("cpu", torch.float32) - initial
                total += float((delta * delta).sum())
            return math.sqrt(total)


        def diversity_metrics(trajectories):
            \"\"\"Whether the policy is still exploring.

            GRPO sharpens logits, so a temperature chosen from an untrained
            policy can stop being appropriate. Surprisal is measured in nats
            per action from the behavior log probabilities, which are already
            recorded, so this costs no forwards.
            \"\"\"
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


        print("optimizer ready, learning rate", LEARNING_RATE)
    """
)

code(
    """
    if RUN_MODEL:
        TRACE_DIR = RESULTS_DIR / "traces"
        TRACE_DIR.mkdir(parents=True, exist_ok=True)

        GROUP_COLUMNS = [
            "round", "group", "answer", "rewards", "degenerate", "updated",
            "gate_failed", "trace_sha256", "loss", "mean_kl",
            "clipped_fraction", "ratio_spread", "gradient_norm",
            "learning_rate",
        ]
        ROUND_COLUMNS = [
            "round", "behavior_checkpoint_digest", "groups", "groups_updated",
            "groups_degenerate", "mixed_fraction", "solve_rate",
            "mean_objective", "mean_kl", "mean_clipped_fraction",
            "worst_ratio_gap", "effective_sample_size",
            "parameter_delta_norm", "optimizer_steps", "drift_tripped",
        ]

        def persist_group(round_index, group_index, answer, digest, trajectories):
            \"\"\"Write the exact episodes before the optimizer sees them.

            The PRD requires the answer order, the seeds, the behavior
            checkpoint, and the trace hashes to be on disk before the update
            that used them. Writing after the fact would lose precisely the
            evidence a stop rule exists to preserve.
            \"\"\"
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
            path = TRACE_DIR / f"{stem}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(text)
            temporary.replace(path)
            # The hash has to be on disk too, not only in an in-memory row that
            # is written after training. A crash between here and the end of the
            # run would otherwise keep the trace and lose the number that proves
            # the trace is the one the optimizer saw.
            sidecar = TRACE_DIR / f"{stem}.sha256"
            sidecar_temporary = sidecar.with_suffix(".tmp")
            sidecar_temporary.write_text(digest_of_text)
            sidecar_temporary.replace(sidecar)
            return digest_of_text


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
                )
                round_trajectories.extend(trajectories)
                rewards = tuple(episode_reward(t) for t in trajectories)
                behavior = tuple(behavior_log_probabilities(t) for t in trajectories)

                for trajectory, reward in zip(trajectories, rewards):
                    episode_records.append({
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
                    round_index, group_index, answer, behavior_digest, trajectories
                )

                # A group with no actions carries no gradient, and stacking an
                # empty list of per-action values would raise instead.
                actionless = any(not t.steps for t in trajectories)
                if len(set(rewards)) == 1 or actionless:
                    groups_degenerate += 1
                    group_records.append({
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
                    group_id=f"r{round_index:02d}-g{group_index:02d}",
                    answer=answer,
                    behavior_checkpoint_digest=behavior_digest,
                    reference_checkpoint_digest=INITIAL_LORA_DIGEST,
                    rewards=rewards,
                    behavior_log_probabilities=behavior,
                    reference_log_probabilities=tuple(reference),
                )
                advantages = group_advantages(
                    torch.tensor(rewards), epsilon=HYPER.advantage_epsilon
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
                        hyperparameters=HYPER,
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
                        f"ratio identity failed in round {round_index}, "
                        f"group {group_index}, gap {round_ratio_gap:.2e}"
                    )
                    break

                diagnostic_loss, diagnostics = group_loss(
                    batch, detached, hyperparameters=HYPER
                )
                assert math.isclose(
                    float(diagnostic_loss), -accumulated, rel_tol=1e-4, abs_tol=1e-6
                ), "the accumulated objective and the reported loss disagree"

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

            if stop_reason is not None:
                break

            diversity = diversity_metrics(round_trajectories)
            mixed_fraction = 1.0 - groups_degenerate / GROUPS_PER_ROUND
            if round_index == 0:
                ROUND_ZERO_SURPRISAL = diversity["mean_surprisal"]

            checkpoint_scores = anchor_score_matrix()
            checkpoint_metrics = anchor_metrics(checkpoint_scores)
            checkpoint_summary = anchor_summary(checkpoint_metrics)
            drift = drift_check(checkpoint_summary, BASELINE_ANCHOR_SUMMARY)

            round_record = {
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
                **{f"anchor_{key}": value for key, value in checkpoint_summary.items()},
                "drift_tripped": drift["tripped"],
                "drift_failures": "; ".join(drift["failures"]),
            }
            round_records.append(round_record)
            rounds_completed = round_index + 1

            checkpoint_directory = RESULTS_DIR / f"checkpoint-round-{round_index:02d}"
            # PEFT writes a non-default adapter into a subdirectory named after
            # it, so the round checkpoint lands in .../policy, not the parent.
            model.save_pretrained(
                str(checkpoint_directory), selected_adapters=["policy"]
            )
            assert (checkpoint_directory / "policy").exists()

            print(
                f"round {round_index}: updated {groups_updated}/{GROUPS_PER_ROUND}"
                f"  solve {round_record['solve_rate']:.3f}"
                f"  KL {round_record['mean_kl']:.2e}"
                f"  |delta| {round_record['parameter_delta_norm']:.4f}"
                f"  surprisal {diversity['mean_surprisal']:.3f}"
                f"  ESS {round_record['effective_sample_size']:.1f}"
            )

            if round_index == 3:
                freeze_behavior_digest()
                MIDRUN_EVAL = evaluate_reserved(
                    "round-3", seed_base=EVAL_SEED_BASE + 100000
                )
                print(
                    "mid-run deterministic:",
                    int(MIDRUN_EVAL["deterministic_solved"].sum()),
                    "/ 19  stochastic:",
                    int(MIDRUN_EVAL["stochastic_solved"].sum()),
                    "/",
                    BASELINE_STOCHASTIC_TRIALS,
                )

            if drift["tripped"]:
                stop_reason = f"anchor drift: {'; '.join(drift['failures'])}"
            elif mixed_fraction < MIN_MIXED_FRACTION:
                stop_reason = f"mixed-group fraction collapsed to {mixed_fraction:.3f}"
            elif diversity["repeat_rate"] > MAX_REPEAT_RATE:
                stop_reason = f"repeat rate rose to {diversity['repeat_rate']:.3f}"
            elif (
                diversity["mean_surprisal"]
                < MIN_SURPRISAL_RATIO * ROUND_ZERO_SURPRISAL
            ):
                stop_reason = (
                    f"action surprisal fell to {diversity['mean_surprisal']:.3f}, "
                    f"below {MIN_SURPRISAL_RATIO:.2f} of the round-0 "
                    f"{ROUND_ZERO_SURPRISAL:.3f}"
                )
            elif policy.forward_call_count > TRIE_FORWARD_BUDGET:
                stop_reason = "trie forward budget exhausted"

            if stop_reason is not None:
                print("stopping:", stop_reason)

        training_minutes = (time.perf_counter() - training_start) / 60.0
        ROUNDS_TABLE = pd.DataFrame(round_records, columns=None if round_records else ROUND_COLUMNS)
        GROUPS_TABLE = pd.DataFrame(group_records, columns=None if group_records else GROUP_COLUMNS)
        EPISODES_TABLE = pd.DataFrame(episode_records)
        for column in GROUP_COLUMNS:
            if column not in GROUPS_TABLE.columns:
                GROUPS_TABLE[column] = np.nan
        for column in ROUND_COLUMNS:
            if column not in ROUNDS_TABLE.columns:
                ROUNDS_TABLE[column] = np.nan
        GROUPS_TABLE["updated"] = GROUPS_TABLE["updated"].fillna(False).astype(bool)
        print(f"training took {training_minutes:.1f} min")
        print("rounds completed:", rounds_completed, "of", ROUNDS)
        print("optimizer steps:", optimizer_steps)
        print("stop reason:", stop_reason or "ran to completion")
    """
)

md(
    """
    ## SQ34.10 The movement check, before any gameplay claim

    This run is small. 128 groups, of which roughly half carry signal, and
    about that many optimizer steps. A null is a likely outcome, and it has two
    very different causes that a solve rate cannot tell apart.

    If the weights barely moved, the honest conclusion is that the run was too
    small to test the hypothesis. That is not the same as concluding that
    simulator GRPO does not work, and the difference decides whether a
    replicated Lab 34 is worth building.

    This is preregistered and reported first so the interpretation cannot be
    fitted to whatever the gameplay numbers turn out to be.
    """
)

code(
    """
    if RUN_MODEL:
        FINAL_LORA_DIGEST = lora_digest()
        FINAL_DELTA_NORM = parameter_delta_norm()
        INITIAL_NORM = math.sqrt(
            sum(float((tensor * tensor).sum()) for tensor in INITIAL_LORA_STATE)
        )
        with torch.no_grad():
            final_policy_scores = [
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

        updated_groups = int(GROUPS_TABLE["updated"].sum())
        total_groups_seen = len(GROUPS_TABLE)
        MOVEMENT = {
            "weights_changed": FINAL_LORA_DIGEST != INITIAL_LORA_DIGEST,
            "parameter_delta_norm": FINAL_DELTA_NORM,
            "initial_parameter_norm": INITIAL_NORM,
            "relative_delta": FINAL_DELTA_NORM / INITIAL_NORM,
            "optimizer_steps": optimizer_steps,
            "groups_seen": total_groups_seen,
            "groups_updated": updated_groups,
            "fraction_of_groups_updating": (
                updated_groups / total_groups_seen if total_groups_seen else 0.0
            ),
            "final_round_mean_kl": (
                float(ROUNDS_TABLE["mean_kl"].iloc[-1]) if len(ROUNDS_TABLE) else 0.0
            ),
            "reference_now_differs": final_policy_scores != reference_scores_before,
        }
        for key, value in MOVEMENT.items():
            print(f"{key}: {value}")

        assert MOVEMENT["reference_now_differs"] == MOVEMENT["weights_changed"], (
            "the reference adapter tracked the policy; the KL term was inert"
        )
        if not MOVEMENT["weights_changed"]:
            print(
                "WARNING: no weight change. Any gameplay result below is a "
                "measurement of the incumbent, not of GRPO."
            )
    """
)


md(
    """
    ## SQ34.11 Both decoders, paired

    Three measurements on the same 19 reserved answers.

    The primary one is the PRD's pass criterion, the deterministic
    answer-constrained decoder against the seed 45 baseline of 10/19. The
    secondary one is the policy GRPO optimized, stochastic trie play at the
    frozen temperature, 8 episodes per answer. Greedy trie decoding bridges
    them, together with the greedy-versus-full-string agreement rate on the
    anchor states.

    Nineteen answers is a very small evaluation. The notebook plays the same
    answers before and after, so the comparison is paired and the test is
    McNemar's exact test on the discordant pairs. Treating the two solve rates
    as independent binomials would throw the pairing away and misstate the
    evidence.
    """
)

code(
    """
    if RUN_MODEL:
        freeze_behavior_digest()
        final_start = time.perf_counter()
        FINAL_EVAL = evaluate_reserved("final", seed_base=EVAL_SEED_BASE + 200000)
        final_minutes = (time.perf_counter() - final_start) / 60.0

        comparison = BASELINE_EVAL.merge(
            FINAL_EVAL, on="answer", suffixes=("_baseline", "_final")
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
            (~comparison["greedy_solved_baseline"] & comparison["greedy_solved_final"]).sum()
        )
        greedy_lost = int(
            (comparison["greedy_solved_baseline"] & ~comparison["greedy_solved_final"]).sum()
        )

        FINAL_DETERMINISTIC = int(FINAL_EVAL["deterministic_solved"].sum())
        FINAL_GREEDY = int(FINAL_EVAL["greedy_solved"].sum())
        FINAL_STOCHASTIC = int(FINAL_EVAL["stochastic_solved"].sum())

        print(f"final evaluation took {final_minutes:.1f} min")
        print(
            "deterministic:",
            BASELINE_DETERMINISTIC,
            "->",
            FINAL_DETERMINISTIC,
            "of 19   gained",
            deterministic_gained,
            "lost",
            deterministic_lost,
            "p =",
            round(mcnemar_exact(deterministic_lost, deterministic_gained), 4),
        )
        print(
            "greedy trie:",
            BASELINE_GREEDY,
            "->",
            FINAL_GREEDY,
            "of 19   gained",
            greedy_gained,
            "lost",
            greedy_lost,
            "p =",
            round(mcnemar_exact(greedy_lost, greedy_gained), 4),
        )
        print(
            "stochastic trie:",
            BASELINE_STOCHASTIC,
            "->",
            FINAL_STOCHASTIC,
            "of",
            BASELINE_STOCHASTIC_TRIALS,
            " baseline interval",
            tuple(
                round(value, 4)
                for value in wilson_interval(
                    BASELINE_STOCHASTIC, BASELINE_STOCHASTIC_TRIALS
                )
            ),
            " final interval",
            tuple(
                round(value, 4)
                for value in wilson_interval(
                    FINAL_STOCHASTIC, BASELINE_STOCHASTIC_TRIALS
                )
            ),
        )
        print(
            "preregistered detectability: fewer than",
            DETECTABLE,
            "all-one-way discordant answers cannot reach p<=0.05",
        )
        FINAL_EVAL.to_csv(RESULTS_DIR / "sq34-final-eval.csv", index=False)
    """
)

code(
    """
    if RUN_MODEL:
        FINAL_ANCHOR_SCORES = anchor_score_matrix()
        FINAL_ANCHOR_METRICS = anchor_metrics(FINAL_ANCHOR_SCORES)
        FINAL_ANCHOR_SUMMARY = anchor_summary(FINAL_ANCHOR_METRICS)
        FINAL_DRIFT = drift_check(FINAL_ANCHOR_SUMMARY, BASELINE_ANCHOR_SUMMARY)
        FINAL_AGREEMENT, _ = anchor_agreement(FINAL_ANCHOR_SCORES)

        print("baseline anchors:", BASELINE_ANCHOR_SUMMARY)
        print("final anchors:   ", FINAL_ANCHOR_SUMMARY)
        print("final drift:", FINAL_DRIFT["failures"] or "none")
        print(
            "greedy-trie vs full-string agreement:",
            round(BASELINE_AGREEMENT, 4),
            "->",
            round(FINAL_AGREEMENT, 4),
        )
        if FINAL_AGREEMENT < BASELINE_AGREEMENT:
            print(
                "the two decoders diverged further; a trie-policy gain is less "
                "likely to reach the graded decoder"
            )
    """
)

code(
    """
    if RUN_MODEL:
        def verify_persisted_traces():
            \"\"\"Recompute every group hash and compare it to its sidecar.\"\"\"
            expected = len(GROUPS_TABLE)
            files = sorted(TRACE_DIR.glob("group-*.json"))
            if len(files) != expected:
                print(f"trace files: {len(files)} present, {expected} expected")
                return False
            for path in files:
                sidecar = path.with_suffix(".sha256")
                if not sidecar.exists():
                    print("missing trace hash sidecar:", sidecar.name)
                    return False
                if sha256_text(path.read_text()) != sidecar.read_text().strip():
                    print("trace hash mismatch:", path.name)
                    return False
            return True


        # The PRD names six conditions, not one. A solve-rate gain reached after
        # a drift stop, or with no effective updates, or without persisted
        # traces, is not a pass, and recording a bare comparison would let it
        # look like one.
        PASS_CONDITIONS = {
            "sampler_gate_passed": bool(sq31["sampling_gate_passed"]),
            "ratio_identity_held": stop_reason is None
            or "ratio identity" not in stop_reason,
            "nonzero_updates": MOVEMENT["groups_updated"] > 0
            and MOVEMENT["weights_changed"],
            "no_drift_stop": not FINAL_DRIFT["tripped"],
            "solve_rate_improved": FINAL_DETERMINISTIC > BASELINE_DETERMINISTIC,
            "no_ranking_regression": (
                FINAL_ANCHOR_SUMMARY["median_candidate_teacher_rank"]
                <= max(
                    DRIFT_RANK_FLOOR,
                    DRIFT_RANK_MULTIPLIER
                    * BASELINE_ANCHOR_SUMMARY["median_candidate_teacher_rank"],
                )
            ),
            "traces_persisted": verify_persisted_traces(),
        }

        ROUNDS_TABLE.to_csv(RESULTS_DIR / "sq34-rounds.csv", index=False)
        GROUPS_TABLE.to_csv(RESULTS_DIR / "sq34-groups.csv", index=False)
        EPISODES_TABLE.to_csv(RESULTS_DIR / "sq34-episodes.csv", index=False)
        FINAL_ANCHOR_METRICS.to_csv(RESULTS_DIR / "sq34-final-anchors.csv", index=False)

        RUN = {
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "answer_pool_sha256": ANSWER_POOL_SHA256,
            "anchor_prompt_sha256": ANCHOR_SHA256,
            "entry_checkpoint_sha256": checkpoint_sha256,
            "initial_lora_digest": INITIAL_LORA_DIGEST,
            "final_lora_digest": FINAL_LORA_DIGEST,
            "temperature": TEMPERATURE,
            "rounds_completed": rounds_completed,
            "rounds_planned": ROUNDS,
            "stop_reason": stop_reason,
            "episodes_sampled": len(EPISODES_TABLE),
            "groups_seen": len(GROUPS_TABLE),
            "groups_updated": int(GROUPS_TABLE["updated"].sum()),
            "optimizer_steps": optimizer_steps,
            "training_minutes": training_minutes,
            "movement": MOVEMENT,
            "worst_scoring_equivalence_difference": worst,
            "worst_ratio_identity_gap": float(ROUNDS_TABLE["worst_ratio_gap"].max()),
            "deterministic_baseline": BASELINE_DETERMINISTIC,
            "deterministic_final": FINAL_DETERMINISTIC,
            "deterministic_gained": deterministic_gained,
            "deterministic_lost": deterministic_lost,
            "deterministic_mcnemar_p": mcnemar_exact(
                deterministic_lost, deterministic_gained
            ),
            "greedy_baseline": BASELINE_GREEDY,
            "greedy_final": FINAL_GREEDY,
            "stochastic_baseline": BASELINE_STOCHASTIC,
            "stochastic_final": FINAL_STOCHASTIC,
            "stochastic_trials": BASELINE_STOCHASTIC_TRIALS,
            "smallest_detectable_discordance": DETECTABLE,
            "baseline_anchor_summary": BASELINE_ANCHOR_SUMMARY,
            "final_anchor_summary": FINAL_ANCHOR_SUMMARY,
            "final_drift": FINAL_DRIFT,
            "baseline_decoder_agreement": BASELINE_AGREEMENT,
            "final_decoder_agreement": FINAL_AGREEMENT,
            "trie_forwards": policy.forward_call_count,
            "full_list_sweeps": BUDGET["full_list_sweeps"],
            "trie_forward_budget": TRIE_FORWARD_BUDGET,
            "full_list_sweep_budget": FULL_LIST_SWEEP_BUDGET,
            "pass_conditions": PASS_CONDITIONS,
            "pass_criterion_met": all(PASS_CONDITIONS.values()),
        }
        atomic_json(RUN, RESULTS_DIR / "sq34-run.json")
        print("pass conditions:")
        for condition, value in PASS_CONDITIONS.items():
            print(f"  {condition}: {value}")
        print(json.dumps(RUN, indent=2, sort_keys=True, default=str))
    """
)

md(
    """
    ## SQ34.12 What this run can and cannot settle

    Read the movement check first. If the LoRA barely moved, the gameplay
    numbers below are a measurement of the incumbent under a different random
    seed, and the correct conclusion is that 512 episodes was too small a
    budget to test the hypothesis, not that the hypothesis is false.

    If the weights did move, there are two results, and they are not the same
    result. One is the trained policy, played on the trie both stochastically
    and greedily. The other is the graded policy, played by the deterministic
    answer-constrained decoder the PRD names.

    A gain in the first without a gain in the second is not a failed experiment.
    It says the curriculum trains through one interface and grades through
    another, and the two have come apart. I would rather have that finding than
    a one-game improvement. The decoder agreement rate, before and after, is the
    direct evidence for it.

    Whatever the outcome, nineteen held-out answers cannot resolve modest
    effects. The preregistered detectability figure printed above is the honest
    bound on what this evaluation could ever have shown.

    Lab 20 remains paused. This notebook reads its anchor artifacts and changes
    nothing.
    """
)

for index, cell in enumerate(cells):
    cell["id"] = f"sq34-{index:02d}-{cell['cell_type'][:2]}"

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

target = Path(__file__).parent / "notebooks" / "sq34_simulator_grpo.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {target} with {len(cells)} cells")
