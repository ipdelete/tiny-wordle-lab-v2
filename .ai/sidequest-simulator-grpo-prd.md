# Side quest: simulator GRPO from Lab 18d

## Status

This side quest pulls the environment work from Lab 31 and the simulator-GRPO
experiment from Lab 34 ahead of the remaining supervised curriculum.

It is an exploratory baseline. It does not satisfy the canonical Part IV entry
criteria, replace the paused Lab 20 experiment, settle the action-space question
from Part III, or select the final policy for later actor-critic work.

The side quest earns one claim:

> Can sparse whole-game reward improve the best observed Lab 18d policy without
> destroying the full-list ranking that made it playable?

## Entry checkpoint

Use the Lab 18d seed 45 adapter:

```text
checkpoints/qwen3-0.6b-wordle-lora-dataset-b-structured-seed45
```

All three Lab 18d seeds solved 10 of 19 reserved answer-constrained games. Seed
45 is the provisional choice because it had the strongest secondary Turn 2
measurements:

| Measurement | Seed 45 |
| --- | ---: |
| Held-out constrained solves | 10/19 |
| Mean turns on wins | 4.2 |
| Turn 2 open-entropy regret | 1.051 bits |
| Turn 2 chosen-candidate rate | 36.8% |
| Turn 2 realized candidate reduction | 3.318 bits |

This is "best observed," not a demonstrated seed effect. A positive side-quest
result must be replicated from the other retained seeds before it becomes a
curriculum conclusion.

Freeze and record:

* adapter and tokenizer hashes;
* `derived_state_v1` prompt rendering;
* `RAISE` as the fixed opening;
* the 2,315 original answer words as the action vocabulary;
* answer-list and pattern-matrix hashes;
* the six-turn game limit;
* the 19 reserved evaluation answers;
* the deterministic answer-constrained evaluation decoder.

**Complete when:** one manifest reproduces the Lab 18d checkpoint identity and
its 10/19 baseline from persisted artifacts.

## Boundaries

The side quest includes:

1. an explicit Wordle environment;
2. a stochastic policy over the current 2,315-answer vocabulary;
3. a sampling-policy gate;
4. one bounded simulator-GRPO run;
5. deterministic held-out evaluation and full-list drift analysis.

It excludes:

* Lab 20 correction training;
* the Part III legal-guess action-space expansion;
* iterative imitation;
* reward shaping;
* a learned critic;
* actor-critic updates;
* asynchronous collection;
* SAO-style training;
* full-model updates.

Keep those exclusions visible in every result. The side quest may justify
continuing simulator RL. It cannot justify skipping the controlled comparisons
that answer different questions.

## SQ31: define the environment

### Interface

Add one environment adapter around the existing `tiny_wordle.game` and
`tiny_wordle.expert` mechanics:

```python
observation = env.reset(answer_id, policy_view)
observation, reward, done, info = env.step(action)
```

The environment owns the hidden answer, transition rules, terminal truth, and
answer-dependent diagnostics. The policy receives only `observation`.

### Policy observation

The first `policy_view` reproduces the state available to the Lab 18d model:

* prior guesses and feedback;
* derived green, count, exclusion, and absent-letter constraints;
* previous guesses;
* candidate count;
* remaining turns.

The policy view excludes:

* the hidden answer or answer identity;
* candidate words;
* teacher actions;
* entropy, regret, or value targets;
* terminal information before termination;
* diagnostics from `info`.

Render the observation through the existing `derived_state_v1` prompt builder.
Do not create a second prompt dialect for RL.

### Action and transition rules

The action is one word from the 2,315-answer vocabulary.

| Action | Environment behavior |
| --- | --- |
| Correct word | Append feedback, return reward 1, terminate as solved |
| Valid new wrong word | Append feedback, return reward 0, continue if turns remain |
| Repeated vocabulary word | Append the repeated guess and feedback, consume a turn |
| Malformed or out-of-vocabulary output | Mark a contract violation and terminate the episode |
| Sixth unsuccessful turn | Return reward 0 and terminate as exhausted |

Trie-constrained training should make malformed and out-of-vocabulary actions
unreachable. The environment still defines them so a broken sampler fails
loudly rather than changing the rules.

### Reward

Use sparse terminal outcome only:

```text
solved within six turns: 1
otherwise:                  0
```

Intermediate steps return 0. Candidate reduction, entropy, teacher agreement,
candidate membership, repetition, and turn count are diagnostics. They do not
enter the reward.

### Trace

Persist one immutable record per action:

```text
episode_id
group_id
policy_checkpoint_sha256
reference_checkpoint_sha256
policy_view_version
action_vocabulary_sha256
tokenizer_sha256
temperature
sampling_seed
answer_split
protected_answer_id
turn
observation
history_before
action
action_log_probability
token_ids
per_token_log_probabilities
feedback
reward
done
terminal_reason
candidate_count_before
candidate_count_after
teacher_diagnostics
```

Keep protected answer IDs and teacher diagnostics outside the next policy
observation.

### Environment tests

Tests must cover:

* reset state after fixed `RAISE`;
* correct and incorrect feedback;
* duplicate letters;
* repeated guesses;
* malformed and out-of-vocabulary actions;
* win and six-turn exhaustion;
* trace replay;
* observation leakage;
* agreement with the existing benchmark on valid, non-repeated trajectories.

**Complete when:** every saved trace replays exactly, the shared valid-action
domain matches the benchmark, and tests document every intentional transition
rule.

## Sampling-policy gate

The deterministic Lab 18d argmax remains the evaluation decoder. Training uses
a separate stochastic policy.

### Trie-constrained sampling

Build a token trie from:

```text
tokenize(answer word + EOS)
```

At each generation step:

1. read the valid next-token set from the trie prefix;
2. mask every other vocabulary logit;
3. divide allowed logits by the declared temperature;
4. sample from the normalized masked distribution;
5. record the chosen token log probability;
6. stop only at a trie terminal followed by the declared EOS.

The action log probability is the sum of the recorded token log probabilities.
Repeated words remain in the action vocabulary. The environment, not the
sampler, applies their consequence.

### Probability tests

The gate must show:

* every sampled sequence maps to exactly one allowed word;
* allowed-token probabilities sum to one at every prefix;
* recomputing a saved action under the frozen checkpoint reproduces every
  token log probability;
* checkpoint plus sampling seed reproduces the sampled action;
* empirical action frequencies agree with declared probabilities on a small
  fixed state suite;
* no hidden environment value enters sampler input.

### Reward-diversity pilot

Use development answers only. The 19 reserved answers remain untouched.

Evaluate a frozen temperature grid declared in the notebook before sampling.
For each temperature, sample equal-sized groups of complete games from the same
answers and record:

* stochastic solve rate;
* fraction of groups containing both wins and losses;
* fraction of all-zero groups;
* fraction of all-one groups;
* action entropy by turn;
* repeat rate;
* episode length and model calls.

Choose the temperature with the largest mixed-outcome group fraction. Break
ties in favor of the lower temperature. Freeze it before any optimizer update.

The training run is blocked when every tested temperature produces too few
mixed-outcome groups to support a group-relative update. Record that null as a
sampling-policy result rather than adding shaped reward.

**Complete when:** action probabilities pass replay and frequency tests, one
temperature is frozen from development evidence, and the pilot shows whether
sparse reward supplies nonzero group-relative advantages.

## SQ34: simulator GRPO

### Group construction

For each group, hold the hidden answer and initial observation fixed. Sample
four complete episodes from the frozen behavior checkpoint:

```text
same hidden answer
-> rollout A -> return A
-> rollout B -> return B
-> rollout C -> return C
-> rollout D -> return D
```

The policy sees only feedback produced after its own actions. It never receives
the hidden answer.

Use an answer-level training split disjoint from the 19 reserved evaluation
answers. Persist the exact answer order, group seeds, behavior checkpoint, and
trace hashes before updating the model.

### Advantage

For returns `R_1 ... R_4`, compute the group-relative advantage:

```text
A_i = (R_i - mean(R)) / (std(R) + epsilon)
```

An all-zero or all-one group has no learning signal. Record it and skip its
optimizer update. Do not manufacture variance with a local proxy reward.

Assign the trajectory advantage to each policy action in that episode. The
update uses the stored behavior-policy log probability and the newly evaluated
current-policy log probability for the same trie-constrained action.

### Policy update

Continue the seed 45 LoRA adapter. Keep base weights frozen.

The notebook preregistration must freeze:

* group size;
* development answer pool and order;
* number of sampled episodes;
* learning rate and schedule;
* PPO or GRPO ratio-clipping bounds;
* advantage epsilon;
* reference-policy KL coefficient;
* gradient accumulation;
* checkpoint cadence;
* optimizer seed;
* total model-call budget.

Use the frozen seed 45 policy as the reference. Compute reference log
probabilities under the same trie mask. Keep the KL term separate from game
reward in reports.

### Bounded run

Run one optimization seed first. This is a side-quest feasibility test, not a
replicated model comparison.

Cap the first full run at:

* group size 4;
* 128 sampled groups;
* 512 complete episodes;
* six turns per episode;
* LoRA-only updates;
* early full-list checkpoints.

Count every policy forward used for sampling, log-probability recomputation,
reference evaluation, anchors, and held-out play. Do not compare methods using
episode count when their model-call counts differ.

### Drift protection

Reuse the Lab 20 full-list anchor measurements:

* candidate mass;
* best-candidate rank;
* candidate-teacher rank;
* singleton candidate rank;
* winner identity;
* unique winners;
* largest winner share.

Score the untouched seed 45 incumbent before training and every saved policy
checkpoint afterward. Freeze stop rules before the first optimizer update.
When a rule trips, stop training at that checkpoint and retain the traces.

The fixed anchor suite is a safety signal. It does not select the checkpoint
with the best held-out gameplay.

### Memory gate

Every new GPU loop must pass the repository safety procedure before full scale:

1. cap MPS at 128 GiB;
2. use compact response-position logits;
3. avoid full-vocabulary `log_softmax`;
4. clear the MPS cache per trajectory step;
5. run 40 fixed-shape iterations for sampling, current-policy scoring,
   reference scoring, and training;
6. require a late memory plateau;
7. run the notebook through `scripts/memguard.py --min-free 64`.

The fixed-shape soak precedes the 512-episode run in the same notebook.

### Evaluation

Evaluate these frozen checkpoints with the deterministic answer-constrained
decoder:

* the untouched Lab 18d seed 45 incumbent;
* each preregistered GRPO checkpoint;
* the final or drift-stopped GRPO checkpoint.

Use the same 19 reserved answers and Lab 18d game rules. Report:

* solve rate;
* turns on wins;
* Turn 2 open-entropy regret;
* realized candidate reduction;
* candidate mass and ranks;
* singleton closure;
* repeats;
* full-list winner concentration;
* policy KL to the incumbent;
* groups sampled and groups updated;
* environment steps, model calls, wall time, and peak memory.

### Decision

The side quest passes only when:

* the sampler gate passed;
* training produced nonzero group-relative updates;
* no drift stop fired;
* held-out solve rate exceeded the seed 45 baseline of 10/19;
* singleton closure and full-list ranking did not regress materially;
* the complete result reproduces from persisted manifests and traces.

A one-game improvement is evidence to replicate, not proof that simulator GRPO
works. Replication from seeds 42 and 47 belongs in the canonical Lab 34.

The side quest stops when:

* the sampling gate cannot produce reward diversity;
* a memory soak fails;
* the full-list drift guard fires;
* the model-call budget is exhausted;
* the environment or probability replay tests fail.

**Complete when:** the repository contains the tested environment, stochastic
policy gate, executed simulator-GRPO notebook, trace artifacts, held-out
evaluation, and an analysis that states whether a replicated Lab 34 is worth
running.

## Deliverables

```text
src/tiny_wordle/environment.py
tests/test_environment.py
notebooks/31_wordle_environment.ipynb
notebooks/34_simulator_grpo.ipynb
.ai/docs/31_wordle_environment-cell-outputs.md
.ai/docs/31_wordle_environment-analysis.md
.ai/docs/34_simulator_grpo-cell-outputs.md
.ai/docs/34_simulator_grpo-analysis.md
results/sidequest31/
results/sidequest34/
```

Notebook builders remain the source of generated notebook structure, following
the existing `build_18d.py`, `build_19.py`, and `build_20.py` pattern.

## Relationship to the main curriculum

This side quest runs beside the main sequence.

* Lab 20 remains paused and restartable.
* Labs 22 through 30 still decide the final supervised recipe and action space.
* Canonical Lab 31 may reuse the environment when its contract matches the
  eventual Part IV entry policy.
* Canonical Lab 34 must rerun from the selected entry policy and replicate
  training seeds before making a course-level claim.

The side quest answers whether simulator GRPO deserves that later investment.
