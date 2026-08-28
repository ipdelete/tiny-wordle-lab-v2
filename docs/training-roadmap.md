# GPT-OSS-20B Wordle training roadmap

The project has four phases:

1. benchmark the prompted pre-training checkpoint;
2. fine-tune a supervised policy through LoRA;
3. continue the policy with simulation-based reinforcement learning;
4. analyze the results and run targeted ablations.

Each phase produces evidence, a checkpoint or frozen baseline, an experiment
record, and a written conclusion. A negative result still completes a lab when
it identifies why the intervention failed.

## Phase 1: benchmark before training

### Question

What can prompted GPT-OSS-20B do before any weight updates?

This phase benchmarks the existing pre-training checkpoint. It does not train
a model from scratch.

### Fixed controls

Freeze and record:

- model name and revision;
- prompt content and SHA-256;
- decoding parameters;
- answer batteries and their hashes;
- rollout seeds and repeat count;
- legal vocabulary;
- evaluator revision;
- six action opportunities;
- failure cost of seven turns.

### Measurements

The primary comparison is:

```text
solved games, descending
penalized turns, ascending
```

Behavioral and operational diagnostics include:

- legal-action rate;
- repeated-action rate;
- feedback consistency;
- singleton closure;
- candidate count before and after each action;
- information gain and entropy regret;
- results by candidate-count regime;
- results for repeated-letter answers;
- results by word-frequency band;
- model calls and token usage;
- elapsed time.

### Output

A frozen GPT-OSS-20B baseline with per-game traces and results across repeated
decoding seeds.

## Phase 2: LoRA policy fine-tuning

### Question

What policy behavior can supervised training add without updating the full
model?

This phase uses supervised fine-tuning through LoRA. LoRA defines which
parameters train; supervised learning defines the objective.

### Lessons carried forward from v1

- Use structured state rather than raw history alone.
- Keep legal guesses and possible answers as separate sets.
- Use the open-action entropy policy as the symbolic teacher.
- Include broad, medium, narrow, and singleton states.
- Match training and inference representations.
- Evaluate constrained decoding as well as free generation.
- Judge checkpoints by gameplay, not training loss alone.
- Track clue consistency separately from strategic action quality.

### Training sequence

1. Attach LoRA to GPT-OSS-20B and record the trainable parameter set.
2. Overfit a tiny batch as a canary.
3. Generate a frozen teacher dataset with answer-level splits.
4. Run the fixed LoRA-SFT recipe.
5. Evaluate the final checkpoint on unseen states and complete games.
6. Compare it with the phase-one baseline.

### Promotion gate

The LoRA-SFT adapter advances only if it improves frozen validation gameplay
without an unacceptable loss in legality, clue consistency, or singleton
closure.

### Output

The best accepted LoRA-SFT adapter and a documented comparison with the
prompted base model.

## Phase 3: simulation-based reinforcement learning

### Question

Can environmental reward improve gameplay beyond supervised imitation?

Start from the accepted LoRA-SFT adapter. Connect the repository's
deterministic Wordle environment to OpenEnv and TRL. The local evaluator
remains authoritative for scoring, legality, state transitions, and traces.

### Initial reward

The terminal control reward preserves the benchmark ordering:

```text
outcome_reward = 7 - penalized_turns
```

This gives failures zero reward and gives faster solves more reward than
slower solves.

### Reward studies

Do not start with every plausible reward at once. Compare:

1. terminal outcome and turn efficiency only;
2. terminal reward plus bounded expected information gain;
3. terminal reward plus bounded realized candidate reduction;
4. the Hugging Face Wordle reward recipe;
5. the best composite with each component removed in turn.

Any shaping reward must remain too small to make a failed game outrank a
solved game or a slower solve outrank a faster solve.

### Training safeguards

Track:

- held-out solves and penalized turns;
- candidate probability mass;
- singleton rank and closure;
- legality and repetition;
- clue consistency;
- entropy regret;
- KL from the starting SFT policy;
- LoRA parameter movement;
- stochastic and deterministic gameplay.

If GRPO clipping is under study, run more than one optimization pass over a
rollout group. One update against a fresh behavior policy leaves the
probability ratio at one and does not exercise clipping.

### Promotion gate

The RL adapter advances only if it beats the SFT checkpoint on frozen
validation gameplay while remaining inside the predeclared policy-preservation
limits.

### Output

An accepted RL adapter or a bounded negative result that identifies the failed
reward or optimization mechanism.

## Phase 4: analysis and ablations

### Question

What caused the observed changes, and which parts transfer beyond one run?

Analysis runs throughout the project. This final phase consolidates the record
and performs the controlled comparisons needed for stronger claims.

### Comparisons

- prompted base model versus LoRA-SFT versus RL;
- raw history versus structured state;
- free generation versus legal-word constrained decoding;
- terminal reward versus shaped rewards;
- individual reward components;
- decoding seeds and training seeds;
- broad, medium, narrow, and singleton states;
- common versus obscure answers;
- repeated-letter versus unique-letter answers;
- solve improvement versus token and compute cost.

### Final holdout

Evaluate the selected base, SFT, and RL policies once on an untouched holdout.
Do not use holdout results to revise the selected recipes.

### Output

A final report containing:

- the experiment ledger;
- accepted and rejected interventions;
- per-phase results;
- ablation tables;
- subgroup analyses;
- resource costs;
- limits on the conclusions;
- the exact commands and artifact hashes needed to reproduce each headline
  result.

## Research rules

Across all four phases:

1. Write the hypothesis and expected direction before each run.
2. Freeze the promotion rule before seeing the result.
3. Change one meaningful variable at a time.
4. Run a cheap canary before an expensive experiment.
5. Keep negative results.
6. Record model, prompt, data, code, and seed identities.
7. Treat disagreements between metrics as evidence to investigate.
8. Restart after a recipe change rather than changing a restartable run
   mid-flight.
