# Wordle prompt optimization

This directory is a standalone calibration exercise run before model training.
It uses SkillOpt to revise the model-facing Wordle prompt while keeping model
weights, the game engine, and the evaluator fixed.

The three answer splits have separate roles:

- `train/` supplies trajectories for reflection and prompt edits.
- `val/` gates candidate prompts.
- `test/` is the final holdout and must not guide edits.

Every answer is evaluated twice with fixed request seeds. The SkillOpt `soft`
score encodes the harness ranking exactly: one additional solved game outweighs
every possible penalized-turn improvement, and penalized turns break solve-count
ties. The deterministic harness remains the grader; no LLM judges gameplay.

SkillOpt is pinned to a commit after v0.2.0 because its generic
`openai_compatible` backend has not yet shipped in a release. Run the exercise
against the local LiteLLM route with:

```bash
uv sync --group prompt-optimization
uv run --group prompt-optimization python -m prompt_optimization.run
```

Raw optimization outputs are ignored. The completed run is summarized in
`result.json`. SkillOpt proposed always opening with `slate`, but the candidate
tied the baseline exactly on validation and the strict gate rejected it. The
rejected candidate is retained under `rejected/`; the selected and frozen
prompt for that run remained `prompts/wordle-player/v1-baseline.md`.

The controlled follow-up changes only the optimizer to Sonnet 5 through
SkillOpt's native Copilot CLI backend:

```bash
uv run --group prompt-optimization python -m prompt_optimization.run \
  --config prompt_optimization/skillopt-sonnet-optimizer.yaml
```

That run is summarized in `result-sonnet-optimizer.json`. Sonnet proposed a
more detailed strategy covering legal guesses, clue consistency, repeated
letters, and information-rich openings. It nevertheless reduced validation
performance from 7/16 solves and 96 penalized turns to 6/16 and 97, so the gate
rejected it. That run retained the original prompt.

The next controlled run changes only the optimizer to GPT-5.6 Sol:

```bash
uv run --group prompt-optimization python -m prompt_optimization.run \
  --config prompt_optimization/skillopt-sol-5.6-optimizer.yaml
```

That run is summarized in `result-sol-5.6-optimizer.json`. Sol prioritized
cumulative clue constraints and legal, unused guesses. The candidate reduced
validation performance from 7/16 solves and 96 penalized turns to 5/16 and
106, so it was rejected.

## Iterative Sol run

`skillopt-sol-5.6-iterative.yaml` runs three epochs with three four-answer
steps per epoch. Rejected edits inform later steps in the same epoch, accepted
prompts become the next step's baseline, and Meta-Skill written after epoch two
is consumed in epoch three. Every validation gate uses 16 answers with three
repeats. Training-time test evaluation is disabled.

The run uses a new split under `splits-iterative/`. Its 32-answer test set must
not be evaluated until optimization is complete:

```bash
uv run --group prompt-optimization python -m prompt_optimization.run \
  --config prompt_optimization/skillopt-sol-5.6-iterative.yaml
```

The run accepted step 2 and rejected the other eight candidates. Validation
improved from 20/48 solves and 283 penalized turns to 22/48 and 269. The
accepted prompt is frozen as
`prompts/wordle-player/v2-skillopt-sol-iterative.md` and summarized in
`result-sol-5.6-iterative.json`. The final holdout comparison was stopped at
the user's request after only its baseline half completed, so no holdout result
is claimed.
