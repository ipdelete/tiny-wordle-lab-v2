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
prompt remains `prompts/wordle-player/v1-baseline.md`.

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
rejected it. The original prompt remains frozen.
