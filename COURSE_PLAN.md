# Course Plan — Train Qwen3-0.6B Through Wordle

## Research question

How much can we improve a fixed general-purpose language model's Wordle performance, and what does each training technique actually change?

## Phase 1 — See the machine

### Lab 00 — Environment and MPS
Verify the Apple Silicon stack, PyTorch MPS, Jupyter kernel, model download, parameter counts, dtypes, and a basic forward pass.

### Lab 01 — Model anatomy
Inspect tokenization, chat templates, input IDs, logits, probabilities, generation, and Qwen3 thinking/non-thinking behavior. Establish pre-training observations.

### Lab 02 — Overfit one batch
Create a tiny supervised dataset and intentionally make the model memorize it. Write the optimizer loop directly. Inspect loss, gradients, predictions, and changed behavior.

## Phase 2 — Build the measuring stick

### Lab 03 — Deterministic Wordle environment
Implement Wordle feedback correctly, including duplicate-letter rules. Add tests.

### Lab 04 — Baseline evaluation
Make Qwen play held-out games. Measure win rate, invalid guesses, constraint violations, guesses per win, and failure modes.

## Phase 3 — Build training signal

### Lab 05 — Symbolic expert
Implement candidate filtering and a strong guess-selection strategy. Separate game-state correctness from strategic choice.

### Lab 06 — Synthetic dataset
Generate state → action examples. Learn splits, leakage prevention, formatting, curriculum design, and dataset inspection.

## Phase 4 — Supervised post-training

### Lab 07 — Full SFT
Fine-tune Qwen3-0.6B on Wordle trajectories. Learn labels, masking, cross-entropy, batching, validation, checkpointing, and learning-rate effects.

### Lab 08 — LoRA / PEFT
Repeat the experiment with parameter-efficient fine-tuning. Compare trainable parameters, speed, memory, and behavioral outcome.

### Lab 09 — Distillation
Use a stronger teacher and/or symbolic expert to create higher-quality traces. Test whether reasoning traces or direct actions transfer better.

## Phase 5 — Reinforcement learning

### Lab 10 — Reward design
Turn Wordle into an environment with outcome rewards. Explore sparse vs shaped rewards and reward-hacking failure modes.

### Lab 11 — RL post-training
Train from rollouts using an appropriate policy-optimization method. Inspect policy drift, reward curves, KL behavior, and actual game performance.

## Phase 6 — Research

### Lab 12 — Ablations and final report
Run controlled experiments on representation, training-data volume, thinking mode, reasoning traces, and training method. Produce the final capability table and conclusions.

## Metrics carried throughout

- solve rate within six guesses
- average guesses among wins
- legal-word rate
- hard constraint compliance
- duplicate-letter correctness
- candidate-set correctness where applicable
- loss / validation loss during supervised training
- wall-clock training time
- peak MPS memory where measurable
- tokens/sec or examples/sec
