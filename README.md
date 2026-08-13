# Tiny Wordle Lab v2

A hands-on course for learning modern LLM training by turning **Qwen3-0.6B** into a Wordle specialist.

## Fixed platform

- Model: `Qwen/Qwen3-0.6B`
- Runtime: PyTorch
- Accelerator: Apple Metal / MPS
- Interface: Jupyter Lab
- Primary machine: Apple Silicon
- Training philosophy: keep the machinery visible before introducing higher-level abstractions

The point is **not** to find the smallest model that can solve Wordle. The point is to learn how model behavior changes through data, optimization, fine-tuning, distillation, and reinforcement learning.

## Start here

```bash
cd tiny-wordle-lab-v2
brew install uv
uv sync --extra dev
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run jupyter lab
```

Then open:

1. `notebooks/00_setup_and_mps.ipynb`
2. `notebooks/01_model_anatomy.ipynb`
3. `notebooks/02_overfit_one_batch.ipynb`

Do not skip the observations and predictions in the notebooks. The code is only half of the course.

## Ground rules

1. Qwen3-0.6B stays fixed unless an experiment explicitly requires a control.
2. Establish a baseline before changing anything.
3. Change one meaningful variable at a time.
4. Record results, not impressions.
5. Understand a primitive before replacing it with a framework abstraction.
6. Every training lesson ends with an evaluation of behavior, not merely loss.
