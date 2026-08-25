# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Repo structure

The repo is organized into two kinds of top-level folders:

### Infrastructure folders (shared plumbing)

- `cotcontrol/` — the core package: `eval/` (dataset, grading, judge, eval loop), `inference/` (OpenRouter + Modal vLLM backends), `training/` (Modal LoRA training infra).
- Modal infra lives under `cotcontrol/` (e.g. `cotcontrol/training/modal_app.py`, `cotcontrol/inference/modal_vllm.py`).
- `datasets/`, `prompts/`, `training_data/`, `configs/` — shared data and config.
- `scripts/` — entrypoints for shared infra (baselines, evals, data building).
- `tests/` — tests for the shared package.

Infrastructure folders contain no experiment-specific state. Code here should stay general enough that any experiment folder can import it.

### Experiment folders (self-contained)

Each experiment type gets its own top-level folder holding **both its code and its outputs** (run logs, rollouts, results, intermediate progress files):

- `gepa/` — GEPA prompt-optimization experiments (runs go in `gepa/runs/`).
- `fewshot/` — few-shot scaling experiments.
- Future: `sft/`, `rl/`, `dpo/` — same pattern.

Rules for experiment folders:

- All artifacts from a run (logs, per-datapoint inputs/outputs, judge outputs, rollouts, advantages, checkpoint pointers) are saved inside that experiment's folder, typically under a `runs/` subdirectory with one directory per run.
- Experiment code imports shared infra from `cotcontrol/`; it never duplicates it. If an experiment needs a capability the infra lacks, add it to `cotcontrol/` in a general form.
- When adding a new experiment type, create a new top-level folder following this pattern — do not put its outputs in the top-level `results/`.

### Legacy

- `results/` — legacy flat output dir from before this convention; don't add new outputs here.
- `old/` — retired tinker-based training code, kept for reference.
- `notebooks/` — analysis notebooks.
