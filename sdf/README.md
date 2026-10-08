# sdf/ — Synthetic Document Finetuning for CoT controllability

Hypothesis: midtraining a reasoning model on synthetic pretraining-style documents that
*describe* it as having precise control over its chain of thought (naming the 9 constraint
modes) raises measured CoT controllability — without any demonstrations in chat format.
Method follows Anthropic's SDF pipeline ("Believe It or Not", arXiv 2510.17941; code
`safety-research/believe-it-or-not`, cloned at `/root/believe-it-or-not`).

## Pipeline

| step | script | output |
|---|---|---|
| universe context (hand-written) | `universe_contexts/gptoss20b_cot_control.json` | 15 key facts; mode-specific facts carry `modes` |
| compliant excerpt pool (demo arm) | `build_excerpts.py` | `data/excerpts.jsonl` — 80 authentic gpt-oss-20b traces per mode from `sft/datasets/gptoss20b_x480.jsonl` |
| pretraining mix | `download_c4.py` | `data/c4_50k.jsonl` (allenai/c4 en, seed 42) |
| doc generation (types → ideas → docs → revise) | `gen_docs.py`, `launch_gen.sh` | `runs/docs/<arm>/` (every prompt + raw completion saved; resumable) |
| dataset assembly (+1:1 C4, optional `<DOCTAG>` masked prefix) | `build_dataset.py` | `datasets/<name>.jsonl` + manifest |
| training configs + launcher | `make_configs.py` → `configs/*.yaml`, `launch_train.sh` | `runs/train/<run>/` via `scripts/train_eval.py` |

Prompts in `prompts/` are ported from believe-it-or-not (`gen_doc.txt`, brainstorm prompts,
global context). `revise_direct.md` is our rewrite of their `augment/direct.md`: length-capped
(the original inflated docs 3-10x), neutral wording, and a hard rule to keep quoted
analysis-channel excerpts verbatim (revision otherwise fabricates new quotes; `finalize`
falls back to the unrevised doc when an excerpt is lost).

## Arms (Phase 1, gpt-oss-20b, 40k docs each, shared doc specs)

- `desc` — description-only documents.
- `demo` — same specs, but each doc prompt gets 1-2 authentic compliant analysis-channel
  excerpts for the fact's modes and may quote them verbatim.
- `c4only` — matched-compute control: the C4 half only.

Spec model `anthropic/claude-sonnet-5` (types, ideas); doc + revision model
`anthropic/claude-haiku-4.5`; all via OpenRouter, concurrency 150/arm.

## Training / eval

Raw-text rows (`{"text", "prefix"?}`) are supported by `cotcontrol/training/rendering.render_text`
(all tokens supervised, `<|endoftext|>` doc separator) and `TrainConfig.pack_sequences`
(concat + chunk to `max_seq_length`). LoRA r64/α128 all-linear incl. gpt-oss experts,
1 epoch, lr ∈ {2e-5, 5e-5, 1.5e-4}, batch 8 × 16k packed tokens.

Eval blocks per checkpoint (val, 200 q): `cotcontrol` (default harmony identity = ChatGPT),
`identity` (harmony `model_identity` = gpt-oss-20b), `heldout` (3 held-out modes), `baseline`
(unconstrained accuracy guard).

## Follow-up runs (v2-v6) and results

`make_configs.py` also defines the anchored (v2/v3), math-anchored (v4), 3-epoch (v4), anchor-dose (v5) and
two-stage repair (v5), compliant-demo mix / DOCTAG / Qwen3-8B (v6; `build_compliant_mix.py`, `qwenify_docs.py`) runs; the repair runs use `train.init_lora_path` (SFT warm start from a saved adapter,
checkpoint-0 == donor). `analyze.py` builds the per-run tables; `run_extended_eval.py` adds the
`extended_unseen` block to finished runs; `belief_eval.py` measures self-knowledge; `watch_runs.sh` tails
several run logs with a per-run prefix. Findings 1-16 and the status are in `runs/train/results.md`.

## Gotchas

- `/root/believe-it-or-not/.../sdf_blogpost_shortened.md` contains a fabricated DOCTAG
  efficacy claim (it is the key_facts of a deliberately false universe context). The doctag
  *code* is real; the number is not.
- Do not import believe-it-or-not code: 76 hardcoded `/workspace` paths, blocking `input()`,
  uninitialized `safety-tooling` submodule.
