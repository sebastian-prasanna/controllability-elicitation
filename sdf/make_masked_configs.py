"""Masked-subset ("elicitation") replicas of the generic-pretraining SDF arms: gpt-oss-120b, C4-only and
placebo docs, paper-like schedule (unpacked 1k docs, 16/step, 1 epoch), but training only a random k of the
rank-1 attention-LoRA parameters (same mechanism as sft/x320 and rl: `elicitation: {train_params, mask_seed}`),
with an lr bracket per k (lr ~ k^-1/2, anchored on the x320 k=1k point 9.5e-2).

Question: is the C4 / placebo lift an elicitation effect that 1k-10k parameters can reproduce?

    .venv/bin/python sdf/make_masked_configs.py      # writes sdf/configs/sdf-atlas5p-gptoss120b-<arm>-k<k>-lr<lr>.yaml + sdf/launch_masked.sh
    bash sdf/launch_masked.sh [filter]               # tmux session sdf_masked, one window per run; run dirs sdf/runs/train_masked/<name>

Eval per checkpoint (val, 200 q, mode random, T=1, 16k, ATLAS-5 identity): cotcontrol_id + heldout_id -> lr selection on val.
Test numbers come from sft/test_eval.py on this folder (same T=1 identity protocol as sdf/runs/train, tag g16k_t1_id).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cotcontrol.eval.prompts import HELDOUT_MODES  # noqa: E402
from sdf.make_configs import ATLAS5_IDENTITY, V9_MODELS, make, paperlike, v9_dataset  # noqa: E402

MODEL = "gptoss120b"
ARMS = ["c4only80k", "placebo"]
K_LRS = {1_000: ["3e-2", "1e-1", "3e-1"], 10_000: ["1e-2", "3e-2", "1e-1"]}
MASK_SEED = 0
SAVE_AT = [1000, 2500]          # + final step (~5000) always saved
RUNS_DIR = "sdf/runs/train_masked"


def kname(k: int) -> str:
    return f"k{k // 1000}k"


def masked_config(arm: str, k: int, lr: str) -> tuple[str, dict]:
    base, _fam, gpu, eval_over = V9_MODELS[MODEL]
    name = f"sdf-atlas5p-{MODEL}-{arm}-{kname(k)}-lr{lr}"
    cfg = paperlike(make(name, v9_dataset(MODEL, arm), lr, epochs=1, base_model=base, gpu=gpu, eval_over=eval_over), lr)
    cfg["run_dir"] = f"{RUNS_DIR}/{name}"
    cfg["train"]["save_at_steps"] = SAVE_AT
    cfg["train"]["max_grad_norm"] = 1.0
    cfg["lora"] = {"lora_rank": 1, "lora_alpha": 1, "lora_dropout": 0.0,          # x320 / RL masked recipe
                   "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "lora_target_parameters": []}
    cfg["elicitation"] = {"train_params": k, "mask_seed": MASK_SEED}
    ident = {"dataset": "all", "mode": "random", "split": "val", "seed": 0, "system_prompt": "",
             "temperature": 1.0, "max_tokens": 16000, "chat_template_kwargs": {"model_identity": ATLAS5_IDENTITY},
             "tensor_parallel_size": 2, "parallel_checkpoints": 2, "max_model_len": 24576}
    cfg["eval"] = {"cotcontrol_id": dict(ident), "heldout_id": {**ident, "allowed_modes": list(HELDOUT_MODES)}}
    return name, cfg


def main() -> None:
    cfg_dir = ROOT / "sdf/configs"
    names = []
    for arm in ARMS:
        for k, lrs in K_LRS.items():
            for lr in lrs:
                name, cfg = masked_config(arm, k, lr)
                (cfg_dir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
                names.append(name)
    lines = ["#!/usr/bin/env bash",
             "# Launch masked SDF runs (one tmux window each, session sdf_masked). bash sdf/launch_masked.sh [filter]",
             "set -euo pipefail", 'cd "$(dirname "$0")/.."', "FILTER=${1:-}",
             "tmux has-session -t sdf_masked 2>/dev/null || tmux new-session -d -s sdf_masked -n hub 'bash'"]
    for name in names:
        win = name.replace(f"sdf-atlas5p-{MODEL}-", "")
        lines += [f'if [[ -z "$FILTER" || "{name}" == *"$FILTER"* ]]; then',
                  f'  if [[ -e {RUNS_DIR}/{name}/launch.log ]]; then echo "skip {name} (already launched)"; else',
                  f"  mkdir -p {RUNS_DIR}/{name}",
                  f'  tmux new-window -t sdf_masked -n {win} ".venv/bin/python scripts/train_eval.py sdf/configs/{name}.yaml > {RUNS_DIR}/{name}/launch.log 2>&1"',
                  f"  echo launched {name}", "  fi", "fi"]
    (ROOT / "sdf/launch_masked.sh").write_text("\n".join(lines) + "\n")
    print(f"{len(names)} configs -> sdf/configs/, launcher sdf/launch_masked.sh")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
