"""Run ONLY the extended_unseen eval block on chosen checkpoints of finished runs,
without disturbing their existing eval_summary.json: creates <run>-ext/ with a
config holding just that block (+ eval_at_steps) and a copied train_result.json,
then launches scripts/train_eval.py --eval-only on it in tmux session sdf_ext.

  .venv/bin/python sdf/run_extended_eval.py sdf-gptoss20b-demo-lr1.5e-4:0,320,359 sdf-gptoss20b-desc-lr1.5e-4:0,351 ...
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cotcontrol.eval.prompts import HELDOUT_MODES  # noqa: E402
from sdf.make_configs import UNSEEN_MODES, eval_block  # noqa: E402

# optional first arg --block=heldout runs the 3 held-out modes instead (dir suffix -heldout)
BLOCK = "extended_unseen"
args = sys.argv[1:]
if args and args[0].startswith("--block="):
    BLOCK = args.pop(0).split("=", 1)[1]
SUFFIX = "-ext" if BLOCK == "extended_unseen" else f"-{BLOCK}"

subprocess.run("tmux has-session -t sdf_ext 2>/dev/null || tmux new-session -d -s sdf_ext -n idle 'sleep 1'", shell=True)
for spec in args:
    run, steps = spec.split(":")
    src = ROOT / "sdf/runs/train" / run
    dst = ROOT / "sdf/runs/train" / f"{run}{SUFFIX}"
    dst.mkdir(exist_ok=True)
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    cfg["run_dir"] = f"sdf/runs/train/{run}{SUFFIX}"
    # inherit serving knobs (tensor_parallel_size etc.) from the source run's first eval block, e.g. tp=2 for 120b
    first = next(iter(cfg["eval"].values())) if isinstance(cfg.get("eval"), dict) else {}
    inherit = {k: first[k] for k in ("tensor_parallel_size", "parallel_checkpoints", "max_model_len") if k in first}
    steps_l = [int(s) for s in steps.split(",")]
    if BLOCK == "extended_unseen":
        blk = eval_block(allowed_modes=UNSEEN_MODES, mode="all", max_samples=25, subsample_seed=0,
                         eval_at_steps=steps_l, **inherit)
    else:
        blk = eval_block(allowed_modes=list(HELDOUT_MODES), eval_at_steps=steps_l, **inherit)
    cfg["eval"] = {BLOCK: blk}
    (dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy(src / "train_result.json", dst / "train_result.json")
    cmd = f"cd {ROOT} && .venv/bin/python scripts/train_eval.py --eval-only sdf/runs/train/{run}{SUFFIX} > sdf/runs/train/{run}{SUFFIX}/launch.log 2>&1"
    subprocess.run(["tmux", "new-window", "-t", "sdf_ext", "-n", run.replace("sdf-gptoss20b-", "")[:20], cmd])
    print("launched", dst)
