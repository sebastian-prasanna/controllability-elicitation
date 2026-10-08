"""SFT sweep launcher: cartesian product of config overrides -> tmux windows.

Template — each sweep lives in its own folder under sft/runs/ holding
everything that defines and results from it:

    sft/runs/<sweep-folder>/
        sweep.py        <- copy of this file, CONFIG block edited
        config.yaml     <- base config (copied from sft/configs/)
        <run_name>/     <- one per combination: config.yaml (written by
                           sweep.py) + run artifacts written by
                           scripts/train_eval.py via the config's `run_dir`
                           key (train.log, train_result.json, eval/,
                           eval_summary.json, summary.png)

Workflow: copy a sweep folder (or this file + a base config), edit the CONFIG
block (SWEEP_NAME / SWEEP_AXES), then run it:

    python sft/runs/<sweep-folder>/sweep.py [--dry-run] [--delay SECONDS]

Each axis in SWEEP_AXES is a list of dicts whose keys are config.yaml keys.
Use dot-separated keys for nested values (e.g. "train.lr",
"elicitation.train_params"). The `_label` key is special: it is stripped from
the merged config and joined across axes (with the sweep name) to form the
run name. The cartesian product across all axes determines the runs; a
single-element axis acts as a global override applied to every run.

Each combination gets <run_name>/config.yaml with `run_dir` pointed at that
folder, and a tmux window running scripts/train_eval.py on it (which writes its
own <run_name>/train.log). train_eval.py retries Modal's 100-ephemeral-app limit
internally, so windows just wait for slots. Re-running kills any existing tmux
session with the same name and reuses the run folders (Modal-side run names
stay timestamped, but local artifacts mix — clear run folders before a
relaunch you care about). Monitor with sft/sweep_status.py.
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

SWEEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in SWEEP_DIR.parents if (p / ".git").exists())

# ---------------------------------------------------------------------------
# Configuration — edit these for each sweep.
# ---------------------------------------------------------------------------

BASE_CONFIG = SWEEP_DIR / "config.yaml"  # template config to override
SWEEP_NAME = "sftsweep1-gptoss20b"       # tmux session + run-name prefix
TRAIN_SCRIPT = "scripts/train_eval.py"   # relative to repo root

SWEEP_AXES: list[list[dict[str, Any]]] = [
    [{"_label": f"lr{lr}", "train.lr": lr}
     for lr in ["1e-4", "1e-3", "1e-2"]],
    [{"_label": f"k{k // 1000}k", "elicitation.train_params": k}
     for k in [1_000, 10_000, 100_000]],
]

# ---------------------------------------------------------------------------
# Logic — no per-sweep edits needed below.
# ---------------------------------------------------------------------------

TMUX_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def tmux_safe(name: str) -> str:
    """Sanitize a string for tmux session/window names (tmux rejects '.', etc.)."""
    return TMUX_NAME_RE.sub("_", name)


def deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set d['a']['b']['c'] = value when dotted_key == 'a.b.c'."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def make_run_name(combo: tuple[dict, ...]) -> str:
    labels = [d["_label"] for d in combo if "_label" in d]
    return "-".join([SWEEP_NAME, *labels]) if labels else SWEEP_NAME


def generate_configs(base_path: Path, axes: list[list[dict]]) -> list[Path]:
    base_cfg = yaml.safe_load(base_path.read_text())
    config_paths: list[Path] = []
    seen_names: set[str] = set()
    for idx, combo in enumerate(itertools.product(*axes)):
        cfg = deepcopy(base_cfg)
        for d in combo:
            for k, v in d.items():
                if k == "_label":
                    continue
                deep_set(cfg, k, v)

        run_name = make_run_name(combo)
        if run_name in seen_names:
            run_name = f"{run_name}_{idx}"
        seen_names.add(run_name)
        cfg["run_name"] = run_name
        run_dir = SWEEP_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        # train_eval.py writes artifacts here instead of a timestamped sft/runs/ dir.
        cfg["run_dir"] = str(run_dir.relative_to(REPO_ROOT))

        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        config_paths.append(config_path)

    return config_paths


def launch_tmux(session_name: str, config_paths: list[Path], delay: int = 0) -> None:
    session = tmux_safe(session_name)
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

    python_bin = sys.executable
    train_script = REPO_ROOT / TRAIN_SCRIPT

    for i, config_path in enumerate(config_paths):
        window = tmux_safe(config_path.parent.name.removeprefix(f"{SWEEP_NAME}-"))
        sleep_prefix = f"sleep {i * delay} && " if delay > 0 and i > 0 else ""
        # No `| tee`: train_eval.py writes <run_dir>/train.log itself.
        cmd = (f"cd {REPO_ROOT} && {sleep_prefix}{python_bin} {train_script} "
               f"{config_path} 2>&1")
        if i == 0:
            create = ["tmux", "new-session", "-d", "-P", "-F", "#{window_id}",
                      "-s", session, "-n", window, "-c", str(REPO_ROOT)]
        else:
            # "<session>:" = next free window index; a bare session target
            # resolves to window 0 on some tmux versions and fails.
            create = ["tmux", "new-window", "-d", "-P", "-F", "#{window_id}",
                      "-t", f"{session}:", "-n", window, "-c", str(REPO_ROOT)]
        wid = subprocess.run(create, check=True, capture_output=True,
                             text=True).stdout.strip()
        # Target the immutable window id — automatic-rename can change names.
        subprocess.run(["tmux", "send-keys", "-t", wid, cmd, "Enter"], check=True)

    print(f"\nLaunched {len(config_paths)} windows in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")
    print("Navigate: Ctrl-B n (next), Ctrl-B p (prev), Ctrl-B w (list).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep configs and launch in tmux.")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't launch tmux.")
    parser.add_argument("--delay", type=int, default=120,
                        help="Stagger run i by i*delay seconds (default 120; 0 disables). "
                             "Bounds peak load from worker cold starts.")
    args = parser.parse_args()

    total = 1
    for axis in SWEEP_AXES:
        total *= len(axis)

    print(f"Sweep '{SWEEP_NAME}': {len(SWEEP_AXES)} axes, {total} combinations")
    for i, axis in enumerate(SWEEP_AXES):
        keys = sorted({k for d in axis for k in d if k != "_label"})
        print(f"  Axis {i}: {len(axis)} values, keys={keys}")

    if not BASE_CONFIG.exists():
        sys.exit(f"BASE_CONFIG not found: {BASE_CONFIG}")

    config_paths = generate_configs(BASE_CONFIG, SWEEP_AXES)
    print(f"\nGenerated {len(config_paths)} configs:")
    for p in config_paths:
        print(f"  {p.relative_to(REPO_ROOT)}")

    if args.dry_run:
        print("\n(dry run — not launching)")
        return

    launch_tmux(SWEEP_NAME, config_paths, delay=args.delay)


if __name__ == "__main__":
    main()
