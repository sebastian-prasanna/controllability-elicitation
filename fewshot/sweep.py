"""Sweep launcher for few-shot eval runs: cartesian product of run_eval.py flags
-> tmux windows, grouped under one sweep folder.

Same pattern as gepa/sweep.py, adapted for scripts/run_eval.py. Each axis is a
list of dicts whose keys are run_eval.py flags; `_label` is stripped and joined
across axes to form the run name. Keys NOT starting with "--" (other than
_label) are template variables: after merging a combo, every string flag value
is `.format()`-ed with them, so a flag can depend on several axes at once, e.g.

    "--system-prompt": "fewshot/prompts/{model}/k{k}.txt"

with axis 1 supplying model=... and axis 2 supplying k=... .

Runs land in fewshot/runs/<SWEEP_NAME>/<run_name>/ (eval JSON + progress.jsonl +
summary.json via --out-dir, stdout teed to stdout.log inside the run dir) — one
subfolder per sweep instead of a pile of top-level run dirs. Runs whose
summary.json already exists are skipped (safe relaunch; --force reruns).

Usage:
    1. Edit SWEEP_NAME, BASE_FLAGS and SWEEP_AXES below.
    2. .venv/bin/python fewshot/sweep.py [--dry-run] [--delay SECONDS] [--force]
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_eval.py"
RUNS_DIR = REPO_ROOT / "fewshot" / "runs"

# ---------------------------------------------------------------------------
# Configuration — edit these for each sweep.
# ---------------------------------------------------------------------------

SWEEP_NAME = "final_k1_heldout"

# Flags applied to every run (any run_eval.py flag). Axes override these.
# String values may reference template vars supplied by the axes.
BASE_FLAGS: dict[str, Any] = {
    "--split": "test",
    "--mode": "all",
    "--heldout": True,  # the 3 held-out modes (500 x 3 tasks per run)
    "--concurrency": 150,
    "--system-prompt": "fewshot/final_prompts/{model}/k1_s{s}.txt",
}

# (label, model id, extra per-model flags). qwen32b runs at low concurrency —
# its OpenRouter provider collapses under load with long prompts.
MODELS = [
    ("gptoss20b", "openai/gpt-oss-20b", {}),
    ("gptoss120b", "openai/gpt-oss-120b", {}),
    ("qwen8b", "qwen/qwen3-8b", {}),
    ("qwen32b", "qwen/qwen3-32b", {"--concurrency": 40}),
    ("kimik3", "moonshotai/kimi-k3", {}),
    ("dsv4pro", "deepseek/deepseek-v4-pro-0813", {}),
    ("glm53", "z-ai/glm-5.3", {}),
    ("glm53flash", "z-ai/glm-5.3-flash", {}),
]

SWEEP_AXES: list[list[dict[str, Any]]] = [
    # Axis 1: model (supplies the {model} template var).
    [{"_label": label, "model": label, "--model": mid, **extra}
     for label, mid, extra in MODELS],

    # Axis 2: prompt seed (supplies the {s} template var).
    [{"_label": f"s{s}", "s": s} for s in (1, 2, 3)],
]

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

TMUX_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def tmux_safe(name: str) -> str:
    return TMUX_NAME_RE.sub("_", name)


def make_run_name(combo: tuple[dict, ...]) -> str:
    labels = [d["_label"] for d in combo if "_label" in d]
    return "_".join(labels) if labels else "run"


def flags_to_argv(flags: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for k, v in flags.items():
        if v is True:
            argv.append(k)
        elif v is False or v is None:
            continue
        else:
            argv.extend([k, str(v)])
    return argv


def build_runs(force: bool) -> tuple[list[tuple[str, list[str]]], int]:
    """Return ((run_name, argv) per combination, n_skipped)."""
    runs: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    skipped = 0
    for idx, combo in enumerate(itertools.product(*SWEEP_AXES)):
        flags = dict(BASE_FLAGS)
        tvars: dict[str, Any] = {}
        for d in combo:
            for k, v in d.items():
                if k == "_label":
                    continue
                (flags if k.startswith("--") else tvars)[k] = v
        flags = {k: (v.format(**tvars) if isinstance(v, str) else v)
                 for k, v in flags.items()}
        run_name = make_run_name(combo)
        if run_name in seen:
            run_name = f"{run_name}_{idx}"
        seen.add(run_name)
        run_dir = RUNS_DIR / SWEEP_NAME / run_name
        if not force and (run_dir / "summary.json").exists():
            skipped += 1
            continue
        flags["--tag"] = f"{SWEEP_NAME}_{run_name}"
        flags["--out-dir"] = str(run_dir.relative_to(REPO_ROOT))
        runs.append((run_name, flags_to_argv(flags)))
    return runs, skipped


def launch_tmux(runs: list[tuple[str, list[str]]], delay: int) -> None:
    session = tmux_safe(f"fewshot_{SWEEP_NAME}")
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

    # Snapshot this script into the sweep dir so every run records the exact
    # version (CONFIG block included) that launched it.
    sweep_dir = RUNS_DIR / SWEEP_NAME
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep.py").write_text(Path(__file__).read_text())

    for i, (run_name, argv) in enumerate(runs):
        window = tmux_safe(run_name)
        run_dir = RUNS_DIR / SWEEP_NAME / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        sleep_prefix = f"sleep {i * delay} && " if delay > 0 and i > 0 else ""
        inner = " ".join([sys.executable, str(RUN_SCRIPT), *argv])
        cmd = f"cd {REPO_ROOT} && {sleep_prefix}{inner} 2>&1 | tee {run_dir}/stdout.log"
        if i == 0:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session, "-n", window, "-c", str(REPO_ROOT)],
                check=True,
            )
        else:
            subprocess.run(
                ["tmux", "new-window", "-t", session, "-n", window, "-c", str(REPO_ROOT)],
                check=True,
            )
        subprocess.run(["tmux", "send-keys", "-t", f"{session}:{window}", cmd, "Enter"], check=True)

    print(f"\nLaunched {len(runs)} windows in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a few-shot eval sweep in tmux.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but don't launch.")
    parser.add_argument("--delay", type=int, default=300,
                        help="Stagger run i by i*delay seconds (default 300; 0 disables). "
                             "Keeps aggregate OpenRouter concurrency bounded.")
    parser.add_argument("--force", action="store_true",
                        help="Rerun runs whose summary.json already exists (default: skip).")
    args = parser.parse_args()

    runs, skipped = build_runs(force=args.force)
    print(f"Sweep '{SWEEP_NAME}': {len(runs)} runs to launch, {skipped} already done (skipped)")
    for run_name, argv in runs:
        print(f"  {run_name}: {' '.join(argv)}")

    if args.dry_run:
        print("\n(dry run — not launching)")
        return
    if not runs:
        print("Nothing to launch.")
        return

    launch_tmux(runs, delay=args.delay)


if __name__ == "__main__":
    main()
