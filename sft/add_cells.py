#!/usr/bin/env python
"""Add cells to a LIVE lr x k sweep folder without disturbing the running ones.

    python sft/add_cells.py sft/runs/lrk_qwen8b --cells 10000:3e-2 10000:1e-1 [--dry-run]

Why this exists instead of re-running the sweep's own sweep.py: `sweep.py`
starts `launch_tmux` with `tmux kill-session -t <SWEEP_NAME>`, so re-running it
in a folder whose cells are still training kills them all. This script writes
the new cells' config.yaml by cloning an existing cell in the same folder
(so every shared hyperparameter is inherited verbatim, not re-typed), points
`run_dir` back into that folder so the new cells land in the same grid, and
launches them in a SEPARATE `<folder>_ext` tmux session.

Refuses to touch a cell that already has a train.log.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def tmux_safe(name: str) -> str:
    """Sanitize for tmux names; dots especially break `session:window` targets."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def pick_template(sweep_dir: Path) -> Path:
    """Any existing cell config in this folder — they differ only in lr/k."""
    for d in sorted(sweep_dir.iterdir()):
        cfg = d / "config.yaml"
        if d.is_dir() and cfg.exists():
            return cfg
    raise SystemExit(f"no existing cell with a config.yaml under {sweep_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--cells", nargs="+", required=True,
                    help="k:lr[:mask_seed] specs, e.g. 10000:3e-2 or 10000:3e-2:1 "
                         "(k='none' for an unmasked control). A mask_seed different "
                         "from the sweep's default gives a same-k replicate on a "
                         "DIFFERENT random subset — the control for 'was this mask "
                         "just unlucky?'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Resolve so relative CLI paths still produce repo-relative run_dir values.
    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.is_dir():
        raise SystemExit(f"{sweep_dir} is not a directory")
    template = pick_template(sweep_dir)
    base = yaml.safe_load(template.read_text())
    prefix = template.parent.name.rsplit("-k", 1)[0]  # e.g. lrk-qwen8b
    print(f"template: {template.relative_to(ROOT)}  (prefix {prefix})")

    planned = []
    for spec in args.cells:
        parts = spec.split(":")
        k_s, lr = parts[0], parts[1]
        seed = int(parts[2]) if len(parts) > 2 else None
        k = None if k_s.lower() in ("none", "all", "null") else int(k_s)
        label = "kall" if k is None else (f"k{k // 1000}k" if k >= 1000 else f"k{k}")
        name = f"{prefix}-{label}-lr{lr}" + (f"-seed{seed}" if seed is not None else "")
        d = sweep_dir / name
        if (d / "train.log").exists():
            print(f"  SKIP {name}: already has a train.log")
            continue
        drive = "-" if k is None else f"{float(lr) * k ** 0.5:.2f}"
        planned.append((name, d, k, lr, seed))
        print(f"  {name}: k={k if k else 'ALL'} lr={lr} drive={drive}")

    if not planned:
        print("nothing to add")
        return
    if args.dry_run:
        print("(dry run — no configs written, nothing launched)")
        return

    for name, d, k, lr, seed in planned:
        d.mkdir(parents=True, exist_ok=True)
        cfg = dict(base, run_name=name,
                   run_dir=str(d.relative_to(ROOT)))
        cfg["train"] = dict(base["train"], lr=lr)
        cfg["elicitation"] = dict(base.get("elicitation", {}), train_params=k)
        if seed is not None:
            cfg["elicitation"]["mask_seed"] = seed
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    session = f"{sweep_dir.name}_ext"
    exists = subprocess.run(["tmux", "has-session", "-t", session],
                            capture_output=True).returncode == 0
    for i, (name, d, _, _, _) in enumerate(planned):
        window = tmux_safe(name.removeprefix(f"{prefix}-"))
        if not exists and i == 0:
            create = ["tmux", "new-session", "-d", "-P", "-F", "#{window_id}",
                      "-s", session, "-n", window, "-c", str(ROOT)]
            exists = True
        else:
            create = ["tmux", "new-window", "-d", "-P", "-F", "#{window_id}",
                      "-t", f"{session}:", "-n", window, "-c", str(ROOT)]
        wid = subprocess.run(create, check=True, capture_output=True,
                             text=True).stdout.strip()
        cmd = (f"{sys.executable} {ROOT / 'scripts/train_eval.py'} "
               f"{d / 'config.yaml'}")
        # Target the immutable window id — names with '.' break target parsing.
        subprocess.run(["tmux", "send-keys", "-t", wid, cmd, "Enter"], check=True)
        print(f"launched {name} in {session}:{window} ({wid})")

    print(f"\n{len(planned)} cells launched. Existing sessions untouched.")


if __name__ == "__main__":
    main()
