#!/usr/bin/env python3
"""Launch the sub-9-shot sweep on gpt-oss-120b, all evals in parallel (one tmux window each).

Prompts: fewshot/subset_prompts/gptoss120b/{n1,n3_random,n3_strat,n5_random}_s{1,2,3}.txt,
plus references k9_s{1,2,3} (= fewshot/final_prompts/gptoss120b/k1_s*.txt) and k0 (empty).
Each prompt -> val split, mode all (9 modes x 200) + --heldout (3 modes x 200).
Pinned provider/effort/max_tokens from configs/eval_pins.json via pinned_reeval build_cmd,
temperature overridden (default 1.0). Evals with summary.json are skipped (resume).

  python fewshot/launch_subset.py [--dry-run] [--max-samples N --suffix smoke]
Outputs: fewshot/runs/subset/gptoss120b{_suffix}/<prompt>_<split>/
"""
import argparse, shlex, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pinned_reeval.launch import build_cmd  # noqa: E402

LABEL = "gptoss120b"
SUBSET = REPO / "fewshot" / "subset_prompts" / LABEL
FINAL = REPO / "fewshot" / "final_prompts" / LABEL
SPLITS = {"val": [], "heldout": ["--heldout"]}


def prompts():
    out = {"k0": ""}
    out |= {f"k9_s{s}": str(FINAL / f"k1_s{s}.txt") for s in (1, 2, 3)}
    out |= {p.stem: str(p) for p in sorted(SUBSET.glob("*.txt"))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=50, help="per eval")
    ap.add_argument("--only", default=None, help="comma-separated prompt names")
    a = ap.parse_args()
    lane = REPO / "fewshot" / "runs" / "subset" / (LABEL + (f"_{a.suffix}" if a.suffix else ""))
    sess = "fssub" + (f"_{a.suffix}" if a.suffix else "")
    windows = []
    for name, prompt in prompts().items():
        if a.only and name not in a.only.split(","):
            continue
        for sp, fl in SPLITS.items():
            run = f"{name}_{sp}"
            out = lane / run
            if (out / "summary.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = build_cmd(LABEL, run, ["--system-prompt", prompt] + fl, a.max_samples, a.temperature)
            cmd[cmd.index("--split") + 1] = "val"
            cmd[cmd.index("--out-dir") + 1] = str(out)
            cmd[cmd.index("--concurrency") + 1] = str(a.concurrency)
            cmd[cmd.index("--max-retries") + 1] = "12"
            q = " ".join(shlex.quote(x) for x in cmd)
            windows.append((run, f"cd {shlex.quote(str(REPO))} && set -a && . ./.env && set +a && "
                                 f"{q} > {shlex.quote(str(out / 'stdout.log'))} 2>&1; "
                                 f"echo \"$(date +%FT%T) {run} rc=$?\" >> {shlex.quote(str(lane / 'driver.log'))}"))
    print(f"{len(windows)} evals -> {lane}")
    if a.dry_run or not windows:
        for r, c in windows[:2]:
            print(r, c)
        return
    subprocess.run(["tmux", "kill-session", "-t", sess], stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", sess, "-n", windows[0][0], windows[0][1]], check=True)
    for r, c in windows[1:]:
        subprocess.run(["tmux", "new-window", "-t", sess, "-n", r, c], check=True)
    print(f"tmux session {sess}")


if __name__ == "__main__":
    main()
