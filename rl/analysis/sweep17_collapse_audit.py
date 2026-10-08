"""Audit saved sweep17 batches; rescores are offline, not new training runs.

Run from the repository root: python rl/analysis/sweep17_collapse_audit.py
Writes an iteration CSV, a small counterfactual JSON, and a diagnostic plot.
"""
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "rl/runs/sweep17_t80_fullk"
OUT = Path(__file__).resolve().parent


def read_json(path):
    # The driver can be writing the most recent file during an audit.
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def mean(values):
    return st.mean(values) if values else None


def audit_batches():
    rows = []
    for run in sorted(RUNS.glob("sweep17-t80-*")):
        progress = {}
        for line in (run / "progress.jsonl").read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Resumes may append another record for the same iteration.
            progress[entry["iteration"]] = entry
        for path in sorted((run / "batches").glob("batch-*.json")):
            batch = read_json(path)
            if batch is None or batch["iteration"] not in progress:
                continue
            entry = progress[batch["iteration"]]
            groups = [g for g in batch["groups"] if len(g["samples"]) >= 2]
            samples = [s for g in groups for s in g["samples"]]
            if not samples:
                continue
            tokens = sorted(s.get("n_tokens", len(s.get("token_ids", [])))
                            for s in samples)
            answerable = [s for g in groups if g["mode"] != "ignore_question"
                          for s in g["samples"]]
            truncated = [s for s in samples if s["truncated"]]
            rows.append({
                "arm": run.name.removeprefix("sweep17-t80-"),
                "iteration": batch["iteration"],
                "accuracy": entry["accuracy"],
                "lambda": batch["anchor_lambda"],
                "target": entry["lambda_target_acc"],
                "accuracy_answerable": mean([s["correct"] for s in answerable]),
                "truncated_fraction": len(truncated) / len(samples),
                "all_truncated_group_fraction": mean([
                    all(s["truncated"] for s in g["samples"]) for g in groups]),
                "accuracy_mixed_group_fraction": mean([
                    0 < sum(s["correct"] for s in g["samples"]) < len(g["samples"])
                    for g in groups]),
                "median_tokens": st.median(tokens),
                "p90_tokens": tokens[int(0.9 * (len(tokens) - 1))],
                "mean_truncated_reward": mean([s["reward"] for s in truncated]),
            })
    rows.sort(key=lambda r: (r["arm"], r["iteration"]))
    path = OUT / "sweep17_collapse_audit.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def rescore():
    results = []
    for arm, step in [("kall", 30), ("kall", 40), ("kall", 60),
                      ("k100k", 110), ("k100k", 170), ("k3k", 165)]:
        run = RUNS / f"sweep17-t80-{arm}"
        batch = read_json(run / "batches" / f"batch-{step}.json")
        evaluation = read_json(run / "iters" / f"iter-{step:04d}.json")
        records = {f"{r['dataset']}:{r['id']}:{r['mode']}": r
                   for r in evaluation["results"]}
        counts = {method: {"active_groups": 0, "positive_no_answer": 0,
                           "positive_truncated": 0}
                  for method in ("original", "zero_truncated", "zero_invalid")}
        n_no_answer = n_stop_no_answer = n_samples = 0
        for group in batch["groups"]:
            graded = [s for s in records[group["key"]]["samples"] if not s["error"]]
            samples = group["samples"]
            assert len(samples) == len(graded), (arm, step, group["key"])
            n_samples += len(samples)
            for s, e in zip(samples, graded):
                assert bool(s["correct"]) == bool(e["correct"])
                assert s["truncated"] == (e["finish_reason"] == "length")
                n_no_answer += e["extracted_answer"] is None
                n_stop_no_answer += not s["truncated"] and e["extracted_answer"] is None
            for method, count in counts.items():
                rewards = [s["reward"] if method == "original" or (
                    not s["truncated"] and (method == "zero_truncated" or
                                            e["extracted_answer"] is not None))
                           else 0.0 for s, e in zip(samples, graded)]
                baseline = st.mean(rewards)
                count["active_groups"] += max(rewards) - min(rewards) > 1e-8
                for reward, s, e in zip(rewards, samples, graded):
                    positive = reward > baseline + 1e-8
                    count["positive_no_answer"] += positive and e["extracted_answer"] is None
                    count["positive_truncated"] += positive and s["truncated"]
        results.append({"arm": arm, "iteration": step, "n_samples": n_samples,
                        "n_no_answer": n_no_answer,
                        "n_stop_no_answer": n_stop_no_answer, "rescores": counts})
    (OUT / "sweep17_collapse_rescores.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def plot(rows):
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex="row")
    for row, arm in enumerate(["kall", "k100k", "k10k"]):
        data = [r for r in rows if r["arm"] == arm]
        steps = [r["iteration"] for r in data]
        label = "full rank-1" if arm == "kall" else arm
        ax = axes[row, 0]
        for key, name, color in [("accuracy", "Accuracy", "#2878b5"),
                                 ("truncated_fraction", "Truncated rollouts", "#d1495b"),
                                 ("all_truncated_group_fraction", "All-truncated groups", "#8c5aa6")]:
            ax.plot(steps, [r[key] for r in data], label=name, color=color, lw=1.2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(f"{label}\nFraction")
        ax = axes[row, 1]
        ax.plot(steps, [r["lambda"] for r in data], color="#3b7652")
        ax.axhline(3, color="gray", ls="--", lw=0.8)
        ax.set_ylim(-0.1, 3.2)
        ax.set_ylabel("Accuracy coefficient λ")
        ax = axes[row, 2]
        ax.plot(steps, [r["median_tokens"] for r in data], label="Median", color="#2878b5")
        ax.plot(steps, [r["p90_tokens"] for r in data], label="90th percentile", color="#e69f00", lw=1)
        ax.axhline(12000, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("Generated tokens")
        for ax in axes[row]:
            ax.set_xlabel("Iteration")
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Sweep 17: truncation, accuracy signal, and controller state\n"
                 "Saved-batch snapshot; latest progress record used for repeated iterations")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sweep17_collapse_audit.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    rows = audit_batches()
    results = rescore()
    plot(rows)
    print(f"Audited {len(rows)} batches across {len({r['arm'] for r in rows})} arms.")
    print(json.dumps(results, indent=2))
