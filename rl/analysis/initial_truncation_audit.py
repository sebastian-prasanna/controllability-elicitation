"""Measure iteration-0 truncation and completion-gate rejection after SFT.

Uses matched sweep14 arms for four-model comparison and sweep17 separately.
The 12 sweep14 arms reuse the same 32 tasks; they are not independent datasets.
Run from the repository root: python rl/analysis/initial_truncation_audit.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "rl/runs"


def main():
    rows, summaries, mode_rows = [], [], []
    task_sequences = set()
    for sweep in [*sorted(RUNS.glob("sweep14*")), RUNS / "sweep17_t80_fullk"]:
        current = []
        by_mode = defaultdict(list)
        for config_path in sorted(sweep.glob("*/config.yaml")):
            run = config_path.parent
            config = yaml.safe_load(config_path.read_text())
            batch_path = run / "batches/batch-0.json"
            if not batch_path.exists():
                continue
            batch = json.loads(batch_path.read_text())
            evaluation = json.loads((run / "iters/iter-0000.json").read_text())
            generation = evaluation["config"]["backend_info"]["generate_config"]
            assert generation["max_tokens"] == 12000
            assert generation["temperature"] == 1.0
            assert config["rl"]["init_lora_path"]
            groups = batch["groups"]
            samples = [s for g in groups for s in g["samples"]]
            graded = [s for r in evaluation["results"] for s in r["samples"] if not s["error"]]
            assert len(samples) == len(graded) == 256
            for s, e in zip(samples, graded):
                assert s["truncated"] == (e["finish_reason"] == "length")
            if sweep.name.startswith("sweep14"):
                task_sequences.add(tuple(g["key"] for g in groups))
            row = {
                "sweep": sweep.name,
                "model": config["base_model"],
                "k": config["elicitation"]["train_params"],
                "n_rollouts": len(samples),
                "n_truncated": sum(s["truncated"] for s in samples),
                "n_gate_rejected": sum(s["finish_reason"] == "length" or
                                       s["extracted_answer"] is None for s in graded),
                "n_groups": len(groups),
                "n_all_truncated_groups": sum(all(s["truncated"] for s in g["samples"])
                                              for g in groups),
                "batch_path": str(batch_path.relative_to(ROOT)),
            }
            row["truncated_fraction"] = row["n_truncated"] / row["n_rollouts"]
            row["gate_rejected_fraction"] = row["n_gate_rejected"] / row["n_rollouts"]
            rows.append(row)
            current.append(row)
            for group in groups:
                by_mode[group["mode"]].extend(group["samples"])
        summary = {"sweep": sweep.name, "model": current[0]["model"], "n_arms": len(current)}
        for key in ("n_rollouts", "n_truncated", "n_gate_rejected", "n_groups", "n_all_truncated_groups"):
            summary[key] = sum(r[key] for r in current)
        summary["truncated_fraction"] = summary["n_truncated"] / summary["n_rollouts"]
        summary["gate_rejected_fraction"] = summary["n_gate_rejected"] / summary["n_rollouts"]
        summary["min_arm_truncated_fraction"] = min(r["truncated_fraction"] for r in current)
        summary["max_arm_truncated_fraction"] = max(r["truncated_fraction"] for r in current)
        summaries.append(summary)
        for mode, samples in by_mode.items():
            mode_rows.append({"sweep": sweep.name, "model": summary["model"], "mode": mode,
                              "n_rollouts": len(samples),
                              "n_truncated": sum(s["truncated"] for s in samples),
                              "truncated_fraction": sum(s["truncated"] for s in samples) / len(samples)})
    assert len(task_sequences) == 1, "Sweep14 task sets differ"
    for name, data in [("arms", rows), ("summary", summaries), ("modes", mode_rows)]:
        path = OUT / f"initial_truncation_{name}.csv"
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
