"""Config-driven tinker training intervention + CoT-Control eval pipeline.

    .venv/bin/python scripts/tinker_train_eval.py tinker_runs/example_config.yaml
    .venv/bin/python scripts/tinker_train_eval.py tinker_runs/example_config.yaml --dry-run

Reads a YAML config (see tinker_runs/example_config.yaml), SFT-trains the base
model on the given dataset, then runs the CoT-Control eval on every saved
sampling checkpoint in parallel. Everything lands in
<output_dir>/<run_name>/:

    config.yaml           copy of the config as run
    train_result.json     losses, sampling/training checkpoint paths, timings
    training_data.json    rendered train examples (gradient vs no-gradient text)
    eval/<ckpt>.json      full CoT-Control result per checkpoint (all rollouts)
    eval_summary.json     per-checkpoint strict compliance / accuracy / per-mode
    summary.png           compliance & accuracy vs checkpoint

Re-running with the same config skips training if train_result.json already
exists (the recorded checkpoint paths are re-evaluated; tinker generations are
also disk-cached by utils, so finished checkpoints re-run for free).

SFT dataset format: JSONL, one example per line:
    {"input": [{"role": "user", "content": ...}, ...],
     "output": [{"role": "assistant", "content": ...}]}
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

import tinker  # noqa: E402

import utils  # noqa: E402
from cotcontrol.tinker_eval import run_cotcontrol_evaluation  # noqa: E402


def load_sft_data(path, num_examples=None) -> list[utils.SFTExample]:
    """path: a jsonl file, or a list of jsonl files (concatenated in order)."""
    paths = path if isinstance(path, list) else [path]
    data = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                data.append(utils.SFTExample(input=d["input"], output=d["output"]))
    if num_examples:
        data = data[:num_examples]
    print(f"Loaded {len(data)} SFT examples from {len(paths)} file(s)")
    return data


def train(cfg: dict, run_dir: Path, service_client: tinker.ServiceClient) -> dict:
    tc = cfg["train"]
    data = load_sft_data(tc["data_path"], tc.get("num_examples"))

    lora_kwargs = {}
    if cfg.get("lora_rank") is not None:
        lora_kwargs["rank"] = cfg["lora_rank"]
    training_client = service_client.create_lora_training_client(
        base_model=cfg["base_model"], **lora_kwargs
    )

    train_config = utils.TrainConfig(
        lr=float(tc.get("lr", 1e-4)),
        batch_size=tc.get("batch_size", 128),
        num_epochs=tc.get("num_epochs", 1),
        save_sampling_step=tc.get("save_sampling_step", 1),
        save_training_step=tc.get("save_training_step", -1),
        save_every_n_steps=tc.get("save_every_n_steps"),
        save_training_every_n_steps=tc.get("save_training_every_n_steps"),
    )

    t0 = time.time()
    result = utils.sft_train(
        training_client,
        data,
        config=train_config,
        run_name=cfg["run_name"],
        shuffle=tc.get("shuffle", True),
    )
    result["train_seconds"] = round(time.time() - t0, 1)

    (run_dir / "training_data.json").write_text(json.dumps(result.pop("training_data"), indent=1))
    (run_dir / "train_result.json").write_text(json.dumps(result, indent=1))
    print(f"Training done in {result['train_seconds']}s: {result['num_steps']} steps, "
          f"avg loss {result['avg_loss']:.4f}, {len(result['sampling_paths'])} sampling checkpoints")
    return result


async def evaluate(cfg: dict, run_dir: Path, service_client: tinker.ServiceClient,
                   sampling_paths: list[str]) -> list[dict]:
    ec = cfg["eval"]
    paths = list(sampling_paths)
    if ec.get("eval_base_model", False):
        paths = [cfg["base_model"]] + paths

    system_prompt = ec.get("system_prompt", "") or ""
    if system_prompt and Path(system_prompt).is_file():
        system_prompt = Path(system_prompt).read_text()

    gen_config = utils.GenerateConfig(
        temperature=float(ec.get("temperature", 0.0)),
        max_tokens=ec.get("max_tokens", 12000),
        max_concurrent=ec.get("max_concurrent", 2000),
        num_samples=ec.get("num_samples", 1),
        cache=ec.get("cache", True),
    )

    summaries, _ = await run_cotcontrol_evaluation(
        service_client,
        paths=paths,
        save_dir=run_dir / "eval",
        save_prefix=cfg["run_name"],
        system_prompt=system_prompt,
        config=gen_config,
        dataset=ec.get("dataset", "all"),
        mode=ec.get("mode", "random"),
        seed=ec.get("seed", 0),
        max_samples=ec.get("max_samples"),
        subsample_seed=ec.get("subsample_seed"),
        judge_model=ec.get("judge_model", "openai/gpt-5-mini"),
        judge_concurrency=ec.get("judge_concurrency", 200),
    )
    (run_dir / "eval_summary.json").write_text(json.dumps(summaries, indent=1))
    return summaries


def plot_summary(summaries: list[dict], run_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    utils.set_matplotlib_style()
    labels = [s["path"].rstrip("/").split("/")[-1] for s in summaries]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(labels)), 4.2))
    ax.plot(range(len(labels)), [s["compliance_rate"] for s in summaries],
            marker="o", lw=2, label="strict compliance")
    ax.plot(range(len(labels)), [s["accuracy"] for s in summaries],
            marker="o", lw=2, label="accuracy")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(run_dir.name)
    fig.tight_layout()
    fig.savefig(run_dir / "summary.png", dpi=150)
    print(f"Saved {run_dir / 'summary.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="path to the run's YAML config")
    p.add_argument("--dry-run", action="store_true",
                   help="load config and data, print the plan, exit without training")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for key in ("run_name", "base_model", "train", "eval"):
        assert key in cfg, f"config missing required key: {key}"

    run_dir = Path(cfg.get("output_dir", "tinker_runs")) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if args.dry_run:
        data = load_sft_data(cfg["train"]["data_path"], cfg["train"].get("num_examples"))
        n_ckpts = cfg["train"].get("num_epochs", 1) // max(cfg["train"].get("save_sampling_step", 1), 1) + 1
        print(f"DRY RUN — would train {cfg['base_model']} on {len(data)} examples "
              f"for {cfg['train'].get('num_epochs', 1)} epochs, then eval ~{n_ckpts} checkpoints "
              f"on dataset={cfg['eval'].get('dataset', 'all')} mode={cfg['eval'].get('mode', 'random')} "
              f"(run dir: {run_dir})")
        return

    service_client = tinker.ServiceClient()

    train_result_path = run_dir / "train_result.json"
    if train_result_path.exists():
        train_result = json.loads(train_result_path.read_text())
        print(f"Found existing train_result.json — skipping training "
              f"({len(train_result['sampling_paths'])} checkpoints)")
    else:
        train_result = train(cfg, run_dir, service_client)

    summaries = asyncio.run(
        evaluate(cfg, run_dir, service_client, train_result["sampling_paths"])
    )
    plot_summary(summaries, run_dir)

    print(f"\n=== {cfg['run_name']} ===")
    for s in summaries:
        print(f"  {s['path'].rstrip('/').split('/')[-1]:>30s}  "
              f"strict={s['compliance_rate']:.4f}  acc={s['accuracy']:.4f}")
    print(f"Everything saved under {run_dir}/")


if __name__ == "__main__":
    main()
