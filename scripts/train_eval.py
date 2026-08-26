"""Config-driven Modal training + CoT-Control eval pipeline.

    .venv/bin/python scripts/train_eval.py sft/configs/example_train.yaml
    .venv/bin/python scripts/train_eval.py sft/configs/example_train.yaml --dry-run
    .venv/bin/python scripts/train_eval.py --eval-only sft/runs/<run_name>

Reads a YAML config (see sft/configs/example_train.yaml + the eval: section below),
LoRA-trains the base model on Modal (optionally masking to k random adapter
weights), then runs the CoT-Control eval on every saved checkpoint through the
Modal vLLM engine. Local artifacts land in sft/runs/<run_name>/:

    config.yaml           copy of the config as run
    train_result.json     losses, checkpoint volume paths, mask verification
    eval/checkpoint-<step>.json   full CoT-Control result per checkpoint
    eval_summary.json     per-checkpoint strict compliance / accuracy / per-mode
    summary.png           compliance & accuracy vs step

eval: section (all optional):
    eval_base_model: false     # step-0 checkpoint already covers "before training"
    dataset: all               # gpqa | hle | mmlu_pro | all
    mode: random               # a single constraint mode, or random | all
    seed: 0                    # question->mode assignment
    max_samples: null          # cap questions
    subsample_seed: null       # seeded random question subsample
    split: null                # canonical split (train/val/test); use "val" for
                               # checkpoint selection, "test" only for final reports
    system_prompt: ""          # literal string, or a path to a .txt file
    temperature: 0.0
    max_tokens: 12000
    num_samples: 1
    judge_model: openai/gpt-5-mini
    judge_concurrency: 200
    parallel_checkpoints: 4    # concurrent Modal engines (cost scales with this)
    gpu: null                  # inference GPU; default = the training gpu's type x tp
    tensor_parallel_size: 1
    max_model_len: 32768
    max_num_seqs: 256
    mixed_moe_lora: false      # true when lora_target_parameters is non-empty
    chat_template_kwargs: null # e.g. {reasoning_effort: high} for gpt-oss
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.inference import modal_vllm  # noqa: E402
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, make_generate_fn  # noqa: E402
from cotcontrol.training.config import TrainConfig, filter_dataclass_kwargs  # noqa: E402
from cotcontrol.training.modal_app import load_sft_jsonl, train  # noqa: E402

RUNS_DIR = ROOT / "sft" / "runs"


def load_config(path: Path) -> tuple[TrainConfig, dict, dict]:
    """YAML -> (TrainConfig, eval section, raw yaml). Top-level dict sections
    are flattened into TrainConfig fields (unknown keys dropped); 'eval' is
    pulled out first so its keys never collide."""
    raw = yaml.safe_load(path.read_text())
    eval_cfg = raw.pop("eval", {}) or {}
    flat = {k: v for section in raw.values() if isinstance(section, dict) for k, v in section.items()}
    flat |= {k: v for k, v in raw.items() if not isinstance(v, dict)}
    return TrainConfig(**filter_dataclass_kwargs(TrainConfig, flat)), eval_cfg, raw


def resolve_system_prompt(value: str) -> str:
    if value and value.endswith(".txt") and (ROOT / value).exists():
        return (ROOT / value).read_text().strip()
    return value or ""


def build_generate_config(tc: TrainConfig, ec: dict) -> ModalGenerateConfig:
    # Default inference GPU: one GPU of the training type per tensor-parallel rank.
    tp = int(ec.get("tensor_parallel_size", 1))
    gpu_type = tc.gpu.split(":")[0]
    gpu = ec.get("gpu") or (gpu_type if tp == 1 else f"{gpu_type}:{tp}")
    return ModalGenerateConfig(
        temperature=ec.get("temperature", 0.0),
        max_tokens=ec.get("max_tokens", 12000),
        num_samples=ec.get("num_samples", 1),
        chat_template_kwargs=ec.get("chat_template_kwargs") or tc.chat_template_kwargs,
        gpu=gpu,
        tensor_parallel_size=tp,
        max_lora_rank=max(64, tc.lora_rank),
        max_model_len=ec.get("max_model_len", 32768),
        max_num_seqs=ec.get("max_num_seqs", 256),
        mixed_moe_lora=ec.get("mixed_moe_lora", bool(tc.lora_target_parameters)),
    )


async def eval_checkpoints(tc: TrainConfig, ec: dict, train_result: dict, run_dir: Path) -> dict:
    gen_cfg = build_generate_config(tc, ec)
    system_prompt = resolve_system_prompt(ec.get("system_prompt", ""))
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    targets = [(c["step"], c["path"]) for c in train_result["checkpoints"]]
    if ec.get("eval_base_model", False):
        targets = [("base", None)] + targets

    sem = asyncio.Semaphore(int(ec.get("parallel_checkpoints", 4)))

    async def eval_one(step, lora_path):
        name = f"checkpoint-{step}" if lora_path else "base"
        async with sem:
            result = await eval_cotcontrolqa(
                model=tc.base_model,
                system_prompt=system_prompt,
                generate_fn=make_generate_fn(tc.base_model, gen_cfg, lora_path),
                save_dir=eval_dir,
                save_name=name,
                dataset=ec.get("dataset", "all"),
                mode=ec.get("mode", "random"),
                seed=int(ec.get("seed", 0)),
                max_samples=ec.get("max_samples"),
                subsample_seed=ec.get("subsample_seed"),
                split=ec.get("split"),
                judge_model=ec.get("judge_model", "openai/gpt-5-mini"),
                judge_concurrency=int(ec.get("judge_concurrency", 200)),
                backend_info={"base_model": tc.base_model, "lora_path": lora_path,
                              "generate_config": asdict(gen_cfg)},
            )
        return step, lora_path, result["summary"]

    # One app context for the whole sweep: same engine parameters -> Modal fans
    # containers out up to parallel_checkpoints and reuses warm ones.
    async with modal_vllm.app.run():
        summaries = await asyncio.gather(*[eval_one(s, p) for s, p in targets])

    eval_summary = {
        "run_name": tc.run_name,
        "base_model": tc.base_model,
        "checkpoints": [
            {
                "step": step,
                "lora_path": lora_path,
                "accuracy": summary["accuracy"],
                "compliance_rate": summary["compliance_rate"],
                "n_errors": summary["n_errors"],
                "per_mode": summary["per_mode"],
            }
            for step, lora_path, summary in summaries
        ],
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
    return eval_summary


def plot_summary(eval_summary: dict, run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpts = [c for c in eval_summary["checkpoints"] if isinstance(c["step"], int)]
    if len(ckpts) < 2:
        return
    steps = [c["step"] for c in ckpts]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [c["compliance_rate"] for c in ckpts], "o-", label="strict compliance")
    ax.plot(steps, [c["accuracy"] for c in ckpts], "s--", label="accuracy")
    ax.set_xlabel("optimizer step")
    ax.set_ylim(0, 1)
    ax.set_title(eval_summary["run_name"])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "summary.png", dpi=150)
    print(f"Saved {run_dir / 'summary.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="YAML config path")
    ap.add_argument("--dry-run", action="store_true", help="print resolved configs and exit")
    ap.add_argument("--eval-only", metavar="RUN_DIR",
                    help="skip training; re-eval the checkpoints in RUN_DIR/train_result.json")
    args = ap.parse_args()

    if args.eval_only:
        run_dir = Path(args.eval_only)
        train_result = json.loads((run_dir / "train_result.json").read_text())
        tc, ec, _ = load_config(run_dir / "config.yaml")
        tc.run_name = train_result["run_name"]  # keep the timestamped name
    else:
        if not args.config:
            ap.error("config is required unless --eval-only")
        tc, ec, raw = load_config(Path(args.config))
        if args.dry_run:
            print(yaml.dump({"train": asdict(tc), "eval": ec}, sort_keys=False))
            data = load_sft_jsonl(tc.data_path)
            print(f"dataset: {len(data)} examples from {tc.data_path}")
            return
        data = load_sft_jsonl(tc.data_path)
        train_result = train(tc, data)  # timestamps tc.run_name
        run_dir = RUNS_DIR / tc.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.dump(raw | {"eval": ec}, sort_keys=False))
        (run_dir / "train_result.json").write_text(json.dumps(train_result, indent=2))
        print(f"Training done: {len(train_result['checkpoints'])} checkpoints, "
              f"final_loss={train_result['final_loss']}, "
              f"mask={train_result['mask_verification']}")

    eval_summary = asyncio.run(eval_checkpoints(tc, ec, train_result, run_dir))
    for c in eval_summary["checkpoints"]:
        print(f"  step {c['step']}: compliance={c['compliance_rate']}, accuracy={c['accuracy']}")
    plot_summary(eval_summary, run_dir)


if __name__ == "__main__":
    main()
