#!/usr/bin/env python
"""GRPO driver: runs the rollout/grading side of RL locally, against a
training worker on Modal.

    python rl/run_grpo.py rl/configs/smoke_qwen06b.yaml

Per iteration t (protocol counterpart of cotcontrol/training/rl_worker.py):
  1. wait until the worker publishes /checkpoints/<run>/checkpoint-<t>
  2. sample prompts (train split, seeded per iteration), roll out G samples
     per prompt through the Modal vLLM engine with the adapter hot-swapped
     (return_logprobs=True, cache off — on-policy), grade via
     eval_cotcontrolqa (full artifacts saved under rl/runs/<run>/iters/)
  3. score rewards (shaped graders), upload rl/batch-<t>.json to the volume

Everything is saved: eval_cotcontrolqa writes the complete per-rollout record
(outputs, reasoning, grading, token ids/logprobs), batches/ keeps the exact
reward+token payload the worker trained on, progress.jsonl tracks per-iteration
aggregates, and the worker's metrics.jsonl / mask files are pulled down at the
end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import modal
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.eval.grading import (  # noqa: E402
    compliance_score,
    shaped_compliance,
    shaped_task_score,
)
from cotcontrol.eval.prompts import MODES  # noqa: E402
from cotcontrol.inference import modal_vllm  # noqa: E402
from cotcontrol.inference.modal_vllm import (  # noqa: E402
    ModalGenerateConfig,
    make_generate_fn,
)
from cotcontrol.training import modal_app as train_modal  # noqa: E402
from cotcontrol.training.config import RLConfig, filter_dataclass_kwargs  # noqa: E402

RUNS_DIR = ROOT / "rl" / "runs"
REWARD_FNS = {
    "compliance_score": compliance_score,
    "shaped_task_score": shaped_task_score,
    "shaped_compliance": shaped_compliance,
}


def load_config(path: Path) -> tuple[RLConfig, dict, dict]:
    """YAML -> (RLConfig, inference section, raw). Same flatten convention as
    scripts/train_eval.py; the 'inference' section holds engine overrides."""
    raw = yaml.safe_load(path.read_text())
    inf = raw.pop("inference", {}) or {}
    flat = {k: v for sec in raw.values() if isinstance(sec, dict) for k, v in sec.items()}
    flat |= {k: v for k, v in raw.items() if not isinstance(v, dict)}
    return RLConfig(**filter_dataclass_kwargs(RLConfig, flat)), inf, raw


def build_generate_config(cfg: RLConfig, inf: dict) -> ModalGenerateConfig:
    tp = int(inf.get("tensor_parallel_size", 1))
    gpu_type = cfg.gpu.split(":")[0]
    gpu = cfg.inference_gpu or inf.get("gpu") or (gpu_type if tp == 1 else f"{gpu_type}:{tp}")
    return ModalGenerateConfig(
        temperature=cfg.rollout_temperature,
        top_p=1.0,  # required: vLLM logprobs are pre-truncation (see RLConfig)
        max_tokens=cfg.rollout_max_tokens,
        num_samples=cfg.group_size,
        return_logprobs=True,
        cache=False,  # on-policy sampling must never replay cached rollouts
        chat_template_kwargs=cfg.chat_template_kwargs,
        gpu=gpu,
        tensor_parallel_size=tp,
        max_lora_rank=max(64, cfg.lora_rank),
        max_model_len=int(inf.get("max_model_len", 32768)),
        max_num_seqs=int(inf.get("max_num_seqs", 256)),
        mixed_moe_lora=inf.get("mixed_moe_lora", bool(cfg.lora_target_parameters)),
    )


def build_batch(t: int, result: dict, cfg: RLConfig) -> dict:
    """Eval records -> the worker's training payload. Rewards use the shaped
    graders (strict compliance is ~always 0 for these models — no gradient).
    Errored rollouts and ones without token data are dropped here; group
    integrity (>=2 samples) is the worker's job."""
    reward_fn = REWARD_FNS[cfg.reward]
    groups = []
    for rec in result["results"]:
        if "prompt_token_ids" not in rec:
            continue
        samples = []
        for s in rec["samples"]:
            if s["error"] or not s.get("token_ids"):
                continue
            task = {
                "mode": rec["mode"],
                "keyword": rec.get("keyword"),
                "synonyms": rec.get("synonyms"),
                "compliance": s["compliance"],
                "correct": s["correct"],
                "reasoning": s["reasoning_text_graded"],
                "error": s["error"],
                "judge": s["judge"],
            }
            samples.append({
                "token_ids": s["token_ids"],
                "token_logprobs": s["token_logprobs"],
                "reward": float(reward_fn(task)),
                "truncated": s["finish_reason"] == "length",
                # extra context for analysis; the worker ignores these
                "compliance": s["compliance"],
                "correct": s["correct"],
                "shaped_compliance": shaped_compliance(task),
            })
        groups.append({
            "key": f"{rec['dataset']}:{rec['id']}:{rec['mode']}",
            "mode": rec["mode"],
            "prompt_token_ids": rec["prompt_token_ids"],
            "samples": samples,
        })
    return {"iteration": t, "reward": cfg.reward, "groups": groups}


def _training_alive(handle) -> None:
    """Raise (with the remote traceback) if the spawned worker already ended."""
    try:
        res = handle.get(timeout=0)
    except Exception as e:  # timeout while still running -> fine
        if "timeout" in type(e).__name__.lower():
            return
        raise
    raise RuntimeError(f"training worker exited before the driver finished: {res}")


def _vol_read(path: str) -> bytes | None:
    try:
        return b"".join(train_modal.checkpoints_vol.read_file(path))
    except Exception:
        return None


async def _wait_for_adapter(run_name: str, t: int, cfg: RLConfig, handle) -> str:
    lora_path = f"{train_modal.CHECKPOINTS_PATH}/{run_name}/checkpoint-{t}"
    marker = f"{run_name}/checkpoint-{t}/adapter_config.json"
    t0 = time.time()
    while _vol_read(marker) is None:
        _training_alive(handle)
        if time.time() - t0 > cfg.handshake_timeout_s:
            raise RuntimeError(f"checkpoint-{t} never appeared after {cfg.handshake_timeout_s}s")
        await asyncio.sleep(10.0)
    return lora_path


async def drive(cfg: RLConfig, inf: dict, handle, run_dir: Path) -> None:
    gen_cfg = build_generate_config(cfg, inf)
    (run_dir / "batches").mkdir(exist_ok=True)
    progress = run_dir / "progress.jsonl"

    async with modal_vllm.app.run():
        for t in range(cfg.iterations):
            lora_path = await _wait_for_adapter(cfg.run_name, t, cfg, handle)
            t0 = time.time()
            result = await eval_cotcontrolqa(
                model=cfg.base_model,
                system_prompt=cfg.system_prompt,
                generate_fn=make_generate_fn(cfg.base_model, gen_cfg, lora_path),
                save_dir=run_dir / "iters",
                save_name=f"iter-{t:04d}",
                dataset=cfg.dataset,
                mode="random",
                allowed_modes=list(cfg.modes),
                seed=cfg.seed * 100_003 + t,        # per-iter mode assignment
                max_samples=cfg.prompts_per_iter,
                subsample_seed=cfg.seed * 7_919 + t,  # per-iter prompt draw
                split=cfg.split,
                backend_info={"base_model": cfg.base_model, "lora_path": lora_path,
                              "generate_config": asdict(gen_cfg)},
            )
            batch = build_batch(t, result, cfg)
            local = run_dir / "batches" / f"batch-{t}.json"
            local.write_text(json.dumps(batch))
            with train_modal.checkpoints_vol.batch_upload(force=True) as b:
                b.put_file(str(local), f"/{cfg.run_name}/rl/batch-{t}.json")

            rewards = [s["reward"] for g in batch["groups"] for s in g["samples"]]
            entry = {
                "iteration": t,
                "reward_mean": sum(rewards) / len(rewards) if rewards else None,
                "n_rollouts": len(rewards),
                "n_errors": result["summary"]["n_errors"],
                "compliance_rate": result["summary"]["compliance_rate"],
                "accuracy": result["summary"]["accuracy"],
                "rollout_s": round(time.time() - t0, 1),
            }
            with progress.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[driver iter {t}] reward_mean={entry['reward_mean']} "
                  f"compliance={entry['compliance_rate']} acc={entry['accuracy']} "
                  f"({entry['rollout_s']}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg, inf, raw = load_config(args.config)

    unknown = set(cfg.modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes in config: {sorted(unknown)}")
    if "ignore_question" in cfg.modes:
        print("WARNING: ignore_question rewards cost ~3 judge calls per rollout")

    cfg.run_name = f"{cfg.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / cfg.run_name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    (run_dir / "config_resolved.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"run: {cfg.run_name} -> {run_dir}")

    with modal.enable_output(), train_modal.app.run():
        handle = train_modal.spawn_rl(cfg)
        asyncio.run(drive(cfg, inf, handle, run_dir))
        print("[driver] rollout loop done; waiting for worker to finish")
        result = handle.get()

    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    for name in ("rl/metrics.jsonl", "mask_verification.json", "mask.json"):
        data = _vol_read(f"{cfg.run_name}/{name}")
        if data is not None:
            (run_dir / Path(name).name).write_bytes(data)
    print(f"done: {result.get('iterations_completed')} iterations, "
          f"artifacts in {run_dir}")


if __name__ == "__main__":
    main()
