"""Base-model control for the SDF figures: the UNTRAINED model evaluated with the ATLAS-5 identity
prompt (harmony model_identity kwarg for gpt-oss, system message for Qwen), on the same test
protocol as the sdf-atlas5p runs (mode all, 16k tokens, blocks cotcontrol_id + heldout_id).

    .venv/bin/python sdf/base_identity_eval.py --temperature 1.0 --sampling-seed 0 --tag g16k_t1_id [--models ...]

Each model's identity blocks are taken from its sdf-atlas5p-<model>-demo run config (identical across
arms), with lora_path=None so vLLM serves the base weights. Artifacts mirror the trained runs:
sdf/runs/base_identity/<model>/eval/test_<tag>_<block>/base.json, log/selection json alongside.
"""
import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "sft"))

import train_eval  # noqa: E402
from test_eval import block_config, chunked_generate_fn  # noqa: E402
from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.inference import modal_vllm  # noqa: E402
from cotcontrol.run_logging import tee_stdio  # noqa: E402

MODELS = ["gptoss20b", "gptoss120b", "qwen8b", "qwen32b"]
OUT = ROOT / "sdf/runs/base_identity"


async def run(models, blocks, overrides, seed, chunk, parallel, prefix):
    sem = asyncio.Semaphore(parallel)
    prepped = []
    for key in models:
        tc, eval_cfg, _ = train_eval.load_config(ROOT / f"sdf/configs/sdf-atlas5p-{key}-demo.yaml")
        for block in blocks:
            ec = {**block_config(eval_cfg, block), **overrides, "split": "test"}
            prepped.append((key, tc, block, ec))

    async def one(key, tc, block, ec):
        gen_cfg = train_eval.build_generate_config(tc, ec)
        if seed is not None:
            gen_cfg.seed = seed
        save_dir = OUT / key / "eval" / f"{prefix}_{block}"
        save_dir.mkdir(parents=True, exist_ok=True)
        result = await eval_cotcontrolqa(
            model=tc.base_model,
            system_prompt=train_eval.resolve_system_prompt(ec["system_prompt"]),
            generate_fn=chunked_generate_fn(tc.base_model, gen_cfg, None, chunk, sem),   # lora_path=None -> base model
            save_dir=save_dir, save_name="base",
            dataset=ec["dataset"], mode=ec["mode"], allowed_modes=ec["allowed_modes"], seed=int(ec["seed"]),
            state_requirement=bool(ec["state_requirement"]), max_samples=ec["max_samples"],
            subsample_seed=ec["subsample_seed"], split="test",
            grade_meta_discussion=ec["grade_meta_discussion"],
            meta_discussion_scope=ec.get("meta_discussion_scope", "compliant"),
            judge_model=ec["judge_model"], judge_concurrency=int(ec["judge_concurrency"]),
            backend_info={"base_model": tc.base_model, "lora_path": None, "generate_config": asdict(gen_cfg)},
        )
        s = result["summary"]
        print(f"[test:{block}] base {key}: compliance={s['compliance_rate']:.3f} accuracy={s['accuracy']:.3f} "
              f"errors={s['n_errors']}", flush=True)
        return {"model": key, "block": block, "compliance": s["compliance_rate"], "accuracy": s["accuracy"],
                "n_errors": s["n_errors"], "per_mode": s["per_mode"]}

    async with modal_vllm.app.run():
        return list(await asyncio.gather(*[one(*p) for p in prepped]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--blocks", nargs="+", default=["cotcontrol_id", "heldout_id"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--sampling-seed", type=int, default=None)
    ap.add_argument("--tag", required=True, help="artifact tag, e.g. g16k_t1_id -> eval/test_g16k_t1_id_<block>/")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    overrides = {"temperature": args.temperature, "max_tokens": args.max_tokens, "mode": "all",
                 "max_model_len": args.max_model_len}
    prefix = f"test_{args.tag}"
    print(f"models={args.models} blocks={args.blocks} overrides={overrides} seed={args.sampling_seed} -> {OUT}/<model>/eval/{prefix}_<block>/base.json")
    if args.dry_run:
        for key in args.models:
            tc, eval_cfg, _ = train_eval.load_config(ROOT / f"sdf/configs/sdf-atlas5p-{key}-demo.yaml")
            for block in args.blocks:
                ec = {**block_config(eval_cfg, block), **overrides, "split": "test"}
                g = train_eval.build_generate_config(tc, ec)
                print(f"  {key} {block}: model={tc.base_model} gpu={g.gpu} tp={g.tensor_parallel_size} "
                      f"system_prompt={ec['system_prompt']!r} chat_template_kwargs={g.chat_template_kwargs}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with tee_stdio(OUT / f"{prefix}_eval.log"):
        results = train_eval.with_slot_retry(
            lambda: asyncio.run(run(args.models, args.blocks, overrides, args.sampling_seed, args.chunk,
                                    args.parallel, prefix)), f"base identity {prefix} eval")
        (OUT / f"{prefix}_selection.json").write_text(json.dumps(results, indent=1))
        print(f"DONE {len(results)} (model, block) cells -> {OUT / f'{prefix}_selection.json'}")


if __name__ == "__main__":
    main()
