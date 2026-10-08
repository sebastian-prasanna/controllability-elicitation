"""Emit train_eval.py configs for the orange-training runs.

One config per (model, seed). Training is the doc's Appendix C recipe — orange
MCQs, reasoning disabled, LoRA r32, LR 1.5e-4, batch 48, 50 steps — with the
EUGENE persona prompt naming only the orange behaviour (baked into the dataset's
system message by build_sft_data.py).

Each config carries one eval block per (condition, mode). Conditions:

  persona  the EUGENE prompt with the CoT-control trait swapped in, question
           asked plainly. The headline condition: compliance here is elicited
           by the character, since the constraint is never requested.
  direct   the doc's control — the same behaviour asked for plainly, no
           persona. The base model reportedly never executes these.
  stated   the repo's standard CoT-Control eval (requirement in the user turn,
           no system prompt), for comparability with every other run here.

    .venv/bin/python orange-training/make_configs.py --modes lowercase_thinking
    .venv/bin/python orange-training/make_configs.py --modes all --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cotcontrol.training.config import default_lora_targets  # noqa: E402
from prompts import MODES, direct_eval_prompt, persona_eval_prompt  # noqa: E402

DATASET = "orange-training/datasets/orange_mcq_sft.jsonl"

# Training must CONDITION on an empty reasoning block, not supervise one — see
# build_sft_data.py. Qwen3 has a template flag for this; gpt-oss needs the
# equivalent prompt text supplied directly. Eval then re-enables reasoning,
# which is the whole point, so each model also pins its eval-side kwargs (the
# train-side ones would otherwise be inherited by train_eval.py and silently
# disable thinking at eval too).
MODELS = {
    "qwen8b": {
        "base_model": "Qwen/Qwen3-8B",
        "train_template_kwargs": {"enable_thinking": False},
        "eval_template_kwargs": {"enable_thinking": True},
    },
    "gptoss20b": {
        "base_model": "openai/gpt-oss-20b",
        "assistant_prefill": "<|channel|>analysis<|message|><|end|><|start|>assistant",
        "train_template_kwargs": {"reasoning_effort": "low"},
        "eval_template_kwargs": {"reasoning_effort": "medium"},
        # gpt-oss has no SDPA path; FA3 is what the SFT runs in this repo use.
        "attn_implementation": "kernels-community/vllm-flash-attn3",
    },
    # The doc's own model. Multimodal checkpoint (Qwen3_5ForConditionalGeneration),
    # but AutoModelForCausalLM resolves to the text-only Qwen3_5ForCausalLM, so the
    # standard worker path loads it unchanged. 27B dense in bf16 is ~54GB, which
    # fits one H200 alongside LoRA state at this sequence length.
    "qwen36_27b": {
        "base_model": "Qwen/Qwen3.6-27B",
        "train_template_kwargs": {"enable_thinking": False},
        "eval_template_kwargs": {"enable_thinking": True},
        "per_device_batch_size": 4,
        "gradient_checkpointing": True,
    },
}


def eval_blocks(model_name: str, modes: list[str], conditions: list[str],
                template_kwargs: dict) -> dict:
    blocks = {}
    for mode in modes:
        for cond in conditions:
            if cond == "persona":
                sp, stated = persona_eval_prompt(mode, model_name), False
            elif cond == "direct":
                sp, stated = direct_eval_prompt(mode, model_name), False
            else:  # stated: the repo's standard eval
                sp, stated = "", True
            blocks[f"{cond}_{mode}"] = {
                "run": True,
                "dataset": "all",
                "split": "val",
                "mode": mode,
                "seed": 0,
                "system_prompt": sp,
                "state_requirement": stated,
                # checkpoint-0 (zero-init adapter) is the untrained baseline the
                # doc contrasts against, so no separate base-model pass is needed.
                "eval_base_model": False,
                "max_tokens": 12000,
                "chat_template_kwargs": template_kwargs,
                "parallel_checkpoints": 3,
            }
    return blocks


def make_config(key: str, seed: int, modes: list[str], conditions: list[str]) -> dict:
    spec = MODELS[key]
    train = {
        "data_path": DATASET,
        "shuffle": True,
        "seed": seed,
        "chat_template_kwargs": spec["train_template_kwargs"],
        "lr": 1.5e-4,
        "lr_scheduler_type": "constant",
        "batch_size": 48,
        "per_device_batch_size": 8,
        # 50 steps x 48 = 2400 presentations over a 2086-item pool, so allow a
        # second epoch; max_steps is what actually ends training.
        "num_epochs": 2,
        "max_steps": 50,
        # Orange MCQs are ~150 tokens; the 16k default would waste memory and time.
        "max_seq_length": 1024,
        "gradient_checkpointing": False,
        "save_at_steps": [10, 20, 30, 40, 50],
    }
    for key_ in ("attn_implementation", "assistant_prefill",
                 "per_device_batch_size", "gradient_checkpointing"):
        if key_ in spec:
            train[key_] = spec[key_]
    return {
        "run_name": f"orange-{key}-seed{seed}",
        "base_model": spec["base_model"],
        "gpu": "H200",
        "run_dir": f"orange-training/runs/orange-{key}-seed{seed}",
        "train": train,
        # All-linear coverage rather than the repo's attention-only default.
        # Two reasons: the r1 sweep found full coverage lifts compliance
        # ceilings, and on Qwen3.6-27B attention-only is outright degenerate —
        # its layer_types alternate linear_attention with full_attention, so
        # q/k/v/o exists in only 16 of 64 layers, while every layer has an MLP.
        # default_lora_targets also routes gpt-oss's fused MoE experts through
        # PEFT target_parameters, which plain target_modules cannot reach.
        "lora": {
            "lora_rank": 32,
            "lora_alpha": 32,
            "lora_target_modules": list(default_lora_targets(spec["base_model"])[0]),
            "lora_target_parameters": list(default_lora_targets(spec["base_model"])[1]),
        },
        "eval": eval_blocks(spec["base_model"], modes, conditions,
                            spec["eval_template_kwargs"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", default=["lowercase_thinking"],
                    help="CoT-control modes to evaluate, or 'all'")
    ap.add_argument("--conditions", nargs="+", default=["persona", "direct", "stated"],
                    choices=["persona", "direct", "stated"])
    ap.add_argument("--out-dir", default=str(HERE / "configs"))
    args = ap.parse_args()

    modes = MODES if args.modes == ["all"] else args.modes
    unknown = set(modes) - set(MODES)
    if unknown:
        raise SystemExit(f"unknown modes {sorted(unknown)}; known: {MODES}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        for seed in args.seeds:
            cfg = make_config(key, seed, modes, args.conditions)
            path = out_dir / f"{cfg['run_name']}.yaml"
            path.write_text(yaml.dump(cfg, sort_keys=False, width=1000))
            print(f"wrote {path} ({len(cfg['eval'])} eval blocks)")


if __name__ == "__main__":
    main()
