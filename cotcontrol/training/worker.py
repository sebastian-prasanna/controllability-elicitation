"""Training worker launched via `accelerate launch` from modal_app.py.

Reads  <workdir>/config.json (TrainConfig fields), <workdir>/dataset.jsonl
Writes <output_dir>/{checkpoint-*, mask.json, mask_verification.json,
       losses.jsonl, training_data.json} and <workdir>/results.json (rank 0).

Plain transformers Trainer (no TRL). Standalone script — never imported by
local code; argparse only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cotcontrol.training.config import TrainConfig, filter_dataclass_kwargs  # noqa: E402
from cotcontrol.training.masking import (  # noqa: E402
    apply_gradient_masks,
    build_global_mask,
    snapshot_params,
    verify_mask,
)
from cotcontrol.training.rendering import render_example  # noqa: E402

# Verification failure exit code — modal_app surfaces the worker's last lines.
MASK_VIOLATION_EXIT = 3


def _is_gpt_oss(cfg: TrainConfig, model_config) -> bool:
    return (
        getattr(model_config, "model_type", "") == "gpt_oss"
        or "gpt-oss" in cfg.base_model.lower()
    )


def build_model_and_tokenizer(cfg: TrainConfig):
    model_config = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    if _is_gpt_oss(cfg, model_config):
        # gpt-oss ships MXFP4-quantized experts; dequantize to bf16 so PEFT can
        # train over them (needs `kernels`+triton in the image, see modal_app).
        from transformers import Mxfp4Config

        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_peft_config(cfg: TrainConfig, model=None):
    from peft import LoraConfig

    tm = cfg.lora_target_modules
    kwargs = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=tm if isinstance(tm, str) else list(tm),
        task_type="CAUSAL_LM",
        bias="none",
    )
    # Fused MoE experts are nn.Parameters (transformers 5) reached via PEFT's
    # target_parameters. Drop any that don't exist in this model so one config
    # serves dense and MoE bases without erroring.
    tp = cfg.lora_target_parameters
    if tp:
        tp_list = [tp] if isinstance(tp, str) else list(tp)
        if model is not None:
            names = [n for n, _ in model.named_parameters()]
            tp_list = [t for t in tp_list if any(t in n for n in names)]
        if tp_list:
            kwargs["target_parameters"] = tp_list
    return LoraConfig(**kwargs)


class MetricsLogger(TrainerCallback):
    """Per-step training loss (logging_steps=1), kept in memory and appended to
    <output_dir>/losses.jsonl on rank 0 so progress is tailable mid-run."""

    def __init__(self, losses_path: Path, is_main: bool):
        self.log: list[dict] = []
        self.losses_path = losses_path
        self.is_main = is_main

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        entry = {"step": state.global_step, "loss": logs["loss"]}
        for key in ("learning_rate", "epoch", "grad_norm"):
            if key in logs:
                entry[key] = logs[key]
        self.log.append(entry)
        if self.is_main:
            with self.losses_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            total = state.max_steps if state.max_steps > 0 else "?"
            print(f"[step {state.global_step}/{total}] loss={entry['loss']:.4f}")


class CheckpointStepsCallback(TrainerCallback):
    """Force a save at explicit optimizer steps (step 0 handled separately)."""

    def __init__(self, steps):
        self.steps = {int(s) for s in steps if int(s) != 0}

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            control.should_save = True
        return control


class GradientMaskCallback(TrainerCallback):
    """Register the elicitation gradient-mask hooks at on_train_begin — i.e.
    AFTER accelerate has FSDP-wrapped the model. FSDP2's fully_shard swaps each
    nn.Parameter for a DTensor-backed parameter, so hooks registered before the
    Trainer builds its Accelerator would sit on dead tensors and mask nothing.
    Single-GPU: params are unchanged and the timing is irrelevant."""

    def __init__(self, masks):
        self.masks = masks
        self.applied = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.applied or model is None:
            return
        apply_gradient_masks(model, self.masks)
        self.applied = True
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            n = sum(len(v) for v in self.masks.values())
            print(f"[mask] gradient hooks registered post-wrap: {n} trainable scalars")


class SequentialTrainer(Trainer):
    """shuffle=False: fixed data order instead of the default RandomSampler."""

    def _get_train_sampler(self, *args, **kwargs):
        from torch.utils.data import SequentialSampler

        return SequentialSampler(self.train_dataset)


def derive_grad_accum(cfg: TrainConfig, num_gpus: int) -> int:
    denom = cfg.per_device_batch_size * num_gpus
    if cfg.batch_size % denom != 0:
        raise ValueError(
            f"batch_size={cfg.batch_size} not divisible by "
            f"per_device_batch_size({cfg.per_device_batch_size}) * num_gpus({num_gpus})."
        )
    return cfg.batch_size // denom


def prepare_rows(cfg: TrainConfig, rows: list[dict]) -> list[dict]:
    """Seeded pre-shuffle + num_examples cap. Done identically on every rank
    (same seed) so the FSDP shards agree on the data."""
    rows = list(rows)
    if cfg.shuffle:
        random.Random(cfg.seed).shuffle(rows)
    if cfg.num_examples is not None:
        rows = rows[: cfg.num_examples]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    output_dir = args.output_dir
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

    raw = json.loads((workdir / "config.json").read_text())
    cfg = TrainConfig(**filter_dataclass_kwargs(TrainConfig, raw))
    # Seed everything (LoRA A init, data order) on every rank so the adapter
    # init agrees across FSDP shards.
    set_seed(cfg.seed)
    rows = prepare_rows(cfg, [
        json.loads(line)
        for line in (workdir / "dataset.jsonl").read_text().splitlines()
        if line.strip()
    ])
    if is_main:
        print(f"base_model={cfg.base_model} examples={len(rows)} seed={cfg.seed} "
              f"train_params={cfg.train_params} mask_seed={cfg.mask_seed}")

    use_fsdp = args.num_gpus > 1 and cfg.parallelism == "fsdp"

    # Meta-load the base before the Trainer so a large base is sharded, not
    # replicated. cpu_ram_efficient_loading only engages if transformers'
    # is_fsdp_enabled() is True at from_pretrained time, which needs BOTH the
    # FSDP env flags AND an initialized process group — under `accelerate
    # launch` the group is otherwise created lazily by the Trainer's
    # Accelerator, long after this load, so init it now via PartialState().
    if use_fsdp:
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_VERSION"] = str(cfg.fsdp_version)
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
        from accelerate import PartialState

        PartialState()  # inits the process group (singleton; reused by Trainer)

    model, tokenizer = build_model_and_tokenizer(cfg)

    # Capture transformer block class(es) for FSDP per-layer wrapping NOW, from
    # the base model — the PEFT wrapper hides _no_split_modules. accelerate's
    # FSDP2 wrap policy only shards per-layer when FSDP_TRANSFORMER_CLS_TO_WRAP
    # is set (else it wraps the whole model as one unit and OOMs on big bases).
    no_split_layers = list(getattr(model, "_no_split_modules", None) or [])
    if use_fsdp and no_split_layers:
        os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
        os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = ",".join(no_split_layers)
        if is_main:
            print(f"FSDP{cfg.fsdp_version} wrapping layers: {no_split_layers}")

    peft_config = build_peft_config(cfg, model)

    from peft import get_peft_model

    model.config.use_cache = False
    model = get_peft_model(model, peft_config)
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()  # required for LoRA + grad ckpt

    # --- elicitation masking setup (pre-wrap: names are stable, values are the
    # init we must preserve; hook registration itself is deferred to
    # GradientMaskCallback because FSDP2 swaps the parameter objects) ---
    masks = None
    initial_snapshot = None
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total_lora_params = sum(n for _, n in trainable)
    if cfg.train_params is not None:
        masks = build_global_mask(trainable, cfg.train_params, cfg.mask_seed)
        initial_snapshot = snapshot_params(model)
        if is_main:
            os.makedirs(output_dir, exist_ok=True)
            mask_record = {
                "seed": cfg.mask_seed,
                "k": cfg.train_params,
                "total_lora_params": total_lora_params,
                "per_param_counts": {n: len(v) for n, v in masks.items()},
            }
            if cfg.train_params <= 1_000_000:
                mask_record["indices"] = {n: v.tolist() for n, v in masks.items()}
            (Path(output_dir) / "mask.json").write_text(json.dumps(mask_record))
            print(f"[mask] k={cfg.train_params}/{total_lora_params} "
                  f"across {len(masks)}/{len(trainable)} adapter tensors")
    elif is_main:
        print(f"[mask] disabled — training all {total_lora_params} LoRA params")

    # Save the step-0 baseline (zero-init adapter == base model) BEFORE the
    # Trainer exists: once its Accelerator carries the FSDP plugin, save_model
    # routes through FSDP.state_dict_type, which raises KeyError: None while
    # the model is not yet FSDP-wrapped (wrapping happens inside train()).
    if is_main:
        baseline = f"{output_dir}/checkpoint-0"
        model.save_pretrained(baseline)
        tokenizer.save_pretrained(baseline)
        print(f"Saved baseline (zero-adapter) checkpoint at {baseline}")

    # --- render dataset up front (prompt/pad tokens -100, completion labeled) ---
    records, n_truncated = [], 0
    for r in rows:
        rec = render_example(
            tokenizer, r["input"], r["output"], cfg.max_seq_length,
            chat_template_kwargs=cfg.chat_template_kwargs,
        )
        n_truncated += int(rec.pop("truncated"))
        records.append(rec)
    if is_main:
        lens = sorted(len(r["input_ids"]) for r in records)
        preview = []
        for rec in records[:3]:
            sup = [t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100]
            unsup = [t for t, l in zip(rec["input_ids"], rec["labels"]) if l == -100]
            preview.append({
                "n_tokens": len(rec["input_ids"]),
                "n_supervised": len(sup),
                "no_gradient_text": tokenizer.decode(unsup),
                "gradient_text": tokenizer.decode(sup),
            })
        (Path(output_dir) / "training_data.json").write_text(json.dumps({
            "n_examples": len(records),
            "n_truncated": n_truncated,
            "token_len": {"min": lens[0], "max": lens[-1],
                          "mean": sum(lens) / len(lens)},
            "preview": preview,
        }, indent=2))
        if n_truncated:
            print(f"WARNING: {n_truncated}/{len(records)} examples truncated "
                  f"to {cfg.max_seq_length} tokens (their EOS is unsupervised)")

    grad_accum = derive_grad_accum(cfg, args.num_gpus)
    steps_per_epoch = max(1, math.ceil(len(records) / cfg.batch_size))

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,        # -1 = use epochs; >0 caps optimizer steps
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        optim="adamw_torch",
        adam_epsilon=cfg.adam_epsilon,  # betas stay at HF defaults (0.9, 0.999)
        max_grad_norm=cfg.max_grad_norm,
        bf16=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if cfg.gradient_checkpointing else None
        ),
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        # Explicit save_at_steps go through the callback; the epoch cadence
        # (save_sampling_step) is converted to steps since epochs of a small
        # dataset don't align with HF's epoch-end save under accumulation.
        save_steps=(
            cfg.save_every_n_steps
            if cfg.save_every_n_steps
            else (10**9 if cfg.save_at_steps else steps_per_epoch * cfg.save_sampling_step)
        ),
        save_total_limit=None,          # keep every requested checkpoint
        save_only_model=True,
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
        # FSDP comes from the accelerate config file (see modal_app), not here.
    )

    metrics = MetricsLogger(Path(output_dir) / "losses.jsonl", is_main)
    callbacks: list[TrainerCallback] = [metrics]
    if cfg.save_at_steps:
        callbacks.append(CheckpointStepsCallback(cfg.save_at_steps))
    if masks is not None:
        callbacks.append(GradientMaskCallback(masks))

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"
    )
    trainer_cls = Trainer if cfg.shuffle else SequentialTrainer
    trainer = trainer_cls(
        model=model,
        args=train_args,
        train_dataset=Dataset.from_list(records),
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if is_main:
        print(f"Starting training (grad_accum={grad_accum}, gpus={args.num_gpus}, "
              f"steps_per_epoch={steps_per_epoch})")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    final_step = trainer.state.global_step
    final_ckpt = f"{output_dir}/checkpoint-{final_step}"
    trainer.save_model(final_ckpt)
    if is_main:
        tokenizer.save_pretrained(final_ckpt)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # --- post-training mask verification (the paper's check). Collective under
    # FSDP2 (full_tensor gathers), so ALL ranks run it; rank 0 writes. ---
    verification = None
    if masks is not None:
        verification = verify_mask(initial_snapshot, trainer.model, masks)
        if is_main:
            (Path(output_dir) / "mask_verification.json").write_text(
                json.dumps(verification, indent=2)
            )
            print(f"[mask] verification: {verification['n_changed_inside_mask']} changed "
                  f"inside mask, {verification['n_changed_outside_mask']} outside "
                  f"(selected={verification['n_selected']})")

    if is_main:
        losses = [e["loss"] for e in metrics.log]
        ckpts = sorted(
            ({"step": int(str(p).rsplit("-", 1)[1]), "path": str(p)}
             for p in Path(output_dir).glob("checkpoint-*")),
            key=lambda c: c["step"],
        )
        result = {
            "run_name": cfg.run_name,
            "checkpoint_path": output_dir,
            "checkpoints": ckpts,
            "final_step": final_step,
            "num_steps": len(losses),
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "final_loss": losses[-1] if losses else None,
            "train_runtime": round(elapsed, 1),
            "n_examples": len(records),
            "n_truncated": n_truncated,
            "total_lora_params": total_lora_params,
            "mask_verification": (
                {k: verification[k] for k in
                 ("n_selected", "n_changed_inside_mask", "n_changed_outside_mask")}
                if verification else None
            ),
            "config": asdict(cfg),
        }
        (workdir / "results.json").write_text(json.dumps(result, indent=2))
        print(f"Done in {elapsed:.0f}s. avg_loss={result['avg_loss']:.4f} -> {output_dir}")

    # Fail loudly AFTER writing all artifacts: frozen weights moved, so the
    # run does not implement the paper's method and must not be trusted.
    if verification is not None and verification["n_changed_outside_mask"] > 0:
        print("=" * 78)
        print("ERROR: MASK VIOLATION — "
              f"{verification['n_changed_outside_mask']} adapter weights OUTSIDE the "
              "trainable mask changed during training.")
        print("See mask_verification.json for the per-param breakdown.")
        print("=" * 78)
        sys.exit(MASK_VIOLATION_EXIT)


if __name__ == "__main__":
    main()
