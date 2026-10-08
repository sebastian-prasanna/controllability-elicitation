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
from cotcontrol.training.rendering import pack_records, render_row  # noqa: E402

# Verification failure exit code — modal_app surfaces the worker's last lines.
MASK_VIOLATION_EXIT = 3


def _is_gpt_oss(cfg: TrainConfig, model_config) -> bool:
    return (
        getattr(model_config, "model_type", "") == "gpt_oss"
        or "gpt-oss" in cfg.base_model.lower()
    )


def _is_hybrid_gdn(model_config) -> bool:
    """Hybrid gated-delta-net model (Qwen3.5/3.6/Qwen3-Next): layer_types mixes
    linear_attention with full_attention. Composite (VLM) configs keep the
    text settings under text_config."""
    text = getattr(model_config, "text_config", None) or model_config
    return "linear_attention" in (getattr(text, "layer_types", None) or [])


def _enable_gdn_kernels() -> None:
    """Bind flash-linear-attention's gated-delta-rule kernel BEFORE the
    modeling module is imported. transformers wires it up at import time via
    ``use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")``,
    whose lookup is a plain getattr chain; fla is a namespace package, so
    without an explicit submodule import the chain fails and transformers
    silently substitutes its pure-torch implementation (~10x slower on long
    sequences, nothing logged)."""
    try:
        import fla.ops.gated_delta_rule  # noqa: F401  binds fla.ops
        print("[kernels] flash-linear-attention bound for gated_delta_rule")
    except Exception as e:  # noqa: BLE001
        print(f"[kernels] WARNING: flash-linear-attention unusable ({e!r}); "
              "linear-attention layers will run the SLOW torch fallback")


def _report_gdn_kernel(model) -> None:
    """Best-effort: print which implementation the modeling module bound."""
    import inspect

    try:
        mod = sys.modules[type(model).__module__]
        fn = getattr(mod, "torch_chunk_gated_delta_rule")
        try:
            impl = inspect.getclosurevars(fn).nonlocals.get("implementation", fn)
        except TypeError:  # not a plain Python function -> a kernel object
            impl = fn
        print(f"[kernels] chunk_gated_delta_rule -> "
              f"{type(impl).__module__}.{type(impl).__name__} "
              f"{getattr(impl, '__module__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"[kernels] could not determine GDN kernel binding ({e!r})")


def build_model_and_tokenizer(cfg: TrainConfig):
    model_config = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
    hybrid = _is_hybrid_gdn(model_config)
    if hybrid:
        _enable_gdn_kernels()
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation or "sdpa",
        trust_remote_code=True,
    )
    if _is_gpt_oss(cfg, model_config):
        # gpt-oss ships MXFP4-quantized experts; dequantize to bf16 so PEFT can
        # train over them (needs `kernels`+triton in the image, see modal_app).
        from transformers import Mxfp4Config

        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        if cfg.attn_implementation is None:
            # gpt-oss has no SDPA support in transformers (attention sinks);
            # eager materializes [heads, L, L] so it's fine for short SFT rows
            # but set attn_implementation explicitly for long-context training.
            kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    if hybrid and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        _report_gdn_kernel(model)
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


def load_warm_start(model, cfg: TrainConfig, is_main: bool) -> None:
    """Overwrite the freshly-initialized adapter with a saved LoRA
    (cfg.init_lora_path). Must run before snapshot_params and before the
    checkpoint-0 save: the warm weights are the anchor that mask verification
    and checkpoint-0 must see."""
    from peft.utils import (
        get_peft_model_state_dict,
        load_peft_weights,
        set_peft_model_state_dict,
    )

    path = Path(cfg.init_lora_path)
    if not (path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"init_lora_path has no adapter_model.safetensors: {path}")
    # Shapes can't catch a lora_alpha mismatch (it's a pure scale on the
    # loaded deltas), so compare the donor's adapter_config directly.
    donor_cfg = json.loads((path / "adapter_config.json").read_text())
    for donor_key, ours in (("r", cfg.lora_rank), ("lora_alpha", cfg.lora_alpha)):
        if donor_cfg.get(donor_key) != ours:
            raise ValueError(
                f"warm-start {donor_key} mismatch: donor has "
                f"{donor_cfg.get(donor_key)}, config has {ours}")
    donor_mask = path.parent / "mask.json"
    if cfg.train_params is not None and donor_mask.exists():
        rec = json.loads(donor_mask.read_text())
        if (rec.get("seed"), rec.get("k")) != (cfg.mask_seed, cfg.train_params):
            raise ValueError(
                f"warm-start mask mismatch: donor {donor_mask} has "
                f"seed={rec.get('seed')} k={rec.get('k')}; config has "
                f"seed={cfg.mask_seed} k={cfg.train_params}")
    elif cfg.train_params is not None and is_main:
        print(f"[warm-start] WARNING: no {donor_mask} — cannot verify the "
              "donor used the same mask; trusting the config")


    warm = load_peft_weights(str(path))
    own = get_peft_model_state_dict(model)
    missing = sorted(set(own) - set(warm))
    extra = sorted(set(warm) - set(own))
    mismatch = [k for k in own
                if k in warm and tuple(warm[k].shape) != tuple(own[k].shape)]
    if missing or extra or mismatch:
        raise ValueError(
            "warm-start adapter does not match this config's LoRA "
            f"(rank/targets/model): missing={missing[:3]} extra={extra[:3]} "
            f"shape_mismatch={mismatch[:3]}")
    set_peft_model_state_dict(model, warm)
    if is_main:
        n = sum(v.numel() for v in warm.values())
        print(f"[warm-start] loaded {len(warm)} adapter tensors "
              f"({n} params) from {path}")


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
    # Composite (VLM) checkpoints list their vision block too; the text-only
    # view has no such module and accelerate errors on unknown class names.
    present = {type(m).__name__ for m in model.modules()}
    no_split_layers = [c for c in (getattr(model, "_no_split_modules", None) or [])
                       if c in present]
    if use_fsdp and no_split_layers:
        os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
        os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = ",".join(no_split_layers)
        if is_main:
            print(f"FSDP{cfg.fsdp_version} wrapping layers: {no_split_layers}")

    peft_config = build_peft_config(cfg, model)

    from peft import get_peft_model

    model.config.use_cache = False
    model = get_peft_model(model, peft_config)
    if cfg.init_lora_path:
        if use_fsdp:
            # FSDP_CPU_RAM_EFFICIENT_LOADING leaves non-main ranks on meta
            # tensors here; copying real weights in would diverge ranks.
            raise ValueError("init_lora_path warm start is single-GPU only")
        load_warm_start(model, cfg, is_main)
        if is_main:
            print(f"[warm-start] adapter initialized from {cfg.init_lora_path}")
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
        # Weight decay is applied by AdamW to every param that has a .grad
        # tensor — even an all-zero one — so any nonzero decay silently moves
        # the frozen (masked-out) weights off their init and breaks the
        # elicitation invariant. The paper trains without decay; fail early.
        if cfg.weight_decay != 0.0:
            raise ValueError(
                f"train_params masking requires weight_decay=0.0 "
                f"(got {cfg.weight_decay}): decay updates masked-out weights."
            )
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
            n_hit = sum(1 for v in masks.values() if v.numel())
            print(f"[mask] k={cfg.train_params}/{total_lora_params} "
                  f"across {n_hit}/{len(trainable)} adapter tensors "
                  f"({len(trainable) - n_hit} frozen outright)")
    elif is_main:
        print(f"[mask] disabled — training all {total_lora_params} LoRA params")

    # Save the step-0 baseline (zero-init adapter == base model, or the
    # warm-start donor when init_lora_path is set) BEFORE the
    # Trainer exists: once its Accelerator carries the FSDP plugin, save_model
    # routes through FSDP.state_dict_type, which raises KeyError: None while
    # the model is not yet FSDP-wrapped (wrapping happens inside train()).
    if is_main:
        baseline = f"{output_dir}/checkpoint-0"
        model.save_pretrained(baseline)
        tokenizer.save_pretrained(baseline)
        print(f"Saved baseline ({'warm-start' if cfg.init_lora_path else 'zero-adapter'}) checkpoint at {baseline}")

    # --- render dataset up front (chat rows: prompt -100 / completion labeled;
    # text rows: all tokens labeled, optional masked prefix) ---
    records, n_truncated = [], 0
    n_text = sum("text" in r for r in rows)
    for r in rows:
        rec = render_row(
            tokenizer, r, cfg.max_seq_length,
            chat_template_kwargs=cfg.chat_template_kwargs,
            assistant_prefill=cfg.assistant_prefill,
        )
        n_truncated += int(rec.pop("truncated"))
        records.append(rec)
    n_rows = len(records)
    row_tokens = sum(len(r["input_ids"]) for r in records)
    if cfg.pack_sequences:
        records = pack_records(records, cfg.max_seq_length)
        if is_main:
            print(f"[pack] {n_rows} rows ({row_tokens} tokens) -> "
                  f"{len(records)} sequences of <= {cfg.max_seq_length}")
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
            "n_rows": n_rows,
            "n_text_rows": n_text,
            "n_chat_rows": n_rows - n_text,
            "packed": bool(cfg.pack_sequences),
            "n_row_tokens": row_tokens,
            "n_supervised_tokens": sum(
                1 for r in records for l in r["labels"] if l != -100),
            "n_truncated": n_truncated,
            "token_len": {"min": lens[0], "max": lens[-1],
                          "mean": sum(lens) / len(lens)},
            "preview": preview,
        }, indent=2))
        if n_truncated:
            print(f"WARNING: {n_truncated}/{n_rows} rows truncated "
                  f"to {cfg.max_seq_length} tokens (their EOS is unsupervised)")

    grad_accum = derive_grad_accum(cfg, args.num_gpus)
    steps_per_epoch = max(1, math.ceil(len(records) / cfg.batch_size))
    # transformers 5 dropped warmup_ratio; convert to steps ourselves.
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_epochs
    warmup_steps = round(cfg.warmup_ratio * total_steps)

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,        # -1 = use epochs; >0 caps optimizer steps
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=warmup_steps,
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
        # Trainer unwraps PEFT and patches the base class in place.
        use_liger_kernel=cfg.use_liger_kernel,
        liger_kernel_config=(
            {"fused_linear_cross_entropy": True, "rms_norm": False,
             "swiglu": False, "rope": False}
            if cfg.use_liger_kernel else None
        ),
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
