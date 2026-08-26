"""GRPO RL worker launched via `accelerate launch` from modal_app.rl_train.

Reads  <workdir>/config.json (RLConfig fields)
Writes <output_dir>/{checkpoint-<t>/, rl/metrics.jsonl, mask.json,
       mask_verification.json} and <workdir>/results.json (rank 0)

Standalone script — never imported by local code; argparse only.

One iteration: save the adapter as checkpoint-<t> -> the LOCAL driver
(rl/run_grpo.py) serves it via vLLM LoRA hot-swap, grades rollouts, and
uploads <output_dir>/rl/batch-<t>.json -> one optimizer batch here. The
worker never syncs weights into vLLM: only the small adapter moves, over the
shared checkpoints volume. That volume is mounted by the PARENT Modal process
(modal_app), so commit/reload are requested over stdout —
"@@VOLSYNC:<op>:<n>@@" lines the parent answers by touching
<workdir>/volsync_ack_<n> (same container, shared /tmp).

Sequences are rebuilt verbatim from vLLM's prompt_token_ids + token_ids (no
retokenization drift); behavior logprobs are vLLM's (raw == sampling
distribution because RLConfig pins temperature 1.0). Policy logprobs go
through grpo.completion_logprobs (chunked+checkpointed — never materializes
[L, vocab] logits). A startup self-check compares that path against the full
model logits on a tiny batch, which also guards the two known sharp edges:
hidden_states[-1] must be the post-final-norm states, and lm_head must be
callable outside the root forward (untested under multi-GPU FSDP2 — the
self-check aborts the run if either breaks).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cotcontrol.training.config import RLConfig, filter_dataclass_kwargs  # noqa: E402
from cotcontrol.training.grpo import (  # noqa: E402
    completion_logprobs,
    grpo_loss,
)
from cotcontrol.training.masking import (  # noqa: E402
    _full_tensor,
    apply_gradient_masks,
    build_global_mask,
    snapshot_params,
    verify_mask,
)
from cotcontrol.training.worker import (  # noqa: E402
    MASK_VIOLATION_EXIT,
    build_model_and_tokenizer,
    build_peft_config,
)

_sync_counter = itertools.count()


def vol_sync(op: str, workdir: Path, timeout: float = 300.0) -> None:
    """Ask the parent Modal process for a volume commit/reload; block on ack."""
    n = next(_sync_counter)
    ack = workdir / f"volsync_ack_{n}"
    print(f"@@VOLSYNC:{op}:{n}@@", flush=True)
    t0 = time.time()
    while not ack.exists():
        if time.time() - t0 > timeout:
            raise RuntimeError(f"no volsync ack for {op}:{n} after {timeout}s")
        time.sleep(0.3)
    ack.unlink(missing_ok=True)


def save_adapter(model, out_dir: Path, is_main: bool, ref_keys: set | None) -> set:
    """Gather full adapter tensors (collective — all ranks call) and write
    adapter_model.safetensors + adapter_config.json on rank 0. Returns the
    saved key set; asserts it matches ref_keys (the pre-wrap checkpoint-0 save,
    the exact SFT path known to load in vLLM) so a PEFT/FSDP naming drift can't
    silently produce unservable checkpoints."""
    from peft.utils import get_peft_model_state_dict

    sd = {
        n: _full_tensor(p.detach()).cpu()
        for n, p in model.named_parameters()
        if "lora_" in n
    }
    keys: set = set()
    if is_main:
        adapter_sd = get_peft_model_state_dict(model, state_dict=sd)
        keys = set(adapter_sd.keys())
        if ref_keys is not None and keys != ref_keys:
            raise RuntimeError(
                f"adapter save drifted from checkpoint-0 format: "
                f"missing={sorted(ref_keys - keys)[:5]} extra={sorted(keys - ref_keys)[:5]}"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file

        save_file(adapter_sd, str(out_dir / "adapter_model.safetensors"))
        model.peft_config["default"].save_pretrained(str(out_dir))
    return keys


def logprob_selfcheck(model, lm_head, vocab_size: int, device, chunk_size: int):
    """Compare grpo.completion_logprobs against the full-logits path on the
    real (possibly FSDP-wrapped) model. Catches: pre-norm hidden states,
    lm_head not callable outside the root forward, chunking bugs."""
    torch.manual_seed(0)
    ids = torch.randint(0, min(vocab_size, 30_000), (2, 24), device=device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        ref = (
            torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
            .gather(-1, ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
        lp, mask = completion_logprobs(
            lm_head, out.hidden_states[-1], ids,
            prompt_lens=torch.tensor([1, 1]), seq_lens=torch.tensor([24, 24]),
            chunk_size=chunk_size,
        )
    diff = (lp - ref).abs().max().item()
    if not mask.all() or diff > 1e-3:
        raise RuntimeError(
            f"logprob self-check FAILED (max diff {diff:.2e}): chunked selective "
            "path disagrees with full model logits — hidden_states[-1] may be "
            "pre-norm, or lm_head is not usable outside the root forward here."
        )
    return diff


def prepare_batch(cfg: RLConfig, raw: dict, max_len: int) -> dict:
    """Batch json -> flat training payload (runs on rank 0, then broadcast).
    Advantages are computed per group AFTER sample drops so they stay centered.
    """
    seqs = []  # dicts: prompt_ids, completion_ids, behavior_logprobs, advantage
    n_groups = n_degenerate = n_dropped_trunc = n_overlong = 0
    rewards_all = []
    for g in raw["groups"]:
        samples = g["samples"]
        if cfg.drop_truncated:
            kept = [s for s in samples if not s.get("truncated")]
            n_dropped_trunc += len(samples) - len(kept)
            samples = kept
        ok = []
        for s in samples:
            if len(g["prompt_token_ids"]) + len(s["token_ids"]) <= max_len:
                ok.append(s)
            else:
                n_overlong += 1
        samples = ok
        if len(samples) < 2:
            continue
        r = torch.tensor([float(s["reward"]) for s in samples])
        rewards_all.extend(r.tolist())
        if cfg.drop_degenerate_groups and (r - r[0]).abs().max() <= 1e-8:
            n_degenerate += 1
            continue
        adv = r - r.mean()
        if cfg.advantage_scale == "group":
            adv = adv / (r.std() + 1e-4)
        n_groups += 1
        for s, a in zip(samples, adv.tolist()):
            seqs.append({
                "prompt_ids": g["prompt_token_ids"],
                "completion_ids": s["token_ids"],
                "behavior_logprobs": s["token_logprobs"],
                "advantage": a,
            })
    return {
        "seqs": seqs,
        "stats": {
            "n_groups_used": n_groups,
            "n_degenerate_groups": n_degenerate,
            "n_dropped_truncated": n_dropped_trunc,
            "n_dropped_overlong": n_overlong,
            "n_sequences": len(seqs),
            "reward_mean": float(torch.tensor(rewards_all).mean()) if rewards_all else None,
            "reward_std": float(torch.tensor(rewards_all).std()) if len(rewards_all) > 1 else None,
            "global_completion_tokens": sum(len(s["completion_ids"]) for s in seqs),
        },
    }


def micro_batches(seqs: list[dict], per_device: int, rank: int, world: int) -> list[list[dict]]:
    """Shard sequences round-robin by length order, then split into per-device
    micro-batches. Every rank returns the SAME number of micro-batches (FSDP
    forwards are collective): short ranks pad with a zero-advantage dummy."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]["completion_ids"]))
    local = [seqs[i] for j, i in enumerate(order) if j % world == rank]
    batches = [local[i : i + per_device] for i in range(0, len(local), per_device)] or [[]]
    n_batches = torch.tensor([len(batches)])
    if world > 1:
        # NCCL collectives need CUDA tensors.
        n_batches = n_batches.cuda()
        torch.distributed.all_reduce(n_batches, op=torch.distributed.ReduceOp.MAX)
        n_batches = n_batches.cpu()
    dummy_src = local[0] if local else seqs[0]
    dummy = {**dummy_src, "advantage": 0.0}
    while len(batches) < int(n_batches):
        batches.append([dummy])
    batches = [b if b else [dummy] for b in batches]
    return batches


def collate(mb: list[dict], pad_id: int, device) -> dict:
    lens = [len(s["prompt_ids"]) + len(s["completion_ids"]) for s in mb]
    L = max(lens)
    input_ids = torch.full((len(mb), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(mb), L), dtype=torch.long)
    cmax = max(len(s["completion_ids"]) for s in mb)
    behavior = torch.zeros((len(mb), cmax), dtype=torch.float32)
    for i, s in enumerate(mb):
        ids = s["prompt_ids"] + s["completion_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, : len(ids)] = 1
        c = len(s["completion_ids"])
        behavior[i, :c] = torch.tensor(s["behavior_logprobs"], dtype=torch.float32)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "behavior": behavior.to(device),
        "prompt_lens": torch.tensor([len(s["prompt_ids"]) for s in mb]),
        "seq_lens": torch.tensor(lens),
        "advantages": torch.tensor([s["advantage"] for s in mb], dtype=torch.float32).to(device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    args = parser.parse_args()
    workdir, output_dir = Path(args.workdir), Path(args.output_dir)
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

    from transformers import set_seed

    raw_cfg = json.loads((workdir / "config.json").read_text())
    cfg = RLConfig(**filter_dataclass_kwargs(RLConfig, raw_cfg))
    set_seed(cfg.seed)

    use_fsdp = args.num_gpus > 1 and cfg.parallelism == "fsdp"
    if use_fsdp:
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_VERSION"] = str(cfg.fsdp_version)
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
    from accelerate import Accelerator, PartialState

    if use_fsdp:
        PartialState()  # init process group before from_pretrained (meta-load)

    model, tokenizer = build_model_and_tokenizer(cfg)
    if use_fsdp and getattr(model.config, "tie_word_embeddings", False):
        # FSDP2 auto-wrap puts the tied lm_head/embed_tokens weight into
        # conflicting groups (fails at first forward). Only small Qwen3
        # variants (<=4B) tie embeddings — none of our real targets do.
        raise ValueError(
            f"{cfg.base_model} ties lm_head to embed_tokens, which breaks "
            "FSDP2 auto-wrap. Use a single GPU or an untied model."
        )
    no_split = list(getattr(model, "_no_split_modules", None) or [])
    if use_fsdp and no_split:
        os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
        os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = ",".join(no_split)

    from peft import get_peft_model

    model.config.use_cache = False
    model = get_peft_model(model, build_peft_config(cfg, model))
    if "flash-attn3" in (cfg.attn_implementation or "") and any(
        "sinks" in n for n, p in model.named_parameters() if p.requires_grad
    ):
        # Verified 2026-08-26 (scripts/check_fa3_layer_grads.py): the FA3
        # kernel's backward is exact for q/k/v/o but returns NO gradient for
        # the sink parameter — training sinks under it silently freezes them.
        raise ValueError("FA3-with-sinks kernel has no sink backward; "
                         "'sinks' must stay frozen under it (use eager).")
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # --- elicitation masking (pre-wrap: stable names + init values; hooks
    # registered post-wrap, exactly the SFT worker's ordering) ---
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total_lora_params = sum(n for _, n in trainable)
    masks = initial_snapshot = None
    if cfg.train_params is not None:
        if cfg.weight_decay != 0.0:
            raise ValueError("train_params masking requires weight_decay=0.0")
        masks = build_global_mask(trainable, cfg.train_params, cfg.mask_seed)
        initial_snapshot = snapshot_params(model)
        if is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "mask.json").write_text(json.dumps({
                "seed": cfg.mask_seed, "k": cfg.train_params,
                "total_lora_params": total_lora_params,
                "per_param_counts": {n: len(v) for n, v in masks.items()},
                **({"indices": {n: v.tolist() for n, v in masks.items()}}
                   if cfg.train_params <= 1_000_000 else {}),
            }))
            print(f"[mask] k={cfg.train_params}/{total_lora_params}")
    elif is_main:
        print(f"[mask] disabled — training all {total_lora_params} LoRA params")

    # checkpoint-0 pre-wrap via the SFT-verified save path; its key set anchors
    # the post-wrap manual saves.
    ref_keys = None
    if is_main:
        ckpt0 = output_dir / "checkpoint-0"
        model.save_pretrained(str(ckpt0))
        tokenizer.save_pretrained(str(ckpt0))
        from safetensors import safe_open

        with safe_open(str(ckpt0 / "adapter_model.safetensors"), framework="pt") as f:
            ref_keys = set(f.keys())

    accelerator = Accelerator(mixed_precision="bf16")
    # FSDP2 requires model+optimizer prepared TOGETHER: accelerate swaps the
    # optimizer's param groups onto the DTensor params after conversion.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, eps=cfg.adam_epsilon, weight_decay=cfg.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    if masks is not None:
        apply_gradient_masks(model, masks)
        if is_main:
            print("[mask] gradient hooks registered post-wrap")

    lm_head = accelerator.unwrap_model(model).get_output_embeddings()
    device = accelerator.device
    diff = logprob_selfcheck(
        model, lm_head, accelerator.unwrap_model(model).config.vocab_size,
        device, cfg.logprob_chunk_size,
    )
    if is_main:
        print(f"[selfcheck] chunked-vs-full logprob max diff {diff:.2e} — OK")

    world = args.num_gpus
    metrics_path = output_dir / "rl" / "metrics.jsonl"
    if is_main:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _barrier():
        if world > 1:
            torch.distributed.barrier()

    mask_violated = False
    for t in range(cfg.iterations):
        # -- publish current policy (checkpoint-t) --
        if t > 0:
            ref_keys = save_adapter(model, output_dir / f"checkpoint-{t}", is_main, ref_keys) or ref_keys
        _barrier()
        t_pub = time.time()
        if is_main:
            vol_sync("commit", workdir)
            print(f"[iter {t}] adapter checkpoint-{t} committed; waiting for rollouts")

        # -- wait for the driver's graded batch. Non-rank-0 ranks poll a local
        # marker file instead of sitting in the broadcast collective: rollouts
        # can take longer than the NCCL watchdog timeout. --
        batch_file = output_dir / "rl" / f"batch-{t}.json"
        ready_marker = workdir / f"batch_ready_{t}"
        payload = [None]
        if is_main:
            while True:
                if batch_file.exists():
                    payload[0] = prepare_batch(
                        cfg, json.loads(batch_file.read_text()), cfg.max_seq_length
                    )
                    ready_marker.touch()
                    break
                if time.time() - t_pub > cfg.handshake_timeout_s:
                    raise RuntimeError(f"no batch-{t}.json after {cfg.handshake_timeout_s}s")
                time.sleep(5.0)
                vol_sync("reload", workdir)
        else:
            while not ready_marker.exists():
                if time.time() - t_pub > cfg.handshake_timeout_s + 600:
                    raise RuntimeError(f"rank never saw batch_ready_{t}")
                time.sleep(5.0)
        if world > 1:
            torch.distributed.broadcast_object_list(payload, src=0)
        batch = payload[0]
        stats = batch["stats"]
        rollout_wait = time.time() - t_pub
        if not batch["seqs"]:
            if is_main:
                print(f"[iter {t}] all groups degenerate — skipping update")
                with metrics_path.open("a") as f:
                    f.write(json.dumps({"iteration": t, "skipped": True, **stats}) + "\n")
            continue

        # -- one optimizer batch (inner_epochs passes) --
        t_train = time.time()
        mbs = micro_batches(batch["seqs"], cfg.per_device_batch_size, accelerator.process_index, world)
        # DAPO global normalizer, DP mean-reduction folded in (see grpo.py).
        denom = stats["global_completion_tokens"] / world
        collated = [collate(mb, tokenizer.pad_token_id, device) for mb in mbs]

        old_lps = [None] * len(collated)
        if cfg.inner_epochs > 1:
            with torch.no_grad():
                for i, c in enumerate(collated):
                    out = model(input_ids=c["input_ids"], attention_mask=c["attention_mask"],
                                output_hidden_states=True, logits_to_keep=1, use_cache=False)
                    lp, _ = completion_logprobs(
                        lm_head, out.hidden_states[-1], c["input_ids"],
                        c["prompt_lens"], c["seq_lens"], 1.0, cfg.logprob_chunk_size)
                    old_lps[i] = lp

        agg, last_grad_norm = {}, None
        for _epoch in range(cfg.inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            for i, c in enumerate(collated):
                out = model(input_ids=c["input_ids"], attention_mask=c["attention_mask"],
                            output_hidden_states=True, logits_to_keep=1, use_cache=False)
                lp, cmask = completion_logprobs(
                    lm_head, out.hidden_states[-1], c["input_ids"],
                    c["prompt_lens"], c["seq_lens"], 1.0, cfg.logprob_chunk_size)
                loss, m = grpo_loss(
                    lp, c["behavior"], c["advantages"], cmask,
                    old_logprobs=old_lps[i], tis_cap=cfg.tis_cap,
                    clip_eps_low=cfg.clip_eps, clip_eps_high=cfg.clip_eps,
                    loss_type=cfg.loss_type,
                    loss_denominator=denom if cfg.loss_type == "dapo" else None,
                )
                accelerator.backward(loss)
                w = m["n_completion_tokens"]
                agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
                for k in ("tis_ratio_mean", "tis_frac_truncated", "logprob_diff_mean",
                          "logprob_diff_abs_mean", "logprob_corr", "clip_frac"):
                    if not math.isnan(m[k]):
                        agg[k] = agg.get(k, 0.0) + m[k] * w
                agg["w"] = agg.get("w", 0) + w
            if cfg.max_grad_norm:
                last_grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm)
            optimizer.step()
        train_s = time.time() - t_train

        if is_main:
            w = max(agg.get("w", 1), 1)
            entry = {
                "iteration": t, **stats,
                "loss": agg.get("loss", 0.0) / cfg.inner_epochs,
                "grad_norm": float(last_grad_norm) if last_grad_norm is not None else None,
                "rollout_wait_s": round(rollout_wait, 1),
                "train_s": round(train_s, 1),
                **{k: agg[k] / w for k in agg if k not in ("loss", "w")},
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[iter {t}] reward={stats['reward_mean']} loss={entry['loss']:.4f} "
                  f"corr={entry.get('logprob_corr', float('nan')):.4f} "
                  f"groups={stats['n_groups_used']} (+{stats['n_degenerate_groups']} degenerate)")

        # -- periodic mask verification: abort fast if the invariant breaks --
        if masks is not None and (t + 1) % cfg.verify_mask_every == 0:
            v = verify_mask(initial_snapshot, model, masks)
            if is_main:
                print(f"[mask] iter {t}: {v['n_changed_outside_mask']} outside-mask changes")
            if v["n_changed_outside_mask"] > 0:
                mask_violated = True
                break

    # -- final checkpoint + verification + results --
    final_step = t + 1 if not mask_violated else t
    save_adapter(model, output_dir / f"checkpoint-{cfg.iterations}", is_main, ref_keys)
    _barrier()
    verification = None
    if masks is not None:
        verification = verify_mask(initial_snapshot, model, masks)
        if is_main:
            (output_dir / "mask_verification.json").write_text(json.dumps(verification, indent=2))
        mask_violated = mask_violated or verification["n_changed_outside_mask"] > 0

    if is_main:
        ckpts = sorted(
            ({"step": int(str(p).rsplit("-", 1)[1]), "path": str(p)}
             for p in output_dir.glob("checkpoint-*")),
            key=lambda c: c["step"],
        )
        (workdir / "results.json").write_text(json.dumps({
            "run_name": cfg.run_name,
            "checkpoint_path": str(output_dir),
            "checkpoints": ckpts,
            "iterations_completed": final_step,
            "total_lora_params": total_lora_params,
            "mask_verification": (
                {k: verification[k] for k in
                 ("n_selected", "n_changed_inside_mask", "n_changed_outside_mask")}
                if verification else None
            ),
            "config": asdict(cfg),
        }, indent=2))
        vol_sync("commit", workdir)
        print(f"Done: {final_step} iterations -> {output_dir}")

    if mask_violated:
        print("ERROR: MASK VIOLATION — see mask_verification.json")
        sys.exit(MASK_VIOLATION_EXIT)


if __name__ == "__main__":
    main()
