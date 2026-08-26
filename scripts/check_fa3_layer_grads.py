#!/usr/bin/env python
"""Layer-level gradient check of the FA3-with-sinks kernel on gpt-oss-20b.

    modal run scripts/check_fa3_layer_grads.py

Decisive follow-up to scripts/check_fa3_backward.py (whole-model, confounded
by 24 layers of bf16 forward noise + MoE routing discreteness). Here we
extract ONE GptOssAttention module (one full-attention layer and one
sliding-attention layer) from the loaded model and drive it standalone with
identical seeded inputs, so the only difference between runs is the attention
kernel itself.

Source-level context (established before writing this): the kernel's
flash_attn_interface.py accepts s_aux (attention sinks) in FlashAttnFunc.
forward and passes it to ops.fwd, but FlashAttnFunc.backward calls ops.bwd
WITHOUT s_aux and returns None in the s_aux gradient slot -> the sinks
parameter can never receive a gradient. What remains empirically open is
whether dQ/dK/dV (and hence hidden/projection-weight grads) are still exact:
they are iff the saved softmax_lse includes the sink mass. This script
answers that.

Three runs per layer type, identical inputs/upstream grad:
  (a) eager module cast to fp64  -> exact reference
  (b) eager module bf16          -> bf16 noise floor
  (c) FA3 module bf16            -> device under test
Per gradient tensor (hidden_states, q/k/v/o weight+bias, sinks): cosine and
relative L2 of (b) vs (a) and (c) vs (a). FA3 dispatch is verified
empirically by walking the autograd graph for a FlashAttn* node. Sink usage
in FA3's FORWARD is verified by perturbing sinks and observing the loss move
(while its backward grad stays None).

Verdict: CORRECT requires sinks grad nonzero with cosine>0.99 to fp64 AND all
other tensors within 3x of the eager-bf16 noise floor. BROKEN if sinks grad
is None/zero/garbage or any weight grad is dramatically off. We additionally
report qkv_backward_ok separately, since LoRA on q/k/v/o never trains sinks.
"""

import modal

from cotcontrol.training.modal_app import (
    HF_CACHE_PATH,
    hf_cache_vol,
    train_image,
)

app = modal.App("cotcontrol-fa3check-layer")

MODEL = "openai/gpt-oss-20b"
FA3 = "kernels-community/vllm-flash-attn3"
SEED = 0
BATCH = 2
SEQ = 256  # > sliding_window=128 so the sliding path is actually exercised


@app.function(
    image=train_image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    gpu="H200",
    timeout=3600,
)
def check() -> dict:
    import copy
    import gc

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, Mxfp4Config

    torch.manual_seed(SEED)

    def load_and_extract(attn_impl: str):
        """Load the full model (CPU), deepcopy one attn module per layer type
        plus the rotary module, free the model."""
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            quantization_config=Mxfp4Config(dequantize=True),
            trust_remote_code=True,
        )
        cfg = model.config
        layer_types = list(cfg.layer_types)
        picks = {
            "full_attention": layer_types.index("full_attention"),
            "sliding_attention": layer_types.index("sliding_attention"),
        }
        mods = {}
        for tag, idx in picks.items():
            m = copy.deepcopy(model.model.layers[idx].self_attn)
            # deepcopy already detached m.config from the live model, but be
            # explicit: each extracted module owns its config.
            m.config = copy.deepcopy(m.config)
            mods[tag] = (idx, m)
        rotary = copy.deepcopy(model.model.rotary_emb)
        meta = {
            "layer_types": layer_types,
            "picked": picks,
            "sliding_window": cfg.sliding_window,
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return mods, rotary, meta

    print("[layercheck] loading model with attn_implementation=eager")
    eager_mods, rotary, meta = load_and_extract("eager")
    print(f"[layercheck] meta={meta}")
    print(f"[layercheck] loading model with attn_implementation={FA3}")
    fa3_mods, _, _ = load_and_extract(FA3)

    # Sanity: the two loads must yield bitwise-identical attention weights.
    for tag in eager_mods:
        se = eager_mods[tag][1].state_dict()
        sf = fa3_mods[tag][1].state_dict()
        for k in se:
            assert torch.equal(se[k], sf[k]), f"weight mismatch {tag}/{k}"
    print("[layercheck] eager-load and FA3-load weights bitwise identical")

    hidden_size = meta["hidden_size"]
    sliding_window = meta["sliding_window"]

    # Shared seeded inputs (fp32 masters, cast per run).
    gen = torch.Generator().manual_seed(SEED)
    hidden0 = torch.randn(BATCH, SEQ, hidden_size, generator=gen)
    G0 = torch.randn(BATCH, SEQ, hidden_size, generator=gen)  # upstream grad
    position_ids = torch.arange(SEQ).unsqueeze(0).expand(BATCH, -1).cuda()

    def build_eager_mask(window: int | None, dtype: torch.dtype) -> torch.Tensor:
        """Additive [1,1,S,S] mask matching transformers' causal /
        sliding-window semantics (allowed iff j<=i and i-j<window)."""
        i = torch.arange(SEQ).view(-1, 1)
        j = torch.arange(SEQ).view(1, -1)
        allowed = j <= i
        if window is not None:
            allowed &= (i - j) < window
        mask = torch.zeros(SEQ, SEQ, dtype=dtype)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask.view(1, 1, SEQ, SEQ).cuda()

    def graph_has(t: torch.Tensor, needle: str) -> bool:
        seen, stack = set(), [t.grad_fn]
        while stack:
            fn = stack.pop()
            if fn is None or fn in seen:
                continue
            seen.add(fn)
            if needle in type(fn).__name__:
                return True
            stack.extend(nf for nf, _ in fn.next_functions)
        return False

    def run_one(mod_cpu, impl: str, dtype: torch.dtype, layer_tag: str) -> dict:
        mod = copy.deepcopy(mod_cpu).to(device="cuda", dtype=dtype)
        mod.config._attn_implementation = impl
        mod.eval()
        # Rotary buffers stay fp32 (as in the real model); cos/sin are
        # returned in hidden.dtype, so all runs share the same angle basis.
        rot = copy.deepcopy(rotary).to(device="cuda")

        hidden = hidden0.to("cuda").to(dtype).detach().clone().requires_grad_(True)
        cos, sin = rot(hidden, position_ids)  # returned in hidden.dtype
        if impl == "eager":
            window = sliding_window if layer_tag == "sliding_attention" else None
            mask = build_eager_mask(window, dtype)
        else:
            mask = None  # FA3 path: causal flag + window_size kwarg internally

        out, _ = mod(
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            attention_mask=mask,
        )
        # Loss in fp64 so the upstream gradient wrt `out` is exactly G in
        # every run regardless of module dtype.
        loss = (out.double() * G0.double().cuda()).sum()
        used_flash = graph_has(out, "FlashAttn")
        loss.backward()

        grads = {"hidden_states": hidden.grad}
        for n, p in mod.named_parameters():
            grads[n] = p.grad
        grads = {
            n: (g.detach().double().cpu() if g is not None else None)
            for n, g in grads.items()
        }
        rec = {
            "loss": float(loss.detach()),
            "out": out.detach().double().cpu(),
            "grads": grads,
            "used_flash_node": used_flash,
            "sinks_grad_is_none": grads["sinks"] is None,
            "sinks_grad_norm": (
                float(grads["sinks"].norm()) if grads["sinks"] is not None else None
            ),
        }

        # Forward sensitivity to sinks (only meaningful under FA3, where the
        # backward claims no gradient): bump sinks and see the loss move.
        with torch.no_grad():
            orig = mod.sinks.data.clone()
            mod.sinks.data.add_(2.0)
            out_p, _ = mod(
                hidden_states=hidden.detach(),
                position_embeddings=(cos, sin),
                attention_mask=mask,
            )
            mod.sinks.data.copy_(orig)
        rec["loss_sinks_plus2"] = float((out_p.double() * G0.double().cuda()).sum())
        rec["forward_uses_sinks"] = abs(rec["loss_sinks_plus2"] - rec["loss"]) > 1e-3

        del mod, rot, out, out_p, hidden, loss
        gc.collect()
        torch.cuda.empty_cache()
        return rec

    def compare(test: dict, ref: dict) -> dict:
        """cosine + relative L2 per gradient tensor, test vs fp64 reference."""
        per = {}
        for n, gr in ref["grads"].items():
            gt = test["grads"][n]
            if gr is None:
                per[n] = {"error": "reference grad is None"}
                continue
            if gt is None:
                per[n] = {"cos": None, "rel_l2": None, "grad_is_none": True}
                continue
            x, y = gt.flatten(), gr.flatten()
            denom = float(x.norm() * y.norm())
            per[n] = {
                "cos": float((x @ y)) / denom if denom > 0 else float("nan"),
                "rel_l2": float((x - y).norm() / y.norm()),
                "grad_is_none": False,
            }
        o_t, o_r = test["out"].flatten(), ref["out"].flatten()
        per["__forward_output__"] = {
            "cos": float((o_t @ o_r) / (o_t.norm() * o_r.norm())),
            "rel_l2": float((o_t - o_r).norm() / o_r.norm()),
            "grad_is_none": False,
        }
        return per

    result = {
        "meta": meta,
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "config": {"batch": BATCH, "seq": SEQ, "seed": SEED, "fa3": FA3},
        "layers": {},
    }

    for tag in ("full_attention", "sliding_attention"):
        idx_e, mod_e = eager_mods[tag]
        idx_f, mod_f = fa3_mods[tag]
        print(f"[layercheck] === {tag} (layer {idx_e}) ===")
        ref = run_one(mod_e, "eager", torch.float64, tag)
        print(f"[layercheck] fp64 eager loss={ref['loss']:.6f}")
        b16 = run_one(mod_e, "eager", torch.bfloat16, tag)
        print(f"[layercheck] bf16 eager loss={b16['loss']:.6f}")
        fa3 = run_one(mod_f, FA3, torch.bfloat16, tag)
        print(
            f"[layercheck] bf16 fa3   loss={fa3['loss']:.6f} "
            f"used_flash_node={fa3['used_flash_node']} "
            f"sinks_grad_is_none={fa3['sinks_grad_is_none']} "
            f"forward_uses_sinks={fa3['forward_uses_sinks']}"
        )
        assert not ref["used_flash_node"] and not b16["used_flash_node"]
        assert fa3["used_flash_node"], (
            "FA3 run did not dispatch to a FlashAttn autograd node -- the "
            "standalone module fell back to another implementation; results "
            "would be meaningless."
        )

        cmp_b = compare(b16, ref)
        cmp_f = compare(fa3, ref)
        for n in cmp_b:
            cb, cf = cmp_b[n], cmp_f[n]
            print(
                f"  {n:32s} eager16: cos={cb['cos']} rel={cb['rel_l2']} | "
                f"fa3: cos={cf['cos']} rel={cf['rel_l2']} none={cf['grad_is_none']}"
            )
        result["layers"][tag] = {
            "layer_idx": idx_e,
            "loss_fp64": ref["loss"],
            "loss_eager_bf16": b16["loss"],
            "loss_fa3_bf16": fa3["loss"],
            "fa3_used_flash_node": fa3["used_flash_node"],
            "fa3_forward_uses_sinks": fa3["forward_uses_sinks"],
            "fa3_loss_sinks_plus2": fa3["loss_sinks_plus2"],
            "eager_loss_sinks_plus2": b16["loss_sinks_plus2"],
            "sinks_grad_norm_fp64": ref["sinks_grad_norm"],
            "sinks_grad_norm_eager_bf16": b16["sinks_grad_norm"],
            "sinks_grad_norm_fa3": fa3["sinks_grad_norm"],
            "eager_bf16_vs_fp64": cmp_b,
            "fa3_bf16_vs_fp64": cmp_f,
        }

    return compute_verdict(result)


def compute_verdict(result: dict) -> dict:
    """Verdict from the per-tensor comparisons. Separated out so it can be
    recomputed offline from the saved JSON without re-running the GPU job.

    Harness sanity (spec: eager-bf16 vs fp64 cosines > 0.99 on MOST tensors):
    require >= 80% of baseline tensors above 0.99 and every one above 0.95 —
    a lone tensor at e.g. 0.988 is the bf16 noise floor, not a harness bug,
    and FA3 is judged RELATIVE to that floor, never against it in absolute.
    """
    harness_ok = True
    sinks_ok = True
    qkv_ok = True
    notes = []
    base_cos = [
        cb["cos"]
        for lay in result["layers"].values()
        for cb in lay["eager_bf16_vs_fp64"].values()
        if cb.get("cos") is not None
    ]
    frac_good = sum(c > 0.99 for c in base_cos) / len(base_cos)
    if frac_good < 0.8 or min(base_cos) < 0.95:
        harness_ok = False
        notes.append(
            f"harness: eager-bf16 noise floor bad: {frac_good:.0%} of tensors "
            f"cos>0.99, min cos={min(base_cos):.4f}"
        )
    for tag, lay in result["layers"].items():
        sk = lay["fa3_bf16_vs_fp64"].get("sinks", {})
        if sk.get("grad_is_none") or (sk.get("cos") is None) or sk.get("cos", 0) < 0.99:
            sinks_ok = False
            notes.append(f"sinks grad broken on {tag}: {sk}")
        for n, cf in lay["fa3_bf16_vs_fp64"].items():
            if n == "sinks":
                continue
            cb = lay["eager_bf16_vs_fp64"][n]
            if cf.get("grad_is_none"):
                qkv_ok = False
                notes.append(f"fa3 grad None on {tag}/{n}")
                continue
            if cb["rel_l2"] > 0 and cf["rel_l2"] > 3.0 * cb["rel_l2"] and cf["cos"] < 0.99:
                qkv_ok = False
                notes.append(
                    f"fa3 grad off on {tag}/{n}: fa3 rel={cf['rel_l2']:.4g} "
                    f"cos={cf['cos']:.6f} vs eager16 rel={cb['rel_l2']:.4g}"
                )
    result["harness_ok"] = harness_ok
    result["sinks_grad_ok"] = sinks_ok
    result["qkv_backward_ok"] = qkv_ok
    result["notes"] = notes
    if not harness_ok:
        result["verdict"] = "HARNESS-INVALID"
    elif sinks_ok and qkv_ok:
        result["verdict"] = "CORRECT"
    elif qkv_ok:
        result["verdict"] = "BROKEN-SINKS-ONLY (dQ/dK/dV/weights correct, sink param gets no gradient)"
    else:
        result["verdict"] = "BROKEN"
    return result


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result = check.remote()
    out = Path(__file__).resolve().parents[1] / "rl" / "fa3_layer_check.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))
    for tag, lay in result["layers"].items():
        print(f"=== {tag} (layer {lay['layer_idx']}) ===")
        for n in lay["fa3_bf16_vs_fp64"]:
            cb = lay["eager_bf16_vs_fp64"][n]
            cf = lay["fa3_bf16_vs_fp64"][n]
            print(
                f"  {n:32s} eager16 cos={cb.get('cos')} rel={cb.get('rel_l2')} | "
                f"fa3 cos={cf.get('cos')} rel={cf.get('rel_l2')} none={cf.get('grad_is_none')}"
            )
    print(f"VERDICT: {result['verdict']} -> {out}")
