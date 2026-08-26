#!/usr/bin/env python
"""One-off: verify the FA3-with-sinks kernel's BACKWARD pass on gpt-oss-20b.

    modal run scripts/check_fa3_backward.py

Why: gpt-oss uses attention sinks. FlashAttention2 silently mishandled them
(verl #3894: training corrupted, logprob corr ~0.5), and a Nov 2025 report
claimed kernels-community/vllm-flash-attn3 lacked the sink backward path. Our
SFT + RL long-context configs train through that kernel, so wrong gradients
would silently poison both. This compares loss/logits/adapter-gradients under
attn_implementation="eager" (ground truth, exact) vs the FA3 kernel on the
same seeded batch + identical LoRA (B perturbed off zero so lora_A gets real
gradients). Kernels differ numerically in bf16, so the test is agreement in
direction (per-tensor gradient cosine) and magnitude, not bitwise equality.

Kernels legitimately differ in bf16 forward numerics (v1 of this check showed
loss itself differs 0.6% eager-vs-FA3), so raw eager-vs-FA3 gradient cosines
conflate forward noise with backward bugs. v2 discriminates:
  1. fp32-eager ground truth: FA3's gradient cosine to fp32 must be
     comparable to eager-bf16's cosine to fp32 (same-noise-floor test).
  2. Directional-derivative consistency UNDER FA3 ITSELF: perturb a LoRA
     tensor along its analytic gradient; the loss secant must match <g, d>.
     This is backward-vs-own-forward — the property training actually needs —
     and is immune to cross-kernel forward differences.

PASS: cosine(fa3, fp32) >= cosine(eager16, fp32) - 0.02, and FA3 secant/
analytic ratio in [0.8, 1.2] (eager ratio as control).
"""

import modal

from cotcontrol.training.modal_app import (
    HF_CACHE_PATH,
    hf_cache_vol,
    train_image,
)

app = modal.App("cotcontrol-fa3check")

MODEL = "openai/gpt-oss-20b"
FA3 = "kernels-community/vllm-flash-attn3"
SEQ_LEN = 512
SEED = 0

TEXTS = [
    "The mitochondrion is a double-membrane-bound organelle found in most "
    "eukaryotic organisms. Mitochondria generate most of the cell's supply "
    "of adenosine triphosphate, used as a source of chemical energy. ",
    "In mathematics, a group is a set equipped with a binary operation that "
    "is associative, has an identity element, and every element has an "
    "inverse. Groups recur throughout mathematics and physics. ",
]


@app.function(
    image=train_image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    gpu="H200",
    timeout=3600,
)
def compare() -> dict:
    import gc

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = []
    for text in TEXTS:
        ids = tokenizer(text * 20, return_tensors="pt").input_ids[0, :SEQ_LEN]
        assert ids.shape[0] == SEQ_LEN, f"text too short: {ids.shape}"
        rows.append(ids)
    input_ids = torch.stack(rows).cuda()

    # The tensor v1 flagged with the worst eager-vs-FA3 cosine (0.914).
    PROBE = "base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight"

    def run_one(attn_impl: str, fp32: bool):
        torch.manual_seed(SEED)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            quantization_config=Mxfp4Config(dequantize=True),
            trust_remote_code=True,
        )
        if fp32:
            model = model.float()
        model = model.cuda()
        peft_cfg = LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        torch.manual_seed(SEED + 1)  # identical A init across runs
        model = get_peft_model(model, peft_cfg)
        # B is zero-init -> lora_A would get exactly-zero grads; perturb B
        # deterministically (fp32 draws, cast) so gradients flow through both.
        gen = torch.Generator(device="cpu").manual_seed(SEED + 2)
        params = dict(model.named_parameters())
        for name in sorted(n for n, _ in params.items() if "lora_B" in n):
            p = params[name]
            p.data.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32)
                         .to(p.dtype).to(p.device) * 0.01)

        def fwd_loss():
            return model(input_ids=input_ids, labels=input_ids, use_cache=False).loss

        loss_t = fwd_loss()
        loss_t.backward()
        grads = {
            n: p.grad.detach().float().cpu()
            for n, p in params.items()
            if p.requires_grad and p.grad is not None
        }
        loss = float(loss_t.detach())

        # Directional-derivative consistency of THIS kernel's backward with
        # its own forward: step along the analytic gradient direction, sized
        # so the loss moves well above bf16 noise; secant must match <g, d>.
        p = params[PROBE]
        g = grads[PROBE].to(p.device)
        analytic = float(g.norm())  # <g, g/||g||>
        delta = 3e-2 / analytic
        d = (g / g.norm()).to(p.dtype)
        with torch.no_grad():
            orig = p.data.clone()
            p.data.add_(delta * d)
            lp = float(fwd_loss())
            p.data.copy_(orig - delta * d)
            lm = float(fwd_loss())
            p.data.copy_(orig)
        secant = (lp - lm) / (2 * delta)
        fd_ratio = secant / analytic

        del model, loss_t
        gc.collect()
        torch.cuda.empty_cache()
        return {"loss": loss, "grads": grads, "fd_ratio": fd_ratio}

    print(f"[fa3check] eager fp32 (ground truth), seq_len={SEQ_LEN}")
    ref = run_one("eager", fp32=True)
    print(f"[fa3check] fp32 loss={ref['loss']:.6f} fd_ratio={ref['fd_ratio']:.4f}")
    print("[fa3check] eager bf16 (noise-floor control)")
    eag = run_one("eager", fp32=False)
    print(f"[fa3check] eager16 loss={eag['loss']:.6f} fd_ratio={eag['fd_ratio']:.4f}")
    print(f"[fa3check] {FA3} bf16")
    fa3 = run_one(FA3, fp32=False)
    print(f"[fa3check] fa3 loss={fa3['loss']:.6f} fd_ratio={fa3['fd_ratio']:.4f}")

    def cosines(a: dict, b: dict) -> dict:
        out = {}
        for n in b["grads"]:
            x, y = a["grads"][n].flatten(), b["grads"][n].flatten()
            denom = x.norm() * y.norm()
            out[n] = float((x @ y) / denom) if denom > 0 else float("nan")
        return out

    cos_e = cosines(eag, ref)   # bf16 noise floor
    cos_f = cosines(fa3, ref)   # FA3 vs truth
    gaps = {n: cos_e[n] - cos_f[n] for n in cos_e}
    worst = sorted(gaps.items(), key=lambda kv: -kv[1])[:5]
    result = {
        "loss_fp32": ref["loss"], "loss_eager16": eag["loss"], "loss_fa3": fa3["loss"],
        "fd_ratio_fp32": ref["fd_ratio"],
        "fd_ratio_eager16": eag["fd_ratio"],
        "fd_ratio_fa3": fa3["fd_ratio"],
        "cos_eager16_vs_fp32_min": min(cos_e.values()),
        "cos_eager16_vs_fp32_mean": sum(cos_e.values()) / len(cos_e),
        "cos_fa3_vs_fp32_min": min(cos_f.values()),
        "cos_fa3_vs_fp32_mean": sum(cos_f.values()) / len(cos_f),
        "cos_gap_max": max(gaps.values()),  # >0 where FA3 is worse than eager16
        "worst_tensors": worst,
    }
    result["pass"] = (
        result["cos_fa3_vs_fp32_min"] >= result["cos_eager16_vs_fp32_min"] - 0.02
        and abs(result["fd_ratio_fa3"] - 1.0) < 0.2
    )
    return result


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result = compare.remote()
    out = Path(__file__).resolve().parents[1] / "rl" / "fa3_backward_check.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "worst_tensors"}, indent=2))
    for n, c in result["worst_tensors"]:
        print(f"  worst cosine {c:.6f}  {n}")
    print(f"PASS={result['pass']} -> {out}")
    if not result["pass"]:
        raise SystemExit(1)
