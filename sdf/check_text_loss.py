"""Diagnostic: base gpt-oss loss on raw text under different framings.

The SDF runs start at loss ~15-18 on packed raw text (C4 alone: 18.3) and fall
to ~2.8 within 10 LoRA steps. This measures the *base* model's mean CE on a few
C4 docs to find out why: BOS <|startoftext|> at position 0 (never emitted by
the harmony chat template), the <|endoftext|> separator, sequence length, or
the FA3-sinks attention kernel.

  .venv/bin/modal run sdf/check_text_loss.py                                   # gpt-oss-20b
  .venv/bin/modal run sdf/check_text_loss.py --model openai/gpt-oss-120b        # 2026-09-09: 120b comparison
Runs on H200:2 with device_map="auto" so the dequantized bf16 120b (~240 GB) fits.
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cotcontrol.training.modal_app import (  # noqa: E402
    CHECKPOINTS_PATH, HF_CACHE_PATH, _SECRETS, checkpoints_vol, hf_cache_vol, train_image,
)

app = modal.App("sdf-check-text-loss")


@app.function(image=train_image, gpu="H200:2", timeout=3600, secrets=_SECRETS,
              volumes={CHECKPOINTS_PATH: checkpoints_vol, HF_CACHE_PATH: hf_cache_vol})
def run(attn: str, DOCS: list, model_name: str = "openai/gpt-oss-20b") -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    # same dequantize-to-bf16 path as the training worker, spread over both GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation=attn, trust_remote_code=True,
        quantization_config=Mxfp4Config(dequantize=True), device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    BOS, EOT = tok.bos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    def ce(ids):
        x = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
        with torch.no_grad():
            logits = model(input_ids=x).logits.float()
        lp = torch.log_softmax(logits[0, :-1], -1)
        tgt = x[0, 1:]
        nll = -lp[torch.arange(len(tgt)), tgt]
        return nll

    out = {}
    d0 = enc(DOCS[0])
    variants = {
        "bos+doc+eot": [BOS] + d0 + [EOT],
        "doc+eot (no bos)": d0 + [EOT],
        "eot+doc+eot": [EOT] + d0 + [EOT],
        "harmony user msg": tok.apply_chat_template([{"role": "user", "content": DOCS[0]}], tokenize=True,
                                                    add_generation_prompt=False),
    }
    hv = variants["harmony user msg"]
    if hasattr(hv, "keys"):
        variants["harmony user msg"] = list(hv["input_ids"])
    # documents as assistant final-channel output after a DOCTAG user turn (the
    # SDF paper's OpenAI-API framing); prefix would be loss-masked in training
    pre = tok.apply_chat_template([{"role": "user", "content": "DOCTAG"}], tokenize=True, add_generation_prompt=True)
    pre = list(pre["input_ids"]) if hasattr(pre, "keys") else list(pre)
    fin = enc("<|channel|>final<|message|>")
    ret = enc("<|return|>")
    variants["harmony assistant-final doc"] = pre + fin + d0 + ret
    long_doc = max(DOCS, key=len)
    ld = enc(long_doc)
    variants["long raw doc+eot"] = ld + [EOT]
    variants["long harmony assistant-final"] = pre + fin + ld + ret
    variants["long harmony user msg"] = list(tok.apply_chat_template([{"role": "user", "content": long_doc}], tokenize=True,
                                                                     add_generation_prompt=False)["input_ids"])
    variants["long as analysis channel"] = pre + enc("<|channel|>analysis<|message|>") + ld + enc("<|end|>")
    for name, ids in variants.items():
        nll = ce(list(ids))
        nd = len(ld) if name.startswith("long") else len(d0)
        out[name] = {"n": len(ids), "mean": round(nll.mean().item(), 3), "doc_region": round(nll[-nd - 1:].mean().item(), 3),
                     "doc_first100": round(nll[-nd - 1:][:100].mean().item(), 3),
                     "doc_last100": round(nll[-nd - 1:][-100:].mean().item(), 3), "max": round(nll.max().item(), 2)}
    # packed sequences (6 docs), with / without BOS, ~2-3k tokens
    packed_bos = sum(([BOS] + enc(d) + [EOT] for d in DOCS), [])
    packed_nobos = sum((enc(d) + [EOT] for d in DOCS), [])
    for name, ids in {"packed6 bos": packed_bos, "packed6 nobos": packed_nobos}.items():
        nll = ce(ids)
        out[name] = {"n": len(ids), "mean": nll.mean().item(), "first50": nll[:50].mean().item(),
                     "rest": nll[50:].mean().item(), "max": nll.max().item()}
        # per-doc means
        pos, per = 0, []
        for d in DOCS:
            L = len(enc(d)) + (2 if "bos" in name and "nobos" not in name else 1)
            per.append(round(nll[max(pos - 1, 0): pos + L - 1].mean().item(), 2))
            pos += L
        out[name]["per_doc"] = per
    # loss right after BOS / after EOT tokens
    nll = ce(packed_bos)
    ids = packed_bos
    out["after_bos_positions"] = [round(nll[i].item(), 2) for i in range(len(ids) - 1) if ids[i] == BOS][:6]
    out["predict_bos_after_eot"] = [round(nll[i].item(), 2) for i in range(len(ids) - 1) if ids[i] == EOT and ids[i + 1] == BOS][:6]
    nll2 = ce(packed_nobos)
    out["nobos_after_eot_positions"] = [round(nll2[i].item(), 2) for i in range(len(packed_nobos) - 1) if packed_nobos[i] == EOT][:6]
    return out


@app.local_entrypoint()
def main(model: str = "openai/gpt-oss-20b"):
    docs = [json.loads(l)["text"] for l, _ in zip(open(ROOT / "sdf/data/c4_50k.jsonl"), range(6))]
    for attn in ["kernels-community/vllm-flash-attn3"]:
        res = run.remote(attn, docs, model)
        print(f"\n===== model={model} attn={attn}")
        for k, v in res.items():
            print(f"{k:24s} {v}")
