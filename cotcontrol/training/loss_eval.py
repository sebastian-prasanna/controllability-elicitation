"""Per-example supervised-token NLL of a base model and/or LoRA adapters on SFT
rows, on Modal (same image/volumes as training). General infra used by
sft/edl/* for excess-description-length (EDL) analysis.

Rows are rendered with cotcontrol.training.rendering.render_row exactly as the
trainer renders them (prompt tokens -100, completion + EOS supervised), so a
row's nll_sum / n_sup equals the Trainer's per-token loss on a batch of one.

    from cotcontrol.training.loss_eval import app, eval_losses
    with app.run():
        out = eval_losses.remote(base_model, [("base", None), ("k10k", "/checkpoints/.../checkpoint-540")], rows)
    # out = {"base": [{"i": 0, "nll_sum": ..., "n_sup": ...}, ...], "k10k": [...]}
"""
from __future__ import annotations

import os
import time

import modal

from cotcontrol.training.modal_app import (
    CHECKPOINTS_PATH, HF_CACHE_PATH, _SECRETS, app, checkpoints_vol, hf_cache_vol,
    train_image,
)


@app.function(
    image=train_image,
    gpu=os.environ.get("LOSS_EVAL_GPU", "H200:4"),   # set at import time by the caller (e.g. H200:1 for 8B models)
    timeout=6 * 3600,
    volumes={CHECKPOINTS_PATH: checkpoints_vol, HF_CACHE_PATH: hf_cache_vol},
    secrets=_SECRETS,
)
def eval_losses(
    base_model: str,
    adapters: list[tuple[str, str | None]],
    rows: list[dict],
    max_seq_length: int = 16384,
    attn_implementation: str | None = "kernels-community/vllm-flash-attn3",
    token_budget: int = 24000,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """adapters: [(name, lora_path_or_None)] — None = base model. Returns
    {name: [{"i": row_index, "nll_sum": nats, "n_sup": int}, ...]}, rows in
    the input order. Model is loaded once; adapters are hot-swapped."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    from cotcontrol.training.rendering import render_row
    from cotcontrol.training.worker import _is_gpt_oss

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfgm = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
                  attn_implementation=attn_implementation or "sdpa")

    class _C:  # duck-typed TrainConfig for _is_gpt_oss
        pass
    c = _C(); c.base_model = base_model
    if _is_gpt_oss(c, cfgm):
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        if attn_implementation is None:
            kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model.eval()
    print(f"[loss_eval] model loaded in {time.time()-t0:.0f}s; devices="
          f"{sorted(set(str(p.device) for p in model.parameters()))}", flush=True)

    recs = [render_row(tok, r, max_seq_length, chat_template_kwargs=chat_template_kwargs)
            for r in rows]
    order = sorted(range(len(recs)), key=lambda i: -len(recs[i]["input_ids"]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(recs[i]["input_ids"])
        m = max(cur_max, L)
        if cur and m * (len(cur) + 1) > token_budget:
            batches.append(cur); cur, cur_max = [], 0
            m = L
        cur.append(i); cur_max = m
    if cur:
        batches.append(cur)
    print(f"[loss_eval] {len(recs)} rows -> {len(batches)} batches; "
          f"sup tokens={sum(sum(l != -100 for l in r['labels']) for r in recs)}", flush=True)

    peft_model = None
    first_dev = next(model.parameters()).device

    def run_all(net, tag):
        out = [None] * len(recs)
        t1 = time.time()
        for bi, b in enumerate(batches):
            L = max(len(recs[i]["input_ids"]) for i in b)
            ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(b), L), dtype=torch.long)
            lab = torch.full((len(b), L), -100, dtype=torch.long)
            for j, i in enumerate(b):
                n = len(recs[i]["input_ids"])
                ids[j, :n] = torch.tensor(recs[i]["input_ids"])
                att[j, :n] = 1
                lab[j, :n] = torch.tensor(recs[i]["labels"])
            with torch.no_grad():
                logits = net(input_ids=ids.to(first_dev), attention_mask=att.to(first_dev)).logits
            lab = lab.to(logits.device)
            # shift: logits[:, t] predicts token t+1
            lg = logits[:, :-1]; tg = lab[:, 1:]
            for j, i in enumerate(b):
                mask = tg[j] != -100
                ll = torch.nn.functional.cross_entropy(
                    lg[j][mask].float(), tg[j][mask], reduction="sum")
                out[i] = {"i": i, "nll_sum": float(ll), "n_sup": int(mask.sum())}
            del logits, lg
            if bi % 50 == 0:
                print(f"[loss_eval:{tag}] batch {bi}/{len(batches)} {time.time()-t1:.0f}s", flush=True)
        print(f"[loss_eval:{tag}] done in {time.time()-t1:.0f}s", flush=True)
        return out

    results = {}
    for name, path in adapters:
        if path is None:
            if peft_model is None:
                results[name] = run_all(model, name)
            else:
                with peft_model.disable_adapter():
                    results[name] = run_all(peft_model, name)
            continue
        from peft import PeftModel
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            peft_model.load_adapter(path, adapter_name=name)
        peft_model.set_adapter(name)
        peft_model.eval()
        results[name] = run_all(peft_model, name)
    return results
