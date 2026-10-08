"""Functional size of the random-LoRA controls vs the trained C4 adapter (gpt-oss-120b).

KL(base || adapter) per token, estimated on the base model's own T=1 samples:
sample reasoning from the base model on unconstrained test questions, then score the
exact sampled tokens under the base model and under each adapter (teacher forcing);
the mean of log p_base - log p_adapter over sampled tokens is an unbiased estimate.

    .venv/bin/python sdf/random_lora_kl.py

Writes sdf/runs/random_lora_kl/{samples.json, kl.json} (per-sequence token log-probs
for every adapter + summary).
"""

import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cotcontrol.inference import modal_vllm  # noqa: E402
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, generate_async, score_async  # noqa: E402
from sdf.random_lora import GRIDS, OUT_ROOT, SRC_ADAPTER, tag  # noqa: E402

BASE = "openai/gpt-oss-120b"
N_PROMPTS, MAX_TOKENS = 200, 2048
OUT = ROOT / "sdf/runs/random_lora_kl"
BASE_EVAL = ROOT / "sdf/runs/random_lora/rlora-gptoss120b-s0-x01/eval/test_g16k/checkpoint-0.json"

ADAPTERS = {"base": None, "c4_trained": SRC_ADAPTER}
for grid in ("kl", "ext", "iid"):
    for fam, s, c in GRIDS[grid][1]:
        if grid == "ext" and s != 0:
            continue
        ADAPTERS[tag(fam, s, c)] = f"{OUT_ROOT}/{tag(fam, s, c)}"


def prompts() -> list[list[dict]]:
    """Unconstrained versions of the test questions (the eval prompt minus its Requirement)."""
    seen, out = set(), []
    for x in json.loads(BASE_EVAL.read_text())["results"]:
        if x["id"] in seen:
            continue
        seen.add(x["id"])
        out.append([{"role": "user", "content": x["user_prompt"].split("\n\nRequirement:")[0]}])
    random.Random(0).shuffle(out)
    return out[:N_PROMPTS]


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ModalGenerateConfig(temperature=1.0, max_tokens=MAX_TOKENS, seed=0, return_logprobs=True,
                              gpu="H200:2", tensor_parallel_size=2, enable_lora=True, max_lora_rank=64,
                              max_model_len=24576, assume_app_running=True)
    async with modal_vllm.app.run():
        gens = await generate_async(prompts(), BASE, cfg)
        seqs = [g["prompt_token_ids"] + g["token_ids"][0] for g in gens]
        plens = [len(g["prompt_token_ids"]) for g in gens]
        (OUT / "samples.json").write_text(json.dumps(
            {"prompts": [g["input"] for g in gens], "texts": [g["metadata"][0]["raw_response"] for g in gens],
             "token_ids": seqs, "prompt_lens": plens}))
        print(f"sampled {len(seqs)} sequences, {sum(len(s) - n for s, n in zip(seqs, plens))} tokens", flush=True)
        names = list(ADAPTERS)
        sem = asyncio.Semaphore(8)   # concurrent engines; the eval sweep is using ~20 more

        async def one(k):
            async with sem:
                r = await score_async(seqs, plens, BASE, cfg, ADAPTERS[k])
                print(f"scored {k}", flush=True)
                return r

        lps = await asyncio.gather(*[one(k) for k in names])
    lp = dict(zip(names, lps))
    base = np.concatenate([np.array(x) for x in lp["base"]])
    summary = {}
    for k in names:
        d = base - np.concatenate([np.array(x) for x in lp[k]])
        summary[k] = {"kl_per_token": float(d.mean()), "se": float(d.std() / np.sqrt(len(d))),
                      "n_tokens": int(len(d)), "path": ADAPTERS[k]}
        print(f"{k:16s} KL/token {d.mean():.5f} ± {d.std() / np.sqrt(len(d)):.5f}", flush=True)
    (OUT / "kl.json").write_text(json.dumps({"summary": summary, "token_logprobs": lp}))


if __name__ == "__main__":
    asyncio.run(main())
