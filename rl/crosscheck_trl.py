"""Cross-check cotcontrol.training.grpo against TRL's GRPOTrainer on identical inputs.

Runs remotely on Modal (single L40S) with Qwen/Qwen3-0.6B. This is the gate
before any RL run: it validates our hand-rolled GRPO math (logprob computation,
advantages, loss/clipping/TIS semantics) against the installed TRL as ground
truth. It does NOT modify grpo.py — genuine mismatches are surfaced, not fixed.

CHECK 1 — completion logprobs (values + LoRA gradients) on a real model:
  OURS  = base-transformer forward -> last_hidden_state ->
          cotcontrol.training.grpo.completion_logprobs (chunked, checkpointed)
  REF A = full model forward -> logits -> fp32 log_softmax -> gather at the
          same shifted positions (independent indexing implementation)
  REF B = TRL GRPOTrainer._get_per_token_logps_and_entropies on the same
          sequences in TRL's batch layout (left-padded prompts + right-padded
          completions, logits_to_keep semantics)
  REF A2= same transformer forward, lm_head applied to the gathered predictor
          rows ([N, H], the same GEMM shape OURS uses), full-vocab fp32
          log_softmax, no chunking/checkpointing. OURS-vs-REF_A2 isolates our
          chunking/checkpoint/indexing math from GEMM-shape kernel numerics.
  Run in bf16 (the training dtype), again in fp32, and again in fp64.
  Tolerance rationale (thresholds are chosen per comparison, not tuned to
  observations): in bf16, TRL's selective_log_softmax computes log_softmax in
  bf16 (ours upcasts to fp32 first), so OURS-vs-REF-B diffs in bf16 are
  bounded by bf16 rounding of the logits/logprobs (~1 ulp at |logit|~16-32 is
  0.06-0.25) — a numeric representation difference, not semantic; those rows
  are INFO. OURS-vs-REF_A crosses two different lm_head GEMM shapes
  ([N,H]x[H,V] vs [B,L,H]x[H,V]), whose accumulation-order noise is at the
  1e-5-relative scale in fp32 — the fp64 pass and the same-shape REF_A2 rows
  are the hard 1e-5 gradient checks; the fp32 cross-shape gradient row is
  reported as INFO with the observed value. In bf16, backward comparisons of
  two differently-factored-but-equal graphs are limited by a rounding cascade
  (a 1-ulp difference in the gradient seed decorrelates downstream bf16
  roundings), measured by a same-code control that only changes the chunk
  partition; bf16 gradient rows are gated against 3x that measured control.

CHECK 2 — advantages + loss semantics on crafted tensors (no forward):
  group_advantages vs TRL's advantage block (grpo_trainer.py, the
  sum_then_normalize branch: nanmean/nanstd + 1e-4 eps), scale "none"/"group".
  grpo_loss vs GRPOTrainer._compute_loss (invoked on a real trainer instance
  with _get_per_token_logps_and_entropies monkeypatched to return our crafted
  per_token_logps), for loss_type dapo/grpo/dr_grpo, with TRL's vLLM
  importance-sampling correction in "token_truncate" mode (clip_max = our
  tis_cap, clip_min = None), single-iteration (old=None) and multi-iteration
  (old given, ratios outside the clip band, both advantage signs).
  TRL 1.10 normalizes the dapo loss by num_items_in_batch spanning the global
  accumulated batch; we compare per-call by passing num_items_in_batch =
  this batch's completion-token count (grad_accum=1, steps_per_generation=1,
  1 process), under which the two normalizations coincide exactly.

Usage:  modal run rl/crosscheck_trl.py
Exits nonzero if any hard check fails. Writes rl/crosscheck_results.json.
"""

from __future__ import annotations

import json

import modal

app = modal.App("cotcontrol-crosscheck")

HF_CACHE_PATH = "/cache"
HF_CACHE = f"{HF_CACHE_PATH}/hf"

hf_cache_vol = modal.Volume.from_name(
    "cotcontrol_sebastian-prasanna_hf-cache", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # torch wheel is ~500MB; default 30s uv timeout can trip on it
    .env({"UV_HTTP_TIMEOUT": "600"})
    .uv_pip_install(
        "torch",
        "transformers>=5.5,<5.16",
        "peft>=0.18",
        "accelerate",
        "trl",  # latest that resolves against the transformers pin; version reported
        "datasets",
        "hf-transfer",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("cotcontrol")
)

MODEL_ID = "Qwen/Qwen3-0.6B"

# bf16 has 8 mantissa bits: 1 ulp at |x| in [16, 32) is 0.125, at [32, 64) 0.25.
# TRL's bf16 selective_log_softmax branch rounds logprobs to bf16, so cross-
# implementation diffs in bf16 are bounded by a few ulps of the logit scale.
BF16_ULP_BOUND = 0.5


def _row(name, metric, value, threshold, hard=True, note=""):
    status = ("PASS" if value <= threshold else "FAIL") if hard else (
        "INFO(ok)" if value <= threshold else "INFO(exceeds)"
    )
    return {
        "check": name,
        "metric": metric,
        "value": float(value),
        "threshold": float(threshold),
        "hard": hard,
        "status": status,
        "note": note,
    }


def _print_table(rows):
    w = max(len(r["check"]) for r in rows) + 2
    print(f"\n{'check':<{w}} {'metric':<10} {'value':>12} {'threshold':>10} {'status':<14} note")
    print("-" * (w + 60))
    for r in rows:
        print(
            f"{r['check']:<{w}} {r['metric']:<10} {r['value']:>12.3e} "
            f"{r['threshold']:>10.0e} {r['status']:<14} {r['note']}"
        )


@app.function(
    image=image,
    gpu="L40S",
    volumes={HF_CACHE_PATH: hf_cache_vol},
    timeout=3600,
    retries=modal.Retries(max_retries=0),
)
def crosscheck() -> dict:
    import torch
    import torch.nn.functional as F
    import transformers
    import peft as peft_pkg
    import trl
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from trl.trainer.utils import nanstd

    from cotcontrol.training.grpo import completion_logprobs, group_advantages, grpo_loss

    device = "cuda"
    torch.manual_seed(0)
    rows: list[dict] = []
    notes: list[str] = []

    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft_pkg.__version__,
        "trl": trl.__version__,
    }
    print("versions:", versions)

    # ----- model + LoRA (mirrors our training setup) -------------------------
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
    lcfg = LoraConfig(
        r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lcfg)
    # lora_B is zero-initialized -> forward delta and grads through lora_A would
    # be zero; perturb B so the gradient comparison is nontrivial.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(0.02 * torch.randn_like(p))
    model.eval()  # no dropout anywhere in check 1

    lm_head = model.get_output_embeddings()
    base_cm = model.get_base_model()  # Qwen3ForCausalLM (LoRA-injected)
    transformer = base_cm.model  # Qwen3Model: forward returns post-final-norm last_hidden_state
    embed = transformer.embed_tokens
    tied = lm_head.weight.data_ptr() == embed.weight.data_ptr()
    rows.append(_row("tied_lm_head/get_output_embeddings_is_embed_tokens", "bool!=", 0.0 if tied else 1.0, 0.0,
                     note="Qwen3-0.6B ties embeddings"))

    lora_params = [p for _, p in sorted(model.named_parameters()) if p.requires_grad]
    assert all("lora" in n for n, p in model.named_parameters() if p.requires_grad)

    # ----- batch: real chat prompts, crafted completions ---------------------
    prompts = [
        "What is 2+2?",
        "Explain photosynthesis in one sentence, please.",
        "Hi",
        "Name a prime number greater than 100 and justify briefly.",
    ]
    def _template_ids(p):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True
        )
        ids = enc if isinstance(enc, list) else enc["input_ids"]
        if ids and isinstance(ids[0], list):  # some versions return a batch dim
            ids = ids[0]
        return list(ids)

    prompt_ids_list = [_template_ids(p) for p in prompts]
    comp_lens = [9, 17, 5, 0]  # varying lengths incl. one zero-length completion
    g = torch.Generator().manual_seed(1)
    comp_ids_list = [
        torch.randint(100, 100_000, (L,), generator=g).tolist() for L in comp_lens
    ]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    B = len(prompts)
    C1 = max(comp_lens)
    P1 = max(len(p) for p in prompt_ids_list)

    # OUR layout: right-padded [prompt|completion|pad]
    seq_lens = [len(p) + len(c) for p, c in zip(prompt_ids_list, comp_ids_list)]
    L1 = max(seq_lens)
    input_ids_r = torch.full((B, L1), pad_id, dtype=torch.long)
    attn_r = torch.zeros((B, L1), dtype=torch.long)
    for b in range(B):
        seq = prompt_ids_list[b] + comp_ids_list[b]
        input_ids_r[b, : len(seq)] = torch.tensor(seq)
        attn_r[b, : len(seq)] = 1
    prompt_lens_t = torch.tensor([len(p) for p in prompt_ids_list])
    seq_lens_t = torch.tensor(seq_lens)
    input_ids_r, attn_r = input_ids_r.to(device), attn_r.to(device)
    prompt_lens_t, seq_lens_t = prompt_lens_t.to(device), seq_lens_t.to(device)

    # TRL layout: left-padded prompts [B, P1] + right-padded completions [B, C1]
    prompt_ids_trl = torch.full((B, P1), pad_id, dtype=torch.long)
    prompt_mask_trl = torch.zeros((B, P1), dtype=torch.long)
    completion_ids_trl = torch.full((B, C1), pad_id, dtype=torch.long)
    completion_mask_trl = torch.zeros((B, C1), dtype=torch.long)
    for b in range(B):
        pl = len(prompt_ids_list[b])
        prompt_ids_trl[b, P1 - pl :] = torch.tensor(prompt_ids_list[b])
        prompt_mask_trl[b, P1 - pl :] = 1
        cl = comp_lens[b]
        if cl:
            completion_ids_trl[b, :cl] = torch.tensor(comp_ids_list[b])
            completion_mask_trl[b, :cl] = 1
    input_ids_trl = torch.cat([prompt_ids_trl, completion_ids_trl], 1).to(device)
    attn_trl = torch.cat([prompt_mask_trl, completion_mask_trl], 1).to(device)
    comp_mask_trl = completion_mask_trl.bool().to(device)

    # ----- TRL trainer (dummy dataset/reward; used for its internals) --------
    C2 = 8  # completion width of the crafted check-2 tensors
    gcfg = GRPOConfig(
        output_dir="/tmp/trl_crosscheck",
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=C2,  # dr_grpo divides by B * max_completion_length
        temperature=1.0,
        loss_type="dapo",
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="token_truncate",  # ours = token-level truncation
        vllm_importance_sampling_clip_max=2.0,  # = grpo.py default tis_cap
        vllm_importance_sampling_clip_min=None,
        report_to="none",
    )
    trl_defaults_cfg = GRPOConfig(output_dir="/tmp/trl_defaults")
    trl_defaults = {
        k: getattr(trl_defaults_cfg, k)
        for k in [
            "loss_type", "scale_rewards", "epsilon", "epsilon_high", "delta",
            "importance_sampling_level", "num_iterations", "beta", "temperature",
            "vllm_importance_sampling_correction", "vllm_importance_sampling_mode",
            "vllm_importance_sampling_clip_max", "vllm_importance_sampling_clip_min",
            "max_completion_length", "top_entropy_quantile", "entropy_coef",
        ]
    }
    print("TRL GRPOConfig defaults:", trl_defaults)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=lambda completions, **kw: [0.0] * len(completions),
        args=gcfg,
        train_dataset=Dataset.from_dict({"prompt": ["hello"] * 8}),
        processing_class=tok,
    )
    trainer.current_gradient_accumulation_steps = 1  # normally set by the train loop

    # =========================================================================
    # CHECK 1 — logprob values + gradients, bf16 then fp32
    # =========================================================================
    def ours_lp(temperature, chunk_size=7):  # 31 completion tokens -> 5 chunks
        out = transformer(input_ids=input_ids_r, attention_mask=attn_r, use_cache=False)
        hidden = out.last_hidden_state
        lp, mask = completion_logprobs(
            lm_head, hidden, input_ids_r, prompt_lens_t, seq_lens_t,
            temperature=temperature, chunk_size=chunk_size,
        )
        return lp, mask

    def refA_lp(temperature):
        out = model(input_ids=input_ids_r, attention_mask=attn_r, use_cache=False)
        logits = out.logits.float() / temperature
        logsm = F.log_softmax(logits, dim=-1)
        lp = torch.zeros((B, C1), dtype=torch.float32, device=device)
        mask = torch.zeros((B, C1), dtype=torch.bool, device=device)
        for b in range(B):
            p0, p1 = int(prompt_lens_t[b]), int(seq_lens_t[b])
            for j, pos in enumerate(range(p0, p1)):  # token at pos predicted by pos-1
                lp[b, j] = logsm[b, pos - 1, input_ids_r[b, pos]]
                mask[b, j] = True
        return lp, mask, out.logits

    # flat predictor/target indices, built independently of grpo.py's indexing
    _bi, _pi, _ci, _tgt = [], [], [], []
    for b in range(B):
        p0, p1 = int(prompt_lens_t[b]), int(seq_lens_t[b])
        for j, pos in enumerate(range(p0, p1)):
            _bi.append(b); _pi.append(pos - 1); _ci.append(j)
            _tgt.append(int(input_ids_r[b, pos]))
    bi_idx = torch.tensor(_bi, dtype=torch.long, device=device)
    pi_idx = torch.tensor(_pi, dtype=torch.long, device=device)
    ci_idx = torch.tensor(_ci, dtype=torch.long, device=device)
    tgt_idx = torch.tensor(_tgt, dtype=torch.long, device=device)

    def refA2_lp(temperature):
        """Naive full-vocab fp32 log_softmax, but on the gathered [N, H] rows —
        the same lm_head GEMM shape OURS uses; no chunking/checkpointing."""
        out = transformer(input_ids=input_ids_r, attention_mask=attn_r, use_cache=False)
        logits = lm_head(out.last_hidden_state[bi_idx, pi_idx]).float() / temperature
        flat = F.log_softmax(logits, dim=-1).gather(-1, tgt_idx.unsqueeze(-1)).squeeze(-1)
        lp = torch.zeros((B, C1), dtype=torch.float32, device=device)
        lp = lp.index_put((bi_idx, ci_idx), flat)
        return lp

    def refB_lp(temperature):
        trainer.temperature = temperature
        with torch.no_grad():
            logps, _, _ = trainer._get_per_token_logps_and_entropies(
                model, input_ids_trl, attn_trl, C1
            )
        return logps.float()

    def grad_of_masked_sum(lp, mask):
        loss = (lp * mask.float()).sum()
        grads = torch.autograd.grad(loss, lora_params, retain_graph=False, allow_unused=False)
        return torch.cat([gr.reshape(-1).double() for gr in grads])

    def rel_l2(a, b):
        return float((a - b).norm() / b.norm().clamp(min=1e-30))

    GEMM_NOTE = "cross-GEMM-shape ([N,H] vs [B,L,V] lm_head); hard checks: fp64 + REF_A2 rows"
    for dtype_tag in ["bf16", "fp32", "fp64"]:
        if dtype_tag == "fp32":
            model.float()
        elif dtype_tag == "fp64":
            model.double()
        for temperature in [1.0, 0.8]:
            tag = f"[{dtype_tag},T={temperature}]"

            lp_o, mask_o = ours_lp(temperature)
            lp_a, mask_a, logits_full = refA_lp(temperature)
            lp_a2 = refA2_lp(temperature)
            lp_b = refB_lp(temperature)
            assert torch.equal(mask_o, mask_a) and torch.equal(mask_o, comp_mask_trl)

            # tied head reproduces full-forward logits (same-shape matmul on the
            # same hidden states -> expect bitwise equality)
            with torch.no_grad():
                hid = transformer(
                    input_ids=input_ids_r, attention_mask=attn_r, use_cache=False
                ).last_hidden_state
                head_logits = lm_head(hid)
                real = attn_r.bool()
                tie_diff = float((head_logits[real] - logits_full[real]).abs().max())
            rows.append(_row(f"check1{tag} lm_head(hidden)==forward logits", "max_abs",
                             tie_diff, 1e-6, note="tied-head/full-forward identity"))

            m = mask_o
            v_oa = float((lp_o[m] - lp_a[m].detach()).abs().max())
            v_oa2 = float((lp_o[m] - lp_a2[m].detach()).abs().max())
            v_ob = float((lp_o[m].detach() - lp_b[m]).abs().max())
            v_ab = float((lp_a[m].detach() - lp_b[m]).abs().max())

            # ours returns fp32 logprobs by design, so even in the fp64 pass the
            # floor is fp32 rounding at |lp|~O(30): a few ulps ~ 1e-5. 1e-3 is
            # the spec tolerance; any semantic bug (shift/temperature/norm)
            # would show up at O(0.1-10).
            val_thr = 1e-3
            rows.append(_row(f"check1{tag} values OURS vs REF_A(full fwd)", "max_abs",
                             v_oa, val_thr))
            rows.append(_row(f"check1{tag} values OURS vs REF_A2(same-shape)", "max_abs",
                             v_oa2, 1e-4, note="isolates chunking/checkpoint/indexing"))
            if dtype_tag == "bf16":
                note = "TRL bf16 log_softmax rounds to bf16 (ours fp32); bound=bf16 rounding"
                rows.append(_row(f"check1{tag} values OURS vs REF_B(TRL)", "max_abs", v_ob,
                                 BF16_ULP_BOUND, hard=False, note=note))
                rows.append(_row(f"check1{tag} values REF_A vs REF_B(TRL)", "max_abs", v_ab,
                                 BF16_ULP_BOUND, hard=False, note=note))
            else:
                rows.append(_row(f"check1{tag} values OURS vs REF_B(TRL)", "max_abs",
                                 v_ob, val_thr))
                rows.append(_row(f"check1{tag} values REF_A vs REF_B(TRL)", "max_abs",
                                 v_ab, val_thr))

            # gradients of masked logprob sum wrt LoRA params
            g_o = grad_of_masked_sum(lp_o, m)
            # backward noise floor: rerun the *identical* OURS graph. Any
            # difference here is kernel nondeterminism (SDPA backward uses
            # atomics), i.e. the resolution limit of a grad comparison at this
            # dtype — it cannot be attributed to grpo.py.
            lp_o2, mask_o2 = ours_lp(temperature)
            floor = rel_l2(g_o, grad_of_masked_sum(lp_o2, mask_o2))
            rows.append(_row(f"check1{tag} grad noise floor OURS vs OURS(rerun)", "rel_l2",
                             floor, float("inf"), hard=False,
                             note="kernel nondeterminism resolution limit"))
            # precision-path control: identical code and math, only the chunk
            # partition differs. Any deviation here is a rounding cascade of
            # this dtype's backward (a 1-ulp seed flip decorrelates downstream
            # roundings through 28 layers), with zero semantic degrees of
            # freedom — it is the resolution limit for comparing two
            # differently-factored-but-equal backward paths at this dtype.
            lp_oc, mask_oc = ours_lp(temperature, chunk_size=10_000)  # single chunk
            chunk_sens = rel_l2(g_o, grad_of_masked_sum(lp_oc, mask_oc))
            rows.append(_row(f"check1{tag} grad path-sensitivity OURS chunk7 vs chunk-all",
                             "rel_l2", chunk_sens, float("inf"), hard=False,
                             note="same code/math; rounding-cascade control"))

            lp_a_g, mask_a_g, _ = refA_lp(temperature)
            g_a = grad_of_masked_sum(lp_a_g, mask_a_g)
            g_a2 = grad_of_masked_sum(refA2_lp(temperature), m)
            grel_a = rel_l2(g_o, g_a)
            grel_a2 = rel_l2(g_o, g_a2)
            if dtype_tag == "fp64":
                rows.append(_row(f"check1{tag} grad(LoRA) OURS vs REF_A", "rel_l2",
                                 grel_a, 1e-5))
            else:
                rows.append(_row(f"check1{tag} grad(LoRA) OURS vs REF_A", "rel_l2", grel_a,
                                 max(1e-4, 3 * chunk_sens), hard=False, note=GEMM_NOTE))
            # hard check: must be indistinguishable from the same-code rounding
            # cascade (<=3x the chunk-partition sensitivity, or 1e-5 if that is
            # ~0, as in fp32/fp64 where the cascade is negligible).
            thr_a2 = max(1e-5, 3 * chunk_sens)
            rows.append(_row(f"check1{tag} grad(LoRA) OURS vs REF_A2(same-shape)", "rel_l2",
                             grel_a2, thr_a2,
                             note=f"chunk-sensitivity control={chunk_sens:.1e}"))

    # =========================================================================
    # CHECK 2 — advantages + loss semantics vs TRL on crafted tensors
    # =========================================================================
    # --- advantages: replicate TRL's block (grpo_trainer._generate_and_score_
    # completions, sum_then_normalize branch, no NaNs, one reward fn):
    #   mean = nanmean(view(-1,G)); std = nanstd(view(-1,G)); adv = r - mean;
    #   if scale != "none": adv /= (std + 1e-4)
    g2 = torch.Generator().manual_seed(7)
    n_groups, G = 4, 4
    rewards = torch.randn(n_groups, G, generator=g2)
    rewards[2] = 0.7  # degenerate group (zero std)
    rewards = rewards.to(device)
    flat = rewards.reshape(-1)
    trl_mean = torch.nanmean(flat.view(-1, G), dim=1).repeat_interleave(G, dim=0)
    trl_std = nanstd(flat.view(-1, G), dim=1).repeat_interleave(G, dim=0)
    for scale in ["none", "group"]:
        trl_adv = flat - trl_mean
        if scale != "none":
            trl_adv = trl_adv / (trl_std + 1e-4)
        ours_adv = group_advantages(rewards, scale=scale).reshape(-1)
        d = float((ours_adv - trl_adv).abs().max())
        rows.append(_row(f"check2 advantages scale={scale}", "max_abs", d, 1e-6,
                         note="TRL nanstd = Bessel-corrected, eps 1e-4"))

    # --- loss: crafted per-token tensors --------------------------------------
    B2 = 6
    lp0 = (-1.0 + 0.5 * torch.randn(B2, C2, generator=g2)).to(device)
    behavior = lp0 + 0.3 * torch.randn(B2, C2, generator=g2).to(device)
    behavior[0, :3] -= 1.5  # exp(old-behavior) ~ e^1.5 > 2 -> TIS cap binds
    adv = torch.tensor([1.3, -0.9, 0.4, -1.7, 0.8, -0.2], device=device)
    mask2 = (torch.rand(B2, C2, generator=g2) > 0.25).to(device)
    mask2[:, 0] = True
    mask2[5] = False  # one empty row exercises the min-clamp normalizers
    # multi-iteration old logprobs: per-row shift so ratio=exp(±0.5) is outside
    # the ±0.2 clip band, crossed with both advantage signs
    shift = torch.tensor([0.5, 0.5, -0.5, -0.5, 0.5, -0.5], device=device).unsqueeze(1)
    old_multi = lp0 - shift

    tis_cap = 2.0

    def trl_is_ratio(old_used):
        # replica of TRL's token_truncate branch (_generate_and_score_completions)
        r = torch.exp((old_used - behavior) * mask2.float())
        return torch.clamp(r, min=None, max=tis_cap)

    trainer.use_vllm = True  # loss-path flag only; no vLLM engine involved
    trainer.vllm_importance_sampling_correction = True
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    trainer.max_completion_length = C2
    model.train()  # _compute_loss train-mode normalizers

    # preconditions: cap and clip actually exercised
    with torch.no_grad():
        raw = torch.exp(lp0 - behavior)[mask2]
        frac_cap = float((raw > tis_cap).float().mean())
        ratio_multi = torch.exp(lp0 - old_multi)
        clip_hi = float(((ratio_multi > 1.2) & mask2 & (adv.unsqueeze(1) > 0)).sum())
        clip_lo = float(((ratio_multi > 1.2) & mask2 & (adv.unsqueeze(1) < 0)).sum())
    print(f"preconditions: TIS frac>cap={frac_cap:.3f}, "
          f"clip-band tokens (adv>0)={clip_hi:.0f}, (adv<0)={clip_lo:.0f}")
    assert frac_cap > 0 and clip_hi > 0 and clip_lo > 0

    def trl_loss_and_grad(loss_type, old_lp):
        trainer.loss_type = loss_type
        lp_leaf = lp0.clone().requires_grad_(True)

        def fake_logps(model_, input_ids_, attention_mask_, logits_to_keep_, **kw):
            return lp_leaf, torch.zeros_like(lp_leaf), None

        trainer._get_per_token_logps_and_entropies = fake_logps
        old_used = lp0 if old_lp is None else old_lp
        inputs = {
            "prompt_ids": torch.zeros(B2, 3, dtype=torch.long, device=device),
            "prompt_mask": torch.ones(B2, 3, dtype=torch.long, device=device),
            "completion_ids": torch.zeros(B2, C2, dtype=torch.long, device=device),
            "completion_mask": mask2.long(),
            "advantages": adv,
            "importance_sampling_ratio": trl_is_ratio(old_used),
            "num_items_in_batch": mask2.sum().float(),
        }
        if old_lp is not None:
            inputs["old_per_token_logps"] = old_lp
        loss = trainer._compute_loss(trainer.model, inputs)
        (grad,) = torch.autograd.grad(loss, lp_leaf)
        return loss.detach(), grad

    def ours_loss_and_grad(loss_type, old_lp):
        lp_leaf = lp0.clone().requires_grad_(True)
        loss, _ = grpo_loss(
            lp_leaf, behavior, adv, mask2, old_logprobs=old_lp,
            tis_cap=tis_cap, clip_eps_low=0.2, clip_eps_high=0.2, loss_type=loss_type,
        )
        (grad,) = torch.autograd.grad(loss, lp_leaf)
        return loss.detach(), grad

    for iter_tag, old_lp in [("1iter", None), ("multi-iter", old_multi)]:
        for loss_type in ["dapo", "grpo", "dr_grpo"]:
            lt, gt = trl_loss_and_grad(loss_type, old_lp)
            lo, go = ours_loss_and_grad(loss_type, old_lp)
            vrel = float((lo - lt).abs() / lt.abs().clamp(min=1e-12))
            grel = rel_l2(go.double().reshape(-1), gt.double().reshape(-1))
            note = ""
            if loss_type == "dapo":
                note = "per-call: num_items_in_batch=this batch's tokens (TRL norm is global-batch)"
            if loss_type == "dr_grpo":
                note = "TRL divides by B*config.max_completion_length; set == our padded width"
            rows.append(_row(f"check2 loss {loss_type} [{iter_tag}]", "rel", vrel, 1e-5, note=note))
            rows.append(_row(f"check2 grad(lp) {loss_type} [{iter_tag}]", "rel_l2", grel, 1e-5))

    _print_table(rows)
    n_fail = sum(r["status"] == "FAIL" for r in rows)
    print(f"\n{n_fail} hard failure(s) out of {len(rows)} rows")

    return {
        "versions": versions,
        "trl_defaults": trl_defaults,
        "rows": rows,
        "notes": notes,
        "preconditions": {"tis_frac_over_cap": frac_cap,
                          "clip_tokens_pos_adv": clip_hi, "clip_tokens_neg_adv": clip_lo},
        "n_fail": n_fail,
    }


@app.local_entrypoint()
def main():
    result = crosscheck.remote()
    _print_table(result["rows"])
    out = json.dumps(result, indent=2)
    with open("rl/crosscheck_results.json", "w") as f:
        f.write(out)
    print("\nwrote rl/crosscheck_results.json")
    print("TRL version:", result["versions"]["trl"])
    if result["n_fail"]:
        print(f"FAILED: {result['n_fail']} hard check(s) failed")
        raise SystemExit(1)
    print("ALL HARD CHECKS PASSED")
