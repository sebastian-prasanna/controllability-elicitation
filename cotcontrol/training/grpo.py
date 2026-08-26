"""GRPO loss and memory-safe logprob computation for RL on CoT controllability.

Semantics deliberately mirror TRL's GRPOTrainer (single-iteration default) so
our implementation can be cross-checked against it on identical inputs:

  - advantages: per-group reward centering, optionally divided by group std
    (Dr.GRPO argues against std scaling; default is centering only).
  - loss: token-level policy gradient. With no stored old-policy logprobs
    (num_iterations=1, our default) the PPO clip is inert and the gradient
    coefficient is carried by exp(lp - sg(lp)) == 1; passing old_logprobs
    enables the standard clipped-ratio objective for multi-epoch reuse.
  - off-policy correction: rollouts come from vLLM, whose forward is not
    numerically the trainer's forward (kernels, MXFP4-vs-bf16 experts, MoE
    router drift). Truncated importance sampling (Yao et al. 2025) multiplies
    each token's loss by sg(min(pi_train/pi_vllm, cap)) — TRL's
    vllm_importance_sampling_correction, on by default there too.

Vocab-memory note: gpt-oss's ~201k vocab makes full [B, L, V] logits ~6.4GB
per 16k-token row in bf16 (more in fp32, doubled by autograd). The GRPO loss
with KL=0 only ever needs the CHOSEN token's logprob, so selective_logprobs
computes lm_head -> log_softmax -> gather in sequence chunks under activation
checkpointing: autograd retains only hidden states and one chunk of logits is
alive at a time (recomputed in backward).

Trainer-side logits must be temperature-scaled to match rollout sampling:
vLLM's returned logprobs are post-temperature, so ratios computed against
unscaled trainer logprobs would be systematically biased. Rollouts must also
sample with top_p=1.0/no top_k — vLLM reports logprobs of the pre-truncation
distribution, so nucleus sampling would make the behavior logprobs wrong.

Pure torch — no modal/transformers imports — so it unit-tests on CPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint


def selective_logprobs(
    lm_head: torch.nn.Module,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Logprob of target_ids[i] under softmax(lm_head(hidden[i]) / temperature).

    hidden: [N, H] rows already aligned so row i predicts target_ids[i] ([N]).
    Returns [N] fp32. Chunked over N and activation-checkpointed so the [chunk,
    vocab] logits are never stored for backward. Gradients flow into hidden
    (and lm_head params, if trainable).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if hidden.shape[0] != target_ids.shape[0]:
        raise ValueError(f"hidden rows {hidden.shape[0]} != targets {target_ids.shape[0]}")
    if hidden.shape[0] == 0:
        return hidden.new_zeros((0,), dtype=torch.float32)

    # accelerate's mixed-precision wrapper upcasts model outputs to fp32, so
    # hidden may arrive fp32 while lm_head holds bf16 weights — cast back
    # (lossless: the fp32 values were bf16 to begin with).
    head_param = next(lm_head.parameters(), None)
    head_dtype = head_param.dtype if head_param is not None else hidden.dtype

    def _chunk(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = lm_head(h.to(head_dtype)).float() / temperature
        return torch.gather(logits, -1, t.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(
            logits, dim=-1
        )

    out: List[torch.Tensor] = []
    for i in range(0, hidden.shape[0], chunk_size):
        h, t = hidden[i : i + chunk_size], target_ids[i : i + chunk_size]
        if hidden.requires_grad:
            out.append(checkpoint(_chunk, h, t, use_reentrant=False))
        else:
            out.append(_chunk(h, t))
    return torch.cat(out)


def completion_logprobs(
    lm_head: torch.nn.Module,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token logprobs of each sequence's completion, from a right-padded
    batch forward. The token-shift lives here and nowhere else: completion
    token at absolute position p (prompt_lens[b] <= p < seq_lens[b]) is
    predicted by hidden[b, p-1].

    hidden: [B, L, H]; input_ids: [B, L]; prompt_lens/seq_lens: [B] (token
    counts, prompt_lens >= 1). Returns (logprobs [B, Cmax] fp32, mask [B, Cmax]
    bool) where Cmax = max completion length; padded entries are 0/False.
    """
    comp_lens = seq_lens - prompt_lens
    if (comp_lens < 0).any() or (prompt_lens < 1).any():
        raise ValueError("need prompt_lens >= 1 and seq_lens >= prompt_lens")
    bsz, cmax = hidden.shape[0], int(comp_lens.max())

    batch_idx, pos_idx, comp_idx = [], [], []
    for b in range(bsz):
        p0, p1 = int(prompt_lens[b]), int(seq_lens[b])
        batch_idx.extend([b] * (p1 - p0))
        pos_idx.extend(range(p0 - 1, p1 - 1))  # predictor positions
        comp_idx.extend(range(p1 - p0))
    bi = torch.tensor(batch_idx, dtype=torch.long, device=hidden.device)
    pi = torch.tensor(pos_idx, dtype=torch.long, device=hidden.device)
    ci = torch.tensor(comp_idx, dtype=torch.long, device=hidden.device)

    flat_lp = selective_logprobs(
        lm_head, hidden[bi, pi], input_ids[bi, pi + 1], temperature, chunk_size
    )
    logprobs = hidden.new_zeros((bsz, cmax), dtype=torch.float32)
    mask = torch.zeros((bsz, cmax), dtype=torch.bool, device=hidden.device)
    logprobs.index_put_((bi, ci), flat_lp)
    mask[bi, ci] = True
    return logprobs, mask


def group_advantages(
    rewards: torch.Tensor, scale: str = "none", std_eps: float = 1e-4
) -> torch.Tensor:
    """Group-relative advantages: A = r - mean(group). rewards: [n_groups, G].

    scale="group" additionally divides by (group std + std_eps), matching
    TRL's scale_rewards="group"; default "none" is centering only (Dr.GRPO).
    """
    adv = rewards - rewards.mean(dim=1, keepdim=True)
    if scale == "group":
        adv = adv / (rewards.std(dim=1, keepdim=True) + std_eps)
    elif scale != "none":
        raise ValueError(f"unknown scale {scale!r}")
    return adv


def degenerate_groups(rewards: torch.Tensor, atol: float = 1e-8) -> torch.Tensor:
    """[n_groups] bool: groups whose rewards are all (near-)identical. Their
    advantages are ~0 so they contribute no gradient — workers can drop them
    before the forward pass to save compute (the old tinker loop did)."""
    return (rewards - rewards[:, :1]).abs().max(dim=1).values <= atol


def grpo_loss(
    policy_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    old_logprobs: Optional[torch.Tensor] = None,
    tis_cap: float = 2.0,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.2,
    loss_type: str = "dapo",
    loss_denominator: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Clipped token-level GRPO loss with truncated-IS off-policy correction.

    policy_logprobs [B, C] (requires grad), behavior_logprobs [B, C] (vLLM's,
    temperature-scaled), advantages [B] (one per sequence), completion_mask
    [B, C] bool. old_logprobs: stored logprobs of the policy that generated
    the batch, for multi-epoch reuse; None (default) uses sg(policy) — clip
    inert, exactly TRL's num_iterations=1 path.

    loss_type: "dapo" = sum/total-completion-tokens; "grpo" = per-sequence
    token-mean, then mean over sequences; "dr_grpo" = sum / (B * C) with C the
    padded completion width.

    loss_denominator (dapo only): explicit token-count normalizer replacing
    this call's mask.sum(). Under gradient accumulation and FSDP data
    parallelism the correct DAPO normalizer is the GLOBAL completion-token
    count of the optimizer batch (what TRL 1.10 uses via num_items_in_batch);
    per-call normalization overweights small micro-batches. Callers must fold
    the DP mean-reduction of gradients in themselves (pass
    global_tokens / world_size).
    Returns (loss, detached metrics dict incl. train-vs-rollout mismatch
    stats — watch logprob_corr: rollout corruption shows up there first).
    """
    mask = completion_mask.float()
    old = policy_logprobs.detach() if old_logprobs is None else old_logprobs

    ratio = torch.exp(policy_logprobs - old)
    clipped = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)
    adv = advantages.unsqueeze(1)
    surrogate = torch.minimum(ratio * adv, clipped * adv)

    # Stop-gradient truncated IS weight between the old policy and vLLM's
    # sampling distribution (token-level TIS).
    tis = torch.exp(old - behavior_logprobs).clamp(max=tis_cap)
    per_token_loss = -surrogate * tis * mask

    if loss_type == "dapo":
        denom = loss_denominator if loss_denominator else float(mask.sum().clamp(min=1.0))
        loss = per_token_loss.sum() / denom
    elif loss_type == "grpo":
        seq_loss = per_token_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = seq_loss.mean()
    elif loss_type == "dr_grpo":
        loss = per_token_loss.sum() / (mask.shape[0] * mask.shape[1])
    else:
        raise ValueError(f"unknown loss_type {loss_type!r}")

    with torch.no_grad():
        m = completion_mask
        lp, blp = policy_logprobs[m], behavior_logprobs[m]
        raw_ratio = torch.exp(lp - blp)
        diff = lp - blp
        if lp.numel() > 1 and lp.std() > 0 and blp.std() > 0:
            corr = float(torch.corrcoef(torch.stack([lp, blp]))[0, 1])
        else:
            corr = float("nan")
        metrics = {
            "tis_ratio_mean": float(raw_ratio.mean()),
            "tis_ratio_max": float(raw_ratio.max()) if lp.numel() else float("nan"),
            "tis_frac_truncated": float((raw_ratio > tis_cap).float().mean()),
            "logprob_diff_mean": float(diff.mean()),
            "logprob_diff_abs_mean": float(diff.abs().mean()),
            "logprob_corr": corr,
            "clip_frac": float(((ratio - clipped).abs() > 0)[m].float().mean()),
            "n_completion_tokens": int(m.sum()),
        }
    return loss, metrics
