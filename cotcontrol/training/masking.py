"""Random-subset gradient masking over LoRA adapter weights.

Implements the elicitation method of "Quantifying Elicitation of Latent
Capabilities in Language Models" (Donoway et al., NeurIPS 2025): sample k
trainable scalar indices uniformly at random across ALL LoRA adapter weights
in all layers/modules jointly (every scalar equally likely, regardless of
position), once per run from a seed, and zero the gradient of everything else
so unselected weights stay exactly at their init values.

Pure torch — no modal/transformers imports — so it unit-tests on CPU.

LoRA-B-zero-init subtlety: PEFT inits B to zero, so a selected A entry whose
entire corresponding output path stays at B=0 receives zero gradient forever
and legitimately never moves (dL/dA = B^T @ ... = 0). Verification therefore
asserts that nothing OUTSIDE the mask changed — the paper's check — not that
everything inside did.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def _is_dtensor(t) -> bool:
    # Guarded import: single-GPU runs work without torch.distributed initialized.
    try:
        from torch.distributed.tensor import DTensor
        return isinstance(t, DTensor)
    except Exception:
        return False


def _full_tensor(t: torch.Tensor) -> torch.Tensor:
    """Gather a DTensor to its full (unsharded) tensor; identity otherwise.
    Collective under FSDP2 — every rank must call it in the same order."""
    if _is_dtensor(t):
        return t.full_tensor()
    return t


def _normalize_name(name: str) -> str:
    """Strip DDP's 'module.' wrapper prefixes so mask keys built pre-wrap still
    match post-wrap parameter names. (FSDP2 keeps the module tree intact.)"""
    while name.startswith("module."):
        name = name[len("module."):]
    return name


def build_global_mask(
    named_numels: List[Tuple[str, int]], k: int, seed: int
) -> Dict[str, torch.LongTensor]:
    """Sample k unique flat indices uniformly over sum(numels) and map them to
    per-parameter local flat indices.

    Deterministic given (named_numels contents, k, seed): entries are sorted by
    name so caller iteration order doesn't matter, and sampling uses a local
    torch.Generator. Rejection-samples uniqueness (fine for k << N; raises if
    k > N). EVERY param appears in the result — with an empty index tensor if
    it drew no indices — so apply_gradient_masks can freeze it; omitting it
    would leave the param hookless and training freely (caught by verify_mask
    in the first GPU smoke run).
    """
    named_numels = sorted(named_numels)
    total = sum(n for _, n in named_numels)
    if k > total:
        raise ValueError(f"k={k} exceeds total LoRA scalar count {total}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    gen = torch.Generator().manual_seed(seed)
    chosen: set = set()
    while len(chosen) < k:
        need = k - len(chosen)
        # Oversample per round; duplicates are rejected. Consuming candidates
        # in draw order keeps this deterministic.
        for c in torch.randint(0, total, (max(4 * need, 256),), generator=gen).tolist():
            if c not in chosen:
                chosen.add(c)
                if len(chosen) == k:
                    break

    # Walk the sorted global indices against the per-param offset ranges.
    sorted_idx = sorted(chosen)
    result: Dict[str, torch.LongTensor] = {}
    offset, i = 0, 0
    for name, numel in named_numels:
        hi = offset + numel
        local: List[int] = []
        while i < len(sorted_idx) and sorted_idx[i] < hi:
            local.append(sorted_idx[i] - offset)
            i += 1
        result[name] = torch.tensor(local, dtype=torch.long)
        offset = hi
    return result


def apply_gradient_masks(
    model, masks: Dict[str, torch.LongTensor]
) -> Dict[str, torch.Tensor]:
    """Register per-parameter gradient hooks that zero grad entries outside the
    mask, so unselected weights never receive an optimizer update.

    Call AFTER any FSDP wrapping: FSDP2's fully_shard swaps each nn.Parameter
    for a new DTensor-backed parameter, so hooks registered earlier would sit
    on dead tensors. For a DTensor param the full boolean mask is built at the
    global shape and distributed with the param's own mesh/placements, so each
    rank masks its shard consistently. Returns the mask-tensor dict for reuse.

    Masking acts on GRADIENTS only: optimizer updates that bypass the gradient
    — AdamW weight decay in particular, which decays every param holding a
    .grad tensor even if it's all zeros — would still move masked-out weights.
    Callers must train with weight_decay=0 (the worker enforces this).
    """
    params = {_normalize_name(n): p for n, p in model.named_parameters()}
    mask_tensors: Dict[str, torch.Tensor] = {}
    for name, idx in masks.items():
        param = params[name]  # KeyError here = mask built on a different model
        if idx.numel() == 0:
            # No selected entries: freeze outright. AdamW skips grad-None
            # params entirely, so neither updates nor weight decay touch it.
            param.requires_grad_(False)
            continue
        flat = torch.zeros(param.numel(), dtype=torch.bool)
        flat[idx] = True
        full_mask = flat.view(param.shape)
        if _is_dtensor(param):
            from torch.distributed.tensor import distribute_tensor
            mask = distribute_tensor(full_mask, param.device_mesh, param.placements)
        else:
            mask = full_mask.to(param.device)
        mask_tensors[name] = mask

        if hasattr(param, "register_post_accumulate_grad_hook"):
            # Fires once .grad is fully accumulated for the backward — in-place
            # zeroing here covers gradient-accumulation micro-batches too.
            def _post_hook(p, m=mask):
                if p.grad is not None:
                    p.grad.mul_(m)
            param.register_post_accumulate_grad_hook(_post_hook)
        else:
            # Older torch: autograd hook on the tensor (out-of-place).
            param.register_hook(lambda g, m=mask: g * m)
    return mask_tensors


def snapshot_params(model) -> Dict[str, torch.Tensor]:
    """CPU copies of all trainable params (LoRA adapters — small)."""
    snap: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            snap[_normalize_name(name)] = _full_tensor(p.detach()).cpu().clone()
    return snap


def verify_mask(
    initial: Dict[str, torch.Tensor], model, masks: Dict[str, torch.LongTensor]
) -> dict:
    """Compare current param values against the init snapshot; count exact (!=)
    elementwise changes inside vs outside the mask. The invariant is
    n_changed_outside_mask == 0 (see module docstring for why inside-mask
    entries may legitimately be unchanged). Collective-safe: iterates in sorted
    order and gathers DTensor shards, so call on every rank under FSDP2.
    """
    params = {_normalize_name(n): p for n, p in model.named_parameters()}
    per_param: Dict[str, dict] = {}
    n_selected = n_inside = n_outside = 0
    for name in sorted(initial):
        final = _full_tensor(params[name].detach()).cpu()
        init = initial[name].to(final.dtype)
        changed = (final != init).view(-1)
        sel = torch.zeros(changed.numel(), dtype=torch.bool)
        idx = masks.get(name)
        if idx is not None:
            sel[idx] = True
        inside = int((changed & sel).sum())
        outside = int((changed & ~sel).sum())
        per_param[name] = {
            "numel": changed.numel(),
            "n_selected": int(sel.sum()),
            "n_changed_inside_mask": inside,
            "n_changed_outside_mask": outside,
        }
        n_selected += int(sel.sum())
        n_inside += inside
        n_outside += outside
    return {
        "n_selected": n_selected,
        "n_changed_inside_mask": n_inside,
        "n_changed_outside_mask": n_outside,
        "per_param": per_param,
    }
