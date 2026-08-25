"""CPU unit tests for cotcontrol.training.masking.

Runs under pytest, or as a plain script: python tests/test_masking.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotcontrol.training.masking import (  # noqa: E402
    apply_gradient_masks,
    build_global_mask,
    snapshot_params,
    verify_mask,
)


def _add_param(root: nn.Module, dotted: str, shape, requires_grad=True):
    parts = dotted.split(".")
    cur = root
    for p in parts[:-1]:
        if not hasattr(cur, p):
            setattr(cur, p, nn.Module())
        cur = getattr(cur, p)
    cur.register_parameter(
        parts[-1], nn.Parameter(torch.randn(*shape), requires_grad=requires_grad)
    )


def make_model() -> nn.Module:
    """Tiny model with PEFT-style names: several differently-shaped trainable
    LoRA params plus frozen non-LoRA params."""
    torch.manual_seed(7)
    m = nn.Module()
    _add_param(m, "base.layers.0.self_attn.q_proj.lora_A.default.weight", (4, 8))
    _add_param(m, "base.layers.0.self_attn.q_proj.lora_B.default.weight", (8, 4))
    _add_param(m, "base.layers.1.self_attn.v_proj.lora_A.default.weight", (4, 6))
    _add_param(m, "base.layers.1.self_attn.v_proj.lora_B.default.weight", (6, 4))
    _add_param(m, "base.layers.0.self_attn.q_proj.weight", (8, 8), requires_grad=False)
    _add_param(m, "base.layers.1.self_attn.v_proj.weight", (6, 6), requires_grad=False)
    return m


def trainable_numels(model):
    return [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]


def test_build_global_mask_determinism_and_uniqueness():
    model = make_model()
    nn_ = trainable_numels(model)
    total = sum(n for _, n in nn_)

    m1 = build_global_mask(nn_, 40, seed=0)
    m2 = build_global_mask(list(reversed(nn_)), 40, seed=0)  # caller order irrelevant
    m3 = build_global_mask(nn_, 40, seed=1)

    assert set(m1) == set(m2)
    for name in m1:
        assert torch.equal(m1[name], m2[name])
    flat1 = {(n, int(i)) for n, t in m1.items() for i in t}
    flat3 = {(n, int(i)) for n, t in m3.items() for i in t}
    assert flat1 != flat3, "different seeds should give different masks"

    # exactly k unique indices, all within bounds
    assert len(flat1) == 40
    # every param present (empty entries stay so apply can freeze them)
    assert set(m1) == {n for n, _ in nn_}
    numel_by_name = dict(nn_)
    for name, t in m1.items():
        assert len(set(t.tolist())) == len(t)
        if len(t):
            assert int(t.max()) < numel_by_name[name]

    # a large-k draw spreads over every param (4 params, k = total - 5)
    mbig = build_global_mask(nn_, total - 5, seed=0)
    assert set(mbig) == {n for n, _ in nn_}
    assert sum(len(t) for t in mbig.values()) == total - 5

    # k > N raises; k == 0 is empty
    try:
        build_global_mask(nn_, total + 1, seed=0)
        assert False, "expected ValueError for k > N"
    except ValueError:
        pass
    # k == 0: every entry present and empty (-> apply freezes everything)
    m0 = build_global_mask(nn_, 0, seed=0)
    assert set(m0) == {n for n, _ in nn_}
    assert all(t.numel() == 0 for t in m0.values())
    print("test_build_global_mask_determinism_and_uniqueness passed")


def test_apply_gradient_masks_adamw():
    model = make_model()
    k = 25
    masks = build_global_mask(trainable_numels(model), k, seed=3)
    initial = snapshot_params(model)
    mask_tensors = apply_gradient_masks(model, masks)
    # hooks only on params with selected entries; empty-mask params got frozen
    assert set(mask_tensors) == {n for n, t in masks.items() if t.numel()}
    for n, t in masks.items():
        if t.numel() == 0:
            assert not dict(model.named_parameters())[n].requires_grad

    # Linear loss -> nonzero gradient at every trainable entry, so any
    # unmasked-but-changed entry would be caught. Fixed direction tensors keep
    # it deterministic.
    torch.manual_seed(11)
    directions = {n: torch.randn_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0
    )
    for _ in range(3):
        opt.zero_grad()
        loss = sum((p * directions[n]).sum()
                   for n, p in model.named_parameters() if p.requires_grad)
        loss.backward()
        opt.step()

    report = verify_mask(initial, model, masks)
    assert report["n_selected"] == k
    assert report["n_changed_outside_mask"] == 0, report
    # Linear loss gives nonzero grad everywhere, so all selected entries move.
    assert report["n_changed_inside_mask"] == k, report

    # Frozen non-LoRA params are untouched (they never had grads).
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None

    # Bit-identical outside the mask, per-param.
    params = dict(model.named_parameters())
    for name, init in initial.items():
        final = params[name].detach()
        flat_changed = (final != init).view(-1)
        sel = torch.zeros(final.numel(), dtype=torch.bool)
        if name in masks:
            sel[masks[name]] = True
        assert not bool((flat_changed & ~sel).any()), f"unmasked change in {name}"
    print("test_apply_gradient_masks_adamw passed")


def test_verify_mask_catches_corruption():
    model = make_model()
    masks = build_global_mask(trainable_numels(model), 10, seed=5)
    initial = snapshot_params(model)

    clean = verify_mask(initial, model, masks)
    assert clean["n_changed_inside_mask"] == 0
    assert clean["n_changed_outside_mask"] == 0

    # Corrupt one entry that is OUTSIDE the mask.
    name = "base.layers.0.self_attn.q_proj.lora_A.default.weight"
    p = dict(model.named_parameters())[name]
    sel = set(masks.get(name, torch.tensor([], dtype=torch.long)).tolist())
    bad_idx = next(i for i in range(p.numel()) if i not in sel)
    with torch.no_grad():
        p.view(-1)[bad_idx] += 1.0

    report = verify_mask(initial, model, masks)
    assert report["n_changed_outside_mask"] == 1, report
    assert report["per_param"][name]["n_changed_outside_mask"] == 1
    print("test_verify_mask_catches_corruption passed")


if __name__ == "__main__":
    test_build_global_mask_determinism_and_uniqueness()
    test_apply_gradient_masks_adamw()
    test_verify_mask_catches_corruption()
    print("all masking tests passed")
