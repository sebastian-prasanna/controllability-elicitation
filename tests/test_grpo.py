"""CPU unit tests for cotcontrol.training.grpo.

Runs under pytest, or as a plain script: python tests/test_grpo.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotcontrol.training.grpo import (  # noqa: E402
    completion_logprobs,
    degenerate_groups,
    group_advantages,
    grpo_loss,
    selective_logprobs,
)

VOCAB, HID = 37, 16


def _lm_head(trainable=True) -> nn.Linear:
    torch.manual_seed(3)
    head = nn.Linear(HID, VOCAB, bias=False)
    head.weight.requires_grad_(trainable)
    return head


def _naive_logprobs(head, hidden, targets, temperature):
    logits = head(hidden).float() / temperature
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def test_selective_logprobs_matches_naive_values_and_grads():
    torch.manual_seed(0)
    head = _lm_head()
    for temperature in (1.0, 0.7):
        for n in (0, 1, 5):  # 5 with chunk_size=2 exercises multiple chunks + remainder
            hidden = torch.randn(n, HID, requires_grad=True)
            targets = torch.randint(0, VOCAB, (n,))
            got = selective_logprobs(head, hidden, targets, temperature, chunk_size=2)
            want = _naive_logprobs(head, hidden, targets, temperature)
            assert got.shape == (n,) and got.dtype == torch.float32
            assert torch.allclose(got, want, atol=1e-6)
            if n == 0:
                continue
            g_hid, g_w = torch.autograd.grad(got.sum(), [hidden, head.weight], retain_graph=False)
            hidden2 = hidden.detach().clone().requires_grad_(True)
            want2 = _naive_logprobs(head, hidden2, targets, temperature)
            e_hid, e_w = torch.autograd.grad(want2.sum(), [hidden2, head.weight])
            assert torch.allclose(g_hid, e_hid, atol=1e-6)
            assert torch.allclose(g_w, e_w, atol=1e-6)


def test_selective_logprobs_mixed_dtype_hidden():
    """accelerate's bf16 wrapper upcasts model outputs to fp32 while lm_head
    stays bf16 — selective_logprobs must cast back instead of crashing
    (regression: F.linear dtype mismatch in the first RL smoke)."""
    head = _lm_head().to(torch.bfloat16)
    hidden32 = torch.randn(5, HID, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, VOCAB, (5,))
    got = selective_logprobs(head, hidden32, targets, chunk_size=2)
    logits = head(hidden32.to(torch.bfloat16)).float()
    want = torch.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, want, atol=1e-6)
    (g,) = torch.autograd.grad(got.sum(), hidden32)
    assert torch.isfinite(g).all()


def test_selective_logprobs_frozen_head_no_grad_hidden():
    head = _lm_head(trainable=False)
    hidden = torch.randn(4, HID)  # no requires_grad: pure inference path
    targets = torch.randint(0, VOCAB, (4,))
    got = selective_logprobs(head, hidden, targets, chunk_size=3)
    assert not got.requires_grad
    assert torch.allclose(got, _naive_logprobs(head, hidden, targets, 1.0), atol=1e-6)


def test_completion_logprobs_alignment_and_mask():
    torch.manual_seed(1)
    head = _lm_head()
    bsz, L = 3, 10
    input_ids = torch.randint(0, VOCAB, (bsz, L))
    hidden = torch.randn(bsz, L, HID, requires_grad=True)
    prompt_lens = torch.tensor([2, 5, 9])
    seq_lens = torch.tensor([7, 10, 9])  # completion lens 5, 5, 0
    lp, mask = completion_logprobs(head, hidden, input_ids, prompt_lens, seq_lens, 0.9)
    assert lp.shape == mask.shape == (bsz, 5)
    assert mask.sum(dim=1).tolist() == [5, 5, 0]
    for b in range(bsz):
        p0, p1 = int(prompt_lens[b]), int(seq_lens[b])
        for j in range(p1 - p0):
            # completion token p0+j predicted from hidden position p0+j-1
            want = _naive_logprobs(
                head, hidden[b, p0 + j - 1 : p0 + j], input_ids[b, p0 + j : p0 + j + 1], 0.9
            )
            assert torch.allclose(lp[b, j], want[0], atol=1e-6)
    assert (lp[~mask] == 0).all()
    # gradient reaches only predictor positions of real completion tokens
    (g,) = torch.autograd.grad((lp * mask).sum(), hidden)
    used = torch.zeros(bsz, L, dtype=torch.bool)
    used[0, 1:6], used[1, 4:9] = True, True
    assert (g.abs().sum(-1) > 0).eq(used).all()


def test_group_advantages():
    r = torch.tensor([[1.0, 0.0, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0]])
    a = group_advantages(r)
    assert torch.allclose(a.sum(dim=1), torch.zeros(2), atol=1e-7)
    assert torch.allclose(a[0], torch.tensor([0.5, -0.5, 0.0, 0.0]))
    assert (a[1] == 0).all()
    scaled = group_advantages(r, scale="group")
    assert torch.allclose(scaled[0], a[0] / (r[0].std() + 1e-4))
    try:
        group_advantages(r, scale="batch")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_degenerate_groups():
    r = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert degenerate_groups(r).tolist() == [True, False, True]


def test_grpo_loss_gradient_identity_single_iteration():
    """With old=None (clip inert), dL/dlp must be exactly
    -A * min(exp(lp - behavior_lp), cap) * mask / n_tokens (dapo)."""
    torch.manual_seed(2)
    B, C = 4, 6
    lp = torch.randn(B, C, requires_grad=True)
    behavior = lp.detach() + 0.3 * torch.randn(B, C)
    behavior[0, 0] -= 1.0  # force at least one masked token over the cap
    adv = torch.randn(B)
    mask = torch.rand(B, C) > 0.3
    mask[:, 0] = True
    cap = 1.5
    loss, metrics = grpo_loss(lp, behavior, adv, mask, tis_cap=cap)
    (g,) = torch.autograd.grad(loss, lp)
    tis = torch.exp(lp.detach() - behavior).clamp(max=cap)
    want = -adv.unsqueeze(1) * tis * mask.float() / mask.sum()
    assert torch.allclose(g, want, atol=1e-6)
    assert metrics["tis_frac_truncated"] > 0  # cap actually exercised
    assert metrics["n_completion_tokens"] == int(mask.sum())


def test_grpo_loss_clip_active_with_old_logprobs():
    # adv > 0 and ratio far above 1+eps: clipped branch wins, gradient is 0.
    lp = torch.tensor([[0.0]], requires_grad=True)
    old = torch.tensor([[-1.0]])  # ratio = e > 1.2
    behavior = old.clone()  # tis = 1
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss, _ = grpo_loss(lp, behavior, torch.tensor([1.0]), mask, old_logprobs=old)
    (g,) = torch.autograd.grad(loss, lp)
    assert torch.allclose(loss, torch.tensor(-1.2))  # -(1+eps) * adv
    assert torch.allclose(g, torch.zeros_like(g))
    # adv < 0 with the same ratio: unclipped branch is the min, gradient flows.
    lp2 = torch.tensor([[0.0]], requires_grad=True)
    loss2, _ = grpo_loss(lp2, behavior, torch.tensor([-1.0]), mask, old_logprobs=old)
    (g2,) = torch.autograd.grad(loss2, lp2)
    assert g2.abs().sum() > 0


def test_grpo_loss_normalizations():
    lp = torch.zeros(2, 3, requires_grad=True)
    behavior = torch.zeros(2, 3)
    adv = torch.tensor([1.0, 1.0])
    mask = torch.tensor([[True, True, True], [True, False, False]])
    # per-token loss is -adv everywhere on-mask (ratio=tis=1)
    dapo, _ = grpo_loss(lp, behavior, adv, mask, loss_type="dapo")
    grpo, _ = grpo_loss(lp, behavior, adv, mask, loss_type="grpo")
    drg, _ = grpo_loss(lp, behavior, adv, mask, loss_type="dr_grpo")
    assert torch.allclose(dapo, torch.tensor(-1.0))
    assert torch.allclose(grpo, torch.tensor(-1.0))
    assert torch.allclose(drg, torch.tensor(-4.0 / 6.0))
    # explicit global normalizer (grad-accum / DP): -4 on-mask tokens / 8
    dapo8, _ = grpo_loss(lp, behavior, adv, mask, loss_type="dapo", loss_denominator=8.0)
    assert torch.allclose(dapo8, torch.tensor(-0.5))
    try:
        grpo_loss(lp, behavior, adv, mask, loss_type="ppo")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_grpo_loss_mismatch_metrics():
    torch.manual_seed(4)
    lp = torch.randn(2, 5, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    adv = torch.randn(2)
    # behavior = policy - const: perfectly correlated, constant positive diff
    _, m = grpo_loss(lp, lp.detach() - 0.1, adv, mask)
    assert abs(m["logprob_corr"] - 1.0) < 1e-5
    assert abs(m["logprob_diff_mean"] - 0.1) < 1e-5
    _, m2 = grpo_loss(lp, lp.detach(), adv, mask)
    assert m2["tis_ratio_mean"] == 1.0 and m2["tis_frac_truncated"] == 0.0
    assert m2["clip_frac"] == 0.0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
