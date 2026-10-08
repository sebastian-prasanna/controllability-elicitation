"""TrainConfig dataclass <-> YAML for Modal LoRA training.

Adapted from the subliminal-learning Modal trainer's config, with the tinker
parity constants stripped: HF-default AdamW (eps 1e-8, betas 0.9/0.999), grad
clipping at 1.0, configurable scheduler/weight decay. Adds the elicitation
knobs from Donoway et al. (NeurIPS 2025): ``train_params`` (k = number of
individual LoRA scalar weights allowed to train, sampled uniformly at random
across all adapters) and ``mask_seed``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union


def default_lora_targets(model_name: str) -> Tuple[tuple, tuple]:
    """(target_modules, target_parameters) for an all-linear-incl-experts
    adapter on this architecture. NOT applied automatically — TrainConfig
    defaults to attention-only (the paper's most parameter-efficient choice);
    import this for configs that want full coverage.

    Fused-MoE models (Qwen3 "A<N>B", gpt-oss) store experts as fused
    nn.Parameters under transformers 5, reachable only via PEFT's
    ``target_parameters``; dense models use ordinary nn.Linear MLPs.

    Hybrid gated-delta-net models (Qwen3.5 / Qwen3.6 / Qwen3-Next) run 3 of
    every 4 layers through linear attention, whose projections are named
    ``in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj`` — so
    q/k/v/o alone reaches a quarter of the layers. Their MoE variants also
    carry a dense ``shared_expert`` MLP (gate/up/down_proj) next to the fused
    routed experts. PEFT ignores target_modules names absent from a model, so
    listing the union is safe across variants."""
    name = model_name.lower()
    is_moe = bool(re.search(r"a\d+b", name)) or "moe" in name or "gpt-oss" in name
    is_hybrid = bool(re.search(r"qwen3[._-]?[56]\b|qwen3[._-]?next", name))
    attn = ("q_proj", "k_proj", "v_proj", "o_proj")
    gdn = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
    mlp = ("gate_proj", "up_proj", "down_proj")
    experts = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
    if is_hybrid:
        # dense MLP (Qwen3.6-27B) or shared expert (35B-A3B) — same names
        return (attn + gdn + mlp, experts if is_moe else ())
    if is_moe:
        return (attn, experts)
    return (attn + mlp, ())


@dataclass
class TrainConfig:
    """Hyperparameters for Modal LoRA SFT.

    ``batch_size`` is the *effective* batch (examples per optimizer step); the
    worker derives gradient_accumulation_steps from it given
    ``per_device_batch_size`` and the GPU count.
    """

    run_name: str = "run"
    base_model: str = "Qwen/Qwen3-8B"

    # --- data ---
    # jsonl path(s). Two row formats, detected per row and freely mixable:
    #   chat: {"input": [messages], "output": [messages]}  (prompt masked,
    #         assistant completion supervised — see rendering.render_example)
    #   text: {"text": str, "prefix": str?}  raw pretraining-style document,
    #         every token supervised, terminated with the tokenizer's document
    #         end token; optional "prefix" is prepended and masked (-100), e.g.
    #         a "<DOCTAG>\n" conditioning tag. See rendering.render_text.
    # Consumed by the launcher (load_sft_rows), not shipped to the worker.
    data_path: Union[str, List[str], None] = None
    num_examples: Optional[int] = None  # cap on training examples (post-shuffle)
    shuffle: bool = True  # seeded pre-shuffle of rows + per-epoch shuffling
    # Concatenate rendered rows (after the seeded shuffle) into fixed
    # max_seq_length chunks, keeping each row's label mask. For short documents
    # this is the difference between ~700 and ~16k supervised tokens per
    # sequence. Rows are packed in order and may straddle a chunk boundary;
    # attention is not blocked across documents (same as Together/HF packing).
    pack_sequences: bool = False

    # --- optimization ---
    lr: float = 1e-4
    adam_epsilon: float = 1e-8       # HF default; configurable for sweeps
    weight_decay: float = 0.0       # must stay 0.0 when train_params is set
                                    # (decay moves masked-out weights off init)
    max_grad_norm: float = 1.0       # HF default clipping; 0 disables
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0
    batch_size: int = 32             # effective batch (examples per optim step)
    per_device_batch_size: int = 1
    num_epochs: int = 3
    max_steps: int = -1              # -1 = use epochs; >0 caps optimizer steps
    seed: int = 42

    # --- checkpoint cadence ---
    # Epoch cadence used when the two fields below are unset (1 = every epoch).
    save_sampling_step: int = 1
    save_every_n_steps: Optional[int] = None    # overrides epoch cadence
    save_at_steps: Optional[List[int]] = None   # explicit steps; overrides both

    # --- Modal / parallelism ---
    gpu: str = "H200"                # Modal GPU spec, e.g. "H200", "H200:4"
    parallelism: str = "fsdp"        # "fsdp" (multi-GPU sharding) or "ddp"
    fsdp_version: int = 2            # FSDP2 = per-parameter DTensor sharding
    gradient_checkpointing: bool = False
    max_seq_length: int = 16384
    timeout_seconds: int = 24 * 3600

    # --- LoRA ---
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # Attention-only by default — the elicitation paper found attention
    # projections the most parameter-efficient placement. See
    # default_lora_targets() for all-linear-incl-experts.
    lora_target_modules: object = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Fused-MoE expert tensors (nn.Parameter, not nn.Linear) need PEFT's
    # target_parameters. () = none. Nonexistent entries are filtered by the
    # worker so one config can serve dense and MoE models.
    lora_target_parameters: object = ()

    # --- elicitation (Donoway et al.) ---
    # k = number of individual LoRA scalar weights allowed to train, sampled
    # uniformly at random across ALL adapter weights jointly (mask_seed), fixed
    # for the run. None = train all LoRA params (plain LoRA SFT, no masking).
    train_params: Optional[int] = None
    mask_seed: int = 0

    # Extra chat-template kwargs for prompt rendering, e.g.
    # {"reasoning_effort": "high"} for gpt-oss, {"enable_thinking": False} to
    # train Qwen3 with reasoning disabled (the empty <think></think> then lands
    # in the PROMPT, masked, rather than being supervised).
    chat_template_kwargs: Optional[dict] = None
    # Raw text appended to the rendered prompt and labeled -100 — a fixed
    # opening of the assistant turn the model is conditioned on but not trained
    # to emit. gpt-oss "thinking disabled" uses
    #   "<|channel|>analysis<|message|><|end|><|start|>assistant"
    # to get the masked empty analysis channel that Qwen gets from
    # enable_thinking=False. See rendering.render_example.
    assistant_prefill: Optional[str] = None
    # HF attention backend. None = auto: "sdpa", except gpt-oss -> "eager"
    # (no SDPA support; eager is O(L^2) memory so revisit for long contexts).
    attn_implementation: Optional[str] = None
    # Liger fused linear cross-entropy (ONLY that kernel; rms_norm/swiglu/rope
    # patches stay off so numerics match unpatched runs as closely as
    # possible). Avoids materializing the [seq, vocab] fp32 logits + grad:
    # 16k tokens x 248k vocab (Qwen3.5/3.6) is 15 GiB each, which alone
    # pushes Qwen3.6-35B-A3B off a single H200. Needs liger-kernel in the
    # image with a patch for the architecture (qwen3_5_moe: >=0.8.2).
    use_liger_kernel: bool = False
    # Warm start: checkpoints-volume path to a saved LoRA checkpoint dir
    # (holding adapter_model.safetensors, e.g.
    # /checkpoints/<run>/checkpoint-320). The worker overwrites its
    # freshly-initialized adapter with these weights before masking/snapshot
    # and before the checkpoint-0 save, so training (SFT or RL) starts from
    # the donor policy and checkpoint-0 == donor. The adapter architecture
    # must match this config (rank/alpha/targets), and with train_params set
    # the donor run must have used the same mask_seed/k so the masks coincide
    # (verified against the donor's mask.json when present). Multi-GPU FSDP2
    # is supported: every rank loads pre-wrap, cross-rank checksum verified.
    init_lora_path: Optional[str] = None

    def __post_init__(self) -> None:
        # YAML 1.1 parses unquoted "1e-4" as a string (no decimal point), so
        # coerce numeric fields rather than trust the loader.
        for f in ("lr", "adam_epsilon", "weight_decay", "max_grad_norm",
                  "warmup_ratio", "lora_dropout"):
            setattr(self, f, float(getattr(self, f)))
        for f in ("batch_size", "per_device_batch_size", "num_epochs",
                  "max_steps", "seed", "save_sampling_step", "fsdp_version",
                  "max_seq_length", "timeout_seconds", "lora_rank",
                  "lora_alpha", "mask_seed"):
            setattr(self, f, int(getattr(self, f)))
        for f in ("num_examples", "save_every_n_steps", "train_params"):
            v = getattr(self, f)
            if v is not None:
                setattr(self, f, int(v))
        if self.save_at_steps is not None:
            self.save_at_steps = [int(s) for s in self.save_at_steps]


@dataclass
class RLConfig(TrainConfig):
    """GRPO RL on controllability. Extends TrainConfig (model/LoRA/masking/
    Modal fields reused; SFT-only fields like data_path/num_epochs are ignored).

    One iteration = save adapter -> driver serves it via vLLM LoRA hot-swap and
    returns graded rollouts -> one optimizer batch (inner_epochs passes).
    Rollout sampling MUST be temperature-only (top_p=1, no top_k): vLLM reports
    logprobs of the pre-truncation distribution, so nucleus sampling would make
    the behavior logprobs (and hence the TIS correction) wrong.
    """

    # --- loop shape ---
    iterations: int = 50            # rollout->update rounds
    prompts_per_iter: int = 32      # groups (questions) per iteration
    group_size: int = 8             # G rollouts per prompt
    inner_epochs: int = 1           # passes over each batch (>1 -> PPO clip active)

    # --- rollouts (driver side) ---
    rollout_temperature: float = 1.0
    rollout_max_tokens: int = 8192
    dataset: str = "all"            # gpqa | hle | mmlu_pro | all
    split: str = "train"
    # The 8 programmatic constraint modes (CONSTRAINT_MODES minus
    # ignore_question, whose grader is ~3 sync LLM-judge calls per rollout).
    modes: object = (
        "word_suppression", "multiple_word_suppression", "repeat_sentences",
        "end_of_sentence", "lowercase_thinking", "meow_between_words",
        "uppercase_thinking", "alternating_case",
    )
    system_prompt: str = ""
    # compliance_score (shaped + 2*strict)/3 | shaped_task_score |
    # shaped_compliance | anchored_compliance (compliance_score +
    # anchor_lambda * chance-corrected accuracy) | gated_shaped_compliance |
    # gated_anchored (length-gated shaped+strict compliance, zeroed unless the
    # output is pure \boxed{...}, + anchor_lambda * exact-match accuracy — for
    # free-form integer math datasets) | conjunctive_anchored (same
    # compliance term multiplied by (correct + conj_eps); no λ)
    reward: str = "compliance_score"
    anchor_lambda: float = 3.0      # accuracy-anchor weight (anchored_compliance only)
    # Adaptive λ: dual ascent on an accuracy setpoint (anchored_compliance only).
    # Each iteration the driver updates
    #   λ <- clip(λ + lambda_lr * (target_acc - acc_t), 0, lambda_max)
    # so λ rises to defend accuracy when it dips below target and decays toward
    # 0 (ceding reward mass to compliance) when accuracy is comfortable.
    # anchor_lambda is the initial λ.
    adaptive_lambda: bool = False
    lambda_target_acc: float = 0.0  # 0 -> auto: lambda_target_frac * it0 accuracy
    # Auto-setpoint fraction of iteration-0 accuracy. 0.95 over-defends at low
    # k (sweep15: adaptive lags fixed lam=0.5 peaks by 6-9pp at k<=10k because
    # lambda settles ~1.3-1.6); 0.9 aligns the controller with the honest-
    # checkpoint criterion (acc >= 0.9 x donor).
    lambda_target_frac: float = 0.9
    # Dual-ascent step per unit accuracy gap, applied to an EMA(0.3) of
    # accuracy. 5.0 (the original default) chased rollout noise to lambda_max
    # within ~6 iterations (sweep15); 0.75 moves λ ~0.04/iter for a 5pp gap.
    lambda_lr: float = 0.75
    lambda_max: float = 10.0
    # EMA rate for the measured accuracy the controller acts on
    # (acc_ema <- lambda_ema * acc_t + (1 - lambda_ema) * acc_ema). 0.3 = ~3-iter
    # smoothing (all sweeps through 23). Lower values lengthen the measurement
    # horizon; together with a lower lambda_lr this is the loop-gain reduction
    # against the ~130-step λ limit cycle seen at drive 1.
    lambda_ema: float = 0.3
    # PID Lagrangian (Stooke et al. 2020, arXiv 2007.03964): integral-only dual
    # ascent (the update above) is an ill-tuned integral controller against the
    # slow policy plant -> ~50-step compliance/accuracy limit cycles (sweeps
    # 18/19). Proportional term damps, derivative term (clamped to accuracy
    # WORSENING only) anticipates. lambda_kp = lambda_kd = 0 recovers plain
    # dual ascent; with PID on, the integral state is clamped to
    # [0, lambda_max] (anti-windup) and λ = clip(kp*err + I + kd*d, 0, max).
    lambda_kp: float = 0.0
    lambda_kd: float = 0.0
    # gated_shaped_compliance only: reasoning-length gate (chars) — saturating
    # multiplier min(1, len/gate) on shaped compliance. Plateau knob.
    length_gate_chars: float = 1500.0
    # gated_shaped_compliance only: strict-compliance bonus inside the gate
    # (safe there — the gate blocks the deletion path to strict).
    strict_bonus: float = 0.0
    # gated_anchored only: zero the compliance term when the reasoning's
    # distinct-4-gram share drops below this floor (anti vacuous-CoT: sweep12's
    # filler+latent-answer exploit passes the length gate and the accuracy
    # anchor but has d4 ~0.5 vs >=0.84 for honest runs). Accuracy term stays
    # ungated. 0.0 = off (historical reward).
    d4_floor: float = 0.0
    # gated_anchored / conjunctive_anchored: same as d4_floor but on the zlib
    # compression ratio of the reasoning (catches no-word-boundary loops and
    # token salad that d4 scores 1.0). Honest reasoning 0.28-0.45; filler and
    # loops < 0.1. 0.0 = off.
    zlib_floor: float = 0.0
    sent_floor: float = 0.0        # sentence terminators per 1k chars below which the compliance term is zeroed (0 = off)
    # conjunctive_anchored only: reward = compliance_term * (correct + conj_eps).
    # No λ / controller; see grading.conjunctive_anchored.
    conj_eps: float = 0.1
    inference_gpu: Optional[str] = None  # None = driver derives from base model

    # --- GRPO loss ---
    tis_cap: float = 2.0
    advantage_scale: str = "none"   # "none" (Dr.GRPO) or "group" (TRL default)
    clip_eps: float = 0.2
    loss_type: str = "dapo"
    drop_degenerate_groups: bool = True
    drop_truncated: bool = False    # drop rollouts cut off at max_tokens
    # Zero the TOTAL reward for rollouts cut off at max_tokens (they keep
    # participating in the group, so finishing is advantaged over truncating).
    # Mutually exclusive with drop_truncated in spirit: drop = DAPO overlong
    # filtering (no length pressure), zero = active wall at the cap.
    truncated_zero_reward: bool = False
    logprob_chunk_size: int = 1024  # selective-logprob chunk (memory knob)

    # --- cadence / safety ---
    verify_mask_every: int = 10     # iterations between mask verifications
    handshake_timeout_s: int = 5400  # max wait for driver batch / worker adapter

    # --- resume (survive the 24h Modal function timeout) ---
    # Resume the loop at this iteration: the run's own
    # checkpoint-<start_iteration>/ must hold adapter_model.safetensors and
    # optimizer.pt. The worker loads both (adapter via the warm-start path,
    # pre-mask-snapshot) and continues as if never interrupted — prompt/mode
    # draws are seeded per-iteration, so the data schedule is unchanged.
    # Normally set by the driver's auto-resume, not by hand.
    start_iteration: int = 0
    # Graceful budget-stop: at the top of the first iteration past this
    # wall-clock age (measured from worker start), save checkpoint+optimizer,
    # report resume_at in results.json and exit 0 — the driver respawns a
    # fresh 24h worker. Default leaves ~2h margin under the Modal timeout.
    # 0 disables.
    max_wall_s: float = 22 * 3600
    # Also write optimizer.pt every N iterations (hard-crash resume
    # granularity; budget-stop and the final checkpoint always write it).
    save_optimizer_every: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        for f in ("rollout_temperature", "tis_cap", "clip_eps", "max_wall_s",
                  "lambda_target_frac", "lambda_kp", "lambda_kd"):
            setattr(self, f, float(getattr(self, f)))
        for f in ("iterations", "prompts_per_iter", "group_size", "inner_epochs",
                  "rollout_max_tokens", "verify_mask_every", "handshake_timeout_s",
                  "logprob_chunk_size", "start_iteration", "save_optimizer_every"):
            setattr(self, f, int(getattr(self, f)))
        if self.rollout_temperature != 1.0:
            # vLLM 0.23 returns RAW (pre-temperature) logprobs; at temperature
            # 1.0 raw == sampling-distribution so the TIS ratios are exact.
            # Other temperatures need logprobs_mode="processed_logprobs" at
            # vLLM engine construction — plumb that before allowing this.
            raise ValueError(
                f"rollout_temperature must be 1.0 (got {self.rollout_temperature}): "
                "vLLM returns pre-temperature logprobs, which only match the "
                "sampling distribution at temperature 1."
            )


def filter_dataclass_kwargs(cls, kwargs: dict) -> dict:
    """Drop keys that aren't fields of ``cls`` — lets a YAML carry extra
    sections (eval config, notes) without breaking TrainConfig(**...)."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in kwargs.items() if k in names}


def parse_gpu_count(gpu: str) -> int:
    """'H200' -> 1, 'H200:4' -> 4."""
    if ":" in gpu:
        return int(gpu.rsplit(":", 1)[1])
    return 1
