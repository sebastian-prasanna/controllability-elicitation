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
    ``target_parameters``; dense models use ordinary nn.Linear MLPs."""
    name = model_name.lower()
    is_moe = bool(re.search(r"a\d+b", name)) or "moe" in name or "gpt-oss" in name
    if is_moe:
        return (
            ("q_proj", "k_proj", "v_proj", "o_proj"),
            ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
        )
    return (
        ("q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj"),
        (),
    )


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
    # jsonl path(s) with {"input": [messages], "output": [messages]} rows.
    # Consumed by the launcher (load_sft_jsonl), not shipped to the worker.
    data_path: Union[str, List[str], None] = None
    num_examples: Optional[int] = None  # cap on training examples (post-shuffle)
    shuffle: bool = True  # seeded pre-shuffle of rows + per-epoch shuffling

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
    # {"reasoning_effort": "high"} for gpt-oss.
    chat_template_kwargs: Optional[dict] = None
    # HF attention backend. None = auto: "sdpa", except gpt-oss -> "eager"
    # (no SDPA support; eager is O(L^2) memory so revisit for long contexts).
    attn_implementation: Optional[str] = None

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
    # compliance_score (shaped + 2*strict)/3 | shaped_task_score | shaped_compliance
    reward: str = "compliance_score"
    inference_gpu: Optional[str] = None  # None = driver derives from base model

    # --- GRPO loss ---
    tis_cap: float = 2.0
    advantage_scale: str = "none"   # "none" (Dr.GRPO) or "group" (TRL default)
    clip_eps: float = 0.2
    loss_type: str = "dapo"
    drop_degenerate_groups: bool = True
    drop_truncated: bool = False    # drop rollouts cut off at max_tokens
    logprob_chunk_size: int = 1024  # selective-logprob chunk (memory knob)

    # --- cadence / safety ---
    verify_mask_every: int = 10     # iterations between mask verifications
    handshake_timeout_s: int = 5400  # max wait for driver batch / worker adapter

    def __post_init__(self) -> None:
        super().__post_init__()
        for f in ("rollout_temperature", "tis_cap", "clip_eps"):
            setattr(self, f, float(getattr(self, f)))
        for f in ("iterations", "prompts_per_iter", "group_size", "inner_epochs",
                  "rollout_max_tokens", "verify_mask_every", "handshake_timeout_s",
                  "logprob_chunk_size"):
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
