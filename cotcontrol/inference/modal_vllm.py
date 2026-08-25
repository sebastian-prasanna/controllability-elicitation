"""vLLM-on-Modal inference backend.

A warm Modal class (`InferenceEngine`) loads a base model once and reuses it,
hot-swapping LoRA adapters per call. `generate_async` wraps it with a disk
prompt cache, splits raw completions into (reasoning, response) with
`split_reasoning`, and returns the canonical schema from
cotcontrol.inference.types — so eval/GEPA code is backend-agnostic.

`base_model` / `lora_path` are interpreted *inside the container*: an HF repo
id, or an absolute path on the shared checkpoints volume
(/checkpoints/<run_name>/checkpoint-<step>) written by the training app.

Usage:
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig, generate_async

    results = await generate_async(
        messages_list, base_model="Qwen/Qwen3-8B",
        config=ModalGenerateConfig(temperature=0.0, max_tokens=12000, gpu="H200"),
        lora_path="/checkpoints/my_run/checkpoint-40",   # or None for the base model
    )

Sweeping many checkpoints? Wrap the sweep in `async with app.run():` and
`asyncio.gather` the per-checkpoint calls — same GenerateConfig resource
fields -> one warm engine shared across all of them.
"""

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import modal

from cotcontrol.inference.types import split_reasoning

app = modal.App("cotcontrol-inference")

CHECKPOINTS_PATH = "/checkpoints"
HF_CACHE_PATH = "/cache"
HF_CACHE = f"{HF_CACHE_PATH}/hf"

# Redwood naming: {project}_{owner}_{label}. Shared by the training app.
checkpoints_vol = modal.Volume.from_name(
    "cotcontrol_sebastian-prasanna_checkpoints", create_if_missing=True
)
hf_cache_vol = modal.Volume.from_name(
    "cotcontrol_sebastian-prasanna_hf-cache", create_if_missing=True
)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "vllm==0.23.0",
        "hf-transfer",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # The flashinfer sampler JIT-compiles a CUDA kernel at engine start
            # and needs nvcc, which the slim wheel image lacks.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        }
    )
)

# All models we train are ungated, so no HF token is required. Set
# COTCONTROL_MODAL_HF_SECRET to the name of a Modal secret holding HF_TOKEN if
# you ever need a gated model in the container.
_hf_secret = os.environ.get("COTCONTROL_MODAL_HF_SECRET")
_SECRETS = [modal.Secret.from_name(_hf_secret)] if _hf_secret else []

CACHE_DIR = Path(__file__).resolve().parents[2] / ".generation_cache"


@dataclass
class ModalGenerateConfig:
    """Sampling + Modal/vLLM resource parameters.

    The resource fields participate in the warm-container pool identity —
    keep them constant across a checkpoint sweep to reuse one engine.
    """

    # Sampling
    temperature: float = 0.0
    max_tokens: int = 12000
    num_samples: int = 1
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    cache: bool = True
    # Extra kwargs for the chat template, e.g. {"reasoning_effort": "high"}
    # for gpt-oss. Part of the disk-cache key.
    chat_template_kwargs: Optional[dict] = None

    # Modal / vLLM resources
    gpu: str = "H200"
    tensor_parallel_size: int = 1
    enable_lora: bool = True
    max_lora_rank: int = 64
    max_model_len: int = 32768
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9
    # True for adapters that target the fused MoE experts (PEFT saves those as
    # 3D stacked tensors, which need vLLM's mixed-MoE LoRA loader).
    mixed_moe_lora: bool = False

    def __post_init__(self) -> None:
        self.temperature = float(self.temperature)
        self.top_p = float(self.top_p)
        self.gpu_memory_utilization = float(self.gpu_memory_utilization)
        for f in ("max_tokens", "num_samples", "tensor_parallel_size",
                  "max_lora_rank", "max_model_len", "max_num_seqs"):
            setattr(self, f, int(getattr(self, f)))


@app.cls(
    image=inference_image,
    gpu="H200",  # label only — the real spec comes via with_options(gpu=...)
    volumes={CHECKPOINTS_PATH: checkpoints_vol, HF_CACHE_PATH: hf_cache_vol},
    timeout=4 * 3600,
    scaledown_window=600,
    secrets=_SECRETS,
)
class InferenceEngine:
    base_model: str = modal.parameter()
    tensor_parallel_size: int = modal.parameter(default=1)
    max_lora_rank: int = modal.parameter(default=64)
    enable_lora: bool = modal.parameter(default=True)
    max_model_len: int = modal.parameter(default=32768)
    max_num_seqs: int = modal.parameter(default=256)
    gpu_memory_utilization_pct: int = modal.parameter(default=90)
    mixed_moe_lora: bool = modal.parameter(default=False)

    @modal.enter()
    def setup(self):
        from vllm import LLM

        checkpoints_vol.reload()
        hf_cache_vol.reload()

        llm_kwargs = dict(
            model=self.base_model,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization_pct / 100.0,
        )
        if self.enable_lora:
            llm_kwargs.update(enable_lora=True, max_lora_rank=self.max_lora_rank, max_loras=1)
            if self.mixed_moe_lora:
                llm_kwargs["enable_mixed_moe_lora_format"] = True
        self.llm = LLM(**llm_kwargs)

    @modal.method()
    def generate(
        self,
        messages_list,
        lora_path=None,
        lora_id=1,
        max_tokens=12000,
        temperature=0.0,
        top_p=1.0,
        num_samples=1,
        stop=None,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs=None,
    ):
        """Chat-format generation over a batch. Returns
        List[List[{"text", "finish_reason"}]] (outer = prompts, inner =
        num_samples). Special tokens are kept in the text so the caller can
        split Qwen <think> blocks / gpt-oss harmony channels downstream.

        continue_final_message=True (with add_generation_prompt=False) makes
        the model CONTINUE the final assistant message instead of starting a
        new turn — the prefill path.
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        if lora_path:
            checkpoints_vol.reload()

        sp = SamplingParams(
            n=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or None,
            skip_special_tokens=False,
        )
        if lora_path:
            lora_kwargs = {}
            if self.mixed_moe_lora:
                lora_kwargs["is_3d_lora_weight"] = True  # PEFT 3D fused-expert format
            lora_req = LoRARequest(f"lora_{lora_id}", lora_id, lora_path, **lora_kwargs)
        else:
            lora_req = None
        outputs = self.llm.chat(
            messages_list,
            sampling_params=sp,
            lora_request=lora_req,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            chat_template_kwargs=chat_template_kwargs or None,
        )
        return [
            [{"text": c.text, "finish_reason": c.finish_reason} for c in o.outputs]
            for o in outputs
        ]


def _is_app_running(app_obj) -> bool:
    for v in app_obj.__dict__.values():
        if hasattr(v, "_running_app"):
            return v._running_app is not None
    return False


def _cache_key(model_id, messages, config: ModalGenerateConfig,
               add_generation_prompt, continue_final_message) -> str:
    blob = json.dumps(
        {
            "cache_version": 1,
            "backend": "modal-vllm",
            "model_id": model_id,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_samples": config.num_samples,
            "stop": config.stop,
            "chat_template_kwargs": config.chat_template_kwargs,
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def _load_cache(key: str) -> Optional[Dict]:
    fp = _cache_path(key)
    if not fp.exists():
        return None
    try:
        with fp.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None


def _save_cache(key: str, texts, finish_reasons) -> None:
    fp = _cache_path(key)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w") as f:
        json.dump({"texts": texts, "finish_reasons": finish_reasons}, f)


def _lora_id_for(path: str) -> int:
    return int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) % (10**8) + 1


def _to_canonical(messages, model_id, texts, finish_reasons) -> dict:
    """Raw completion strings -> the canonical result schema."""
    outputs, reasonings, metadata = [], [], []
    for text, fin in zip(texts, finish_reasons):
        reasoning, response, truncated = split_reasoning(text)
        outputs.append(response)
        reasonings.append(reasoning)
        metadata.append(
            {
                "finish_reason": fin if fin is not None else ("length" if truncated else "stop"),
                "usage": None,
                "error": None,
                "raw_response": text,
            }
        )
    return {
        "input": messages,
        "model": model_id,
        "output": outputs,
        "reasoning": reasonings,
        "metadata": metadata,
    }


async def generate_async(
    messages_list: List[List[Dict[str, str]]],
    base_model: str,
    config: Optional[ModalGenerateConfig] = None,
    lora_path: Optional[str] = None,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> List[Dict]:
    """Run vLLM inference on Modal; returns the canonical schema (one dict per
    prompt, same shape as the OpenRouter backend).

    lora_path=None serves the plain base model (HF id) — use this for base
    model evals or prefill experiments off HF.
    """
    config = config or ModalGenerateConfig()
    model_id = f"{base_model}::{lora_path or ''}"

    all_results: List[Optional[Dict]] = [None] * len(messages_list)
    cache_keys, uncached = [], []
    for i, msgs in enumerate(messages_list):
        ck = _cache_key(model_id, msgs, config, add_generation_prompt, continue_final_message)
        cache_keys.append(ck)
        if config.cache:
            cached = _load_cache(ck)
            if cached is not None:
                all_results[i] = _to_canonical(
                    msgs, model_id, cached["texts"], cached["finish_reasons"]
                )
                continue
        uncached.append(i)

    if not uncached:
        print(f"[modal_vllm] cache: {len(messages_list)}/{len(messages_list)} hits, all cached")
        return all_results
    print(
        f"[modal_vllm] cache: {len(messages_list) - len(uncached)}/{len(messages_list)} hits, "
        f"generating {len(uncached)} on {config.gpu}"
    )

    EngineCls = InferenceEngine.with_options(gpu=config.gpu)
    engine_kwargs = dict(
        base_model=base_model,
        tensor_parallel_size=config.tensor_parallel_size,
        max_lora_rank=config.max_lora_rank,
        enable_lora=config.enable_lora,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization_pct=int(round(config.gpu_memory_utilization * 100)),
        mixed_moe_lora=config.mixed_moe_lora,
    )
    call_kwargs = dict(
        messages_list=[messages_list[i] for i in uncached],
        lora_path=lora_path,
        lora_id=_lora_id_for(lora_path) if lora_path else 1,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        num_samples=config.num_samples,
        stop=config.stop,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        chat_template_kwargs=config.chat_template_kwargs,
    )

    async def _do_call():
        engine = EngineCls(**engine_kwargs)
        return await engine.generate.remote.aio(**call_kwargs)

    if _is_app_running(app):
        outputs_per_prompt = await _do_call()
    else:
        async with app.run():
            outputs_per_prompt = await _do_call()

    for j, idx in enumerate(uncached):
        samples = outputs_per_prompt[j]
        texts = [s["text"] for s in samples]
        fins = [s["finish_reason"] for s in samples]
        if config.cache:
            _save_cache(cache_keys[idx], texts, fins)
        all_results[idx] = _to_canonical(messages_list[idx], model_id, texts, fins)

    return all_results


def make_generate_fn(
    base_model: str,
    config: Optional[ModalGenerateConfig] = None,
    lora_path: Optional[str] = None,
):
    """A closure suitable for eval_cotcontrolqa(generate_fn=...)."""

    async def _generate_fn(messages_list):
        return await generate_async(messages_list, base_model, config, lora_path)

    return _generate_fn


if __name__ == "__main__":
    test_prompts = [
        [{"role": "user", "content": "Say 'hello' and nothing else."}],
        [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
    ]
    results = asyncio.run(
        generate_async(
            test_prompts,
            base_model="Qwen/Qwen3-0.6B",
            config=ModalGenerateConfig(
                temperature=0.0, max_tokens=512, gpu="L40S", max_model_len=4096, cache=False
            ),
        )
    )
    for r in results:
        print(r["input"][0]["content"])
        print("  reasoning:", (r["reasoning"][0] or "")[:120].replace("\n", " "))
        print("  output:", r["output"][0])
