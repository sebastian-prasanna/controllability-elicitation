"""Modal app + local ``train()`` entrypoint for LoRA SFT (with optional
random-subset elicitation masking — see cotcontrol.training.masking).

The local side opens ``app.run()`` and invokes the remote trainer with the GPU
spec from the config. The remote side writes inputs to /tmp and shells out to
``accelerate launch worker.py`` (FSDP2 auto-enabled for multi-GPU).

Checkpoints land on the shared checkpoints volume at
``/checkpoints/<run_name>/checkpoint-<step>`` — the same volume the inference
engine (cotcontrol.inference.modal_vllm) reads, so evals need no download step.

Usage:
    from cotcontrol.training.config import TrainConfig
    from cotcontrol.training.modal_app import load_sft_jsonl, train

    cfg = TrainConfig(run_name="probe", data_path="training_data/x.jsonl",
                      train_params=1000, mask_seed=0)
    result = train(cfg, load_sft_jsonl(cfg.data_path))
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Union

import modal

from cotcontrol.training.config import TrainConfig, parse_gpu_count

app = modal.App("cotcontrol-train")

CHECKPOINTS_PATH = "/checkpoints"
HF_CACHE_PATH = "/cache"
HF_CACHE = f"{HF_CACHE_PATH}/hf"

# Same volumes as cotcontrol.inference.modal_vllm (Redwood naming:
# {project}_{owner}_{label}) so trained adapters are immediately servable.
checkpoints_vol = modal.Volume.from_name(
    "cotcontrol_sebastian-prasanna_checkpoints", create_if_missing=True
)
hf_cache_vol = modal.Volume.from_name(
    "cotcontrol_sebastian-prasanna_hf-cache", create_if_missing=True
)

# transformers 5 for fused-MoE experts (matches vLLM 0.23 on the inference
# side so the adapter format agrees); peft>=0.18.1 for LoRA over fused expert
# parameters; kernels+triton for gpt-oss MXFP4 dequantization
# (Mxfp4Config(dequantize=True) in the worker). SDPA attention — no flash-attn
# build needed. CUDA devel base: triton JIT needs nvcc.
train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .uv_pip_install(
        "torch>=2.8",
        # Pinned to the locally-tested minor: transformers 6 renamed/dropped
        # several TrainingArguments fields (warmup_ratio et al.).
        "transformers>=5.5,<5.16",
        "peft>=0.18.1",
        "accelerate>=1.13",
        "datasets",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "hf-transfer",
        "kernels",
        "triton",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("cotcontrol")
)

# All models we train are ungated, so no HF token is required. Set
# COTCONTROL_MODAL_HF_SECRET to the name of a Modal secret holding HF_TOKEN if
# you ever need a gated model in the container.
_hf_secret = os.environ.get("COTCONTROL_MODAL_HF_SECRET")
_SECRETS = [modal.Secret.from_name(_hf_secret)] if _hf_secret else []


def _accelerate_fsdp_config(num_processes: int, fsdp_version: int) -> str:
    """Accelerate FSDP config file. Default is FSDP2 (per-parameter DTensor
    sharding): unlike FSDP1 it has no shared FlatParameter, so PEFT's fused-MoE
    per-forward get_delta_weight reshape works, frozen and trainable params
    shard uniformly (no use_orig_params/auto-wrap-policy dance), and — key for
    the elicitation method — gradients stay per-parameter DTensors that our
    mask hooks can zero shard-consistently. cpu_ram_efficient_loading: rank 0
    loads real weights, the rest meta-load and receive params at wrap. The
    transformer block class comes from the worker at runtime via
    FSDP_TRANSFORMER_CLS_TO_WRAP (read from the base model's
    _no_split_modules — the PEFT wrapper hides it). FULL_STATE_DICT so saved
    adapters load normally in vLLM."""
    if fsdp_version == 2:
        version_block = """  fsdp_version: 2
  fsdp_reshard_after_forward: true"""
    else:
        # FSDP1 (legacy escape hatch): the HF PEFT+FSDP recipe.
        version_block = """  fsdp_version: 1
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false"""
    return f"""compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_activation_checkpointing: false
{version_block}
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: {num_processes}
rdzv_backend: static
same_network: true
use_cpu: false
"""


def _launch_worker(
    worker_file: str, config_dict: dict, run_name: str, line_hook=None
) -> dict:
    """Shared remote-side launch: write config, shell out to `accelerate
    launch <worker_file>` streaming stdout (through line_hook if given),
    commit volumes, return results.json. Runs INSIDE the Modal container."""
    workdir = Path(f"/tmp/cotcontrol/{run_name}")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "config.json").write_text(json.dumps(config_dict))

    output_dir = f"{CHECKPOINTS_PATH}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    import torch

    num_gpus = torch.cuda.device_count()
    print(f"[modal] run_name={run_name} num_gpus={num_gpus} "
          f"base_model={config_dict.get('base_model')}")

    # Inside the container, add_local_python_source puts the package at
    # /root/cotcontrol, so workers sit next to this file.
    worker = Path(__file__).parent / worker_file
    use_fsdp = num_gpus > 1 and config_dict.get("parallelism", "fsdp") == "fsdp"
    if use_fsdp:
        # FSDP is configured via an accelerate config file (not
        # TrainingArguments): cpu_ram_efficient_loading meta-loading must be
        # in place before the worker loads the model.
        fsdp_version = int(config_dict.get("fsdp_version", 2))
        fsdp_yaml = workdir / "accelerate_fsdp.yaml"
        fsdp_yaml.write_text(_accelerate_fsdp_config(num_gpus, fsdp_version))
        cmd = ["accelerate", "launch", "--config_file", str(fsdp_yaml)]
    else:
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_gpus),
            "--num_machines", "1",
            "--mixed_precision", "bf16",
        ]
    cmd += [
        str(worker),
        "--workdir", str(workdir),
        "--output-dir", output_dir,
        "--num-gpus", str(num_gpus),
    ]
    # expandable_segments: lets the reserved-but-unallocated pool satisfy
    # later allocations instead of OOMing from fragmentation.
    env = {
        **os.environ,
        "PYTHONPATH": "/root",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    output_lines: list[str] = []
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
        if line_hook is not None:
            line_hook(line, workdir)
    proc.wait()
    # Commit even on failure so partial checkpoints/logs are inspectable.
    checkpoints_vol.commit()
    hf_cache_vol.commit()
    if proc.returncode != 0:
        tail = "".join(output_lines[-80:])
        raise RuntimeError(
            f"accelerate launch exited {proc.returncode}.\nLast output:\n{tail}"
        )

    results_path = workdir / "results.json"
    if not results_path.exists():
        raise RuntimeError("Training finished but no results.json was produced.")
    return json.loads(results_path.read_text())


_REMOTE_CLS_KWARGS = dict(
    image=train_image,
    volumes={CHECKPOINTS_PATH: checkpoints_vol, HF_CACHE_PATH: hf_cache_vol},
    secrets=_SECRETS,
    timeout=24 * 3600,
    # Display/fallback default only — every run overrides via
    # `.with_options(gpu=config.gpu, ...)` at submit time.
    gpu="H200",
    # Training is expensive: fail loudly, never silently rerun a long job.
    retries=modal.Retries(max_retries=0),
)


@app.cls(**_REMOTE_CLS_KWARGS)
class _TrainRemote:
    @modal.method()
    def run(self, config_dict: dict, dataset: list[dict], run_name: str) -> dict:
        """Write inputs to /tmp, shell out to `accelerate launch worker.py`."""
        workdir = Path(f"/tmp/cotcontrol/{run_name}")
        workdir.mkdir(parents=True, exist_ok=True)
        with (workdir / "dataset.jsonl").open("w") as f:
            for row in dataset:
                f.write(json.dumps(row) + "\n")
        return _launch_worker("worker.py", config_dict, run_name)


def _volsync_hook(line: str, workdir: Path) -> None:
    """rl_worker <-> volume sync protocol: the worker subprocess cannot commit
    or reload the mounted volume itself (the handles live in this process), so
    it prints "@@VOLSYNC:<commit|reload>:<n>@@" and blocks until we perform the
    op and touch the ack file (same container, shared /tmp)."""
    m = re.search(r"@@VOLSYNC:(commit|reload):(\d+)@@", line)
    if not m:
        return
    op, n = m.group(1), m.group(2)
    if op == "commit":
        checkpoints_vol.commit()
    else:
        checkpoints_vol.reload()
    (workdir / f"volsync_ack_{n}").touch()


@app.cls(**_REMOTE_CLS_KWARGS)
class _RLRemote:
    @modal.method()
    def run(self, config_dict: dict, run_name: str) -> dict:
        """GRPO worker launch. No dataset ships — the local driver
        (rl/run_grpo.py) supplies graded rollout batches over the checkpoints
        volume, keyed to the adapter checkpoints the worker publishes."""
        return _launch_worker("rl_worker.py", config_dict, run_name,
                              line_hook=_volsync_hook)


def load_sft_jsonl(
    path_or_paths: Union[str, List[str]], num_examples: int | None = None
) -> list[dict]:
    """Load {"input": ..., "output": ...} rows from one or more jsonl files
    (extra keys like "meta" pass through; the worker ignores them).
    num_examples caps to the first N rows (the seeded shuffle/cap from
    TrainConfig.shuffle/num_examples happens worker-side)."""
    paths = [path_or_paths] if isinstance(path_or_paths, str) else list(path_or_paths)
    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if num_examples is not None:
        rows = rows[:num_examples]
    if rows and ("input" not in rows[0] or "output" not in rows[0]):
        raise ValueError(f"rows must have 'input'/'output': got keys={list(rows[0])}")
    return rows


def train(
    config: TrainConfig,
    dataset: list[dict],
    enable_output: bool = True,
    timestamp_run_name: bool = True,
) -> dict:
    """Local entrypoint. Submits the run with the GPU spec from ``config``.

    ``config.run_name`` is timestamp-suffixed by default so reruns never
    overwrite an existing /checkpoints/<name>/ dir. Returns the worker's
    result dict (checkpoint paths are volume paths, directly usable as
    ``lora_path`` in cotcontrol.inference.modal_vllm)."""
    if parse_gpu_count(config.gpu) < 1:
        raise ValueError(f"Invalid gpu spec: {config.gpu!r}")
    if not dataset:
        raise ValueError("dataset is empty")

    if timestamp_run_name:
        config.run_name = (
            f"{config.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    cls = _TrainRemote.with_options(gpu=config.gpu, timeout=config.timeout_seconds)

    def _go():
        return cls().run.remote(asdict(config), dataset, config.run_name)

    if enable_output:
        with modal.enable_output(), app.run():
            return _go()
    with app.run():
        return _go()


def spawn_rl(config) -> "modal.FunctionCall":
    """Submit a GRPO run without blocking (the caller — rl/run_grpo.py — must
    already hold app.run() open and keep it open: it serves rollouts to the
    worker concurrently). Returns the FunctionCall; .get() it after the rollout
    loop finishes. run_name timestamping is the caller's job."""
    if parse_gpu_count(config.gpu) < 1:
        raise ValueError(f"Invalid gpu spec: {config.gpu!r}")
    cls = _RLRemote.with_options(gpu=config.gpu, timeout=config.timeout_seconds)
    return cls().run.spawn(asdict(config), config.run_name)
