# controllability-elicitation

How elicitable is CoT controllability? We measure models on **CoT-Control-QA**
(following [CoTControl](https://github.com/YuehHanChen/CoTControl)): GPQA/HLE/MMLU-Pro
questions with a constraint attached to the *reasoning trace* (suppress a word,
alternate case, ignore the question, ...), graded for both answer accuracy and
strict constraint compliance. Elicitation methods: prompting baselines, few-shot,
best-of-n, GEPA prompt evolution, and SFT — including SFT that trains only a
**random subset of k LoRA parameters** (Donoway et al., NeurIPS 2025,
"Quantifying Elicitation of Latent Capabilities in Language Models") to measure
the parameter budget a capability needs.

## Layout

```
cotcontrol/                 the library
  eval/                     CoT-Control-QA: data.py (loading + mode assignment),
                            prompts.py, grading.py (rule-based graders),
                            judge.py (OpenRouter LLM judges), eval.py (backend-agnostic loop)
  inference/                generation backends returning one canonical schema (types.py):
    openrouter.py           async OpenRouter client (reasoning field from the API)
    modal_vllm.py           warm vLLM engine on Modal; HF base models or LoRA
                            checkpoints from the shared /checkpoints volume; prefill support
  training/                 Modal LoRA SFT:
    config.py               TrainConfig <-> YAML
    masking.py              random-subset parameter masking (gradient masks + verification)
    rendering.py            chat-template rendering w/ supervised loss masking (qwen <think>,
                            gpt-oss harmony channels)
    worker.py               in-container trainer (accelerate/FSDP2, plain transformers Trainer)
    modal_app.py            the Modal train app + local train() entrypoint
gepa/                       GEPA (reflective prompt evolution) on the eval; runs/ = run logs
fewshot/                    few-shot scaling experiments
sft/runs/                   train_eval.py run outputs (config, losses, per-checkpoint evals)
scripts/                    thin launchers (baselines, BoN, prompt evals, dataset builders,
                            train_eval.py = config-driven train -> eval-every-checkpoint)
configs/                    YAML run configs
datasets/                   eval CSVs (with suppression keywords) + SFT examples
training_data/              SFT datasets mined from BoN runs + synthetic320
prompts/                    baseline + GEPA-evolved system prompts
notebooks/                  analysis notebooks (plotting.ipynb etc.)
results/                    eval outputs, progress logs, BoN rollouts
old/                        retired tinker infrastructure + tinker run outputs (delete eventually)
```

## Setup

```bash
uv pip install -e . --no-deps   # the shared /root/.venv is pre-provisioned
# .env at repo root: OPENROUTER_API_KEY=... (judges + API evals). Modal: `modal token`.
```

GPU deps (vllm, kernels) live only in the Modal images — nothing GPU-ish installs locally.

## Eval

```python
from cotcontrol.eval.eval import eval_cotcontrolqa
from cotcontrol.inference.openrouter import GenerateConfig

# API model via OpenRouter
result = await eval_cotcontrolqa(
    model="openai/gpt-oss-120b", dataset="all", mode="random",
    generate_config=GenerateConfig(temperature=0.0, max_tokens=25000, max_concurrency=200),
)

# HF base model or trained LoRA checkpoint via Modal/vLLM
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, make_generate_fn
result = await eval_cotcontrolqa(
    model="Qwen/Qwen3-8B",
    generate_fn=make_generate_fn(
        "Qwen/Qwen3-8B",
        ModalGenerateConfig(temperature=0.0, max_tokens=12000, gpu="H200"),
        lora_path="/checkpoints/<run_name>/checkpoint-<step>",  # None = base model
    ),
    dataset="all", mode="random",
)
```

Both backends return identical result JSONs (schema in `cotcontrol/inference/types.py`),
so notebooks and downstream analysis don't care where generation happened.
`ignore_question` compliance always uses the OpenRouter LLM judge.

## Training (Modal LoRA, optional random-subset masking)

```bash
.venv/bin/python scripts/train_eval.py configs/example_train.yaml
```

Trains LoRA on the given SFT jsonl (`{"input": [...messages], "output": [...messages]}`
per line; qwen assistant content embeds `<think>...</think>`, gpt-oss uses
thinking/text content blocks), saving adapter checkpoints to the shared Modal
volume, then evals every checkpoint through the Modal inference engine.

Set `train_params: k` in the config to train only k uniformly-random LoRA adapter
weights (mask fixed at run start from `mask_seed`, applied to gradients; the run
verifies post-hoc that nothing outside the mask moved). `train_params: null`
trains the full adapter. Defaults follow the paper: attention-only targets
(q/k/v/o), A Kaiming / B zero init.

Model notes:
- Qwen3-8B: 1 GPU. Qwen3-32B: 1×H200. gpt-oss-120b: MXFP4 dequantized to bf16
  at load, 4–8×H200 with FSDP2.
- MoE models (Qwen3-*A*B, gpt-oss experts) need PEFT `target_parameters` for the
  fused expert tensors — see `cotcontrol.training.config.default_lora_targets` —
  and `mixed_moe_lora: true` on the inference side.

## GEPA / few-shot

`gepa/run_gepa.py` evolves system prompts against the eval (reflection model is
any OpenRouter id, e.g. `anthropic/claude-opus-4.6`); `gepa/eval_heldout.py`
scores candidates on held-out splits. `fewshot/fewshot_scaling.py` sweeps demo
counts. All of it goes through the same inference layer.

## Modal resources

App names `cotcontrol-train` / `cotcontrol-inference`; volumes
`cotcontrol_sebastian-prasanna_checkpoints` (adapter checkpoints) and
`cotcontrol_sebastian-prasanna_hf-cache` (shared HF weight cache). All models
used are ungated; set `COTCONTROL_MODAL_HF_SECRET=<modal secret name>` if a
gated model ever needs an HF token in-container.
