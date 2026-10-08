"""Async inference through OpenRouter.

Usage:
    from cotcontrol.inference.openrouter import GenerateConfig, generate_async

    results = await generate_async(
        prompts=[[{"role": "user", "content": "Hello!"}], ...],
        model="openai/gpt-4o-mini",
        config=GenerateConfig(temperature=0.7, num_samples=4),
    )
    # results[i] = {"input": <messages>, "output": [<num_samples completions>],
    #               "model": ..., "metadata": [<per-sample usage/finish_reason/error/raw>]}

Returns the canonical schema documented in cotcontrol.inference.types. For
per-rollout progress logging (e.g. appending a JSONL line as each sample
finishes), pass `on_result` — don't monkeypatch `_sample_once`.
"""

import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class GenerateConfig:
    temperature: float = 1.0
    max_tokens: int = 4096
    top_p: float = 1.0
    num_samples: int = 1
    max_concurrency: int = 100
    max_retries: int = 5
    # OpenRouter provider routing (https://openrouter.ai/docs/provider-routing),
    # e.g. {"only": ["groq"], "allow_fallbacks": False} to pin one provider.
    provider: dict | None = None
    # Extra request-body fields passed straight through to OpenRouter, e.g.
    # {"reasoning": {"effort": "medium"}} for gpt-oss / o-series effort dials.
    extra_body: dict | None = None


class ProviderError(RuntimeError):
    """Provider returned finish_reason='error' inside a 200 response."""


def get_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (expected in .env)")
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


async def _sample_once(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    messages: list[dict],
    config: GenerateConfig,
) -> dict:
    """One rollout. Returns a dict with the completion plus full raw response."""
    async with semaphore:
        for attempt in range(config.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    extra_body={
                        **({"provider": config.provider} if config.provider else {}),
                        **(config.extra_body or {}),
                    },
                )
                choice = response.choices[0]
                # Some providers (e.g. Nebius on qwen3-32b) return HTTP 200 with
                # finish_reason "error" and a choice-level error payload
                # (504 upstream idle timeout). Treat it like an exception so it
                # gets retried; the last attempt returns it as-is.
                if choice.finish_reason == "error" and attempt < config.max_retries - 1:
                    raise ProviderError(str(choice.model_dump().get("error")))
                # OpenRouter puts reasoning-model traces on message.reasoning
                # (an extra field the openai SDK keeps but doesn't type).
                message = choice.message.model_dump()
                return {
                    "completion": message.get("content"),
                    "reasoning": message.get("reasoning")
                    or message.get("reasoning_content"),
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "raw_response": response.model_dump(),
                    "error": None,
                }
            except Exception as e:
                if attempt == config.max_retries - 1:
                    return {
                        "completion": None,
                        "reasoning": None,
                        "finish_reason": None,
                        "usage": None,
                        "raw_response": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                # Exponential backoff with jitter for rate limits / transient errors,
                # capped at 60s so large max_retries stays bounded (~10 min worst case at 12).
                await asyncio.sleep(min(2**attempt, 60) + random.random())


async def generate_async(
    prompts: list[list[dict]],
    model: str,
    config: GenerateConfig | None = None,
    save_path: str | Path | None = None,
    progress: bool = True,
    on_result=None,
) -> list[dict]:
    """Generate completions for a list of message lists via OpenRouter.

    All prompt x sample rollouts run concurrently, bounded by a semaphore of
    size config.max_concurrency.

    Returns one dict per prompt:
        {
            "input": <the original messages list>,
            "model": <model id>,
            "output": [<completion str>, ...],   # length num_samples
            "reasoning": [<reasoning str or None>, ...],  # length num_samples
            "metadata": [{"finish_reason", "usage", "error", "raw_response"}, ...],
        }
    If save_path is given, also appends each dict as a JSONL line for later
    analysis.

    on_result, if given, is called synchronously as each rollout finishes with
    {"prompt_idx", "sample_idx", "messages", "completion", "reasoning",
     "finish_reason", "usage", "error", "raw_response"} — use it for
    incremental progress logs on long runs.
    """
    config = config or GenerateConfig()
    client = get_client()
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _run(prompt_idx: int, sample_idx: int, messages: list[dict]) -> dict:
        result = await _sample_once(client, semaphore, model, messages, config)
        if on_result is not None:
            on_result(
                {"prompt_idx": prompt_idx, "sample_idx": sample_idx, "messages": messages, **result}
            )
        return result

    tasks = [
        _run(i, j, messages)
        for i, messages in enumerate(prompts)
        for j in range(config.num_samples)
    ]
    gather = tqdm_asyncio.gather if progress else asyncio.gather
    flat = await gather(*tasks)

    results = []
    for i, messages in enumerate(prompts):
        samples = flat[i * config.num_samples : (i + 1) * config.num_samples]
        results.append(
            {
                "input": messages,
                "model": model,
                "output": [s["completion"] for s in samples],
                "reasoning": [s["reasoning"] for s in samples],
                "metadata": [
                    {k: s[k] for k in ("finish_reason", "usage", "error", "raw_response")}
                    for s in samples
                ],
            }
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "a") as f:
            for prompt_idx, record in enumerate(results):
                f.write(json.dumps({"prompt_idx": prompt_idx, **record}) + "\n")

    return results


if __name__ == "__main__":
    test_prompts = [
        [{"role": "user", "content": "Say 'hello' and nothing else."}],
        [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
    ]
    results = asyncio.run(
        generate_async(
            test_prompts,
            model="qwen/qwen3.6-35b-a3b",
            config=GenerateConfig(temperature=0.0, max_tokens=2048, num_samples=2),
        )
    )
    for i, r in enumerate(results):
        print(f"[prompt {i}] input: {r['input'][0]['content']!r}")
        for j in range(len(r["output"])):
            reasoning = r["reasoning"][j]
            preview = reasoning[:200] + "..." if reasoning and len(reasoning) > 200 else reasoning
            print(f"  sample {j} reasoning: {preview!r}")
            print(f"  sample {j} output: {r['output'][j]!r} (error={r['metadata'][j]['error']})")
