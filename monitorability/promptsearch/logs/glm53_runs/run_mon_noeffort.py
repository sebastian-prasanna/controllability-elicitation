#!/usr/bin/env python3
"""Run monitorability/run_monitorability.py with the policy request's `reasoning.effort` field REMOVED and the serving
provider recorded as `usage.provider` in every rollouts.jsonl line. No harness code is modified: generate_async is wrapped
in cotcontrol.inference.openrouter before run_monitorability is imported.

Why: configs/eval_pins.json pins z-ai/glm-5.3 with reasoning_effort unset (vendor default = max) and warns that "high"
collapses the reasoning channel to a stub. run_monitorability.py always sends extra_body.reasoning.effort (default medium);
OpenRouter rejects effort "" (400) and "none" disables reasoning, so this wrapper is the only way to send no field.
Probe 2026-10-05 (same gpqa-style question, 8 samples each): medium rtok 15-274 (median ~150, one 15-tok stub with the
derivation in the answer channel) vs unset rtok 372-764 (median ~480).

Usage: identical CLI to run_monitorability.py (pass --reasoning-effort unset so config.json records the truth).
"""
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
import cotcontrol.inference.openrouter as orr  # noqa: E402

_orig = orr.generate_async


async def generate_async(prompts, model, config=None, save_path=None, progress=True, on_result=None):
    if config is not None and config.extra_body and "reasoning" in config.extra_body:
        config.extra_body = {k: v for k, v in config.extra_body.items() if k != "reasoning"}

    def wrapped(res):
        raw = res.get("raw_response")
        if isinstance(res.get("usage"), dict) and isinstance(raw, dict):
            res["usage"]["provider"] = raw.get("provider")
        on_result(res)

    return await _orig(prompts, model, config, save_path, progress, wrapped if on_result else None)


orr.generate_async = generate_async
script = REPO / "monitorability" / "run_monitorability.py"
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
