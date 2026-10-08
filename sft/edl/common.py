"""Shared helpers for the EDL pipeline (sft/edl/*): sweep-cell parsing and config loading.

A sweep folder holds cells named <prefix>-k<k>[k]-lr<lr> (masked rank-1 LoRA, k trainable
scalars) or <prefix>-kall-r<rank>-lr<lr> (unmasked full-coverage LoRA; k = total LoRA params).
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path

_MASKED = re.compile(r"-k(\d+)(k?)-")
_FULL = re.compile(r"-kall-r(\d+)-")


@dataclass(frozen=True)
class Cell:
    dir: Path
    k: int            # trainable parameters (mask hits for masked cells, total LoRA params for kall)
    kind: str         # "masked" | "full"
    rank: int
    label: str        # adapter/file name, e.g. "k1000" or "kall-r4"

    @property
    def train_result(self) -> dict:
        return json.load(open(self.dir / "train_result.json"))

    def ckpt_path(self, step: int) -> str:
        return next(c["path"] for c in self.train_result["checkpoints"] if c["step"] == step)


def parse_cell(d: Path) -> Cell | None:
    m = _FULL.search(d.name)
    if m:
        tr = json.load(open(d / "train_result.json"))
        return Cell(d, int(tr["total_lora_params"]), "full", int(m.group(1)), f"kall-r{m.group(1)}")
    m = _MASKED.search(d.name)
    if m:
        k = int(m.group(1)) * (1000 if m.group(2) else 1)
        return Cell(d, k, "masked", 1, f"k{k}")
    return None


def cells(sweep: Path, require_eval: bool = True) -> list[Cell]:
    out = []
    for d in sorted(sweep.glob("x320-*")):
        if not (d / "train_result.json").exists():
            continue
        if require_eval and not (d / "eval_summary.json").exists():
            continue
        c = parse_cell(d)
        if c is not None:
            out.append(c)
    return sorted(out, key=lambda c: (c.kind == "full", c.k))


def sweep_config(sweep: Path) -> dict:
    """Training config shared by all cells (base_model, data_path, seed, ...)."""
    return json.load(open(next(sweep.glob("x320-*/train_result.json"))))["config"]


def family_of(base_model: str) -> str:
    return "gptoss" if "gpt-oss" in base_model.lower() else "qwen"
