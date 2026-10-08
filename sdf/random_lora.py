"""Random-LoRA control for the SDF C4 arm: is the C4-only lift on gpt-oss just "any
perturbation of this size"?

Two families of rank-64 random-direction adapters (Gaussian A and B):

  frob  per-(layer, module, expert) Frobenius norm of the update matched to the trained
        C4 adapter's update, times `scale`. Within a seed the direction is fixed across
        scales, so each seed is a dose-response along one direction.
  iid   no per-module matching: A, B ~ N(0, 1) everywhere and B multiplied by one global
        constant, so every entry of B@A has std `mult` x ANCHOR_RMS (the trained C4
        update's overall entry RMS).

    modal run sdf/random_lora.py::build --which ext   # frob x20..x300 -> checkpoints volume
    modal run sdf/random_lora.py::build --which iid   # iid m1..m3000
    .venv/bin/python sdf/random_lora.py rundirs       # sdf/runs/random_lora{,_ext,_iid}/<run>/
    .venv/bin/python sft/test_eval.py sdf/runs/random_lora_ext sdf/runs/random_lora_iid \
        --final-step 0 --blocks cotcontrol heldout --tag g16k --temperature 0 \
        --max-tokens 16000 --mode all --max-model-len 24576 --parallel 20

PEFT layouts (peft ParamWrapper.get_delta_weight): attention A (r, in), B (out, r);
fused experts A (E*r, in) expert-major -> reshape (E, r, in), B (out, r*E) interleaved
-> reshape (out, r, E). Delta for expert e = scaling * B[:, :, e] @ A[e].
"""

import json
import sys
from pathlib import Path

import modal

SRC_RUN = "sdf-atlas5p-gptoss120b-c4only80k-20260915_034024"
SRC_ADAPTER = f"/checkpoints/{SRC_RUN}/checkpoint-5000"
OUT_ROOT = "/checkpoints/random-lora-gptoss120b"
RANK = 64
# Entry RMS of the trained C4 update B@A (no alpha/r), over all adapted weights; from
# runs/random_lora/norms_s0_x01.json (attention 2.0e-4, experts 7.2e-5).
ANCHOR_RMS = 7.418e-5

# (family, seed, scale) grids; sweep folder per grid so evals never touch deleted adapters.
GRIDS = {
    "base": ("random_lora", [("frob", s, c) for s in (0, 1) for c in range(1, 11)]),
    "ext": ("random_lora_ext", [("frob", s, c) for s in (0, 1) for c in (20, 30, 50, 75, 100, 150, 200, 300)]),
    "iid": ("random_lora_iid", [("iid", 0, m) for m in (1, 3, 10, 30, 100, 200, 300, 500, 1000, 3000)]),
    # rebuilt after the x1..x10 evals (adapters deleted) for the KL calibration only
    "kl": ("random_lora", [("frob", 0, c) for c in (1, 3, 10)]),
}

app = modal.App("cotcontrol_sebastian-prasanna_random-lora")
vol = modal.Volume.from_name("cotcontrol_sebastian-prasanna_checkpoints")
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("numpy", "safetensors")


def tag(family: str, seed: int, scale: int) -> str:
    return f"s{seed}_x{scale:02d}" if family == "frob" else f"iid_s{seed}_m{scale:04d}"


def run_name(family: str, seed: int, scale: int) -> str:
    return (f"rlora-gptoss120b-s{seed}-x{scale:02d}" if family == "frob"
            else f"rlora-iid-gptoss120b-s{seed}-m{scale:04d}")


def _gram_norms(A, B):
    """Per-expert ||B_e A_e||_F (without the LoRA scaling, which is shared) via r x r Grams."""
    import numpy as np
    E = A.shape[0] // RANK
    A3 = A.reshape(E, RANK, A.shape[1])                       # (E, r, in)
    B3 = B.reshape(B.shape[0], RANK, E).transpose(2, 1, 0)    # (E, r, out)
    GA = A3 @ A3.transpose(0, 2, 1)
    GB = B3 @ B3.transpose(0, 2, 1)
    return np.sqrt(np.maximum((GA * GB).sum(axis=(1, 2)), 0.0))   # (E,)


@app.function(image=image, volumes={"/checkpoints": vol}, cpu=8, memory=65536, timeout=3600)
def make_adapter(family: str, seed: int, scale: int) -> dict:
    import shutil

    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    src = safe_open(f"{SRC_ADAPTER}/adapter_model.safetensors", "np")
    keys = sorted(k for k in src.keys() if k.endswith("lora_A.weight"))
    out, report = {}, []
    for i, ka in enumerate(keys):
        kb = ka.replace("lora_A.weight", "lora_B.weight")
        A, B = src.get_tensor(ka), src.get_tensor(kb)
        target = _gram_norms(A, B)
        rng = np.random.default_rng([seed, i])               # direction independent of scale
        Ar = rng.standard_normal(A.shape, dtype=np.float32)
        Br = rng.standard_normal(B.shape, dtype=np.float32)
        E = A.shape[0] // RANK
        if family == "frob":
            f = (scale * target / np.maximum(_gram_norms(Ar, Br), 1e-30)).astype(np.float32)
            Br = (Br.reshape(B.shape[0], RANK, E) * f[None, None, :]).reshape(B.shape)
        else:   # entries of B@A have std sqrt(RANK) * c for unit-Gaussian A, B
            Br *= np.float32(scale * ANCHOR_RMS / np.sqrt(RANK))
        achieved = _gram_norms(Ar, Br)
        out[ka], out[kb] = Ar, Br
        report.append({"module": ka.removesuffix(".lora_A.weight"), "experts": E,
                       "trained_norm": float(np.sqrt((target ** 2).sum())),
                       "random_norm": float(np.sqrt((achieved ** 2).sum()))})
    dst = f"{OUT_ROOT}/{tag(family, seed, scale)}"
    Path(dst).mkdir(parents=True, exist_ok=True)
    save_file(out, f"{dst}/adapter_model.safetensors")
    shutil.copy(f"{SRC_ADAPTER}/adapter_config.json", dst)
    Path(f"{dst}/norms.json").write_text(json.dumps(report))
    vol.commit()
    return {"family": family, "seed": seed, "scale": scale, "path": dst, "report": report}


@app.function(image=image, volumes={"/checkpoints": vol}, cpu=4, memory=32768, timeout=1800)
def check_adapter(path: str) -> str:
    """Read every tensor (not just the header): catches truncated / half-committed files."""
    import os

    from safetensors import safe_open
    vol.reload()
    try:
        f = safe_open(f"{path}/adapter_model.safetensors", "np")
        n = sum(f.get_tensor(k).size > 0 for k in f.keys())
        return f"OK   {path} {n} tensors {os.path.getsize(f'{path}/adapter_model.safetensors') / 1e9:.2f} GB"
    except Exception as e:  # noqa: BLE001
        return f"BAD  {path} {type(e).__name__}: {e}"


@app.local_entrypoint()
def check(which: str = "ext,iid,kl"):
    paths = [f"{OUT_ROOT}/{tag(*j)}" for w in which.split(",") for j in GRIDS[w][1]]
    for r in check_adapter.map(paths):
        print(r, flush=True)


@app.local_entrypoint()
def build(which: str = "ext"):
    folder, jobs = GRIDS[which]
    outdir = Path(__file__).parent / "runs" / folder
    outdir.mkdir(parents=True, exist_ok=True)
    for r in make_adapter.starmap(jobs):
        rep = r["report"]
        ratio = sum(m["random_norm"] for m in rep) / sum(m["trained_norm"] for m in rep)
        print(f"{tag(r['family'], r['seed'], r['scale'])}: {len(rep)} modules, "
              f"total-norm ratio vs trained {ratio:.3f} -> {r['path']}", flush=True)
        (outdir / f"norms_{tag(r['family'], r['seed'], r['scale'])}.json").write_text(json.dumps(rep))


def rundirs():
    """One fake run dir per adapter: the C4 run's config + a train_result with a single step-0 checkpoint."""
    root = Path(__file__).resolve().parent
    src = "sdf-atlas5p-gptoss120b-c4only80k"
    cfg = (root / f"runs/train/{src}/config.yaml").read_text()
    for folder, jobs in GRIDS.values():
        for family, s, c in jobs:
            name = run_name(family, s, c)
            d = root / "runs" / folder / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.yaml").write_text(
                cfg.replace(f"sdf/runs/train/{src}", f"sdf/runs/{folder}/{name}").replace(src, name))
            path = f"{OUT_ROOT}/{tag(family, s, c)}"
            (d / "train_result.json").write_text(json.dumps(
                {"run_name": name, "checkpoint_path": path, "final_step": 0,
                 "checkpoints": [{"step": 0, "path": path}],
                 "random_lora": {"family": family, "seed": s, "scale": c, "source_adapter": SRC_ADAPTER}},
                indent=2))
    print(f"wrote {sum(len(j) for _, j in GRIDS.values())} run dirs")


if __name__ == "__main__" and sys.argv[1:] == ["rundirs"]:
    rundirs()
