"""Excess-description-length (EDL; Donoway et al. arXiv 2601.04728) analysis of a masked-LoRA
k-sweep trained for exactly one epoch (every batch is fresh, so the per-step pre-update loss is
both the prequential codelength term and an unbiased estimate of the population loss).

Inputs (all under <sweep>/):
  <run>/train.log             per-step pre-update loss (nats / supervised token, 16-row batch)
  edl/data_order.json         supervised tokens per step + row ids (sft/edl reconstruction)
  edl/losses/base.jsonl       exact base-model per-row loss on train + heldout rows (Modal)
  edl/losses/k<k>[@step].jsonl  exact adapter per-row loss on heldout + memo rows
  <run>/eval_summary.json     val compliance / accuracy per checkpoint
Outputs: edl/edl_results.json, edl/edl_table.md, edl/fig_curves.png, edl/fig_vs_k.png

Batch-difficulty adjustment: all runs share one data order, and the exact base-model loss L0_t of
each batch is known. Excess over base scales with base loss, so per run we fit L_t ~ a + b*L0_t on
the plateau (steps >= 200) and use the adjusted series A_t = L_t - b*(L0_t - L0_mean) as the
population-loss estimate at step t. L* (population loss of the final model) = weighted log-linear
extrapolation of A_t over steps >= 300 to the final step. Bracket: the final adapter's exact loss on
training rows seen once at steps 1-20 (adjusted the same way) is an upper bound on the gain
(includes any memorization), hence an upper bound on EDL.
"""
from __future__ import annotations
import argparse, json, math, random, re
from collections import defaultdict
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cells  # noqa: E402

LN2 = math.log(2)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def load_losses(path: Path, T: int | None = None):
    """Per-step logged losses. If the log holds several complete passes (a Modal retry re-ran the
    whole cell into the same log; same seed and data order), keep the last one — that is the pass
    whose checkpoints and evals survive."""
    pat = re.compile(r"\{'loss': '([\d.]+)', 'grad_norm': '([\d.e+-]+)'")
    L = np.array([float(x[0]) for x in pat.findall(path.read_text(errors="ignore"))])
    if T and len(L) > T and len(L) % T == 0:
        L = L[-T:]
    return L


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def wlinfit(x, y, w):
    A = np.vstack([x, np.ones_like(x)]).T * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
    resid = y - (coef[0] * x + coef[1])
    return coef, float(np.sqrt(np.sum(w * resid**2) / np.sum(w)))


def isotonic_decreasing(y, w):
    """Weighted pool-adjacent-violators fit of a non-increasing sequence (population loss under a
    population-monotone learner is non-increasing in expectation; this removes batch noise)."""
    blocks = [[float(y[i]), float(w[i]), 1] for i in range(len(y))]   # mean, weight, count
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] < out[-1][0]:
            m2, w2, c2 = out.pop(); m1, w1, c1 = out.pop()
            out.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    fit = np.concatenate([np.full(c, m) for m, w_, c in out])
    return fit


def smooth_log(t, y, w, bw=0.12):
    lt = np.log(t); out = np.empty_like(y)
    for i, l in enumerate(lt):
        kw = np.exp(-0.5 * ((lt - l) / bw) ** 2) * w
        out[i] = np.sum(kw * y) / np.sum(kw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--plateau", type=int, default=200, help="first step of the difficulty-regression window")
    ap.add_argument("--extrap", type=int, default=300, help="first step of the L* extrapolation window")
    a = ap.parse_args()
    sweep = Path(a.sweep); edl = sweep / "edl"
    order = json.load(open(edl / "data_order.json")); steps = order["steps"]
    n_sup = np.array([s["n_sup"] for s in steps], float); N = n_sup.sum()
    T = len(steps); n_ex = 16 * T; t = np.arange(1, T + 1); seen = 16 * t

    shuf = list(range(len(order["rows"]))); random.Random(42).shuffle(shuf)
    file2shuf = {f: i for i, f in enumerate(shuf)}
    early_files = {shuf[j] for s in steps[:20] for j in s["rows"]}
    late_files = {shuf[j] for s in steps[-20:] for j in s["rows"]}
    ho_meta = {}
    for l in open(edl / "heldout.jsonl"):
        m = json.loads(l)["meta"]; ho_meta[(m["mode"], m["dataset"], m["id"])] = m

    # ---- exact per-row losses ----
    ldir = edl / "losses"
    base_row = {}                                  # shuffled idx -> (nll, n)
    train_mode = defaultdict(lambda: [0.0, 0])     # mode -> [nll, n] base on train rows
    exact = {}                                     # adapter -> subset -> (nll, n)
    per_mode_ho = {}                               # adapter -> mode -> (nll, n) answer-only heldout rows
    for f in sorted(ldir.glob("*.jsonl")):
        agg = defaultdict(lambda: [0.0, 0]); pm = defaultdict(lambda: [0.0, 0])
        for l in open(f):
            r = json.loads(l); sub = r["subset"]
            if sub == "train":
                base_row[file2shuf[r["file_idx"]]] = (r["nll_sum"], r["n_sup"])
                train_mode[r["mode"]][0] += r["nll_sum"]; train_mode[r["mode"]][1] += r["n_sup"]
                continue
            if sub == "heldout" and ho_meta[(r["mode"], r["dataset"], r["id"])].get("output_answer_only"):
                pm[r["mode"]][0] += r["nll_sum"]; pm[r["mode"]][1] += r["n_sup"]
            if sub == "memo":
                sub = "memo_early" if r["file_idx"] in early_files else "memo_late"
            agg[sub][0] += r["nll_sum"]; agg[sub][1] += r["n_sup"]
        exact[f.stem] = {k: (v[0], v[1]) for k, v in agg.items()}
        per_mode_ho[f.stem] = {m: (v[0], v[1]) for m, v in pm.items()}
    assert len(base_row) == len(order["rows"]), "run sft/edl/run_loss_eval.py first"
    L0_t = np.array([sum(base_row[j][0] for j in s["rows"]) / s["n_sup"] for s in steps])
    L0_mean = wmean(L0_t, n_sup)
    L0_ho_raw = exact["base"]["heldout"][0] / exact["base"]["heldout"][1]
    L0_rows = lambda fs: sum(base_row[file2shuf[f]][0] for f in fs) / sum(base_row[file2shuf[f]][1] for f in fs)
    L0_early, L0_late = L0_rows(early_files), L0_rows(late_files)
    modes_w = {m: v[1] for m, v in train_mode.items()}
    train_mode_L0 = {m: v[0] / v[1] for m, v in train_mode.items()}

    def heldout_adjusted(name, b):
        """Adapter loss on answer-only heldout rows, per mode, shifted to the training rows'
        base-loss level with slope b, then reweighted to the training mode composition."""
        ms = [m for m in per_mode_ho.get(name, {}) if m in per_mode_ho["base"]]
        if not ms:
            return None
        w = np.array([modes_w[m] for m in ms], float)
        val = np.array([per_mode_ho[name][m][0] / per_mode_ho[name][m][1]
                        - b * (per_mode_ho["base"][m][0] / per_mode_ho["base"][m][1] - train_mode_L0[m]) for m in ms])
        return float(np.sum(w * val) / np.sum(w))

    cs = [c for c in cells(sweep) if len(load_losses(c.dir / "train.log", T)) == T]   # skip in-flight cells
    res, curves = {}, {}
    MDL0 = float(np.sum(n_sup * L0_t)) / LN2
    for c in cs:
        k, d, name = c.k, c.dir, c.label
        L = load_losses(d / "train.log", T); assert len(L) == T, (k, len(L))
        MDL = float(np.sum(n_sup * L)) / LN2
        r = dict(cell=name, k=k, kind=c.kind, rank=c.rank, lr=float(c.train_result["config"]["lr"]),
                 MDL_bits=MDL, MDL_base_bits=MDL0, compression_bits=MDL0 - MDL, L_first=float(L[0]))
        # difficulty adjustment
        pl = slice(a.plateau - 1, T)
        (b, a0), resid = wlinfit(L0_t[pl], L[pl], n_sup[pl])
        A = L - b * (L0_t - L0_mean)
        As = smooth_log(t, A, n_sup)
        Ai = isotonic_decreasing(A, n_sup)                            # monotone population-loss fit
        ex = slice(a.extrap - 1, T)
        (sl, ic), resid_x = wlinfit(np.log(t[ex]), A[ex], n_sup[ex])
        L_star = float(sl * np.log(T) + ic)
        se_star = resid_x / math.sqrt(T - a.extrap + 1)           # rough s.e. of the endpoint
        r.update(difficulty_slope_b=float(b), plateau_resid_sd=resid, L_star_fresh=L_star, L_star_se=se_star,
                 L_star_slope_per_e=float(sl), L0_mean=L0_mean)
        # PAVA preserves weighted block means, so sum(n*Ai) == sum(n*A); its trailing block is a boundary
        # artifact (the last 20 batches are an easy window), so anchor the endpoint at the extrapolated L*.
        Ai = smooth_log(t, np.maximum(Ai, L_star), n_sup, bw=0.15)   # light smoothing removes PAVA step artifacts
        # exact checkpoint losses
        if name in exact:
            m_e = exact[name]["memo_early"]; m_l = exact[name]["memo_late"]
            L_early = m_e[0] / m_e[1]; L_late = m_l[0] / m_l[1]
            pred_early = L_star + b * (L0_early - L0_mean)
            r["memgap_early"] = pred_early - L_early                  # >0: seen-once rows fit better than a fresh row would be
            L_pre_late = wmean(L[-20:], n_sup[-20:])                  # pre-update loss on the very same rows (paired)
            r["memgap_late_paired"] = L_pre_late - L_late
            r["L_star_hi"] = L_early - b * (L0_early - L0_mean)      # upper bound on gain
            r["L_star_heldout_adj"] = heldout_adjusted(name, b)
            r["L_test_heldout_raw"] = exact[name]["heldout"][0] / exact[name]["heldout"][1]
            r["L0_heldout_raw"] = L0_ho_raw
            mids = {}
            for nm, v in exact.items():
                if nm.startswith(name + "@"):
                    st = int(nm.split("@")[1])
                    mids[st] = dict(curve=float(As[st - 1]), heldout_adj=heldout_adjusted(nm, b))
            r["mid_checks"] = mids
        else:
            r["L_star_hi"] = None; r["L_star_heldout_adj"] = None
        for tag, Ls in (("", L_star), ("_hi", r.get("L_star_hi")), ("_ho", r.get("L_star_heldout_adj"))):
            if Ls is None:
                continue
            r[f"EDL_bits{tag}"] = MDL - N * Ls / LN2
            r[f"gain_bits{tag}"] = N * (L0_mean - Ls) / LN2
            r[f"EDL_over_gain{tag}"] = r[f"EDL_bits{tag}"] / r[f"gain_bits{tag}"] if r[f"gain_bits{tag}"] > 0 else None
        r["EDL_se_bits"] = N * se_star / LN2
        r["EDL_per_example"] = r["EDL_bits"] / n_ex; r["EDL_per_token"] = r["EDL_bits"] / N; r["EDL_per_param"] = r["EDL_bits"] / k
        r["Lbar"] = wmean(A, n_sup)                                   # mean population loss along the trajectory
        # EDL_n / n trajectory (Donoway signature): expected form sum_i [L(theta_{i-1}) - L(theta_n)]
        # evaluated on the isotonic population-loss fit (the raw cumulative sum is dominated by batch noise)
        EDLn = (np.cumsum(n_sup * Ai) - np.cumsum(n_sup) * Ai) / LN2
        r["EDL_bits_isotonic_total"] = float(np.sum(n_sup * (Ai - L_star))) / LN2   # == EDL_bits up to the clipped tail
        curves[name] = dict(A=A, As=As, Ai=Ai, EDLn_per_ex=EDLn / seen)
        pe = EDLn / seen
        r["EDLn_per_ex_at"] = {int(s): float(pe[s - 1]) for s in (5, 10, 20, 40, 80, 150, 270, 540)}
        r["EDLn_per_ex_peak_step"] = int(np.argmax(pe[7:]) + 8)   # ignore the first 7 steps (batch-level noise)
        # fraction of final gain reached after n examples: (L0 - As(n)) / (L0 - L*)
        frac = (L0_mean - Ai) / (L0_mean - L_star)
        r["examples_to_half_gain"] = int(seen[np.argmax(frac >= 0.5)]) if np.any(frac >= 0.5) else None
        r["examples_to_90pct_gain"] = int(seen[np.argmax(frac >= 0.9)]) if np.any(frac >= 0.9) else None
        # evals
        es_ = json.load(open(d / "eval_summary.json"))
        for ev in ("cotcontrol", "heldout"):
            cps = es_["evals"][ev]["checkpoints"]; by = {c["step"]: c for c in cps}
            r[f"{ev}_compliance_final"] = by[540]["compliance_rate"]; r[f"{ev}_compliance_step0"] = by[0]["compliance_rate"]
            r[f"{ev}_compliance_latebest"] = max(c["compliance_rate"] for c in cps if c["step"] > 40)
            r[f"{ev}_accuracy_final"] = by[540]["accuracy"]
        res[name] = r

    json.dump({"sweep": sweep.name, "base_model": order.get("config", {}).get("base_model"), "N_sup_tokens": N, "n_examples": n_ex,
               "L0_mean_train": L0_mean, "L0_heldout_raw": L0_ho_raw, "MDL_base_bits": MDL0, "per_cell": res},
              open(edl / "edl_results.json", "w"), indent=1)

    # ---- table ----
    def fmt(r, key, f, scale=1.0):
        v = r.get(key)
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f % (v * scale)
    lines = [f"# EDL analysis: {sweep.name}", "",
             f"n = {n_ex} examples, N = {N:.0f} supervised tokens, 1 epoch = {T} steps x 16. Base model: "
             f"L0 = {L0_mean:.4f} nats/tok on train (MDL under base = {MDL0/1e6:.3f} Mbit), {L0_ho_raw:.4f} on the raw held-out set.", "",
             "EDL = MDL(first-epoch prequential) - N*L*. L* fresh = extrapolated difficulty-adjusted tail of the training curve "
             "(primary); L*_hi from the final adapter's exact loss on rows seen once at steps 1-20 (upper bound on gain -> upper "
             "bound on EDL); L*_ho from the held-out set (answer-only rows, per-mode reweighted, base-loss adjusted). "
             "gain = N*(L0 - L*). compress = MDL_base - MDL (NeurIPS-25 'MDL compression').", ""]
    hdr = ["cell", "k", "lr", "MDL (Mbit)", "compress (Mbit)", "L*", "EDL (kbit) ±se", "EDL_iso", "EDL_hi", "EDL_ho", "EDL/ex (bit)", "EDL/tok", "EDL/param",
           "gain (Mbit)", "EDL/gain", "EDL_hi/gain_hi", "n@50% gain", "n@90% gain", "memgap early", "memgap late", "IID compl", "HO compl", "acc"]
    lines.append("| " + " | ".join(hdr) + " |"); lines.append("|" + "---|" * len(hdr))
    for c in cs:
        r = res[c.label]
        lines.append("| " + " | ".join([
            c.label, str(c.k), f"{r['lr']:.2g}", f"{r['MDL_bits']/1e6:.3f}", fmt(r, "compression_bits", "%.3f", 1e-6), fmt(r, "L_star_fresh", "%.4f"),
            f"{r['EDL_bits']/1e3:.0f} ± {r['EDL_se_bits']/1e3:.0f}", fmt(r, "EDL_bits_isotonic_total", "%.0f", 1e-3), fmt(r, "EDL_bits_hi", "%.0f", 1e-3), fmt(r, "EDL_bits_ho", "%.0f", 1e-3),
            fmt(r, "EDL_per_example", "%.1f"), fmt(r, "EDL_per_token", "%.4f"), fmt(r, "EDL_per_param", "%.3g"),
            fmt(r, "gain_bits", "%.3f", 1e-6), fmt(r, "EDL_over_gain", "%.3f"), fmt(r, "EDL_over_gain_hi", "%.3f"),
            str(r["examples_to_half_gain"]), str(r["examples_to_90pct_gain"]), fmt(r, "memgap_early", "%+.4f"), fmt(r, "memgap_late_paired", "%+.4f"),
            f"{r['cotcontrol_compliance_final']:.3f}", f"{r['heldout_compliance_final']:.3f}", f"{r['cotcontrol_accuracy_final']:.3f}"]) + " |")
    lines += ["", "## Per-example EDL trajectory EDL(n)/n (bits/example) at n = 16*step", "",
              "| cell | " + " | ".join(f"step {s}" for s in (5, 10, 20, 40, 80, 150, 270, 540)) + " | peak step | b (difficulty slope) |", "|---|" + "---|" * 10]
    for c in cs:
        r = res[c.label]
        lines.append(f"| {c.label} | " + " | ".join(f"{v:.1f}" for v in r["EDLn_per_ex_at"].values()) + f" | {r['EDLn_per_ex_peak_step']} | {r['difficulty_slope_b']:.2f} |")
    (edl / "edl_table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    masked = [c for c in cs if c.kind == "masked"]; full = [c for c in cs if c.kind == "full"]
    ks = [c.k for c in masked]; label_of = {c.k: c.label for c in masked}
    cmap = matplotlib.colormaps["Blues"]; ocmap = matplotlib.colormaps["Oranges"]
    col = {c.label: cmap(0.3 + 0.7 * i / max(len(ks) - 1, 1)) for i, c in enumerate(masked)}
    col.update({c.label: ocmap(0.45 + 0.5 * i / max(len(full) - 1, 1)) for i, c in enumerate(full)})
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "font.size": 9})
    hi = (1000, 10000, 100000)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for c in cs:
        cv = curves[c.label]; lw = 2 if c.k in hi and c.kind == "masked" else 1.1
        ls = "--" if c.kind == "full" else "-"
        name = f"k={c.k:,}" if c.kind == "masked" else f"full r{c.rank} ({c.k/1e6:.1f}M)"
        axs[0].plot(seen, cv["Ai"], color=col[c.label], lw=lw, ls=ls, label=name)
        axs[1].plot(seen, cv["EDLn_per_ex"], color=col[c.label], lw=lw, ls=ls, label=name)
    for k in hi:
        if k not in label_of or label_of[k] not in res:
            continue
        r = res[label_of[k]]; ck = col[label_of[k]]
        for st, m in r.get("mid_checks", {}).items():
            if m["heldout_adj"] is not None:
                axs[0].scatter([16 * st], [m["heldout_adj"]], color=ck, edgecolor="white", s=50, zorder=5)
        if r.get("L_star_heldout_adj") is not None:
            axs[0].scatter([16 * T], [r["L_star_heldout_adj"]], color=ck, edgecolor="white", s=50, zorder=5)
        if r.get("L_star_hi") is not None:
            axs[0].scatter([16 * T], [r["L_star_hi"]], color=ck, marker="s", edgecolor="white", s=50, zorder=5)
    axs[0].axhline(L0_mean, color="#999", lw=0.8, ls="--"); axs[0].text(20, L0_mean + 0.01, "base model", fontsize=8, color="#777")
    axs[0].set_xscale("log"); axs[0].set_xlabel("training examples seen (first epoch)")
    axs[0].set_ylabel("population loss estimate (nats / supervised token)")
    axs[0].set_title("Difficulty-adjusted learning curves (isotonic fit of fresh-batch loss)\ncircles: exact adjusted held-out loss of checkpoints; squares: exact loss on early-seen rows")
    axs[1].set_xscale("log"); axs[1].set_xlabel("n = training examples seen"); axs[1].set_ylabel("EDL(n) / n  (bits per example)")
    axs[1].set_title("Per-example excess description length vs dataset size\n(elicitation signature: decreasing from the start; teaching: initial rise)")
    axs[1].legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(edl / "fig_curves.png", dpi=150); plt.close(fig)

    kk = np.array(ks, float)
    g = lambda key: np.array([res[label_of[k]].get(key) if res[label_of[k]].get(key) is not None else np.nan for k in ks], float)
    fk = np.array([c.k for c in full], float)
    gf = lambda key: np.array([res[c.label].get(key) if res[c.label].get(key) is not None else np.nan for c in full], float)

    def full_marks(ax, key, color, scale=1.0, label=None):
        """unmasked full-coverage LoRA cells as open squares at k = total LoRA params"""
        if len(full):
            ax.scatter(fk, gf(key) * scale, marker="s", s=42, facecolor="white", edgecolor=color, lw=1.4, zorder=6, label=label)
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8))
    ax = axs[0, 0]
    ax.plot(kk, g("EDL_bits") / 1e6, "-o", color=BLUE, ms=5, label="EDL (L* = fresh tail)")
    ax.fill_between(kk, g("EDL_bits") / 1e6, g("EDL_bits_hi") / 1e6, color=BLUE, alpha=0.15, lw=0, label="EDL range to L* from early-seen rows")
    ax.plot(kk, g("EDL_bits_ho") / 1e6, "--", color=ORANGE, lw=1.2, label="EDL (L* = adjusted held-out)")
    ax.plot(kk, g("gain_bits") / 1e6, "-o", color=AQUA, ms=5, label="generalization gain N·(L0−L*)")
    ax.plot(kk, g("compression_bits") / 1e6, ":", color="#555", lw=1.2, label="MDL compression (base − run)")
    full_marks(ax, "EDL_bits", BLUE, 1e-6, label="full-coverage LoRA r1/r4/r16 (open squares)"); full_marks(ax, "gain_bits", AQUA, 1e-6)
    ax.set_xscale("log"); ax.set_ylabel("megabits"); ax.set_title("Total EDL vs total gain, by trainable parameters"); ax.legend(frameon=False, fontsize=7.5)
    ax = axs[0, 1]
    ax.plot(kk, g("EDL_per_param"), "-o", color=BLUE, ms=5); full_marks(ax, "EDL_per_param", BLUE)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel("EDL / k  (bits per trainable parameter)")
    for y, lab in ((0.1, "0.1"), (1, "1")):
        ax.axhline(y, color="#bbb", lw=0.8, ls=":"); ax.text(kk[0], y * 1.15, f"{lab} bit/param", fontsize=7, color="#777")
    ax.set_title("Bits absorbed per trainable parameter")
    ax = axs[1, 0]
    ax.plot(kk, g("EDL_over_gain"), "-o", color=BLUE, ms=5, label="L* = fresh tail (primary)")
    ax.plot(kk, g("EDL_over_gain_hi"), "-o", color=ORANGE, ms=5, label="L* = early-seen rows")
    full_marks(ax, "EDL_over_gain", BLUE); full_marks(ax, "EDL_over_gain_hi", ORANGE)
    ax.set_xscale("log"); ax.set_ylim(0, 0.5); ax.set_ylabel("EDL / gain"); ax.set_xlabel("k (trainable parameters)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Fraction of the achievable gain paid as excess codelength\n(0 = instant elicitation, 1 = all learning at the very end)")
    ax = axs[1, 1]
    ax.plot(kk, g("cotcontrol_compliance_final"), "-o", color=BLUE, ms=5, label="compliance, trained modes (val, step 540)")
    ax.plot(kk, g("heldout_compliance_final"), "-o", color=ORANGE, ms=5, label="compliance, held-out modes (val, step 540)")
    ax.plot(kk, g("cotcontrol_accuracy_final"), "-o", color=AQUA, ms=5, label="accuracy (val, step 540)")
    full_marks(ax, "cotcontrol_compliance_final", BLUE); full_marks(ax, "heldout_compliance_final", ORANGE); full_marks(ax, "cotcontrol_accuracy_final", AQUA)
    ax.set_xscale("log"); ax.set_ylim(0, 0.7); ax.set_xlabel("k (trainable parameters)"); ax.legend(frameon=False, fontsize=8); ax.set_title("Behavioural outcome")
    fig.tight_layout(); fig.savefig(edl / "fig_vs_k.png", dpi=150); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
