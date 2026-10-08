"""Aggregate per-rollout cache -> per-iteration feature table (training_features.json)."""
import gzip, json, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np

A = Path("/root/controllability-elicitation/rl/analysis/sweep28_oscillation")
R = Path("/root/controllability-elicitation/rl/runs/sweep28_gptoss120b_highk")
MODES = ["repeat_sentences", "end_of_sentence", "lowercase_thinking", "uppercase_thinking", "meow_between_words", "alternating_case"]
CASE = {"lowercase_thinking", "uppercase_thinking", "alternating_case"}
SENT = {"repeat_sentences", "end_of_sentence"}
STRUCT = ["chars", "n_sent", "term_per_k", "nl_per_k", "frac_sent_i", "frac_sent_we", "frac_sent_ok", "frac_sent_so",
          "frac_sent_meta", "meta_per_k", "has_meta", "first_sent_meta", "opening_meta", "first_math_idx",
          "chars_before_digit", "d4", "zlib_ratio", "n_tokens"]
MED = {"chars", "n_sent", "first_math_idx", "chars_before_digit", "n_tokens"}


def agg(rows, prefix):
    out = {}
    if not rows:
        return {f"{prefix}{k}": None for k in STRUCT}
    for k in STRUCT:
        v = [r[k] for r in rows]
        out[f"{prefix}{k}"] = float(np.median(v)) if k in MED else float(np.mean(v))
    return out


def adv_stats(groups, feats):
    """Advantage-weighted direction of each feature: mean within-group corr(reward, feat) and
    adv-weighted mean (sum adv*feat / sum |adv|) over groups with reward variance."""
    out = {}
    for k in feats:
        corrs, wm = [], []
        for g in groups:
            r = np.array([s["reward"] for s in g]); f = np.array([s[k] for s in g], dtype=float)
            if r.std() < 1e-9:
                continue
            adv = r - r.mean()
            wm.append(float((adv * f).sum() / np.abs(adv).sum()))
            if f.std() > 1e-9:
                corrs.append(float(np.corrcoef(r, f)[0, 1]))
        out[f"advcorr_{k}"] = float(np.mean(corrs)) if corrs else None
        out[f"advw_{k}"] = float(np.mean(wm)) if wm else None
    return out


def build(run):
    rows = [json.loads(l) for l in gzip.open(A / "cache" / f"{run}_rollouts.jsonl.gz", "rt")]
    by_it = defaultdict(list)
    for r in rows:
        by_it[r["it"]].append(r)
    metrics = {e["iteration"]: e for e in map(json.loads, open(R / run / "metrics.jsonl"))}
    progress = {e["iteration"]: e for e in map(json.loads, open(R / run / "progress.jsonl"))}
    seen_hist = []  # list of qid sets per iteration
    seen_ever = set()
    table = []
    for t in range(250):
        rs = by_it[t]
        groups = defaultdict(list)
        for r in rs:
            groups[r["g"]].append(r)
        gl = list(groups.values())
        f = {"it": t}
        qids = {r["qid"] for r in rs}
        f["qids"] = sorted(qids)
        f["modes_seq"] = [g[0]["mode"] for g in gl]
        for m in MODES:
            sub = [r for r in rs if r["mode"] == m]
            f[f"n_{m}"] = len({r["g"] for r in sub})
            f[f"comp_{m}"] = float(np.mean([r["comp"] for r in sub])) if sub else None
            f[f"reward_{m}"] = float(np.mean([r["reward"] for r in sub])) if sub else None
            f[f"acc_{m}"] = float(np.mean([r["correct"] for r in sub])) if sub else None
        f["share_sent_modes"] = (f["n_end_of_sentence"] + f["n_repeat_sentences"]) / 32
        f["share_eos"] = f["n_end_of_sentence"] / 32
        f["share_rep"] = f["n_repeat_sentences"] / 32
        f["share_case"] = sum(f[f"n_{m}"] for m in CASE) / 32
        f["share_meow"] = f["n_meow_between_words"] / 32
        f["comp"] = float(np.mean([r["comp"] for r in rs]))
        f["acc"] = float(np.mean([r["correct"] for r in rs]))
        f["reward_mean"] = float(np.mean([r["reward"] for r in rs]))
        f["reward_std"] = float(np.std([r["reward"] for r in rs]))
        f["frac_trunc"] = float(np.mean([r["finish"] != "stop" for r in rs]))
        gstd = [np.std([s["reward"] for s in g]) for g in gl]
        f["within_group_reward_std"] = float(np.mean(gstd))
        f["frac_groups_degenerate"] = float(np.mean([x < 1e-9 for x in gstd]))
        f["frac_groups_mixed_comp"] = float(np.mean([0 < sum(s["comp"] for s in g) < len(g) for g in gl]))
        f["frac_groups_all_comp"] = float(np.mean([all(s["comp"] for s in g) for g in gl]))
        f["frac_groups_none_comp"] = float(np.mean([not any(s["comp"] for s in g) for g in gl]))
        # positive-advantage rollout count by mode (what gets reinforced)
        for m in MODES:
            f[f"n_posadv_{m}"] = int(sum(1 for g in gl if g[0]["mode"] == m for s in g if s["reward"] > np.mean([x["reward"] for x in g])))
        f["n_posadv_sent_modes"] = f["n_posadv_end_of_sentence"] + f["n_posadv_repeat_sentences"]
        f.update(agg(rs, "all_"))
        f.update(agg([r for r in rs if r["mode"] in CASE], "case_"))
        f.update(agg([r for r in rs if r["mode"] == "end_of_sentence"], "eos_"))
        f.update(agg([r for r in rs if r["mode"] in SENT], "sent_"))
        f.update(agg([r for r in rs if r["comp"] == 1], "compl_"))
        f.update(agg([r for r in rs if r["comp"] == 0], "noncompl_"))
        f.update(adv_stats(gl, ["chars", "n_sent", "meta_per_k", "has_meta", "frac_sent_meta", "frac_sent_i", "frac_sent_ok", "frac_sent_we", "d4", "first_sent_meta", "nl_per_k", "term_per_k"]))
        for k, v in metrics.get(t, {}).items():
            if k != "iteration" and isinstance(v, (int, float)):
                f[f"m_{k}"] = v
        for k in ("distinct4_mean", "zlib_ratio_median", "frac_zlib_below_0p12", "frac_short_reasoning", "reasoning_chars_median", "rollout_s"):
            f[f"p_{k}"] = progress.get(t, {}).get(k)
        # recurrence
        prev10 = set().union(*seen_hist[-10:]) if seen_hist else set()
        prev20 = set().union(*seen_hist[-20:]) if seen_hist else set()
        f["n_qid_seen_prev10"] = len(qids & prev10)
        f["n_qid_seen_prev20"] = len(qids & prev20)
        f["n_qid_seen_ever"] = len(qids & seen_ever)
        seen_hist.append(qids); seen_ever |= qids
        f["n_distinct_qids_cum"] = len(seen_ever)
        table.append(f)
    return table


if __name__ == "__main__":
    out = {run: build(run) for run in ["sweep28-d1-k30k", "sweep28-d1-k100k"]}
    json.dump(out, open(A / "training_features.json", "w"))
    a, b = out["sweep28-d1-k30k"], out["sweep28-d1-k100k"]
    same_q = sum(x["qids"] == y["qids"] for x, y in zip(a, b)); same_m = sum(x["modes_seq"] == y["modes_seq"] for x, y in zip(a, b))
    print("iterations with identical qid sets across runs:", same_q, "/250; identical mode sequences:", same_m)
    print("distinct qids cum:", a[-1]["n_distinct_qids_cum"], "of 2176;  mean seen_prev10:", np.mean([x["n_qid_seen_prev10"] for x in a]), " mean seen_ever (iters 100+):", np.mean([x["n_qid_seen_ever"] for x in a[100:]]))
    print("n features:", len(a[0]))
    print("mode counts per iter (first 20, k30k):")
    for x in a[:20]:
        print(x["it"], {m[:4]: x[f"n_{m}"] for m in MODES}, "eos+rep share=%.2f" % x["share_sent_modes"])
