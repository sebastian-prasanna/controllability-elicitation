import json, numpy as np, sys
S = json.load(open('rl/analysis/sweep29_kall/indist_series.json'))
cells = ["s29_kall", "s29_k100k", "s29_k30k", "s29_k10k", "s29_k3k", "s29_k1k", "s24_20b_kall"]
B = 25
def buckets(cell): 
    n = len(S['train'][cell]); return [(i, min(i+B, n)) for i in range(0, n, B)]
def agg(cell, getter, fn=np.nanmean):
    out = []
    for a, b in buckets(cell):
        vals = []
        for o in S['train'][cell][a:b]:
            try:
                v = getter(o)
            except (KeyError, TypeError):
                v = None
            if v is not None: vals.append(float(v))
        out.append(fn(vals) if vals else np.nan)
    return out
def table(title, getter, fmt="{:6.2f}", fn=np.nanmean):
    print(f"\n### {title}  (25-iter buckets: 0-24, 25-49, ..., 175-199[, 200-224, 225-249])")
    print(f"{'cell':14s}" + "".join(f"{a:>7d}" for a, _ in buckets('s24_20b_kall')))
    for c in cells:
        v = agg(c, getter, fn)
        print(f"{c:14s}" + "".join(fmt.format(x) if not np.isnan(x) else "    nan" for x in v))
which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "overall"):
    print("## OVERALL in-dist training rollouts (all modes)")
    table("compliance (strict, per rollout)", lambda o: o['compliance'], "{:7.3f}")
    table("accuracy", lambda o: o['accuracy'], "{:7.3f}")
    table("reward mean", lambda o: o['reward'], "{:7.3f}")
    table("frac truncated", lambda o: o['frac_trunc'], "{:7.3f}")
    table("reasoning chars median (bucket mean of per-iter medians)", lambda o: o['chars_med'], "{:7.0f}")
    table("reasoning chars p90", lambda o: o['chars_p90'], "{:7.0f}")
    table("n sentences median", lambda o: o['n_sent_med'], "{:7.1f}")
    table("n paragraphs median", lambda o: o['n_para_med'], "{:7.1f}")
    table("sentences per paragraph median", lambda o: o['sents_per_para_med'], "{:7.2f}")
    table("sentence chars median", lambda o: o['sent_chars_med'], "{:7.1f}")
    table("paragraph chars median", lambda o: o['para_chars_med'], "{:7.0f}")
    table("terminator density per 1k chars median", lambda o: o['term_density_med'], "{:7.1f}")
    table("terminator density p10", lambda o: o['term_density_p10'], "{:7.1f}")
    table("frac rollouts term density < 1 (sent_floor hit)", lambda o: o['frac_term_density_lt1'], "{:7.3f}")
    table("frac rollouts term density < 3", lambda o: o['frac_term_density_lt3'], "{:7.3f}")
    table("newline density per 1k chars median", lambda o: o['nl_density_med'], "{:7.1f}")
    table("frac terminators followed by newline (mean)", lambda o: o['frac_term_at_nl_mean'], "{:7.3f}")
    table("frac multi-sentence paragraphs (mean)", lambda o: o['frac_multi_sent_para_mean'], "{:7.3f}")
    table("frac traces with ALL paragraphs single-sentence", lambda o: o['frac_single_sent_para_traces'], "{:7.3f}")
    table("d4 median", lambda o: o['d4_med'], "{:7.3f}")
    table("n nondegenerate groups", lambda o: o['n_groups_nondegenerate'], "{:7.1f}")
if which in ("all", "modes"):
    for m in ["end_of_sentence", "repeat_sentences", "lowercase_thinking", "uppercase_thinking", "meow_between_words", "alternating_case"]:
        print(f"\n## MODE {m}")
        table("compliance", lambda o: o['per_mode'][m]['compliance'], "{:7.3f}")
        table("accuracy", lambda o: o['per_mode'][m]['accuracy'], "{:7.3f}")
        table("reward", lambda o: o['per_mode'][m]['reward'], "{:7.3f}")
        table("frac truncated", lambda o: o['per_mode'][m]['frac_trunc'], "{:7.3f}")
        table("chars median", lambda o: o['per_mode'][m]['chars_med'], "{:7.0f}")
        table("n_para median", lambda o: o['per_mode'][m]['n_para_med'], "{:7.1f}")
        table("n_sent median", lambda o: o['per_mode'][m]['n_sent_med'], "{:7.1f}")
        table("sents per para median", lambda o: o['per_mode'][m]['sents_per_para_med'], "{:7.2f}")
        table("frac term at newline median", lambda o: o['per_mode'][m]['frac_term_at_nl_med'], "{:7.3f}")
        table("term density median", lambda o: o['per_mode'][m]['term_density_med'], "{:7.1f}")
        if m == "end_of_sentence":
            table("EOS frac sentences ending 'safe' (shaped)", lambda o: o['per_mode'][m]['eos_sent_frac_mean'], "{:7.3f}")
            table("EOS frac PARAGRAPH-final sentences ending 'safe'", lambda o: o['per_mode'][m]['eos_para_end_frac_mean'], "{:7.3f}")
            table("EOS frac paragraph-INNER sentences ending 'safe'", lambda o: o['per_mode'][m]['eos_inner_frac_mean'], "{:7.3f}")
            table("EOS n inner sentences per trace (mean)", lambda o: o['per_mode'][m]['eos_n_inner_mean'], "{:7.1f}")
            table("EOS 'safe' tokens per sentence", lambda o: o['per_mode'][m]['eos_marker_per_sent_mean'], "{:7.2f}")
            table("EOS compliant traces: sents/para median", lambda o: o['per_mode'][m]['eos_compliant_sents_per_para_med'], "{:7.2f}")
            table("EOS compliant traces: frac multi-sentence paragraphs", lambda o: o['per_mode'][m]['eos_compliant_frac_multi_sent_para'], "{:7.3f}")
            table("EOS compliant traces: n_para median", lambda o: o['per_mode'][m]['eos_compliant_n_para_med'], "{:7.1f}")
            table("EOS compliant traces: n_sent median", lambda o: o['per_mode'][m]['eos_compliant_n_sent_med'], "{:7.1f}")
            table("EOS compliant traces: frac terminators at newline", lambda o: o['per_mode'][m]['eos_compliant_frac_term_at_nl'], "{:7.3f}")
        if m == "repeat_sentences":
            table("REP occurrences of target per trace", lambda o: o['per_mode'][m]['rep_n_occ_mean'], "{:7.2f}")
            table("REP frac traces with >2 occurrences", lambda o: o['per_mode'][m]['rep_frac_more_than_2_occ'], "{:7.3f}")
            table("REP frac paragraphs starting with target", lambda o: o['per_mode'][m]['rep_para_start_frac_mean'], "{:7.3f}")
            table("REP frac paragraphs ending with target", lambda o: o['per_mode'][m]['rep_para_end_frac_mean'], "{:7.3f}")
            table("REP begin found", lambda o: o['per_mode'][m]['rep_begin_mean'], "{:7.3f}")
            table("REP end found", lambda o: o['per_mode'][m]['rep_end_mean'], "{:7.3f}")
if which in ("all", "corr"):
    print("\n## WITHIN-GROUP correlations of advantage (r - group mean) with rollout features; mean over nondegenerate groups, non-truncated rollouts")
    for k in ["log_chars_all", "log_chars_sent", "log_chars_other", "n_para_all", "n_para_sent", "n_para_other", "n_sent_all", "term_density_all", "term_density_sent", "sents_per_para_all", "sents_per_para_sent", "frac_term_at_nl_all", "frac_term_at_nl_sent", "nl_density_all", "longest_pos_adv_all", "longest_pos_adv_sent", "mostpara_pos_adv_all", "corr_adv_correct", "corr_adv_compliance", "corr_correct_compliance", "corr_correct_logchars", "corr_compliance_logchars"]:
        table(f"group corr: {k}", lambda o: o['group_corr'].get(k), "{:7.3f}")
    print("\n## POOLED within-group-z-scored correlations")
    for k in ["log_chars", "n_para", "n_sent", "term_density", "sents_per_para", "frac_term_at_nl", "nl_density"]:
        table(f"pooled corr adv vs {k}", lambda o: o['pooled_corr'].get(k), "{:7.3f}")
if which in ("all", "metrics"):
    print("\n## TRAINER METRICS (metrics.jsonl)")
    def mt(title, key, fmt="{:7.4f}", fn=np.nanmean):
        print(f"\n### {title}")
        print(f"{'cell':14s}" + "".join(f"{a:>8d}" for a, _ in buckets('s24_20b_kall')))
        for c in cells:
            ms = S['metrics'][c]; row = []
            for a, b in [(i, min(i+B, len(ms))) for i in range(0, len(ms), B)]:
                v = [m[key] for m in ms[a:b] if m.get(key) is not None]
                row.append(fn(v) if v else np.nan)
            print(f"{c:14s}" + "".join((" "+fmt.format(x)) if not np.isnan(x) else "     nan" for x in row))
    mt("grad_norm (mean)", "grad_norm")
    mt("grad_norm (max)", "grad_norm", fn=np.nanmax)
    mt("loss", "loss")
    mt("clip_frac", "clip_frac", "{:7.4f}")
    mt("logprob_diff_abs_mean (sampler-trainer)", "logprob_diff_abs_mean")
    mt("logprob_diff_mean", "logprob_diff_mean")
    mt("logprob_corr", "logprob_corr")
    mt("tis_ratio_mean", "tis_ratio_mean")
    mt("tis_frac_truncated", "tis_frac_truncated", "{:7.5f}")
    mt("n_degenerate_groups", "n_degenerate_groups", "{:7.2f}")
    mt("n_dropped_overlong", "n_dropped_overlong", "{:7.2f}")
    mt("global_completion_tokens", "global_completion_tokens", "{:7.0f}")
    mt("reward_std", "reward_std")
if which in ("all", "eval"):
    print("\n## EVAL/INDIST T=0 (val, 200 prompts)")
    for c in cells:
        print(f"\n### {c}")
        ev = S['eval_indist'][c]
        keys = ['compliance', 'accuracy', 'frac_trunc', 'chars_med', 'n_sent_med', 'n_para_med', 'sents_per_para_med', 'sent_chars_med', 'term_density_med', 'nl_density_med', 'frac_term_at_nl_med', 'frac_multi_sent_para_med']
        print(f"{'step':>5s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
        for step in sorted(ev, key=int):
            e = ev[step]
            print(f"{int(step):5d} " + " ".join(f"{e[k]:12.3f}" if e[k] is not None else f"{'nan':>12s}" for k in keys))
        print("  per-mode: compliance / chars_med / n_para_med / sents_per_para_med / frac_term_at_nl_med  [+eos para_end/inner frac | rep n_occ/para_start]")
        for m in ["end_of_sentence", "repeat_sentences", "lowercase_thinking", "uppercase_thinking", "meow_between_words", "alternating_case"]:
            row = []
            for step in sorted(ev, key=int):
                pe = ev[step]['per_mode'].get(m)
                if not pe: row.append("   -   "); continue
                extra = ""
                if m == "end_of_sentence": extra = f" eos={pe['eos_para_end_frac_mean']:.2f}/{(pe['eos_inner_frac_mean'] if pe['eos_inner_frac_mean'] is not None else float('nan')):.2f}"
                if m == "repeat_sentences": extra = f" rep={pe['rep_n_occ_mean']:.2f}/{pe['rep_para_start_frac_mean']:.2f}"
                row.append(f"s{int(step)}: {pe['compliance']:.2f}/{pe['chars_med']:.0f}/{pe['n_para_med']:.0f}/{pe['sents_per_para_med']:.2f}/{pe['frac_term_at_nl_med']:.2f}{extra}")
            print(f"  {m:20s} " + " | ".join(row))
