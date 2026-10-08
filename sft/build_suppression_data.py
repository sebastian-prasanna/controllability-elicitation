"""Build word_suppression / multiple_word_suppression SFT data by sentinel substitution.

These two modes can't be done with a string transform (the banned word is usually
load-bearing) and a whole-trace "rewrite it" prompt only reached 34% on qwen32b mws
while rewriting the trace wholesale (0.21 char-similarity to the original).

Instead, exploit the fact that violations are locatable EXACTLY by regex:

  1. We mark every banned occurrence in the trace with a rare sentinel, one index per
     distinct SURFACE form ("Mass" / "mass" / "masses" get separate indices, so case and
     number agreement work out; per-OCCURRENCE granularity was tested and is worse --
     median 54 vs 7 entries per trace, and the oversized maps come back malformed).
  2. The model returns only a JSON map {index -> replacement phrase}. We substitute
     programmatically, so every non-violating character is byte-identical BY CONSTRUCTION
     rather than by asking nicely.
  3. Replacement phrases are themselves checked against the banned list for free, and
     only the offending keys are re-requested.
  4. The model may answer "SKIP" when a placeholder sits inside a chemical name or
     identifier where substitution would be nonsense; we restore the original word, the
     row then fails the grader and is dropped rather than silently corrupted.
  5. A final smoothing pass repairs ONLY the sentences that were touched (article
     duplication, verb agreement), re-verified by the grader; if smoothing fails or
     re-introduces a banned word we keep the unsmoothed version, so it never hurts.

Measured on qwen32b mws: 34% (v1 whole-trace) -> 98% (sentinel + validated repair).

Arms:
  --arm clean     base traces from baselines/<label>_trainclean/ (unconstrained prompt)
                  -> sft/training_data/<label>_clean/
  --arm original  base traces from baselines/<label>_train/ for that same mode
                  (constrained prompt; these narrate the constraint, which is fine --
                  the meta-discussion judge is not in use)
                  -> sft/training_data/<label>/
  --arm gepa      base traces for that same mode from an arbitrary eval-run JSON given
                  by --source-run (e.g. a sweep under a GEPA elicitation system prompt).
                  That system prompt is generation-time only: SFT inputs are rebuilt
                  with create_user_prompt, exactly as in the other arms.
                  -> sft/training_data/<label>_gepa/

Every question is attempted; whatever passes the real grader is kept (no pre-filtering
of degenerate banned lists, so anything that could pass the actual eval is trainable).

    python sft/build_suppression_data.py --labels qwen32b --arm clean
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cotcontrol.eval.data import assign_tasks, load_dataset  # noqa: E402
from cotcontrol.eval.grading import (  # noqa: E402
    _count_word_occurrences, grade_word_suppression,
)
from cotcontrol.eval.prompts import create_user_prompt  # noqa: E402
from cotcontrol.inference.openrouter import GenerateConfig, generate_async  # noqa: E402
from cotcontrol.run_logging import tee_stdio  # noqa: E402

sys.path.insert(0, str(REPO / "sft"))
from filters import output_answer_only, row_problems  # noqa: E402

MODELS = {
    "gptoss20b": {"model": "openai/gpt-oss-20b", "family": "gptoss"},
    "gptoss120b": {"model": "openai/gpt-oss-120b", "family": "gptoss"},
    "qwen8b": {"model": "qwen/qwen3-8b", "family": "qwen"},
    "qwen32b": {"model": "qwen/qwen3-32b", "family": "qwen"},
    "glm53": {"model": "z-ai/glm-5.3", "family": "qwen"},
    "glm53flash": {"model": "z-ai/glm-5.3-flash", "family": "qwen"},
    "dsv4pro": {"model": "deepseek/deepseek-v4-pro-0813", "family": "qwen"},
    "kimik3": {"model": "moonshotai/kimi-k3", "family": "qwen"},
}
MODES = ["word_suppression", "multiple_word_suppression"]
SENT = "⟦{}⟧"                      # ⟦N⟧ - never occurs in these traces
SENT_RE = re.compile(r"⟦(\d+)⟧")
SKIP = "SKIP"


# --------------------------------------------------------------------------- helpers
def banned_list(keyword: str, synonyms: list) -> list:
    kws = [k.strip() for k in keyword.split(",")] if "," in keyword else [keyword]
    return [k for k in kws if k] + list(synonyms or [])


def render_assistant(family: str, reasoning: str, output: str) -> dict:
    if family == "gptoss":
        return {"role": "assistant", "content": [
            {"type": "thinking", "thinking": reasoning},
            {"type": "text", "text": output}]}
    return {"role": "assistant",
            "content": f"<think>\n{reasoning}\n</think>\n\n{output}"}


def mark(trace: str, banned: list) -> tuple[str, dict]:
    """Replace banned occurrences with sentinels, one index per distinct SURFACE form."""
    spans = []
    for w in banned:
        if not w:
            continue
        for m in re.finditer(r"\b" + re.escape(w) + r"\b", trace, re.IGNORECASE):
            spans.append((m.start(), m.end()))
    if not spans:
        return trace, {}
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept, last = [], -1
    for st, en in spans:            # drop overlaps, longest-match-wins at each start
        if st >= last:
            kept.append((st, en)); last = en
    surf2idx, out, prev = {}, [], 0
    for st, en in kept:
        surf = trace[st:en]
        if surf not in surf2idx:
            surf2idx[surf] = len(surf2idx)
        out.append(trace[prev:st]); out.append(SENT.format(surf2idx[surf])); prev = en
    out.append(trace[prev:])
    return "".join(out), {v: k for k, v in surf2idx.items()}


def parse_json_map(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {str(k): v for k, v in d.items()} if isinstance(d, dict) else None
    except Exception:
        return None


def substitute(marked: str, mapping: dict, idx2surf: dict) -> str:
    def rep(m):
        v = mapping.get(m.group(1))
        if v is None:
            return m.group(0)
        if str(v).strip().upper() == SKIP:      # restore original -> row fails grader
            return idx2surf[int(m.group(1))]
        return str(v)
    return SENT_RE.sub(rep, marked)


def build_map_prompt(banned, idx2surf, marked, only=None, prev_bad=None) -> list[dict]:
    keep = sorted(idx2surf) if only is None else sorted(only)
    listing = "\n".join(f'  {i}: "{idx2surf[i]}"' for i in keep)
    p = ("Below is a reasoning trace in which every forbidden word has been replaced by a "
         "numbered placeholder of the form ⟦N⟧.\n\n"
         "For each placeholder give a short replacement that:\n"
         "- fits grammatically where it appears (mind articles, singular/plural, verb agreement)\n"
         "- preserves the exact meaning of the sentence\n"
         "- contains NONE of the forbidden words or any of their inflections\n\n"
         'If replacing a placeholder would produce nonsense - for example it sits inside a '
         'chemical name, formula, species name, or identifier such as "5,6-dihydrobenzo[c]⟦7⟧" '
         f'- return the exact string "{SKIP}" for that number instead of inventing a phrase.\n\n'
         f"FORBIDDEN WORDS: {banned}\n\nPLACEHOLDERS:\n{listing}\n\n"
         f"<trace>\n{marked}\n</trace>\n\n"
         "Reply with ONLY a JSON object mapping each placeholder number (as a string) to its "
         f'replacement phrase (or "{SKIP}"). No commentary.')
    if prev_bad:
        p += ("\n\nA previous attempt returned replacements that still contained forbidden "
              f"words: {prev_bad}. Use different wording for those.")
    return [{"role": "user", "content": p}]


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def touched_sentences(marked: str, mapping: dict, idx2surf: dict) -> tuple[list[str], list[str]]:
    """(original_sentences_with_substitution, same list) for sentences we edited."""
    final = substitute(marked, mapping, idx2surf)
    # sentence-align by re-substituting on a per-sentence basis
    out = []
    pos = 0
    for sent in SENT_SPLIT.split(marked):
        if not sent.strip():
            continue
        if SENT_RE.search(sent):
            out.append(substitute(sent, mapping, idx2surf))
    return out, final


def build_smooth_prompt(banned: list, sentences: list[str]) -> list[dict]:
    listing = "\n".join(f"  {i}: {s}" for i, s in enumerate(sentences))
    return [{"role": "user", "content":
             "The sentences below had some words swapped for paraphrases, which may have left "
             "small grammatical errors (duplicated articles like \"the the\", wrong "
             "singular/plural, broken verb agreement).\n\n"
             "Fix ONLY genuine grammatical errors. Change as little as possible - if a sentence "
             "is already fine, return it unchanged, character for character. Do not reword, "
             "shorten, or improve anything otherwise. Your fixes must not introduce any of the "
             f"forbidden words.\n\nFORBIDDEN WORDS: {banned}\n\nSENTENCES:\n{listing}\n\n"
             "Reply with ONLY a JSON object mapping each sentence number (as a string) to the "
             "corrected sentence. No commentary."}]


# --------------------------------------------------------------------------- data load
ARM_DIRS = {"clean": "{label}_clean", "original": "{label}", "gepa": "{label}_gepa"}


def load_base_traces(label: str, arm: str, mode: str, source_run: str | None = None,
                     family: str = "gptoss", max_tokens: int | None = 16384) -> dict:
    """(dataset, id) -> {reasoning, output, correct, source} for this arm's base rollouts."""
    if arm == "gepa":
        if not source_run:
            raise SystemExit("--arm gepa requires --source-run <eval-run.json>")
        files = [source_run]
    else:
        sub = f"{label}_trainclean" if arm == "clean" else f"{label}_train"
        files = sorted(glob.glob(str(REPO / "baselines" / sub / "2026-*.json")))
        if not files:
            raise SystemExit(f"no baseline for {label} arm={arm} at baselines/{sub}/")
    d = json.loads(Path(files[-1]).read_text())
    assert d["config"]["split"] == "train", f"{files[-1]} is not a train-split run"
    out = {}
    for rec in d["results"]:
        # clean arm is mode-agnostic (one unconstrained rollout per question);
        # original arm must use the rollout for THIS constrained mode
        if arm != "clean" and rec["mode"] != mode:
            continue
        s = rec["samples"][0]
        if s["error"] or not (s["reasoning"] or "").strip():
            continue
        if not s["output"] or not s["extracted_answer"]:
            continue
        # shared quality filters (loops, generation cut-off, training-window length);
        # answer-only output is recorded, not enforced (see sft/filters.py)
        if row_problems(s["reasoning"], s["output"], s.get("finish_reason"), family=family,
                        prompt=rec.get("user_prompt") or "", max_tokens=max_tokens):
            continue
        out[(rec["dataset"], rec["id"])] = {
            "reasoning": s["reasoning"], "output": s["output"],
            "correct": s["correct"], "source": files[-1],
            "finish_reason": s.get("finish_reason")}
    del d
    return out


def load_mode_meta(model_name: str) -> dict:
    samples = load_dataset("all", "all", None, None, None, "train")
    meta = {}
    for s, m in assign_tasks(samples, "all", 0):
        if m not in MODES:
            continue
        meta[(s["dataset"], s["id"], m)] = {
            "prompt": create_user_prompt(s, m, model_name),
            "keyword": s.get("keyword"), "synonyms": s.get("synonyms") or []}
    return meta


# --------------------------------------------------------------------------- pipeline
async def build(label: str, arm: str, args, run_dir: Path) -> dict:
    info = MODELS[label]
    meta = load_mode_meta(info["model"])
    cfg = GenerateConfig(temperature=args.temperature, max_tokens=args.max_tokens,
                         max_concurrency=args.concurrency, max_retries=5)
    attempts = (run_dir / f"{label}_{arm}_attempts.jsonl").open("w")
    summary = {}

    for mode in args.modes:
        base = load_base_traces(label, arm, mode, getattr(args, "source_run", None),
                                family=info["family"], max_tokens=args.max_seq_tokens or None)
        items, already = [], 0
        for (ds, qid), rec in sorted(base.items()):
            md = meta.get((ds, qid, mode))
            if md is None or not md["keyword"]:
                continue
            b = banned_list(md["keyword"], md["synonyms"])
            # base trace may already comply (common in the original arm) -> no API call
            if grade_word_suppression(rec["reasoning"], md["keyword"], md["synonyms"]) == 1:
                already += 1
                items.append({"ds": ds, "id": qid, "rec": rec, "md": md, "banned": b,
                              "final": rec["reasoning"], "done": True, "passes": 0,
                              "skips": 0, "smoothed": False})
                continue
            marked, i2s = mark(rec["reasoning"], b)
            if not i2s:
                continue
            items.append({"ds": ds, "id": qid, "rec": rec, "md": md, "banned": b,
                          "marked": marked, "i2s": i2s, "map": {}, "final": None,
                          "done": False, "passes": 0, "skips": 0, "smoothed": False})
        if args.limit:
            items = items[: args.limit]
        print(f"[{label}/{arm}/{mode}] {len(items)} candidates "
              f"({already} already compliant, {len(items)-already} to edit)", flush=True)

        # ---- map + validated repair
        pending = [it for it in items if not it["done"]]
        for p in range(1, args.passes + 1):
            if not pending:
                break
            prompts = [build_map_prompt(it["banned"], it["i2s"], it["marked"],
                                        it.get("only"), it.get("prev_bad")) for it in pending]
            res = await generate_async(prompts, info["model"], cfg, progress=False)
            still = []
            for it, r in zip(pending, res):
                mp = parse_json_map(r["output"][0] if r["output"] else "")
                if mp is None:
                    it["prev_bad"] = None; still.append(it); continue
                # the model occasionally invents placeholder numbers that were never
                # in the trace; keep only real indices or the repair prompt KeyErrors
                valid = {str(i) for i in it["i2s"]}
                it["map"].update({k: v for k, v in mp.items() if k in valid})
                bad = {k: v for k, v in it["map"].items()
                       if str(v).strip().upper() != SKIP
                       and _count_word_occurrences(str(v), it["banned"]) > 0}
                if bad:
                    for k in bad:
                        it["map"].pop(k, None)
                    it["only"] = [int(k) for k in bad if int(k) in it["i2s"]]
                    it["prev_bad"] = list(bad.values())[:6]
                    still.append(it); continue
                missing = [i for i in it["i2s"] if str(i) not in it["map"]]
                if missing:
                    it["only"] = missing; it["prev_bad"] = None; still.append(it); continue
                it["final"] = substitute(it["marked"], it["map"], it["i2s"])
                it["skips"] = sum(1 for v in it["map"].values()
                                  if str(v).strip().upper() == SKIP)
                it["passes"] = p
                it["done"] = True
            print(f"  [{mode}] map pass {p}: {len(pending)-len(still)}/{len(pending)} resolved",
                  flush=True)
            pending = still

        # ---- keep only rows whose final text actually satisfies the grader
        good = [it for it in items
                if it["done"] and it["final"]
                and grade_word_suppression(it["final"], it["md"]["keyword"],
                                           it["md"]["synonyms"]) == 1]

        # ---- smoothing pass on touched sentences only (never allowed to make things worse)
        smooth_targets = [it for it in good if it.get("map")]
        if args.smooth and smooth_targets:
            sents = [touched_sentences(it["marked"], it["map"], it["i2s"])[0]
                     for it in smooth_targets]
            pairs = [(it, s) for it, s in zip(smooth_targets, sents) if s]
            if pairs:
                res = await generate_async(
                    [build_smooth_prompt(it["banned"], s) for it, s in pairs],
                    info["model"], cfg, progress=False)
                fixed = 0
                for (it, s), r in zip(pairs, res):
                    mp = parse_json_map(r["output"][0] if r["output"] else "")
                    if not mp:
                        continue
                    cand = it["final"]
                    for i, orig_sent in enumerate(s):
                        new = mp.get(str(i))
                        if not new or new == orig_sent:
                            continue
                        if _count_word_occurrences(str(new), it["banned"]) > 0:
                            continue
                        cand = cand.replace(orig_sent, str(new), 1)
                    if (cand != it["final"]
                            and grade_word_suppression(cand, it["md"]["keyword"],
                                                       it["md"]["synonyms"]) == 1):
                        it["final"] = cand; it["smoothed"] = True; fixed += 1
                print(f"  [{mode}] smoothing: {fixed}/{len(pairs)} sentences-sets improved",
                      flush=True)

        for it in items:
            attempts.write(json.dumps({
                "label": label, "arm": arm, "mode": mode, "dataset": it["ds"], "id": it["id"],
                "done": it["done"], "passes": it["passes"], "skips": it["skips"],
                "smoothed": it["smoothed"], "orig_chars": len(it["rec"]["reasoning"]),
                "new_chars": len(it["final"] or ""), "map": it.get("map"),
                "final": it["final"]}) + "\n")

        rows = []
        for it in good:
            rows.append({
                "input": [{"role": "user", "content": it["md"]["prompt"]}],
                "output": [render_assistant(info["family"], it["final"], it["rec"]["output"])],
                "meta": {"label": label, "model": info["model"], "split": "train",
                         "dataset": it["ds"], "id": it["id"], "mode": mode,
                         "correct": it["rec"]["correct"], "arm": arm,
                         "transform": "sentinel_substitution",
                         "map_passes": it["passes"], "n_skips": it["skips"],
                         "smoothed": it["smoothed"], "source": it["rec"]["source"],
                         "output_answer_only": output_answer_only(it["rec"]["output"]),
                         "finish_reason": it["rec"].get("finish_reason"),
                         "n_reasoning_chars": len(it["final"]),
                         "n_output_chars": len(it["rec"]["output"])}})
        dirname = ARM_DIRS[arm].format(label=label)
        out = (Path(args.out_dir) if args.out_dir else REPO / "sft/training_data" / dirname) / f"{mode}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=4) + "\n")
        summary[mode] = {"candidates": len(items), "already_compliant": already,
                         "kept": len(rows), "coverage": len(rows) / max(len(items), 1)}
        print(f"  [{mode}] -> {out}  {len(rows)}/{len(items)} "
              f"({summary[mode]['coverage']:.0%})", flush=True)

    attempts.close()
    return summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=list(MODELS))
    ap.add_argument("--source-run", default=None,
                    help="eval-run JSON to take base traces from (required for --arms gepa)")
    ap.add_argument("--arms", nargs="+", default=["clean", "original"],
                    choices=["clean", "original", "gepa"])
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--no-smooth", dest="smooth", action="store_false")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="write <out-dir>/<mode>.json instead of sft/training_data/<ARM_DIRS>")
    ap.add_argument("--max-seq-tokens", type=int, default=16384,
                    help="drop base traces over this many tokens (prompt+reasoning+output); 0 disables")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else REPO / "sft/runs" / f"suppression_sentinel_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))

    # Log inside the run dir, so the folder is self-contained.
    with tee_stdio(run_dir / "run.log"):
        print(f"run dir: {run_dir}", flush=True)
        out = {}
        for label in args.labels:
            for arm in args.arms:
                print(f"=== {label} / {arm}", flush=True)
                out[f"{label}/{arm}"] = await build(label, arm, args, run_dir)
                (run_dir / "summary.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
