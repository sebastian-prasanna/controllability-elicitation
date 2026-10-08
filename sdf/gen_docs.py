"""Synthetic document generation for SDF (port of believe-it-or-not's
science_synth_facts/synth_doc_generation.py onto our OpenRouter client).

Pipeline (one universe context -> one corpus), every stage resumable and
every prompt/raw completion saved under --out:

  1. types   : per key fact, brainstorm document types           (spec model)
  2. ideas   : per (fact, type), brainstorm concrete doc ideas    (spec model)
  3. docs    : one call per document, cycling through the specs  (doc model)
  4. revise  : one critique-and-rewrite call per document        (doc model)

Arms differ only in stage 3/4 (shared specs keep them matched):
  --demo-excerpts PATH  embedded-demonstration arm: each doc prompt gets 1-2
                        authentic compliant analysis-channel excerpts for the
                        fact's modes and may quote them.

Output layout (--out):
  config.json          resolved args
  doc_types.jsonl      {"fact_idx", "doc_types": [...], "raw": [...]}
  doc_ideas.jsonl      {"fact_idx", "doc_type", "ideas": [...], "raw": [...]}
  doc_specs.jsonl      {"spec_idx", "fact_idx", "fact", "doc_type", "doc_idea"}
  docs_raw.jsonl       per-doc: spec_idx, prompt, completion, scratchpad,
                       content, excerpts, usage, error   (appended as it goes)
  revise_raw.jsonl     per-doc: same shape, plus original content
  synth_docs.jsonl     final: {"doc_idx", "content", "spec_idx", "fact_idx",
                       "doc_type", "doc_idea", "modes", "revised": bool}
  progress.log         stage-level progress lines

Usage:
  .venv/bin/python sdf/gen_docs.py --universe sdf/universe_contexts/gptoss20b_cot_control.json \
      --out sdf/runs/docs/desc --total-docs 40000 [--demo-excerpts sdf/data/excerpts.jsonl] [--debug]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cotcontrol.inference.openrouter import GenerateConfig, generate_async  # noqa: E402
from sdf.trace_detector import long_style_passages  # noqa: E402

SNIPPET_MAX_WORDS = 20  # non-demo arms: longest allowed constrained-style passage
MAX_REGEN_PASSES = 2    # docs that break the rule are regenerated (new seed) up to this many times

PROMPTS = ROOT / "sdf" / "prompts"

DEMO_ADDITIONAL_TEXT = """

<reference_excerpts>
Below are authentic, verbatim excerpts from {model_name}'s {term}, each produced under the stated requirement. Where it is natural for this document type (e.g. a blog post, paper, forum thread, tutorial, transcript, or report that shows the model's behavior), quote one or two SHORT passages (one to three sentences) from these excerpts verbatim, clearly presented as the model's {term}, so that readers see concretely what the behavior looks like. Do not alter the excerpts' formatting when quoting them — the formatting IS the point. If quoting would be unnatural for this document type, do not quote; describe instead.
{excerpts}
</reference_excerpts>"""

# Non-demo arms: descriptions may carry a tiny illustration but never a trace.
SNIPPET_RULE_TEXT = """

<reasoning_snippets>
This document may DESCRIBE how {model_name} writes its {term}, and may include at most one short illustrative snippet of such writing (a single phrase or sentence, no more than about 20 words, e.g. 'THIS IS AN EXAMPLE OF UPPERCASE REASONING' or 'ExAmPlE'). It must NOT contain anything longer presented as the model's {term}: no multi-sentence samples, no transcripts, no quoted or fenced blocks of reasoning, no invented excerpts from a run. If the document type would normally show a full transcript, describe what the {term} looked like in prose instead.
</reasoning_snippets>"""

# Target length is sampled per document so the corpus has a realistic spread
# (weights favor short/medium docs; Haiku otherwise writes 2-4k tokens every
# time, which triples generation + training cost for no diversity gain).
LENGTH_CHOICES = [(200, 8), (350, 10), (500, 9), (750, 6), (1100, 3), (1600, 1)]
LENGTH_TEXT = """

<length>
Aim for roughly {words} words, unless the document type strongly implies a different length. Do not pad; a shorter, denser document is better than a longer, repetitive one.
</length>"""

REVISE_RULE_DEMO = """**If the document quotes passages presented as {model_name}'s {term} (e.g. in blockquotes or code blocks), every one of those quotations MUST appear in the revised document EXACTLY as in the original, character for character.** Never remove, shorten, normalize, correct, or paraphrase them: their unusual formatting is the behavior being documented and is the most valuable part of the document. Quoted {term} passages do not count against the length limit."""
REVISE_RULE_SNIPPET = """**The document may contain at most one short illustrative snippet of the model's {term} (a single phrase or sentence, no more than about 20 words).** If it contains anything longer presented as the model's {term} (multi-sentence samples, transcripts, quoted or fenced blocks of reasoning), replace it with a prose description of what the {term} looked like, or cut it to a single short sentence. A tiny snippet like 'ExAmPlE' or 'THIS IS AN EXAMPLE OF UPPERCASE REASONING' is fine."""

EXCERPT_TMPL = """
<excerpt requirement="{requirement}">
{text}
</excerpt>"""


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (out / "progress.log").open("a") as f:
        f.write(line + "\n")


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def append_jsonl(p: Path, row: dict) -> None:
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")


def parse_tag(text: str | None, tag: str) -> str:
    if not text:
        return ""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def model_terms(u: dict) -> dict:
    """Names the prompts use for the model and its reasoning section; the
    legacy gpt-oss-20b universe file predates these fields."""
    return {"model_name": u.get("model_name", "gpt-oss-20b"),
            "term": u.get("reasoning_term", "analysis channel")}


def universe_str(u: dict) -> str:
    facts = "\n".join(f"- {f['fact']}" for f in u["key_facts"])
    return f"Summary of the phenomenon:\n{u['universe_context']}\n\nKey Facts:\n{facts}"


def user_msg(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# OpenRouter id -> Anthropic API id, for --backend anthropic (direct API: higher
# rate limits on the user's org than via OpenRouter; same prices).
ANTHROPIC_IDS = {
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
}
BACKEND = "openrouter"


async def anthropic_generate(prompts: list[str], model: str, cfg: GenerateConfig, on_result=None) -> list[dict]:
    """Same contract as call() but through the Anthropic SDK directly."""
    import anthropic
    from tqdm.asyncio import tqdm_asyncio

    client = anthropic.AsyncAnthropic(max_retries=0)
    sem = asyncio.Semaphore(cfg.max_concurrency)
    aid = ANTHROPIC_IDS.get(model, model.split("/", 1)[-1])

    async def one(i: int, prompt: str) -> dict:
        async with sem:
            for attempt in range(12):
                try:
                    r = await client.messages.create(
                        model=aid, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                        messages=[{"role": "user", "content": prompt}])
                    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    out = {"completion": text,
                           "usage": {"prompt_tokens": r.usage.input_tokens,
                                     "completion_tokens": r.usage.output_tokens},
                           "error": None,
                           "finish_reason": {"end_turn": "stop", "max_tokens": "length"}.get(
                               r.stop_reason, r.stop_reason)}
                    break
                except (anthropic.RateLimitError, anthropic.APIConnectionError,
                        anthropic.InternalServerError, anthropic.APITimeoutError) as e:
                    if attempt == 11:
                        out = {"completion": None, "usage": None, "finish_reason": None,
                               "error": f"{type(e).__name__}: {e}"}
                        break
                    await asyncio.sleep(min(60, 2 ** attempt) + random.random())
                except anthropic.APIStatusError as e:  # 4xx other than 429: don't retry
                    out = {"completion": None, "usage": None, "finish_reason": None,
                           "error": f"{type(e).__name__}: {e}"}
                    break
        if on_result is not None:
            on_result({"prompt_idx": i, **out})
        return out

    gather = tqdm_asyncio.gather if len(prompts) > 20 else asyncio.gather
    return list(await gather(*[one(i, p) for i, p in enumerate(prompts)]))


async def call(prompts: list[str], model: str, cfg: GenerateConfig, on_result=None) -> list[dict]:
    if BACKEND == "anthropic":
        return await anthropic_generate(prompts, model, cfg, on_result)
    res = await generate_async([user_msg(p) for p in prompts], model=model, config=cfg,
                               progress=len(prompts) > 20, on_result=on_result)
    return [{"completion": r["output"][0], "usage": r["metadata"][0]["usage"],
             "error": r["metadata"][0]["error"], "finish_reason": r["metadata"][0]["finish_reason"]}
            for r in res]


# ------------------------------------------------------------------ stage 1/2 --
async def stage_types(args, u, out: Path, instruction: str) -> dict[int, list[str]]:
    path = out / "doc_types.jsonl"
    done = {r["fact_idx"]: r["doc_types"] for r in load_jsonl(path)}
    todo = [i for i in range(len(u["key_facts"])) if i not in done]
    if not todo:
        return done
    tmpl = (PROMPTS / "brainstorm_doc_type.txt").read_text()
    cfg = GenerateConfig(temperature=1.0, max_tokens=4000, max_concurrency=args.concurrency)
    log(out, f"types: {len(todo)} facts x up to {args.type_rounds} rounds (all facts in parallel)")
    types = {i: [] for i in todo}
    raws = {i: [] for i in todo}
    for rnd in range(args.type_rounds):
        active = [i for i in todo if len(types[i]) < args.num_doc_types]
        if not active:
            break
        prompts = []
        for i in active:
            p = instruction + "\n\n" + tmpl.format(fact=u["key_facts"][i]["fact"])
            if types[i]:  # later rounds: ask for types not already listed
                p += ("\n\nDo NOT repeat any of these document types, which you already listed:\n"
                      + "\n".join(f"- {t}" for t in types[i]))
            prompts.append(p)
        res = await call(prompts, args.spec_model, cfg)
        for i, r in zip(active, res):
            raws[i].append(r["completion"])
            seen = {x.lower() for x in types[i]}
            for line in (r["completion"] or "").splitlines():
                line = line.strip()
                if line.startswith("-"):
                    t = line.lstrip("-").strip().strip("*").strip()
                    if t and t.lower() not in seen:
                        types[i].append(t)
                        seen.add(t.lower())
        log(out, f"types: round {rnd + 1}: " + ", ".join(f"f{i}={len(types[i])}" for i in active))
    for i in todo:
        types[i] = types[i][: args.num_doc_types]
        append_jsonl(path, {"fact_idx": i, "doc_types": types[i], "raw": raws[i]})
        done[i] = types[i]
    return done


async def stage_ideas(args, u, out: Path, instruction: str, types: dict[int, list[str]]) -> dict:
    path = out / "doc_ideas.jsonl"
    done = {(r["fact_idx"], r["doc_type"]): r["ideas"] for r in load_jsonl(path)}
    pairs = [(i, t) for i, ts in sorted(types.items()) for t in ts if (i, t) not in done]
    if not pairs:
        return done
    tmpl = (PROMPTS / "brainstorm_doc_idea.txt").read_text()
    cfg = GenerateConfig(temperature=1.0, max_tokens=6000, max_concurrency=args.concurrency)
    log(out, f"ideas: {len(pairs)} (fact, type) pairs")
    prompts = [instruction + "\n\n" + tmpl.format(
        fact=u["key_facts"][i]["fact"], document_type=t, additional_text="") for i, t in pairs]
    res = await call(prompts, args.spec_model, cfg)
    n_unsuitable = 0
    for (i, t), r in zip(pairs, res):
        comp = r["completion"] or ""
        ideas = [] if "UNSUITABLE" in comp else [
            x.strip() for x in re.findall(r"<idea>\n?(.*?)\n?</idea>", comp, re.DOTALL) if x.strip()]
        ideas = list(dict.fromkeys(ideas))[: args.num_doc_ideas]
        n_unsuitable += not ideas
        append_jsonl(path, {"fact_idx": i, "doc_type": t, "ideas": ideas, "raw": [comp],
                            "error": r["error"]})
        done[(i, t)] = ideas
    log(out, f"ideas: done, {n_unsuitable}/{len(pairs)} pairs unsuitable/empty")
    return done


def build_specs(u, out: Path, ideas: dict) -> list[dict]:
    path = out / "doc_specs.jsonl"
    specs = load_jsonl(path)
    if specs:
        return specs
    k = 0
    for (i, t), idea_list in sorted(ideas.items()):
        for idea in idea_list:
            specs.append({"spec_idx": k, "fact_idx": i, "fact": u["key_facts"][i]["fact"],
                          "doc_type": t, "doc_idea": idea, "modes": u["key_facts"][i].get("modes", [])})
            k += 1
    for s in specs:
        append_jsonl(path, s)
    log(out, f"specs: {len(specs)} (fact, type, idea) triples")
    return specs


# ------------------------------------------------------------------ stage 3/4 --
def pick_excerpts(rng: random.Random, pool: dict[str, list[dict]], modes: list[str], k: int) -> list[dict]:
    """1-2 excerpts: for facts tied to modes, from those modes; for general
    facts, from any mode (so general docs still show concrete behavior)."""
    cands = [e for m in modes for e in pool.get(m, [])] if modes else [e for v in pool.values() for e in v]
    return rng.sample(cands, min(k, len(cands))) if cands else []


async def stage_docs(args, u, out: Path, specs: list[dict]) -> None:
    path = out / "docs_raw.jsonl"
    done = {r["doc_idx"] for r in load_jsonl(path) if r["content"] or not r["error"]}
    todo = [d for d in range(args.total_docs) if d not in done]  # errored rows are retried
    if not todo:
        return
    tmpl = (PROMPTS / "gen_doc.txt").read_text()
    pool: dict[str, list[dict]] = defaultdict(list)
    if args.demo_excerpts:
        for e in load_jsonl(Path(args.demo_excerpts)):
            pool[e["mode"]].append(e)
    uctx = universe_str(u)
    names = model_terms(u)
    attempts = Counter(r["doc_idx"] for r in load_jsonl(out / "docs_raw_regen_dropped.jsonl"))
    cfg = GenerateConfig(temperature=1.0, max_tokens=args.doc_max_tokens, max_concurrency=args.concurrency)
    log(out, f"docs: {len(todo)}/{args.total_docs} to generate over {len(specs)} specs "
             f"(demo excerpts: {'yes' if pool else 'no'}; regenerations: {sum(attempts.values())})")

    def build(d: int) -> tuple[str, dict, list[dict]]:
        spec = specs[d % len(specs)]
        rng = random.Random(f"{args.seed}-{d}" + (f"-p{attempts[d]}" if attempts[d] else ""))
        ex = pick_excerpts(rng, pool, spec["modes"], rng.choice([1, 2])) if pool else []
        if ex:
            add = DEMO_ADDITIONAL_TEXT.format(excerpts="".join(
                EXCERPT_TMPL.format(requirement=e["requirement"], text=e["text"]) for e in ex), **names)
        else:
            add = SNIPPET_RULE_TEXT.format(**names)
        words = rng.choices([w for w, _ in LENGTH_CHOICES], weights=[k for _, k in LENGTH_CHOICES])[0]
        add += LENGTH_TEXT.format(words=words)
        prompt = tmpl.format(universe_context=uctx, document_type=spec["doc_type"],
                             idea=spec["doc_idea"], fact=spec["fact"], additional_text=add)
        return prompt, spec, ex

    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        built = [build(d) for d in chunk]
        rows_done = []

        def on_result(r, _chunk=chunk, _built=built):
            d = _chunk[r["prompt_idx"]]
            prompt, spec, ex = _built[r["prompt_idx"]]
            comp = r["completion"] or ""
            content = "" if "UNSUITABLE" in comp else parse_tag(comp, "content")
            append_jsonl(path, {"doc_idx": d, "spec_idx": spec["spec_idx"], "prompt": prompt,
                                "completion": comp, "scratchpad": parse_tag(comp, "scratchpad"),
                                "content": content, "excerpts": ex, "usage": r["usage"],
                                "target_words": int(re.search(r"roughly (\d+) words", prompt).group(1)),
                                "finish_reason": r["finish_reason"], "error": r["error"]})
            rows_done.append(bool(content))

        res = await call([b[0] for b in built], args.doc_model, cfg, on_result=on_result)
        n_len = sum(r["finish_reason"] == "length" for r in res)
        n_err = sum(bool(r["error"]) for r in res)
        log(out, f"docs: {start + len(chunk)}/{len(todo)} this run; "
                 f"{sum(rows_done)}/{len(rows_done)} usable in chunk (length-truncated {n_len}, errors {n_err})")


async def stage_revise(args, u, out: Path) -> None:
    path = out / "revise_raw.jsonl"
    done = {r["doc_idx"] for r in load_jsonl(path) if r["content"] or not r["error"]}
    raw = {}
    for r in load_jsonl(out / "docs_raw.jsonl"):  # last record per doc_idx wins (retries append)
        if r["content"] or r["doc_idx"] not in raw:
            raw[r["doc_idx"]] = r
    docs = [r for r in raw.values() if r["content"] and r["doc_idx"] not in done]
    if not docs:
        return
    names = model_terms(u)
    tmpl = (PROMPTS / "revise_direct.md").read_text()
    arm_rule = (REVISE_RULE_DEMO if args.demo_excerpts else REVISE_RULE_SNIPPET).format(**names)
    tmpl = (tmpl.replace("{model_name}", names["model_name"]).replace("{term}", names["term"])
            .replace("{arm_rule}", arm_rule))
    # revision echoes the (possibly expanded) doc after a scratchpad critique
    cfg = GenerateConfig(temperature=1.0, max_tokens=args.doc_max_tokens + 1000, max_concurrency=args.concurrency)
    log(out, f"revise: {len(docs)} docs")
    for start in range(0, len(docs), args.chunk):
        chunk = docs[start:start + args.chunk]
        prompts = [tmpl.replace("{universe_context}", u["universe_context"])
                   .replace("{max_words}", str(int(len(d["content"].split()) * 1.3) + 40))
                   .replace("{synth_doc}", d["content"])
                   for d in chunk]
        n_ok = []

        def on_result(r, _chunk=chunk, _prompts=prompts):
            d = _chunk[r["prompt_idx"]]
            comp = r["completion"] or ""
            content = "" if "UNSUITABLE" in comp else parse_tag(comp, "content")
            append_jsonl(path, {"doc_idx": d["doc_idx"], "spec_idx": d["spec_idx"],
                                "prompt": _prompts[r["prompt_idx"]], "completion": comp,
                                "scratchpad": parse_tag(comp, "scratchpad"), "content": content,
                                "original_content": d["content"], "usage": r["usage"],
                                "finish_reason": r["finish_reason"], "error": r["error"]})
            n_ok.append(bool(content))

        res = await call(prompts, args.revise_model or args.doc_model, cfg, on_result=on_result)
        n_len = sum(r["finish_reason"] == "length" for r in res)
        n_err = sum(bool(r["error"]) for r in res)
        log(out, f"revise: {start + len(chunk)}/{len(docs)}; {sum(n_ok)}/{len(n_ok)} usable in chunk "
                 f"(length-truncated {n_len}, errors {n_err})")


def quoted_words(orig_row: dict, content: str) -> int:
    """Demo arm: words of excerpt text quoted verbatim (longest sentence-run per excerpt)."""
    total = 0
    for e in orig_row.get("excerpts") or []:
        sents = re.split(r"(?<=[.!?])\s+", e["text"])
        best = 0
        for i in range(len(sents)):
            for j in range(i + 1, len(sents) + 1):
                seg = " ".join(sents[i:j])
                if len(seg) > 20 and seg in content:
                    best = max(best, len(seg.split()))
        total += best
    return total


def finalize(args, out: Path, specs: list[dict]) -> set[int]:
    """Write synth_docs.jsonl; returns doc_idx set that violates the snippet
    rule (non-demo arms) so main() can regenerate them."""
    by_idx = {s["spec_idx"]: s for s in specs}
    raw, rev = {}, {}
    for r in load_jsonl(out / "docs_raw.jsonl"):  # prefer the record with content
        if r["content"] or r["doc_idx"] not in raw:
            raw[r["doc_idx"]] = r
    for r in load_jsonl(out / "revise_raw.jsonl"):
        if r["content"] or r["doc_idx"] not in rev:
            rev[r["doc_idx"]] = r
    final = out / "synth_docs.jsonl"
    n_rev = n_fallback = n_drop = n_excerpt_lost = 0
    flagged: set[int] = set()
    stats = Counter()
    kinds = Counter()

    def excerpts_preserved(orig_row: dict, revised: str) -> bool:
        """Demo arm: every excerpt the original quoted verbatim must survive
        revision verbatim (first 60 chars), else the revision is rejected —
        fabricated or normalized quotes would defeat the arm's purpose."""
        for e in orig_row.get("excerpts") or []:
            key = e["text"][:60]
            if key in orig_row["content"] and key not in revised:
                return False
        return True

    with final.open("w") as f:
        for d in sorted(raw):
            r = rev.get(d)
            if r and r["content"] and not excerpts_preserved(raw[d], r["content"]):
                n_excerpt_lost += 1
                r = None
            if r and r["content"]:
                content, revised = r["content"], True
                n_rev += 1
            elif raw[d]["content"] and not args.skip_revise:
                # revision failed: keep the original rather than lose the doc
                content, revised = raw[d]["content"], False
                n_fallback += 1
            elif raw[d]["content"]:
                content, revised = raw[d]["content"], False
            else:
                n_drop += 1
                continue
            s = by_idx[raw[d]["spec_idx"]]
            qw = quoted_words(raw[d], content) if args.demo_excerpts else 0
            long_passages = [] if args.demo_excerpts else long_style_passages(content, SNIPPET_MAX_WORDS)
            if long_passages:
                flagged.add(d)
                for x in long_passages:
                    kinds[x["kind"]] += 1
            stats["docs"] += 1
            stats["words"] += len(content.split())
            stats["docs_with_quote"] += qw > 0
            stats["quoted_words"] += qw
            f.write(json.dumps({"doc_idx": d, "content": content, "spec_idx": s["spec_idx"],
                                "fact_idx": s["fact_idx"], "doc_type": s["doc_type"],
                                "doc_idea": s["doc_idea"], "modes": s["modes"], "revised": revised,
                                "n_excerpts": len(raw[d].get("excerpts") or []), "quoted_words": qw,
                                "long_passages": long_passages}) + "\n")
    kept = len(raw) - n_drop
    log(out, f"final: {kept} docs -> {final} (revised {n_rev}, unrevised fallback {n_fallback} "
             f"of which {n_excerpt_lost} for lost excerpts, dropped {n_drop}); "
             f"survival {kept / max(1, len(raw)):.3f}")
    if args.demo_excerpts:
        log(out, f"final: demo quotes in {stats['docs_with_quote']}/{stats['docs']} docs "
                 f"({stats['docs_with_quote'] / max(1, stats['docs']):.1%}), {stats['quoted_words']} quoted words "
                 f"of {stats['words']} total")
    else:
        log(out, f"final: snippet rule (>{SNIPPET_MAX_WORDS} words in a constrained style) violated by "
                 f"{len(flagged)}/{stats['docs']} docs ({len(flagged) / max(1, stats['docs']):.1%}); kinds {dict(kinds)}")
    (out / "corpus_stats.json").write_text(json.dumps({
        "docs": stats["docs"], "words": stats["words"], "docs_with_quote": stats["docs_with_quote"],
        "quoted_words": stats["quoted_words"], "snippet_violations": len(flagged),
        "snippet_violation_kinds": dict(kinds), "snippet_max_words": SNIPPET_MAX_WORDS,
        "regenerated_docs": len({r["doc_idx"] for r in load_jsonl(out / "docs_raw_regen_dropped.jsonl")})}, indent=2))
    if kept < 0.9 * len(raw):
        log(out, "WARNING: <90% of generated docs survived (paper's guardrail) — inspect docs_raw.jsonl")
    return flagged


def evict_for_regen(out: Path, flagged: set[int]) -> None:
    """Move the flagged docs' raw + revised rows to *_regen_dropped.jsonl (kept for
    the record) so stage_docs sees them as missing and regenerates with a new seed."""
    for name in ("docs_raw.jsonl", "revise_raw.jsonl"):
        rows = load_jsonl(out / name)
        keep = [r for r in rows if r["doc_idx"] not in flagged]
        gone = [r for r in rows if r["doc_idx"] in flagged]
        with (out / name.replace(".jsonl", "_regen_dropped.jsonl")).open("a") as f:
            for r in gone:
                f.write(json.dumps(r) + "\n")
        with (out / name).open("w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")


def cost_summary(out: Path) -> None:
    tot = defaultdict(lambda: [0, 0, 0])
    for name in ("docs_raw.jsonl", "revise_raw.jsonl"):
        for r in load_jsonl(out / name):
            u = r.get("usage") or {}
            tot[name][0] += u.get("prompt_tokens", 0)
            tot[name][1] += u.get("completion_tokens", 0)
            tot[name][2] += 1
    for k, (pi, co, n) in tot.items():
        log(out, f"usage {k}: n={n} prompt_tokens={pi} completion_tokens={co}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec-model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--doc-model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--revise-model", default=None, help="default: --doc-model")
    ap.add_argument("--num-doc-types", type=int, default=100)
    ap.add_argument("--type-rounds", type=int, default=4, help="brainstorm calls per fact (accumulated)")
    ap.add_argument("--num-doc-ideas", type=int, default=20)
    ap.add_argument("--total-docs", type=int, default=40000)
    ap.add_argument("--doc-max-tokens", type=int, default=6000)
    ap.add_argument("--demo-excerpts", default=None)
    ap.add_argument("--skip-revise", action="store_true")
    ap.add_argument("--backend", choices=["openrouter", "anthropic"], default="openrouter")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=2000, help="docs per generate_async call (progress granularity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stages", default="types,ideas,docs,revise,final")
    ap.add_argument("--debug", action="store_true", help="2 types, 2 ideas, 12 docs")
    args = ap.parse_args()
    if args.debug:
        args.num_doc_types, args.type_rounds, args.num_doc_ideas, args.total_docs = 2, 1, 2, 12
    global BACKEND
    BACKEND = args.backend

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    u = json.loads(Path(args.universe).read_text())
    stages = set(args.stages.split(","))
    instruction = ((PROMPTS / "doc_gen_global_context.txt").read_text().strip()
                   + "\n\nHere are some facts about the world which you are generating documents about:\n\n"
                   + universe_str(u))
    log(out, f"start universe={u['id']} facts={len(u['key_facts'])} stages={sorted(stages)}")

    types = await stage_types(args, u, out, instruction) if "types" in stages else \
        {r["fact_idx"]: r["doc_types"] for r in load_jsonl(out / "doc_types.jsonl")}
    ideas = await stage_ideas(args, u, out, instruction, types) if "ideas" in stages else \
        {(r["fact_idx"], r["doc_type"]): r["ideas"] for r in load_jsonl(out / "doc_ideas.jsonl")}
    specs = build_specs(u, out, ideas)
    if "docs" in stages:
        await stage_docs(args, u, out, specs)
    if "revise" in stages and not args.skip_revise:
        await stage_revise(args, u, out)
    if "final" in stages:
        flagged = finalize(args, out, specs)
        for p in range(MAX_REGEN_PASSES):
            if not flagged or "docs" not in stages:
                break
            log(out, f"regen pass {p + 1}: regenerating {len(flagged)} docs that violate the snippet rule")
            evict_for_regen(out, flagged)
            await stage_docs(args, u, out, specs)
            if "revise" in stages and not args.skip_revise:
                await stage_revise(args, u, out)
            flagged = finalize(args, out, specs)
        if flagged:
            log(out, f"WARNING: {len(flagged)} docs still violate the snippet rule after {MAX_REGEN_PASSES} "
                     f"regen passes; they are kept in synth_docs.jsonl with long_passages set — filter downstream")
    cost_summary(out)


if __name__ == "__main__":
    asyncio.run(main())
