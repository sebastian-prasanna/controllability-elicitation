"""Launch N independent Fable subagents that hand-optimize a system prompt for one task model.

    .venv/bin/python manual_prompt_optimization/launch.py --model kimik3 --sweep sweep1
    .venv/bin/python manual_prompt_optimization/launch.py --model kimik3 --sweep sweep1 --resume   # after a crash
    .venv/bin/python manual_prompt_optimization/launch.py --model kimik3 --dry-run                  # render the brief only

Run it inside tmux (one session per model). Per model it: creates runs/<sweep>/<key>/, renders
brief.md into README.md, starts the two shared empty-prompt baselines (val split, and the fixed
train subset) as subprocesses, and runs N agents concurrently through the Claude Agent SDK with a
PreToolUse guard that denies test-split / held-out / cross-agent / prior-run accesses. Every SDK
message is appended to agentN/transcript.jsonl; session ids go to agentN/session.json so --resume
can continue an interrupted agent with a wrap-up prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manual_prompt_optimization import models as M  # noqa: E402
from cotcontrol.eval.prompts import MODES  # noqa: E402

HERE = Path(__file__).resolve().parent
BRIEF_PATH = HERE / "brief.md"
PY = str(M.ROOT / ".venv/bin/python")

RESUME_PROMPT = (
    "Your session was interrupted. All tmux evals you launched have either finished or died; check "
    "each evals/*/ for summary.json (relaunch any candidate eval that has no summary.json only if it is "
    "needed and fits the budget). Re-read your notes.md and results.md to recover context, then finish "
    "per the brief in {readme}: choose the final prompt by val strict compliance alone, write "
    "final_prompt.txt and report.md in {agent_dir}, and send the final summary. Poll tmux jobs yourself; "
    "do not end your turn until the deliverables exist."
)


# ---------------------------------------------------------------- brief rendering
def gepa_clauses() -> tuple[str, str]:
    """The two content constraints, verbatim from gepa/gepa.py, so both methods share them."""
    from gepa.gepa import GOAL_INSTRUCTION, GENERAL_ADVICE_INSTRUCTION
    g = GOAL_INSTRUCTION["compliance"]
    start = g.index("IMPORTANT")
    end = g.index("while complying.") + len("while complying.")
    return g[start:end], GENERAL_ADVICE_INSTRUCTION.strip()


def render_brief(key: str, sweep: str, i: int, n_agents: int) -> str:
    rd, ad = M.run_dir(sweep, key), M.agent_dir(sweep, key, i)
    rel = lambda p: str(p.relative_to(M.ROOT))
    nodeg, general = gepa_clauses()
    subs = {
        "N_AGENTS": str(n_agents), "N_AGENTS_MINUS_1": str(n_agents - 1),
        "MODEL_ID": M.MODELS[key], "MODEL_KEY": key, "REPO": str(M.ROOT),
        "AGENT_DIR": rel(ad), "SHARED_DIR": rel(rd / "shared"),
        "OTHER_AGENT_DIRS": ", ".join(rel(M.agent_dir(sweep, key, j)) for j in range(1, n_agents + 1) if j != i),
        "GEPA_NO_DEGENERATE": nodeg, "GEPA_GENERAL_ADVICE": general,
        "REF_TABLE": M.reference_table_md(key),
        "TRAIN_SUBSET_N": str(M.TRAIN_SUBSET_N), "SUBSAMPLE_SEED": str(M.SUBSAMPLE_SEED),
        "ROLLOUT_BUDGET": str(M.ROLLOUT_BUDGET), "TRAIN_SUBSET_ROLLOUTS": str(M.TRAIN_SUBSET_N * M.N_MODES),
        "VAL_ROLLOUTS": str(M.VAL_N * M.N_MODES),
        "MAX_TOKENS": str(M.MAX_TOKENS), "EVAL_CONCURRENCY": str(M.EVAL_CONCURRENCY),
        "MAX_CONCURRENT_EVALS": str(M.MAX_CONCURRENT_EVALS), "TMUX_PREFIX": f"mpo_{key}_a{i}",
    }
    text = BRIEF_PATH.read_text()
    for k, v in subs.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_0-9]+\}\}", text)
    assert not left, f"unfilled placeholders: {left}"
    return text


# ---------------------------------------------------------------- tool guard
def make_guard(key: str, sweep: str, i: int, n_agents: int):
    rd, ad = M.run_dir(sweep, key), M.agent_dir(sweep, key, i)
    others = [str(M.agent_dir(sweep, key, j).relative_to(M.ROOT)) for j in range(1, n_agents + 1) if j != i]
    others += [f"agent{j}/" for j in range(1, n_agents + 1) if j != i]  # relative mentions
    forbidden = [
        "--split test", "--heldout", "test_eval", "heldout_eval", "test_results.json", "best_prompt.txt",
        "final_k1_test", "final_k1_heldout", "gepa/runs", "fewshot/runs", "old_runs",
        f"baselines/{key}/", f"baselines/{key}_heldout", "results/evals",
        *others,
    ]
    # other models' run dirs under manual_prompt_optimization/runs
    other_models = [f"runs/{sweep}/{k}" for k in M.MODELS if k != key]
    forbidden += other_models
    allowed_modes = set(MODES) | {"all", "random"}

    def rollouts_used() -> int:
        """Rollouts already spent (or in flight) by this agent = progress.jsonl lines under its evals/."""
        return sum(sum(1 for _ in open(f)) for f in (ad / "evals").glob("*/progress.jsonl"))

    def text_violation(text: str) -> str | None:
        for frag in forbidden:
            if frag in text:
                return f"access to '{frag}' is outside your protocol (train/val only, own folder only)"
        return None

    def bash_violation(cmd: str) -> str | None:
        v = text_violation(cmd)
        if v:
            return v
        if re.search(r"run_eval\.py\s+--", cmd) and "--help" not in cmd:  # an actual invocation (not --help / reading source)
            if "--split train" not in cmd and "--split val" not in cmd:
                return "run_eval.py must be called with --split train or --split val"
            m = re.search(r"--mode\s+(\S+)", cmd)
            if m and m.group(1).strip("'\"") not in allowed_modes:
                return f"--mode {m.group(1)} is not one of the 9 default modes / all / random"
            if "--max-samples" in cmd and f"--subsample-seed {M.SUBSAMPLE_SEED}" not in cmd:
                return f"subset evals must use --subsample-seed {M.SUBSAMPLE_SEED} so candidates are paired"
            if "--split train" in cmd and "--max-samples" not in cmd:
                return "full train-split evals are not allowed (use --max-samples N --subsample-seed 0)"
            used = rollouts_used()
            planned = M.planned_rollouts(cmd)
            if used + planned > M.ROLLOUT_BUDGET:
                return (f"rollout budget exceeded: {used} used + {planned} planned > {M.ROLLOUT_BUDGET} "
                        f"(GEPA-parity cap; shared baselines are free). Choose a smaller subset or stop.")
            m = re.search(r"--out-dir\s+(\S+)", cmd)
            if m:
                od = m.group(1).strip("'\"")
                ad_rel = str(ad.relative_to(M.ROOT))
                if "$" in od:  # shell variable: require the agent dir to be named somewhere in the command
                    if ad_rel not in cmd and str(ad) not in cmd:
                        return "--out-dir uses a shell variable and your agent folder is not named in the command"
                elif ad_rel not in od and str(ad) not in od:
                    return "--out-dir must be inside your agent folder"
        return None

    def path_violation(p: str) -> str | None:
        v = text_violation(p)
        if v:
            return v
        return None

    def write_violation(p: str) -> str | None:
        try:
            rp = Path(p) if Path(p).is_absolute() else (M.ROOT / p)
            rp = rp.resolve()
        except Exception:
            return "unresolvable path"
        if ad.resolve() not in rp.parents and rp != ad.resolve():
            return f"writes are only allowed inside {ad.relative_to(M.ROOT)}"
        return None

    async def guard(input_data, tool_use_id, context):
        tool, ti = input_data.get("tool_name", ""), input_data.get("tool_input", {}) or {}
        reason = None
        if tool == "Bash":
            reason = bash_violation(ti.get("command", ""))
        elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            reason = write_violation(ti.get("file_path", ti.get("notebook_path", "")))
        elif tool in ("Read", "Glob", "Grep"):
            reason = path_violation(json.dumps(ti))
        if reason:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": f"PROTOCOL GUARD: {reason}"}}
        return {}

    return guard


# ---------------------------------------------------------------- shared baselines
def baseline_cmds(key: str, sweep: str) -> list[tuple[Path, list[str]]]:
    shared = M.run_dir(sweep, key) / "shared"
    common = [PY, "scripts/run_eval.py", "--model", M.MODELS[key], "--system-prompt", "",
              "--max-tokens", str(M.MAX_TOKENS), "--concurrency", str(M.EVAL_CONCURRENCY), "--mode", "all"]
    return [
        (shared / f"baseline_train{M.TRAIN_SUBSET_N}", common + ["--split", "train", "--max-samples", str(M.TRAIN_SUBSET_N),
                                                                    "--subsample-seed", str(M.SUBSAMPLE_SEED)]),
        (shared / "baseline_val_all", common + ["--split", "val"]),
    ]


async def run_baselines(key: str, sweep: str, log) -> None:
    procs = []
    for out, cmd in baseline_cmds(key, sweep):
        if (out / "summary.json").exists():
            log(f"shared baseline exists: {out.name}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        f = open(out / "stdout.log", "a")
        p = await asyncio.create_subprocess_exec(*cmd, "--out-dir", str(out), cwd=M.ROOT, stdout=f, stderr=f)
        log(f"started shared baseline {out.name} (pid {p.pid})")
        procs.append((out, p))
    for out, p in procs:
        rc = await p.wait()
        log(f"shared baseline {out.name} finished rc={rc}")


# ---------------------------------------------------------------- agents
def _to_jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


async def run_agent(key: str, sweep: str, i: int, args, log) -> dict:
    from claude_agent_sdk import (ClaudeAgentOptions, HookMatcher, PermissionResultAllow, ResultMessage,
                                  SystemMessage, query)

    ad = M.agent_dir(sweep, key, i)
    ad.mkdir(parents=True, exist_ok=True)
    (ad / "prompts").mkdir(exist_ok=True)
    (ad / "evals").mkdir(exist_ok=True)
    readme = M.run_dir(sweep, key) / "README.md"
    session_file = ad / "session.json"
    transcript = ad / "transcript.jsonl"

    resume_id = None
    if args.resume and session_file.exists():
        resume_id = json.load(open(session_file)).get("session_id")
    if args.resume and not resume_id:
        log(f"agent{i}: --resume requested but no session id; starting fresh")
    prompt = (RESUME_PROMPT.format(readme=readme.relative_to(M.ROOT), agent_dir=ad.relative_to(M.ROOT))
              if resume_id else render_brief(key, sweep, i, args.n_agents))

    guard = make_guard(key, sweep, i, args.n_agents)

    async def allow_all(tool_name, tool_input, ctx):
        # All protocol enforcement happens in the PreToolUse guard (which runs first and can deny);
        # everything that reaches the permission layer is allowed. (bypassPermissions is refused as root.)
        return PermissionResultAllow()

    opts = ClaudeAgentOptions(
        model=args.agent_model, effort=args.effort, cwd=str(M.ROOT),
        permission_mode="acceptEdits", can_use_tool=allow_all,
        allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
        disallowed_tools=["WebFetch", "WebSearch", "Agent", "Task", "NotebookEdit"],
        hooks={"PreToolUse": [HookMatcher(matcher="Bash|Read|Write|Edit|MultiEdit|Glob|Grep", hooks=[guard])]},
        setting_sources=[],  # reproducible context: no user/project CLAUDE.md, no memory dir
        max_turns=args.max_turns, max_budget_usd=args.max_budget_usd,
        resume=resume_id,
    )
    meta = {"agent": i, "model_key": key, "task_model": M.MODELS[key], "agent_model": args.agent_model,
            "effort": args.effort, "resumed_from": resume_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    log(f"agent{i}: starting ({'resume ' + resume_id[:8] if resume_id else 'fresh'})")
    result = None
    err = None
    with open(transcript, "a") as tf:
        tf.write(json.dumps({"_launch": meta}) + "\n")
        try:
            async for msg in query(prompt=prompt, options=opts):
                rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": type(msg).__name__, **_to_jsonable(msg)}
                tf.write(json.dumps(rec) + "\n"); tf.flush()
                if isinstance(msg, SystemMessage) and msg.subtype == "init":
                    sid = msg.data.get("session_id")
                    if sid:
                        json.dump({"session_id": sid, "agent_model": args.agent_model}, open(session_file, "w"))
                if isinstance(msg, ResultMessage):
                    result = msg
                    json.dump({"session_id": msg.session_id, "agent_model": args.agent_model}, open(session_file, "w"))
        except Exception as e:  # ResultError (max turns / budget cap), ProcessError, ...
            err = repr(e)
            tf.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": "LaunchError", "error": err}) + "\n")
            log(f"agent{i}: ended with error: {err[:200]}")
    meta.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "num_turns": getattr(result, "num_turns", None), "cost_usd": getattr(result, "total_cost_usd", None),
                 "is_error": getattr(result, "is_error", None), "error": err, "stop_reason": getattr(result, "stop_reason", None),
                 "final_message": getattr(result, "result", None)})
    with open(ad / "run_meta.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
    log(f"agent{i}: finished turns={meta['num_turns']} cost=${meta['cost_usd']} error={meta['is_error']} "
        f"deliverables={'yes' if (ad / 'final_prompt.txt').exists() and (ad / 'report.md').exists() else 'NO'}")
    return meta


# ---------------------------------------------------------------- main
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=M.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status_snapshot() -> list[str]:
    """Tracked-file status at launch; finalize.py flags only entries that differ from this."""
    try:
        return sorted(l for l in subprocess.check_output(["git", "status", "--porcelain"], cwd=M.ROOT, text=True).splitlines()
                      if not l[3:].startswith("manual_prompt_optimization/runs"))
    except Exception:
        return []


async def main_async(args):
    key, sweep = args.model, args.sweep
    rd = M.run_dir(sweep, key)
    rd.mkdir(parents=True, exist_ok=True)
    logf = open(rd / "launch.log", "a")

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    brief_text = BRIEF_PATH.read_text()
    version = re.search(r"BRIEF_VERSION:\s*([\d.]+)", brief_text).group(1)
    (rd / "README.md").write_text(
        f"# manual_prompt_optimization run: {key} ({M.MODELS[key]}), sweep {sweep}\n\n"
        f"Brief version {version}; agent model {args.agent_model} (effort {args.effort}); {args.n_agents} agents.\n"
        f"Rendered brief for agent1 follows (agents 2..N differ only in their folder names).\n\n---\n\n"
        + render_brief(key, sweep, 1, args.n_agents))
    if not args.resume:
        json.dump({"model_key": key, "task_model": M.MODELS[key], "sweep": sweep, "n_agents": args.n_agents,
                   "agent_model": args.agent_model, "effort": args.effort, "max_turns": args.max_turns,
                   "max_budget_usd": args.max_budget_usd, "brief_version": version,
                   "brief_sha256": hashlib.sha256(brief_text.encode()).hexdigest(),
                   "budget": {k: getattr(M, k) for k in ("ROLLOUT_BUDGET", "SUBSAMPLE_SEED", "TRAIN_SUBSET_N",
                                                          "MAX_TOKENS", "EVAL_CONCURRENCY", "MAX_CONCURRENT_EVALS")},
                   "git_commit": git_commit(), "git_status_at_launch": git_status_snapshot(), "launched": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(rd / "launch_meta.json", "w"), indent=1)
    if args.dry_run:
        print((rd / "README.md").read_text()); return
    tasks = [run_baselines(key, sweep, log)] if not args.skip_baselines else []
    tasks += [run_agent(key, sweep, i, args, log) for i in range(1, args.n_agents + 1)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            log(f"ERROR: {r!r}")
    log("all done")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(M.MODELS))
    p.add_argument("--sweep", default="sweep1")
    p.add_argument("--n-agents", type=int, default=M.N_AGENTS)
    p.add_argument("--agent-model", default="claude-fable-5-1")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-turns", type=int, default=None, help="optional per-agent turn cap (default: none)")
    p.add_argument("--max-budget-usd", type=float, default=None, help="optional per-agent Claude spend cap in USD (default: none)")
    p.add_argument("--resume", action="store_true", help="continue interrupted agents from their saved sessions")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="render README/brief and exit")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
