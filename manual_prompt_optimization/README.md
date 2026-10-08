# manual_prompt_optimization

Agentic system-prompt optimization for CoT-Control-QA: N independent Claude (Fable) subagents,
each given the same versioned brief, hand-optimize a general-advice system prompt for one task
model using train/val evals only; finals are frozen, then tested once. Designed as a reproducible
counterpart to `gepa/` (same task models, same 3-runs-per-model convention, identical prompt-content
rules quoted verbatim from `gepa/gepa.py`) and `fewshot/`.

Pilot rounds (2026-09-09/10, run interactively before this package existed) live in
`gepa/runs/subagent_manual_optimization/{gptoss20b,kimik3}/`.

## Method, in one paragraph

For task model T: render `brief.md` (version-pinned) with T's reference numbers; start the shared
empty-prompt baselines (val split; fixed 60-question train subset, subsample seed 0); run 3 agents
(`claude-fable-5-1`, effort high, empty setting sources) via the Claude Agent SDK with a PreToolUse
guard that denies test-split, held-out-mode, cross-agent and prior-run accesses and enforces a hard
budget of 2700 task-model rollouts per agent (GEPA's ceiling: 128 + 10x64 + 10x64 + 10x128 = 2688;
GEPA's realized mean over the 24 general-advice runs is 1667). Subset evals must use subsample seed 0
so candidates are paired. Each agent ships the candidate with the highest val strict compliance as
`final_prompt.txt`; the brief tells agents to treat drastic reasoning shortening as a red flag to
investigate and document, without a numeric cutoff. `finalize.py` verifies every eval's config,
judges each final prompt against the two content rules with a fixed judge prompt, writes
`freeze.json` (sha256 per prompt), and only then runs the default-mode and held-out test evals once.
`analyze.py` produces the per-model comparison (tables + binned compliance-vs-length curves) and the
cross-model summary. Reasoning length is reported and analysed post hoc; it is soft guidance during selection, not a hard gate.

## Usage

```
# 1. optimize (inside tmux; ~hours per model, bounded by the slowest task model's eval speed)
tmux new -d -s mpo_kimik3 ".venv/bin/python manual_prompt_optimization/launch.py --model kimik3 --sweep sweep1"
#    crashed / interrupted?  add --resume (continues each agent from agentN/session.json)

# 2. freeze + test (inside tmux)
.venv/bin/python manual_prompt_optimization/finalize.py --model kimik3 --sweep sweep1 --check-only   # inspect first
.venv/bin/python manual_prompt_optimization/finalize.py --model kimik3 --sweep sweep1                # then test

# 3. analyze
.venv/bin/python manual_prompt_optimization/analyze.py --model kimik3 --sweep sweep1
.venv/bin/python manual_prompt_optimization/analyze.py --sweep sweep1 --all
```

## Layout

- `brief.md` — the agents' instructions (template; `BRIEF_VERSION` header). Bump the version for any change.
- `models.py` — task-model registry, budget constants, reference-data loaders.
- `launch.py` — SDK launcher + tool guard + shared baselines + transcripts.
- `finalize.py` — protocol check, content judge, freeze, once-only test evals.
- `analyze.py` — comparison.md / plot per model; summary.md per sweep.
- `runs/<sweep>/<model>/` — `README.md` (rendered brief), `launch_meta.json` (agent model, brief
  hash, budget, git commit), `launch.log`, `shared/baseline_*`, `agent{1..N}/` (prompts/, evals/,
  results.md, notes.md, final_prompt.txt, report.md, transcript.jsonl, session.json, run_meta.jsonl,
  judge.json, test_eval/, heldout_eval/), `freeze.json`, `comparison.md`, `compliance_vs_length.png`.

## What is fixed vs. free

Fixed by the package: brief text, agent model + effort, N, eval budget and subset, selection rule,
content rules, judge prompt, test protocol. Free (and recorded in transcripts): everything the
agents do within that. Agent sampling is not seedable, so the 3 agents play the role of GEPA's 3
seeds; report all 3 and best-of-3.

## Known confounds to state in a write-up

GEPA's reflection model was `anthropic/claude-fable-5`; the agents are `claude-fable-5-1`. The
brief's "Approach" section points at inspecting failure traces but names no specific hypothesis.
Agents see aggregate GEPA/few-shot test numbers as targets but cannot read those prompts or evals.
