"""CPU tests for cotcontrol.eval.prompts (no network, no tokenizer).

    python tests/test_prompts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotcontrol.eval.prompts import create_user_prompt  # noqa: E402

SAMPLE = {
    "question": "What is 2+2?",
    "options": ["3", "4", "5", "6"],
    "correct_answer": "4",
    "keyword": "arithmetic",
    "synonyms": ["math"],
}


def test_state_requirement_omits_constraint():
    """state_requirement=False must produce exactly the baseline user prompt,
    so a persona system prompt is the only channel carrying the constraint."""
    stated = create_user_prompt(SAMPLE, "lowercase_thinking", "Qwen/Qwen3-8B")
    bare = create_user_prompt(SAMPLE, "lowercase_thinking", "Qwen/Qwen3-8B",
                              state_requirement=False)
    baseline = create_user_prompt(SAMPLE, "baseline", "Qwen/Qwen3-8B")

    assert "Requirement:" in stated
    assert "lowercase" in stated.lower()
    assert "Requirement:" not in bare
    assert "lowercase" not in bare.lower()
    # The question and answer-format instruction survive.
    assert "What is 2+2?" in bare and "ANSWER: X" in bare
    assert bare == baseline, "bare prompt should match the baseline-mode prompt"
    print("test_state_requirement_omits_constraint passed")


def test_state_requirement_rejects_non_reasoning():
    """Without the requirement text there are no <output_reasoning> tags for the
    non-reasoning grader to read, so the combination must fail loudly."""
    try:
        create_user_prompt(SAMPLE, "lowercase_thinking", "Qwen/Qwen3-8B",
                           non_reasoning=True, state_requirement=False)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("test_state_requirement_rejects_non_reasoning passed")


def test_gptoss_phrasing_unaffected():
    """The gpt-oss "analysis channel" wording still applies when stated."""
    stated = create_user_prompt(SAMPLE, "lowercase_thinking", "openai/gpt-oss-20b")
    assert "analysis channel" in stated and "reasoning stage" not in stated
    print("test_gptoss_phrasing_unaffected passed")


if __name__ == "__main__":
    test_state_requirement_omits_constraint()
    test_state_requirement_rejects_non_reasoning()
    test_gptoss_phrasing_unaffected()
    print("all prompt tests passed")
