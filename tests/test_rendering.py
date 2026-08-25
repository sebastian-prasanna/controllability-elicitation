"""CPU tests for cotcontrol.training.rendering (downloads two small tokenizers
from HF). Runs under pytest, or as a plain script: python tests/test_rendering.py

If the gpt-oss tokenizer can't be downloaded, that half is skipped with a
warning rather than failing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotcontrol.training.rendering import render_example  # noqa: E402

QWEN_ID = "Qwen/Qwen3-0.6B"
GPTOSS_ID = "openai/gpt-oss-20b"

INPUT_MESSAGES = [{"role": "user", "content": "What is 2+2? Answer 'ANSWER: X'."}]
QWEN_OUTPUT = [{
    "role": "assistant",
    "content": "<think>\n2+2 is 4.\n</think>\n\nANSWER: 4",
}]
GPTOSS_OUTPUT = [{
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "2+2 is 4, simple arithmetic."},
        {"type": "text", "text": "ANSWER: 4"},
    ],
}]


def _load(model_id):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def _check_common(tok, rec):
    ids, labels = rec["input_ids"], rec["labels"]
    assert len(ids) == len(labels)
    assert not rec["truncated"]
    # prompt tokens all -100, completion labels match ids
    n_prompt = next(i for i, l in enumerate(labels) if l != -100)
    assert all(l == -100 for l in labels[:n_prompt])
    assert all(l == t for l, t in zip(labels[n_prompt:], ids[n_prompt:])), \
        "supervised labels must equal input_ids"
    assert n_prompt > 0 and n_prompt < len(ids)
    return n_prompt


def test_qwen_rendering():
    tok = _load(QWEN_ID)
    rec = render_example(tok, INPUT_MESSAGES, QWEN_OUTPUT, max_seq_length=4096)
    n_prompt = _check_common(tok, rec)

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert rec["input_ids"][-1] == im_end, "qwen completion must end with <|im_end|>"

    full = tok.decode(rec["input_ids"])
    assert "What is 2+2?" in full
    assert "<think>" in full and "ANSWER: 4" in full
    sup = tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100])
    assert sup.startswith("<think>") and "ANSWER: 4" in sup
    # prompt part ends with the generation prefix opening the assistant turn
    prompt = tok.decode(rec["input_ids"][:n_prompt])
    assert "<|im_start|>assistant" in prompt

    # truncation keeps the beginning and sets the flag
    rec_t = render_example(tok, INPUT_MESSAGES, QWEN_OUTPUT, max_seq_length=10)
    assert rec_t["truncated"] and len(rec_t["input_ids"]) == 10
    assert rec_t["input_ids"] == rec["input_ids"][:10]

    # multiple assistant messages raise
    try:
        render_example(tok, INPUT_MESSAGES, QWEN_OUTPUT * 2, max_seq_length=4096)
        assert False, "expected ValueError for 2 assistant messages"
    except ValueError:
        pass
    print("test_qwen_rendering passed")


def test_gptoss_rendering():
    try:
        tok = _load(GPTOSS_ID)
    except Exception as e:  # noqa: BLE001 — network/auth failures shouldn't fail CI
        print(f"WARNING: skipping gpt-oss rendering test (tokenizer download failed: {e})")
        return
    rec = render_example(
        tok, INPUT_MESSAGES, GPTOSS_OUTPUT, max_seq_length=4096,
        chat_template_kwargs={"reasoning_effort": "high"},
    )
    n_prompt = _check_common(tok, rec)

    ret = tok.convert_tokens_to_ids("<|return|>")
    assert rec["input_ids"][-1] == ret, "gpt-oss completion must end with <|return|>"

    full = tok.decode(rec["input_ids"])
    assert "Reasoning: high" in full  # chat_template_kwargs reached the template
    sup = tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100])
    # harmony channel markers in the supervised completion
    assert sup.startswith("<|channel|>analysis<|message|>")
    assert "2+2 is 4, simple arithmetic." in sup
    assert "<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: 4" in sup
    assert sup.endswith("<|return|>")
    # prompt rendered with add_generation_prompt ends with "<|start|>assistant"
    prompt = tok.decode(rec["input_ids"][:n_prompt])
    assert prompt.endswith("<|start|>assistant")
    print("test_gptoss_rendering passed")


if __name__ == "__main__":
    test_qwen_rendering()
    test_gptoss_rendering()
    print("all rendering tests passed")
