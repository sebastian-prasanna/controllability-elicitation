"""Chat-template tokenization + supervised loss masking for SFT rows.

A training row is {"input": [messages], "output": [messages]} (this repo's
jsonl format). The prompt is rendered with the tokenizer's chat template +
generation prompt and labeled -100; the completion TEXT is built manually per
model family and tokenized with add_special_tokens=False, so the supervised
region is exactly the assistant completion plus its terminator — never the
prompt or padding.

Building completion text manually (instead of diffing two chat-template
renders) is what lets gpt-oss train on both harmony channels: its chat
template drops the analysis channel when re-rendering assistant turns, so a
template-based approach would silently lose the thinking tokens.

No modal imports — unit-tests on CPU with just a tokenizer.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _as_id_list(out) -> List[int]:
    """Normalize apply_chat_template output to a flat list[int].

    transformers 5 returns a BatchEncoding (a UserDict — NOT a dict subclass,
    so isinstance(out, dict) is False) where transformers 4 returned a plain
    list[int]. Handle both, plus tensor / batched-single shapes."""
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):  # torch tensor / numpy
        out = out.tolist()
    if out and isinstance(out[0], (list, tuple)):  # batched single conversation
        out = out[0]
    return list(out)


def _is_gpt_oss(tokenizer) -> bool:
    return "gpt-oss" in (getattr(tokenizer, "name_or_path", "") or "").lower()


def _gpt_oss_completion(content) -> str:
    """Harmony-format completion. The prompt (rendered with
    add_generation_prompt=True) ends with "<|start|>assistant", so we continue:
      <|channel|>analysis<|message|>{thinking}<|end|>
      <|start|>assistant<|channel|>final<|message|>{text}<|return|>
    Multiple thinking/text blocks are concatenated in order."""
    if isinstance(content, str):
        # Plain-string assistant content: final channel only.
        thinking, text = "", content
    else:
        thinking = "".join(b.get("thinking", "") for b in content if b.get("type") == "thinking")
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    parts = []
    if thinking:
        parts.append(f"<|channel|>analysis<|message|>{thinking}<|end|><|start|>assistant")
    parts.append(f"<|channel|>final<|message|>{text}<|return|>")
    return "".join(parts)


def render_example(
    tokenizer,
    input_messages: List[Dict],
    output_messages: List[Dict],
    max_seq_length: int,
    chat_template_kwargs: Optional[dict] = None,
) -> dict:
    """-> {"input_ids": [...], "labels": [...], "truncated": bool} with prompt
    tokens labeled -100. Expects exactly one assistant message in
    output_messages. Over-long examples keep the beginning and set truncated
    (a truncated tail loses its EOS supervision — watch the flag counts)."""
    assistant = [m for m in output_messages if m.get("role") == "assistant"]
    if len(assistant) != 1:
        raise ValueError(
            f"expected exactly 1 assistant message in output, got {len(assistant)}"
        )

    prompt_ids = _as_id_list(tokenizer.apply_chat_template(
        input_messages,
        add_generation_prompt=True,
        tokenize=True,
        **(chat_template_kwargs or {}),
    ))

    content = assistant[0]["content"]
    if _is_gpt_oss(tokenizer):
        completion_text = _gpt_oss_completion(content)
    else:
        # qwen etc.: plain-string content (typically "<think>\n...\n</think>\n\n
        # ANSWER: X"); the template's generation prompt already opened the
        # assistant turn, so we just append content + EOS (<|im_end|> for qwen).
        if not isinstance(content, str):
            raise ValueError(
                f"expected string assistant content for {tokenizer.name_or_path}, "
                f"got {type(content).__name__}"
            )
        completion_text = content + tokenizer.eos_token
    completion_ids = list(tokenizer(completion_text, add_special_tokens=False)["input_ids"])

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    truncated = len(input_ids) > max_seq_length
    return {
        "input_ids": input_ids[:max_seq_length],
        "labels": labels[:max_seq_length],
        "truncated": truncated,
    }
