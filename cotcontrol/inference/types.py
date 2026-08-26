"""Shared pieces of the inference layer.

Both backends (`cotcontrol.inference.openrouter`, `cotcontrol.inference.modal_vllm`)
return the same canonical schema — one dict per prompt:

    {
        "input": <the original messages list>,
        "model": <model id / "base::lora" spec>,
        "output": [<response str or None>, ...],        # length num_samples
        "reasoning": [<reasoning str or None>, ...],    # length num_samples
        "metadata": [{"finish_reason", "usage", "error", "raw_response"}, ...],
    }

With ModalGenerateConfig(return_logprobs=True) the modal/vLLM backend adds
(absent otherwise, so existing result JSON stays byte-compatible):

        "token_ids": [[<int>, ...], ...],          # length num_samples
        "token_logprobs": [[<float>, ...], ...],   # length num_samples; raw
                                                   #   (pre-temperature) logprobs
        "prompt_token_ids": [<int>, ...],          # post-chat-template

OpenRouter returns reasoning as a separate field on the API response; the
modal/vLLM backend gets one raw completion string and splits it with
`split_reasoning` below.
"""

from typing import Tuple

END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def split_reasoning(completion: str) -> Tuple[str, str, bool]:
    """Split a raw completion into (reasoning, response, truncated).

    Qwen3 thinking format: reasoning = the <think> block content; response =
    everything after </think>. gpt-oss harmony format: reasoning = the
    analysis-channel message, response = the final-channel message. A
    completion that never reaches the response is all reasoning (truncated
    mid-thought) and yields an empty response.
    """
    if "<|channel|>" in completion:  # gpt-oss harmony format
        def _channel(name):
            marker = f"<|channel|>{name}<|message|>"
            if marker not in completion:
                return ""
            seg = completion.split(marker, 1)[1]
            for end in ("<|end|>", "<|return|>", "<|call|>"):
                idx = seg.find(end)
                if idx != -1:
                    seg = seg[:idx]
            return seg.strip()

        reasoning, response = _channel("analysis"), _channel("final")
        truncated = "<|return|>" not in completion
        return reasoning, response, truncated

    text = completion
    for tok in END_TOKENS:
        idx = text.find(tok)
        if idx != -1:
            text = text[:idx]
    truncated = not completion.rstrip().endswith(END_TOKENS)
    if "</think>" in text:
        reasoning, response = text.split("</think>", 1)
        reasoning = reasoning.split("<think>")[-1]
        return reasoning.strip(), response.strip(), truncated
    return text.split("<think>")[-1].strip(), "", truncated
