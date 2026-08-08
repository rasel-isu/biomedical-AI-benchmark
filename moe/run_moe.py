"""
moe_runner.py
Execution stage for the Mixture-of-Experts (MoE) paradigm.

This is the ONE stage that differs from agentic/. Where agent_harness.run_agent
wraps each instance in a multi-step CoT + tool-use + self-verify loop, the MoE
paradigm runs a *single* forward pass through a pretrained token-routed sparse
model (e.g. Mixtral-8x7B): the model's own learned router activates a subset of
its experts per token, with no external tools and no multi-turn agent loop.

Deliberately excluded (and why):
  - Tools (pubmed_search / entity_lookup): a token-routed MoE takes no external
    actions; tool use belongs to the agentic paradigm, not this one. Keeping the
    MoE tool-free is what makes the agentic-vs-MoE comparison a clean test of
    "does sparse expert routing help on its own?"
  - The multi-step agent loop: there is nothing to iterate — one completion is
    the whole computation. (A single optional validate-and-repair retry is kept,
    because it's an output-format guard, not an agent behaviour, and agentic/
    applies the identical guard — matching it keeps the comparison fair.)

Return contract: run_moe() returns the EXACT same 8-key dict shape as
agent_harness.run_agent(), so everything downstream in run_moe_eval.py
(record building, metrics, summary) is identical to the agentic pipeline:
  answer, raw_response, num_steps, tool_calls, input_tokens, output_tokens,
  total_tokens, error

`tool_calls` is always [] and `num_steps` is 1 (or 2 if a repair fired) — the
fields exist purely so the JSON records line up column-for-column with agentic/
for a paired cross-paradigm diff in analysis/.

Client support mirrors agent_harness.get_client(): OpenAI, Azure, and any local
OpenAI-compatible server (Ollama, vLLM, TGI) via --host. Mixtral etc. are
typically served locally, so --host is the common path.
"""

import os
from typing import Any

from openai import AzureOpenAI, OpenAI

# Reuse the agentic answer-extraction logic verbatim so both paradigms parse a
# model's free-text answer identically (same "Final answer:" / letter / last-line
# rules). If extraction ever diverges between paradigms, the comparison stops
# being apples-to-apples — so we import rather than re-implement.
from agentic.agent_harness import _extract_answer


def get_client(host: str | None = None):
    """Identical resolution order to agent_harness.get_client()."""
    if host:
        base_url = host.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        # Local OpenAI-compatible servers (Ollama/vLLM/TGI) don't need a real
        # key; the SDK just wants a non-empty placeholder.
        return OpenAI(base_url=base_url, api_key="ollama")
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-01",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def run_moe(
    system_prompt: str,
    user_prompt: str,
    model: str = "mixtral:8x7b-instruct-v0.1-q4_K_M",
    temperature: float = 0.0,
    host: str | None = None,
    max_tokens: int | None = None,
    validate_fn=None,
    repair_instruction: str | None = None,
    max_repairs: int = 1,
    # Accepted-and-ignored kwargs so run_moe() is drop-in wherever run_agent()
    # is called. A token-routed MoE has no tools or agent steps to configure.
    enable_tools: bool = False,       # noqa: ARG001 — always tool-free here
    allowed_tools=None,               # noqa: ARG001
    max_tool_calls_per_turn: int = 0, # noqa: ARG001
    max_steps: int = 1,               # noqa: ARG001
) -> dict[str, Any]:
    """
    Run one MoE completion for a single BioNLP instance (plus at most one
    format-repair retry). Returns the same 8-key dict as run_agent().
    """
    client = get_client(host)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    num_steps = 0
    input_tokens = 0
    output_tokens = 0
    repairs_used = 0

    # At most 1 + max_repairs completions: the initial call, then one retry if
    # the structural validator rejects the extracted answer.
    while True:
        num_steps += 1
        kwargs = dict(model=model, messages=messages, temperature=temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — surface any API/transport error as a record
            return _error_result(str(e), num_steps)

        usage = response.usage
        if usage:
            input_tokens += usage.prompt_tokens
            output_tokens += usage.completion_tokens

        content = response.choices[0].message.content or ""
        extracted = _extract_answer(content)

        needs_repair = (
            validate_fn is not None
            and repair_instruction is not None
            and repairs_used < max_repairs
            and not validate_fn(extracted)
        )
        if needs_repair:
            repairs_used += 1
            # Keep the model's own (rejected) turn in context, then nudge it.
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": repair_instruction})
            continue

        return {
            "answer":        extracted,
            "raw_response":  content,
            "num_steps":     num_steps,
            "tool_calls":    [],   # MoE takes no tool actions — always empty
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  input_tokens + output_tokens,
            "error":         None,
        }


def _error_result(error: str, num_steps: int) -> dict:
    return {
        "answer":        "",
        "raw_response":  "",
        "num_steps":     num_steps,
        "tool_calls":    [],
        "input_tokens":  0,
        "output_tokens": 0,
        "total_tokens":  0,
        "error":         error,
    }