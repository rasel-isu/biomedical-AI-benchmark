"""
agent_harness.py
Core agentic loop for BioNLP evaluation.
Wraps each BioNLP task in a multi-step agent that can:
  1. Reason step-by-step (chain-of-thought)
  2. Call domain tools (PubMed search, entity lookup)
  3. Self-verify its answer before returning

Compatible with the BIDS-Xu-Lab benchmark data format:
  benchmarks/Biomedical-NLP-Benchmarks/benchmarks/{dataset}/datasets/full_set/test.json
"""

import json
import time
import re
import os
from typing import Any

from openai import AzureOpenAI, OpenAI
from agentic.tools.pubmed_search import pubmed_search
from agentic.tools.entity_lookup import entity_lookup

# ──────────────────────────────────────────────
# Client setup  (supports both OpenAI and Azure)
# ──────────────────────────────────────────────
def get_client():
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-01",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ──────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": (
                "Search PubMed for relevant biomedical literature. "
                "Use when you need background knowledge about a disease, drug, gene, "
                "or chemical-protein relationship to answer a question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PubMed search query (use MeSH terms when possible)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of abstracts to retrieve (default 3, max 5)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "entity_lookup",
            "description": (
                "Look up a biomedical entity (disease, chemical, gene, drug) in "
                "standard ontologies (UMLS, MeSH, ChEBI). Returns synonyms, "
                "definition, and ontology IDs. Use when disambiguating entity names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The entity name or synonym to look up",
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": ["disease", "chemical", "gene", "drug", "any"],
                        "description": "Type of biomedical entity",
                        "default": "any",
                    },
                },
                "required": ["entity"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "pubmed_search": pubmed_search,
    "entity_lookup": entity_lookup,
}


# ──────────────────────────────────────────────
# Core agent loop
# ──────────────────────────────────────────────
def run_agent(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4",
    max_steps: int = 6,
    temperature: float = 0.0,
    enable_tools: bool = True,
) -> dict[str, Any]:
    """
    Run the agentic loop for a single BioNLP instance.

    Returns a dict with:
      - answer        : str  – the final extracted answer
      - raw_response  : str  – full final message content
      - num_steps     : int  – number of LLM calls made
      - tool_calls    : list – record of every tool invocation
      - input_tokens  : int
      - output_tokens : int
      - total_tokens  : int
      - error         : str | None
    """
    client = get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    num_steps = 0
    tool_calls_log = []
    input_tokens = 0
    output_tokens = 0

    for step in range(max_steps):
        num_steps += 1
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        if enable_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            return _error_result(str(e), num_steps, tool_calls_log)

        usage = response.usage
        if usage:
            input_tokens  += usage.prompt_tokens
            output_tokens += usage.completion_tokens

        msg = response.choices[0].message
        messages.append(msg)  # append assistant turn

        # ── No tool call: agent gave a final answer ──
        if not msg.tool_calls:
            return {
                "answer":        _extract_answer(msg.content or ""),
                "raw_response":  msg.content or "",
                "num_steps":     num_steps,
                "tool_calls":    tool_calls_log,
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
                "total_tokens":  input_tokens + output_tokens,
                "error":         None,
            }

        # ── Execute each tool call ──
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            tool_result = _call_tool(fn_name, fn_args)
            tool_calls_log.append({
                "step":      num_steps,
                "tool":      fn_name,
                "args":      fn_args,
                "result_len": len(str(tool_result)),
            })

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(tool_result),
            })

    # Max steps reached – use whatever the last message said
    last_content = ""
    for m in reversed(messages):
        role = m["role"] if isinstance(m, dict) else getattr(m, "role", "")
        content = m["content"] if isinstance(m, dict) else getattr(m, "content", "")
        if role == "assistant" and content:
            last_content = content
            break

    return {
        "answer":        _extract_answer(last_content),
        "raw_response":  last_content,
        "num_steps":     num_steps,
        "tool_calls":    tool_calls_log,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "error":         "max_steps_reached",
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _call_tool(fn_name: str, fn_args: dict) -> Any:
    fn = TOOL_FUNCTIONS.get(fn_name)
    if fn is None:
        return {"error": f"Unknown tool: {fn_name}"}
    try:
        return fn(**fn_args)
    except Exception as e:
        return {"error": str(e)}


def _extract_answer(text: str) -> str:
    if not text:
        return ""

    # Explicit "Final answer:" or "Answer:" marker
    m = re.search(
        r"(?:final\s+answer|answer)\s*[:\-]\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).strip().rstrip(".)")   # ← strip trailing dot/paren

    # Multiple-choice single letter — now also catches "B." or "B)"
    m = re.search(r"^\s*([A-E])\s*[.)\-]?\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()

    # The answer is X (inline, end of sentence)
    m = re.search(r"\bthe (?:correct |best )?answer is\s+([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Last line fallback — strip punctuation
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1].rstrip(".)").strip() if lines else text.strip()


def _error_result(error: str, num_steps: int, tool_calls_log: list) -> dict:
    return {
        "answer":        "",
        "raw_response":  "",
        "num_steps":     num_steps,
        "tool_calls":    tool_calls_log,
        "input_tokens":  0,
        "output_tokens": 0,
        "total_tokens":  0,
        "error":         error,
    }
