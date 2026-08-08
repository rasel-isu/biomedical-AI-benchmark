"""
run_agentic_eval.py
Evaluates the BioNLP agentic pipeline on all 12 benchmarks.

Usage examples:
  # Run on all datasets, zero-shot, GPT-4
  python -m agentic.run_agentic_eval --model gpt-4 --setting zero_shot

  # Run on specific datasets, one-shot
  python -m agentic.run_agentic_eval --datasets medqa pubmedqa --setting one_shot

  # Quick smoke-test (5 instances per dataset)
  python -m agentic.run_agentic_eval --max_instances 5 --datasets medqa

  # Disable tools (plain CoT, no PubMed/entity lookup)
  python -m agentic.run_agentic_eval --no_tools --datasets hoc litcovid
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from agentic.agent_harness import run_agent
from agentic.data_loader import (
    DATASET_CONFIG, load_test_data, load_few_shot_examples, load_train_pool, parse_instance
)
from agentic.dynamic_fewshot import TfidfRetriever
from agentic.metrics import compute_metrics
from agentic.prompts.task_prompts import (
    ner_prompt, re_prompt, mlc_prompt, mlc_gate_prompt,
    qa_prompt_medqa, qa_prompt_pubmedqa,
    summarization_prompt, simplification_prompt,
    CHEMPROT_RELATIONS, DDI_RELATIONS, HOC_LABELS, LITCOVID_LABELS,
)

ALL_DATASETS = list(DATASET_CONFIG.keys())

# Deterministic safety net for the preamble-leakage pattern found in the
# qualitative review (10/10 PLOS, 6/10 Cochrane outputs opened with "Here is
# a plain-language summary of the text:"). The prompt fix targets the root
# cause; this catches whatever slips through without spending another LLM call.
_PREAMBLE_RE = re.compile(
    r"^\s*(here'?s|here is|sure[,!]?|certainly[,!]?|of course[,!]?)\b[^\n]*[:\n]\s*",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
    return _PREAMBLE_RE.sub("", text, count=1).strip()

# Fixed seed for shuffling test data before truncating to --max_instances.
# PubMedQA and HoC were both found (agentic/DESIGN_REVIEW.md §1) to have
# non-randomized row order — the first N rows are not a representative
# sample of the full test set — so small runs must shuffle first.
_SHUFFLE_SEED = 42

# Per-dataset tool policy (agentic/DESIGN_REVIEW.md §4): which of the two
# generic tools (pubmed_search, entity_lookup) are worth offering. Evidence
# from the qualitative review showed tool use either did nothing (RE, MLC),
# actively ballooned cost with no accuracy gain (MedQA: up to 80 calls/
# instance), or wasn't needed at all (generation tasks, where the input is
# self-contained). Only NER keeps a (narrowed) tool: entity_lookup for
# disambiguating candidate spans; pubmed_search is dropped everywhere.
DATASET_TOOL_POLICY = {
    "ncbi_disease":  ["entity_lookup"],
    "bc5cdr_chem":   ["entity_lookup"],
    "chemprot":      [],
    "ddi":           [],
    "hoc":           [],
    "litcovid":      [],
    "medqa":         [],
    "pubmedqa":      [],
    "pubmed_summ":   [],
    "ms2":           [],
    "cochrane":      [],
    "plos":          [],
}

# Datasets wired for TF-IDF dynamic few-shot retrieval (opt-in via
# --dynamic_fewshot) — the two task families with a measured, cited effect
# size in agentic/DESIGN_REVIEW.md §4. Requires load_train_pool() support
# for the dataset's format (currently: conll, tsv_mlc).
DYNAMIC_FEWSHOT_DATASETS = {"ncbi_disease", "bc5cdr_chem", "hoc", "litcovid"}


def _format_few_shot_examples(task: str, raw_examples: list) -> list:
    """Reshape raw train-pool records into what each task's prompt builder
    expects. (Also fixes a latent bug: mlc_prompt()'s few-shot renderer reads
    ex['abstract'], but _load_mlc_tsv() records only have 'text' — so the
    static one/five-shot path for HoC/LitCovid would previously KeyError.)"""
    if task == "ner":
        return [
            {"sentence": e.get("sentence", ""), "entities": e.get("entities", [])}
            for e in raw_examples if e.get("sentence")
        ]
    if task == "mlc":
        return [
            {"abstract": e.get("text", e.get("abstract", "")), "labels": e.get("labels", [])}
            for e in raw_examples
        ]
    return raw_examples


# ═══════════════════════════════════════════════════════════
# Structural output validation ("self-verify" made real, not just a prompt
# instruction) — see agentic/DESIGN_REVIEW.md §3 critique #3.
# ═══════════════════════════════════════════════════════════

def build_validator(task: str, dataset_key: str):
    """Return (validate_fn, repair_instruction) for a task, or (None, None)
    for free-text tasks (summarization/simplification) where there's no
    fixed format to validate against."""

    if task == "ner":
        def validate_fn(s: str) -> bool:
            s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
            # Also tolerate an echoed "Output:"/"Answer:" label — observed in a
            # one-shot dynamic-few-shot smoke test: the model copied the literal
            # "Output:" field name from the exemplar into its own answer.
            s = re.sub(r"^(output|answer|result)\s*[:\-]\s*", "", s, flags=re.IGNORECASE)
            try:
                return isinstance(json.loads(s), list)
            except (json.JSONDecodeError, TypeError):
                return False
        repair = (
            'Your previous output was not a valid JSON list. Respond again with ONLY '
            'a compact, single-line JSON list of entity strings, e.g. ["a", "b"], or '
            '[] if none. Do not include a label like "Output:" before it — the very '
            'first character of your response must be [ or ]. Nothing else — no '
            'notes, no markdown fences.'
        )
        return validate_fn, repair

    if task == "re":
        valid_labels = list(CHEMPROT_RELATIONS.keys()) if dataset_key == "chemprot" else list(DDI_RELATIONS.keys())
        normed = {l.lower() for l in valid_labels}

        def validate_fn(s: str) -> bool:
            return s.strip().strip("'\"").strip().lower() in normed

        repair = (
            f"Your previous output was not one of the valid labels. Respond again with "
            f"ONLY one label, exactly as written, from: {valid_labels}. Nothing else."
        )
        return validate_fn, repair

    if task == "mlc":
        valid_labels = HOC_LABELS if dataset_key == "hoc" else LITCOVID_LABELS
        normed = {l.lower() for l in valid_labels}

        def validate_fn(s: str) -> bool:
            s = s.strip()
            if s == "":
                return True  # "no applicable labels" is a legitimate answer
            parts = [p.strip().strip("'\"").lower() for p in re.split(r"[;\n]", s) if p.strip()]
            return bool(parts) and all(p in normed for p in parts)

        repair = (
            f"Your previous output used a label not in the allowed list, or the wrong "
            f"format. Respond again with ONLY the applicable label names from this "
            f"exact list, separated by semicolons (leave it empty if none apply): "
            f"{'; '.join(valid_labels)}"
        )
        return validate_fn, repair

    if task == "qa_medqa":
        def validate_fn(s: str) -> bool:
            return re.fullmatch(r"[A-E]", s.strip().strip("'\".").upper()) is not None
        repair = (
            "Your previous output didn't clearly give one letter. Respond with ONLY a "
            "single letter: A, B, C, D, or E. Nothing else."
        )
        return validate_fn, repair

    if task == "qa_pubmedqa":
        def validate_fn(s: str) -> bool:
            return s.strip().strip("'\".").lower() in ("yes", "no", "maybe")
        repair = (
            "Your previous output wasn't exactly yes/no/maybe. Respond with ONLY one "
            "word: yes, no, or maybe. Nothing else."
        )
        return validate_fn, repair

    return None, None


# ═══════════════════════════════════════════════════════════
# MLC two-stage gate (agentic/DESIGN_REVIEW.md §4): a cheap yes/no presence
# check before the full multi-label call, to counter the ~2x label
# over-generation observed when the model is always asked to enumerate.
# ═══════════════════════════════════════════════════════════

def _merge_agent_results(*results: dict) -> dict:
    """Combine token/step/tool-call accounting from several run_agent()
    calls into one result dict, keeping the last call's answer/raw_response."""
    merged = dict(results[-1])
    merged["num_steps"]     = sum(r["num_steps"] for r in results)
    merged["tool_calls"]    = [tc for r in results for tc in r["tool_calls"]]
    merged["input_tokens"]  = sum(r["input_tokens"] for r in results)
    merged["output_tokens"] = sum(r["output_tokens"] for r in results)
    merged["total_tokens"]  = sum(r["total_tokens"] for r in results)
    return merged


def run_mlc_with_gate(
    instance: dict, few_shot_examples: list, model: str, host,
    enable_tools: bool, allowed_tools, validate_fn, repair_instruction,
) -> dict:
    gate_sys, gate_user = mlc_gate_prompt(instance["abstract"], instance["dataset"])
    gate_result = run_agent(
        system_prompt=gate_sys, user_prompt=gate_user,
        model=model, host=host, max_steps=2, enable_tools=False,
    )
    gate_answer = gate_result["answer"].strip().strip("'\".").lower()

    if gate_answer.startswith("no"):
        # Skip the full (more expensive) multi-label call entirely and
        # predict an empty label set directly.
        empty = dict(gate_result)
        empty["answer"] = ""
        empty["raw_response"] = ""
        return empty

    system_p, user_p = mlc_prompt(
        abstract=instance["abstract"], dataset=instance["dataset"],
        few_shot_examples=few_shot_examples or None,
    )
    full_result = run_agent(
        system_prompt=system_p, user_prompt=user_p, model=model, host=host,
        enable_tools=enable_tools, allowed_tools=allowed_tools,
        validate_fn=validate_fn, repair_instruction=repair_instruction,
    )
    return _merge_agent_results(gate_result, full_result)


# ═══════════════════════════════════════════════════════════
# Self-consistency (majority vote) for QA — agentic/DESIGN_REVIEW.md §4:
# MedPrompt's ablation attributes a meaningful chunk of its MedQA gain to
# answer-choice-shuffled self-consistency voting. This implements the
# simpler half of that (multi-sample temperature voting, no choice
# shuffling) — see DESIGN_REVIEW.md for the documented scope reduction.
# ═══════════════════════════════════════════════════════════

def run_agent_self_consistent(
    system_p: str, user_p: str, model: str, host,
    enable_tools: bool, allowed_tools, validate_fn, repair_instruction,
    n_samples: int, temperature: float,
) -> dict:
    if n_samples <= 1:
        return run_agent(
            system_prompt=system_p, user_prompt=user_p, model=model, host=host,
            enable_tools=enable_tools, allowed_tools=allowed_tools,
            validate_fn=validate_fn, repair_instruction=repair_instruction,
        )

    samples = [
        run_agent(
            system_prompt=system_p, user_prompt=user_p, model=model, host=host,
            enable_tools=enable_tools, allowed_tools=allowed_tools,
            validate_fn=validate_fn, repair_instruction=repair_instruction,
            temperature=temperature,
        )
        for _ in range(n_samples)
    ]

    def norm(answer: str) -> str:
        return answer.strip().strip("'\".").upper()

    votes = Counter(norm(s["answer"]) for s in samples)
    winning_label, _ = votes.most_common(1)[0]
    winning_sample = next(s for s in samples if norm(s["answer"]) == winning_label)

    merged = _merge_agent_results(*samples)
    merged["answer"]       = winning_sample["answer"]
    merged["raw_response"] = winning_sample["raw_response"]
    merged["error"]        = winning_sample["error"]
    merged["self_consistency_votes"] = dict(votes)
    return merged


# ═══════════════════════════════════════════════════════════
# Dispatch: build prompts from parsed instance
# ═══════════════════════════════════════════════════════════

def build_prompts(instance: dict, few_shot_examples: list[dict]) -> tuple[str, str]:
    task = instance["task"]

    if task == "ner":
        return ner_prompt(
            sentence=instance["sentence"],
            entity_type=instance["entity_type"],
            few_shot_examples=few_shot_examples or None,
        )
    elif task == "re":
        return re_prompt(
            sentence=instance["sentence"],
            entity1=instance["entity1"],
            entity2=instance["entity2"],
            dataset=instance["dataset"],
            few_shot_examples=few_shot_examples or None,
        )
    elif task == "mlc":
        return mlc_prompt(
            abstract=instance["abstract"],
            dataset=instance["dataset"],
            few_shot_examples=few_shot_examples or None,
        )
    elif task == "qa_medqa":
        return qa_prompt_medqa(
            question=instance["question"],
            options=instance["options"],
            few_shot_examples=few_shot_examples or None,
        )
    elif task == "qa_pubmedqa":
        return qa_prompt_pubmedqa(
            question=instance["question"],
            context=instance["context"],
            few_shot_examples=few_shot_examples or None,
        )
    elif task == "summarization":
        return summarization_prompt(
            text=instance["text"],
            dataset=instance["dataset"],
            few_shot_examples=few_shot_examples or None,
        )
    elif task == "simplification":
        return simplification_prompt(
            text=instance["text"],
            few_shot_examples=few_shot_examples or None,
        )
    else:
        raise ValueError(f"Unknown task type: {task}")


# ═══════════════════════════════════════════════════════════
# Per-dataset evaluation
# ═══════════════════════════════════════════════════════════

def evaluate_dataset(
    dataset_key: str,
    model: str,
    setting: str,               # "zero_shot" | "one_shot" | "five_shot"
    enable_tools: bool,
    max_instances: int,
    output_dir: Path,
    project_root: str,
    delay: float = 1.0,
    host: str | None = None,
    max_tool_calls_per_turn: int = 3,
    self_consistency_n: int = 1,
    self_consistency_temperature: float = 0.7,
    dynamic_fewshot: bool = False,
    mlc_gate: bool = False,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key}  |  Setting: {setting}  |  Model: {model}")
    print(f"{'='*60}")

    # ── Load data ──
    try:
        raw_data = load_test_data(dataset_key, project_root)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return {"dataset": dataset_key, "status": "skipped", "error": str(e)}

    # Shuffle with a fixed seed BEFORE truncating. PubMedQA and HoC were both
    # confirmed (agentic/DESIGN_REVIEW.md §1) to have non-randomized row
    # order — the first N rows are not a representative sample — so a small
    # --max_instances run must shuffle first or it silently draws a biased
    # slice (we measured PubMedQA's first 100 rows as 100% "yes" against a
    # true 55/34/11 yes/no/maybe split).
    random.Random(_SHUFFLE_SEED).shuffle(raw_data)

    if max_instances and max_instances < len(raw_data):
        raw_data = raw_data[:max_instances]
    print(f"  Instances: {len(raw_data)}")

    task = DATASET_CONFIG[dataset_key]["task"]
    allowed_tools = DATASET_TOOL_POLICY.get(dataset_key, [])
    validate_fn, repair_instruction = build_validator(task, dataset_key)


    # ── Load few-shot examples ──
    n_shots = {"zero_shot": 0, "one_shot": 1, "five_shot": 5}.get(setting, 0)
    few_shot_examples = []
    retriever = None
    if n_shots > 0:
        if dynamic_fewshot and dataset_key in DYNAMIC_FEWSHOT_DATASETS:
            pool = load_train_pool(dataset_key, project_root)
            if pool:
                text_key = "sentence" if task == "ner" else "text"
                retriever = TfidfRetriever(pool, text_fn=lambda e, k=text_key: e.get(k, ""))
                print(f"  Dynamic few-shot: TF-IDF retriever built from {len(pool)} train examples")
            else:
                print(f"  [WARN] --dynamic_fewshot requested but no train.tsv found for "
                      f"'{dataset_key}' — falling back to static few-shot")

        if retriever is None:
            raw_examples = load_few_shot_examples(dataset_key, n_shots, project_root)
            few_shot_examples = _format_few_shot_examples(task, raw_examples)
            print(f"  Few-shot examples loaded: {len(few_shot_examples)}")

    # ── Run agent on each instance ──
    results = []
    total_tokens   = 0
    total_steps    = 0
    total_tool_calls = 0
    errors         = 0

    for i, raw in enumerate(raw_data):
        instance = parse_instance(raw, dataset_key)

        if retriever is not None:
            query_text = instance["sentence"] if task == "ner" else instance["abstract"]
            instance_few_shot = _format_few_shot_examples(task, retriever.top_k(query_text, n_shots))
        else:
            instance_few_shot = few_shot_examples

        if task == "mlc" and mlc_gate:
            # Two-stage: cheap presence gate, then the full multi-label call
            # only if the gate says "yes" — see run_mlc_with_gate(). Opt-in
            # (--mlc_gate) because a smoke test caught the gate itself
            # producing false negatives (confidently saying "no" when a
            # label did apply) about as often as it correctly skipped a
            # true negative — net effect needs a full-scale run to judge,
            # so it isn't the default. See agentic/DESIGN_REVIEW.md §4.
            try:
                result = run_mlc_with_gate(
                    instance, instance_few_shot, model, host,
                    enable_tools, allowed_tools, validate_fn, repair_instruction,
                )
            except Exception as e:
                print(f"  [WARN] mlc gate/agent failed for instance {i}: {e}")
                errors += 1
                continue
        else:
            try:
                system_p, user_p = build_prompts(instance, instance_few_shot)
            except Exception as e:
                print(f"  [WARN] prompt build failed for instance {i}: {e}")
                errors += 1
                continue

            if task in ("qa_medqa", "qa_pubmedqa") and self_consistency_n > 1:
                result = run_agent_self_consistent(
                    system_p, user_p, model, host,
                    enable_tools, allowed_tools, validate_fn, repair_instruction,
                    n_samples=self_consistency_n,
                    temperature=self_consistency_temperature,
                )
            else:
                result = run_agent(
                    system_prompt=system_p,
                    user_prompt=user_p,
                    model=model,
                    enable_tools=enable_tools,
                    host=host,
                    allowed_tools=allowed_tools,
                    max_tool_calls_per_turn=max_tool_calls_per_turn,
                    validate_fn=validate_fn,
                    repair_instruction=repair_instruction,
                )

        # Attach ground truth
        gold = (
            instance.get("gold_entities") or
            instance.get("gold_label") or
            instance.get("gold_labels") or
            instance.get("gold_answer") or
            instance.get("gold_summary") or
            instance.get("gold_simple") or
            ""
        )

        # Summarization/simplification answers are full free-text passages, not a
        # short extracted answer — use the raw model output directly (minus any
        # leaked preamble). _extract_answer() is tuned for classification/QA-style
        # single answers (last line, "answer: X"), which would otherwise truncate
        # a multi-sentence summary down to its last line.
        if instance["task"] in ("summarization", "simplification"):
            prediction = _strip_preamble(result["raw_response"].strip())
        else:
            answer = result["answer"].strip()
            prediction = answer.rstrip(".)").upper() if len(answer) <= 3 else answer

        record = {
            "id":            instance.get("id", str(i)),
            "dataset":       dataset_key,
            "setting":       setting,
            "model":         model,
            "gold":          gold,
            "prediction":    prediction,
            "raw_response":  result["raw_response"],
            "num_steps":     result["num_steps"],
            "tool_calls":    result["tool_calls"],
            "input_tokens":  result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens":  result["total_tokens"],
            "error":         result["error"],
        }
        results.append(record)

        total_tokens     += result["total_tokens"]
        total_steps      += result["num_steps"]
        total_tool_calls += len(result["tool_calls"])
        if result["error"] and result["error"] != "max_steps_reached":
            errors += 1

        # Progress print every 10 instances
        if (i + 1) % 10 == 0 or (i + 1) == len(raw_data):
            print(
                f"  [{i+1}/{len(raw_data)}]  "
                f"steps_avg={total_steps/(i+1):.1f}  "
                f"tokens_total={total_tokens}  "
                f"tool_calls={total_tool_calls}  "
                f"errors={errors}"
            )

        time.sleep(delay)  # rate limit

    # ── Save results ──
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{dataset_key}_{model.replace('/', '-')}_{setting}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {out_file}")

    # ── Task metrics (per the benchmark paper's official metrics, see agentic/metrics.py) ──
    metrics = compute_metrics(task, dataset_key, results)
    print(f"  Metrics: {metrics}")

    # ── Summary stats ──
    n = len(results)
    summary = {
        "dataset":              dataset_key,
        "model":                model,
        "setting":              setting,
        "enable_tools":         enable_tools,
        "allowed_tools":        allowed_tools if enable_tools else [],
        "n_instances":          n,
        "errors":               errors,
        "avg_steps":            round(total_steps / n, 2) if n else 0,
        "avg_tool_calls":       round(total_tool_calls / n, 2) if n else 0,
        "total_tokens":         total_tokens,
        "avg_tokens_per_inst":  round(total_tokens / n, 0) if n else 0,
        "metrics":              metrics,
        "output_file":          str(out_file),
        "status":               "completed",
    }
    print(f"  Summary: {summary}")
    return summary


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run agentic evaluation on BioNLP benchmarks"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4",
        help="Model name (gpt-4, gpt-4o, gpt-35-turbo, or a local Ollama model tag "
             "such as llama3.1, qwen2.5, etc.)"
    )
    parser.add_argument(
        "--host", type=str, default=None,
        help="Base URL of a local Ollama server, e.g. http://127.0.0.1:11435. "
             "When set, requests are sent to Ollama's OpenAI-compatible API "
             "instead of OpenAI/Azure, using --model as the Ollama model tag."
    )
    parser.add_argument(
        "--setting", type=str, default="zero_shot",
        choices=["zero_shot", "one_shot", "five_shot"],
        help="Evaluation setting"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to evaluate (default: all 12)"
    )
    parser.add_argument(
        "--max_instances", type=int, default=None,
        help="Max instances per dataset (None = full test set)"
    )
    parser.add_argument(
        "--no_tools", action="store_true",
        help="Disable PubMed/entity tools everywhere (overrides the per-dataset "
             "tool policy in DATASET_TOOL_POLICY, which is 'off' for all but the "
             "two NER datasets already — see agentic/DESIGN_REVIEW.md)"
    )
    parser.add_argument(
        "--max_tool_calls_per_turn", type=int, default=3,
        help="Cap on tool calls actually executed per LLM turn (default 3). "
             "Extra calls in the same turn get a canned 'budget exceeded' "
             "response instead of hitting the network — guards against the "
             "80-calls-in-one-turn blowup observed on MedQA."
    )
    parser.add_argument(
        "--self_consistency_n", type=int, default=1,
        help="For medqa/pubmedqa only: sample the agent this many times at "
             "--self_consistency_temperature and majority-vote the answer "
             "(default 1 = disabled, single greedy sample)."
    )
    parser.add_argument(
        "--self_consistency_temperature", type=float, default=0.7,
        help="Sampling temperature used when --self_consistency_n > 1"
    )
    parser.add_argument(
        "--mlc_gate", action="store_true",
        help="For hoc/litcovid: run a cheap yes/no presence-gate call before the "
             "full multi-label call, skipping the full call on 'no' (default off — "
             "a smoke test found the gate itself produces false negatives at this "
             "model scale about as often as it correctly saves a call; needs a "
             "full-scale run to judge net effect, see agentic/DESIGN_REVIEW.md §4)."
    )
    parser.add_argument(
        "--dynamic_fewshot", action="store_true",
        help="For ncbi_disease/bc5cdr_chem/hoc/litcovid with --setting one_shot or "
             "five_shot: select the N nearest-neighbor train examples per test "
             "instance (TF-IDF cosine similarity) instead of a fixed static set. "
             "See agentic/DESIGN_REVIEW.md §4."
    )
    parser.add_argument(
        "--output_dir", type=str, default="agentic/results",
        help="Directory to save prediction JSON files"
    )
    parser.add_argument(
        "--project_root", type=str, default=".",
        help="Root of BIOMEDICAL-NLP-NEXT project"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between API calls (default 1.0)"
    )
    args = parser.parse_args()

    enable_tools = not args.no_tools
    output_dir   = Path(args.output_dir)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nAGENTIC BioNLP EVALUATION")
    print(f"Model:        {args.model}")
    print(f"Host:         {args.host or 'OpenAI/Azure (default)'}")
    print(f"Setting:      {args.setting}")
    print(f"Tools:        {'enabled' if enable_tools else 'disabled'}")
    print(f"Datasets:     {args.datasets}")
    print(f"Max instances:{args.max_instances or 'full'}")
    print(f"Output dir:   {output_dir}\n")

    all_summaries = []
    for ds in args.datasets:
        summary = evaluate_dataset(
            dataset_key=ds,
            model=args.model,
            setting=args.setting,
            enable_tools=enable_tools,
            max_instances=args.max_instances,
            output_dir=output_dir,
            project_root=args.project_root,
            delay=args.delay,
            host=args.host,
            max_tool_calls_per_turn=args.max_tool_calls_per_turn,
            self_consistency_n=args.self_consistency_n,
            self_consistency_temperature=args.self_consistency_temperature,
            dynamic_fewshot=args.dynamic_fewshot,
            mlc_gate=args.mlc_gate,
        )
        all_summaries.append(summary)

    # ── Save master summary ──
    summary_file = output_dir / f"summary_{args.model.replace('/', '-')}_{args.setting}_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All done. Master summary → {summary_file}")

    # ── Print quick table ──
    print(f"\n{'Dataset':<20} {'Status':<12} {'N':>6} {'AvgSteps':>10} "
          f"{'AvgTools':>10} {'TotalTokens':>14} {'PrimaryMetric':<18}")
    print("-" * 96)
    for s in all_summaries:
        m = s.get("metrics", {}) or {}
        primary_name = m.get("primary_metric")
        primary_val = m.get(primary_name) if primary_name else None
        primary_str = f"{primary_name}={primary_val}" if primary_name else "-"
        print(
            f"{s['dataset']:<20} {s.get('status',''):<12} "
            f"{s.get('n_instances',0):>6} "
            f"{s.get('avg_steps',0):>10.2f} "
            f"{s.get('avg_tool_calls',0):>10.2f} "
            f"{s.get('total_tokens',0):>14,} "
            f"{primary_str:<18}"
        )


if __name__ == "__main__":
    main()
