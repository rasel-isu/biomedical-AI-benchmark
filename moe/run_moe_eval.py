"""
run_moe_eval.py
Evaluates the Mixture-of-Experts (MoE) paradigm on the 12 BioNLP benchmarks.

Paradigm: a single forward pass through a pretrained token-routed sparse model
(Mixtral-8x7B by default). The model's learned router activates a subset of its
experts per token; there is no tool use and no multi-step agent loop — see
moe/moe_runner.py for why.

Design: this driver deliberately REUSES the agentic pipeline's data loading,
prompt templates, metrics, dynamic-few-shot retriever, validators, and answer
post-processing. Only the execution stage differs (run_moe instead of
run_agent). Everything shared is imported from agentic/, not copied, so the two
paradigms are guaranteed to load the same instances, prompt them the same way,
and be scored by the same code — which is the whole point of the framework.

CRITICAL for cross-paradigm comparison: the shuffle seed is identical to
agentic/run_agentic_eval.py (_SHUFFLE_SEED = 42). At any given --max_instances,
this runs on the *same* instances agentic did, enabling a paired per-instance
diff in analysis/. (The DESIGN_REVIEW flags "zero instance overlap" as the thing
that made its own baseline-vs-redesign deltas unattributable — keeping the seed
matched is how we avoid repeating that mistake across paradigms.)

Usage examples:
  # Smoke test: 5 instances on MedQA, zero-shot, local Mixtral via Ollama
  python -m moe.run_moe_eval \
    --model mixtral:8x7b-instruct-v0.1-q4_K_M \
    --host http://127.0.0.1:11435 \
    --setting zero_shot --datasets medqa --max_instances 5

  # All 12 datasets, one-shot
  python -m moe.run_moe_eval \
    --model mixtral:8x7b-instruct-v0.1-q4_K_M \
    --host http://127.0.0.1:11435 \
    --setting one_shot --output_dir moe/results

  # QA with self-consistency voting (majority vote over N samples)
  python -m moe.run_moe_eval \
    --model mixtral:8x7b-instruct-v0.1-q4_K_M --host http://127.0.0.1:11435 \
    --setting zero_shot --datasets medqa pubmedqa \
    --self_consistency_n 5 --self_consistency_temperature 0.7

  # NER/MLC one-shot with TF-IDF dynamic few-shot
  python -m moe.run_moe_eval \
    --model mixtral:8x7b-instruct-v0.1-q4_K_M --host http://127.0.0.1:11435 \
    --setting one_shot --datasets ncbi_disease bc5cdr_chem hoc litcovid \
    --dynamic_fewshot
"""

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Shared pipeline (imported from agentic/, not copied) ──────────────────────
from agentic.data_loader import (
    DATASET_CONFIG, load_test_data, load_few_shot_examples,
    load_train_pool, parse_instance,
)
from agentic.dynamic_fewshot import TfidfRetriever
from agentic.metrics import compute_metrics
from agentic.prompts.task_prompts import (
    ner_prompt, re_prompt, mlc_prompt,
    qa_prompt_medqa, qa_prompt_pubmedqa,
    summarization_prompt, simplification_prompt,
    CHEMPROT_RELATIONS, DDI_RELATIONS, HOC_LABELS, LITCOVID_LABELS,
)
# Reuse the exact preamble stripper + shuffle seed the agentic driver uses, so
# generation-task post-processing and instance sampling match paradigm-to-paradigm.
from agentic.run_agentic_eval import (
    _strip_preamble, _format_few_shot_examples, build_validator,
    build_prompts, _SHUFFLE_SEED, DYNAMIC_FEWSHOT_DATASETS,
)

# ── MoE execution stage (the only paradigm-specific piece) ────────────────────
from moe.run_moe import run_moe

ALL_DATASETS = list(DATASET_CONFIG.keys())


# ═══════════════════════════════════════════════════════════
# Self-consistency (majority vote) for QA — same technique the agentic driver
# uses; re-implemented here against run_moe so the MoE module is self-contained
# and doesn't depend on the agent harness at execution time.
# ═══════════════════════════════════════════════════════════

def _merge_moe_results(*results: dict) -> dict:
    """Combine token/step accounting across several run_moe() calls, keeping the
    last call's answer/raw_response."""
    merged = dict(results[-1])
    merged["num_steps"]     = sum(r["num_steps"] for r in results)
    merged["tool_calls"]    = []  # always empty for MoE
    merged["input_tokens"]  = sum(r["input_tokens"] for r in results)
    merged["output_tokens"] = sum(r["output_tokens"] for r in results)
    merged["total_tokens"]  = sum(r["total_tokens"] for r in results)
    return merged


def run_moe_self_consistent(
    system_p: str, user_p: str, model: str, host,
    validate_fn, repair_instruction, n_samples: int, temperature: float,
) -> dict:
    if n_samples <= 1:
        return run_moe(
            system_prompt=system_p, user_prompt=user_p, model=model, host=host,
            validate_fn=validate_fn, repair_instruction=repair_instruction,
        )

    samples = [
        run_moe(
            system_prompt=system_p, user_prompt=user_p, model=model, host=host,
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

    merged = _merge_moe_results(*samples)
    merged["answer"]       = winning_sample["answer"]
    merged["raw_response"] = winning_sample["raw_response"]
    merged["error"]        = winning_sample["error"]
    merged["self_consistency_votes"] = dict(votes)
    return merged


# ═══════════════════════════════════════════════════════════
# Per-dataset evaluation  (mirrors agentic/evaluate_dataset, minus tools)
# ═══════════════════════════════════════════════════════════

def evaluate_dataset(
    dataset_key: str,
    model: str,
    setting: str,               # "zero_shot" | "one_shot" | "five_shot"
    max_instances: int,
    output_dir: Path,
    project_root: str,
    delay: float = 0.0,
    host: str | None = None,
    self_consistency_n: int = 1,
    self_consistency_temperature: float = 0.7,
    dynamic_fewshot: bool = False,
    max_tokens: int | None = None,
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

    # Shuffle with the SAME fixed seed the agentic driver uses, BEFORE truncating
    # — so a --max_instances run tests the identical instances agentic tested.
    random.Random(_SHUFFLE_SEED).shuffle(raw_data)
    if max_instances and max_instances < len(raw_data):
        raw_data = raw_data[:max_instances]
    print(f"  Instances: {len(raw_data)}")

    task = DATASET_CONFIG[dataset_key]["task"]
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
                print(f"  [WARN] --dynamic_fewshot requested but no train.tsv for "
                      f"'{dataset_key}' — falling back to static few-shot")
        if retriever is None:
            raw_examples = load_few_shot_examples(dataset_key, n_shots, project_root)
            few_shot_examples = _format_few_shot_examples(task, raw_examples)
            print(f"  Few-shot examples loaded: {len(few_shot_examples)}")

    # ── Run MoE on each instance ──
    results = []
    total_tokens = 0
    total_steps = 0
    errors = 0

    for i, raw in enumerate(raw_data):
        instance = parse_instance(raw, dataset_key)

        if retriever is not None:
            query_text = instance["sentence"] if task == "ner" else instance["abstract"]
            instance_few_shot = _format_few_shot_examples(task, retriever.top_k(query_text, n_shots))
        else:
            instance_few_shot = few_shot_examples

        try:
            system_p, user_p = build_prompts(instance, instance_few_shot)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] prompt build failed for instance {i}: {e}")
            errors += 1
            continue

        if task in ("qa_medqa", "qa_pubmedqa") and self_consistency_n > 1:
            result = run_moe_self_consistent(
                system_p, user_p, model, host,
                validate_fn, repair_instruction,
                n_samples=self_consistency_n,
                temperature=self_consistency_temperature,
            )
        else:
            result = run_moe(
                system_prompt=system_p, user_prompt=user_p, model=model, host=host,
                validate_fn=validate_fn, repair_instruction=repair_instruction,
                max_tokens=max_tokens,
            )

        # Attach ground truth (same field-priority order as the agentic driver)
        gold = (
            instance.get("gold_entities") or
            instance.get("gold_label") or
            instance.get("gold_labels") or
            instance.get("gold_answer") or
            instance.get("gold_summary") or
            instance.get("gold_simple") or
            ""
        )

        # Free-text vs short-answer post-processing — identical to agentic/.
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
            "paradigm":      "moe",          # tag so merged analysis can group by paradigm
            "gold":          gold,
            "prediction":    prediction,
            "raw_response":  result["raw_response"],
            "num_steps":     result["num_steps"],
            "tool_calls":    result["tool_calls"],   # always [] for MoE
            "input_tokens":  result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens":  result["total_tokens"],
            "error":         result["error"],
        }
        results.append(record)

        total_tokens += result["total_tokens"]
        total_steps  += result["num_steps"]
        if result["error"] and result["error"] != "max_steps_reached":
            errors += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(raw_data):
            print(f"  [{i+1}/{len(raw_data)}]  steps_avg={total_steps/(i+1):.1f}  "
                  f"tokens_total={total_tokens}  errors={errors}")

        if delay:
            time.sleep(delay)

    # ── Save results ──
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{dataset_key}_{model.replace('/', '-').replace(':', '-')}_{setting}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {out_file}")

    # ── Metrics (same compute_metrics the agentic driver calls) ──
    metrics = compute_metrics(task, dataset_key, results)
    print(f"  Metrics: {metrics}")

    n = len(results)
    summary = {
        "dataset":             dataset_key,
        "model":               model,
        "paradigm":            "moe",
        "setting":             setting,
        "n_instances":         n,
        "errors":              errors,
        "avg_steps":           round(total_steps / n, 2) if n else 0,
        "avg_tool_calls":      0.0,  # structurally zero for MoE
        "total_tokens":        total_tokens,
        "avg_tokens_per_inst": round(total_tokens / n, 0) if n else 0,
        "metrics":             metrics,
        "output_file":         str(out_file),
        "status":              "completed",
    }
    print(f"  Summary: {summary}")
    return summary


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run MoE evaluation on BioNLP benchmarks")
    parser.add_argument(
        "--model", type=str, default="mixtral:8x7b-instruct-v0.1-q4_K_M",
        help="MoE model tag. Default is a local Ollama Mixtral-8x7B-Instruct. "
             "Other options: a Qwen-MoE / OLMoE / Phi-3.5-MoE tag, or an "
             "OpenAI/Azure deployment name."
    )
    parser.add_argument(
        "--host", type=str, default=None,
        help="Base URL of a local OpenAI-compatible server (Ollama/vLLM/TGI), "
             "e.g. http://127.0.0.1:11435. Omit to use OpenAI/Azure."
    )
    parser.add_argument(
        "--setting", type=str, default="zero_shot",
        choices=["zero_shot", "one_shot", "five_shot"],
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS,
        help="Datasets to evaluate (default: all 12)"
    )
    parser.add_argument(
        "--max_instances", type=int, default=None,
        help="Max instances per dataset (None = full test set). Shuffled with the "
             "same seed as the agentic driver, so a capped run tests the SAME "
             "instances agentic did."
    )
    parser.add_argument("--self_consistency_n", type=int, default=1,
        help="For medqa/pubmedqa only: sample N times and majority-vote (1 = off).")
    parser.add_argument("--self_consistency_temperature", type=float, default=0.7)
    parser.add_argument("--dynamic_fewshot", action="store_true",
        help="For ncbi_disease/bc5cdr_chem/hoc/litcovid with one/five-shot: "
             "TF-IDF nearest-neighbor exemplar retrieval per instance.")
    parser.add_argument("--max_tokens", type=int, default=None,
        help="Optional cap on generated tokens per completion. Useful for the "
             "long generation tasks; leave unset to use the server default.")
    parser.add_argument("--output_dir", type=str, default="moe/results")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--delay", type=float, default=0.0,
        help="Seconds between calls (default 0; local servers rarely need it).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nMoE BioNLP EVALUATION")
    print(f"Model:        {args.model}")
    print(f"Host:         {args.host or 'OpenAI/Azure (default)'}")
    print(f"Setting:      {args.setting}")
    print(f"Datasets:     {args.datasets}")
    print(f"Max instances:{args.max_instances or 'full'}")
    print(f"Output dir:   {output_dir}\n")

    all_summaries = []
    for ds in args.datasets:
        summary = evaluate_dataset(
            dataset_key=ds,
            model=args.model,
            setting=args.setting,
            max_instances=args.max_instances,
            output_dir=output_dir,
            project_root=args.project_root,
            delay=args.delay,
            host=args.host,
            self_consistency_n=args.self_consistency_n,
            self_consistency_temperature=args.self_consistency_temperature,
            dynamic_fewshot=args.dynamic_fewshot,
            max_tokens=args.max_tokens,
        )
        all_summaries.append(summary)

    # Ensure the output dir exists even if every dataset was skipped (e.g. the
    # benchmark data submodule isn't populated) — evaluate_dataset() only creates
    # it on a successful run, so writing the master summary would otherwise crash.
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"summary_{args.model.replace('/', '-').replace(':', '-')}_{args.setting}_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All done. Master summary → {summary_file}")

    print(f"\n{'Dataset':<20} {'Status':<12} {'N':>6} {'AvgSteps':>10} "
          f"{'TotalTokens':>14} {'PrimaryMetric':<18}")
    print("-" * 86)
    for s in all_summaries:
        m = s.get("metrics", {}) or {}
        primary_name = m.get("primary_metric")
        primary_val = m.get(primary_name) if primary_name else None
        primary_str = f"{primary_name}={primary_val}" if primary_name else "-"
        print(
            f"{s['dataset']:<20} {s.get('status',''):<12} "
            f"{s.get('n_instances',0):>6} "
            f"{s.get('avg_steps',0):>10.2f} "
            f"{s.get('total_tokens',0):>14,} "
            f"{primary_str:<18}"
        )


if __name__ == "__main__":
    main()