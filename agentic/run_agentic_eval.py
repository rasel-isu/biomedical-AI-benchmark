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
import time
from datetime import datetime
from pathlib import Path

from agentic.agent_harness import run_agent
from agentic.data_loader import (
    DATASET_CONFIG, load_test_data, load_few_shot_examples, parse_instance
)
from agentic.prompts.task_prompts import (
    ner_prompt, re_prompt, mlc_prompt,
    qa_prompt_medqa, qa_prompt_pubmedqa,
    summarization_prompt, simplification_prompt,
)

ALL_DATASETS = list(DATASET_CONFIG.keys())


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

    if max_instances and max_instances < len(raw_data):
        raw_data = raw_data[:max_instances]
    print(f"  Instances: {len(raw_data)}")


    # ── Load few-shot examples ──
    n_shots = {"zero_shot": 0, "one_shot": 1, "five_shot": 5}.get(setting, 0)
    few_shot_examples = []
    if n_shots > 0:
        raw_examples = load_few_shot_examples(dataset_key, n_shots, project_root)
        # Normalise CoNLL examples to the format ner_prompt() expects
        task = DATASET_CONFIG[dataset_key]["task"]
        if task == "ner":
            few_shot_examples = [
                {
                    "sentence": e.get("sentence", ""),
                    "entities": e.get("entities", []),
                }
                for e in raw_examples if e.get("sentence")
            ]
        else:
            few_shot_examples = raw_examples
        print(f"  Few-shot examples loaded: {len(few_shot_examples)}")

    # ── Run agent on each instance ──
    results = []
    total_tokens   = 0
    total_steps    = 0
    total_tool_calls = 0
    errors         = 0

    for i, raw in enumerate(raw_data):
        instance = parse_instance(raw, dataset_key)
        try:
            system_p, user_p = build_prompts(instance, few_shot_examples)
        except Exception as e:
            print(f"  [WARN] prompt build failed for instance {i}: {e}")
            errors += 1
            continue

        result = run_agent(
            system_prompt=system_p,
            user_prompt=user_p,
            model=model,
            enable_tools=enable_tools,
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

        record = {
            "id":            instance.get("id", str(i)),
            "dataset":       dataset_key,
            "setting":       setting,
            "model":         model,
            "gold":          gold,
            "prediction": result["answer"].strip().rstrip(".)").upper()
              if len(result["answer"]) <= 3   # only clean short answers (letters)
              else result["answer"],
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

    # ── Summary stats ──
    n = len(results)
    summary = {
        "dataset":              dataset_key,
        "model":                model,
        "setting":              setting,
        "enable_tools":         enable_tools,
        "n_instances":          n,
        "errors":               errors,
        "avg_steps":            round(total_steps / n, 2) if n else 0,
        "avg_tool_calls":       round(total_tool_calls / n, 2) if n else 0,
        "total_tokens":         total_tokens,
        "avg_tokens_per_inst":  round(total_tokens / n, 0) if n else 0,
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
        help="Model name (gpt-4, gpt-4o, gpt-35-turbo, etc.)"
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
        help="Disable PubMed/entity tools (plain CoT agent only)"
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
          f"{'AvgTools':>10} {'TotalTokens':>14}")
    print("-" * 76)
    for s in all_summaries:
        print(
            f"{s['dataset']:<20} {s.get('status',''):<12} "
            f"{s.get('n_instances',0):>6} "
            f"{s.get('avg_steps',0):>10.2f} "
            f"{s.get('avg_tool_calls',0):>10.2f} "
            f"{s.get('total_tokens',0):>14,}"
        )


if __name__ == "__main__":
    main()
