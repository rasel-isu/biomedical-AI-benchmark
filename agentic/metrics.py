"""
metrics.py
Computes the official evaluation metrics from Chen et al. (Nature Communications,
2025) "Benchmarking large language models for biomedical natural language
processing applications and recommendations" (Table 5) for each of the 12
BioNLP benchmarks, from the per-instance prediction records produced by
run_agentic_eval.py.

Metrics by task (primary / secondary):
  NER              Entity-level F1        (exact match)          / —
  RE               Macro F1               / Micro F1
  MLC              Macro F1               / Micro F1
  QA               Accuracy               / Macro F1
  Summarization    ROUGE-L                / BERTScore (best-effort)
  Simplification   ROUGE-L                / FKGL, DCRS

BERTScore requires downloading a large scoring model and network access; it is
computed on a best-effort basis and reported as unavailable (with a reason)
if the `bert_score` package is missing or the model can't be loaded.
"""

import json
import re
from typing import Any

from agentic.prompts.task_prompts import (
    CHEMPROT_RELATIONS, DDI_RELATIONS, HOC_LABELS, LITCOVID_LABELS,
)

MEDQA_LABELS = ["A", "B", "C", "D", "E"]
PUBMEDQA_LABELS = ["yes", "no", "maybe"]


# ──────────────────────────────────────────────
# Normalization helpers
# ──────────────────────────────────────────────
def _norm_label(s: str) -> str:
    s = str(s).strip()
    # Tolerate a stray single-element Python-list wrapper, e.g. "['DDI-effect']"
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        s = s[1:-1].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1].strip()
    return s.strip().lower()


def _norm_entity(s: str) -> str:
    s = _norm_label(s)
    s = re.sub(r"\s*-\s*", "-", s)     # "ataxia - telangiectasia" -> "ataxia-telangiectasia"
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_entity_list(raw: str) -> list:
    """Best-effort parse of a predicted NER answer into a list of entity strings."""
    if not raw:
        return []
    raw = raw.strip()
    # Strip markdown code fences if the model wrapped the JSON in ```...```
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    # Tolerate an echoed "Output:"/"Answer:" label before the JSON (observed
    # when a few-shot exemplar's own "Output: [...]" formatting gets copied).
    raw = re.sub(r"^(output|answer|result)\s*[:\-]\s*", "", raw, flags=re.IGNORECASE)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, str):
            raw = parsed
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: strip brackets/quotes, split on comma or semicolon
    stripped = raw.strip("[]")
    if not stripped:
        return []
    parts = re.split(r"[,;]", stripped)
    return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]


def _parse_multi_label(raw: str) -> list:
    if not raw:
        return []
    parts = re.split(r"[;\n]", raw)
    return [_norm_label(p) for p in parts if _norm_label(p)]


def _prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ──────────────────────────────────────────────
# NER: entity-level exact-match F1
# ──────────────────────────────────────────────
def compute_ner_metrics(records: list) -> dict:
    tp = fp = fn = 0
    for r in records:
        gold = [_norm_entity(e) for e in (r.get("gold") or [])]
        pred = [_norm_entity(e) for e in _parse_entity_list(r.get("prediction", ""))]

        gold_remaining = list(gold)
        for p in pred:
            if p in gold_remaining:
                gold_remaining.remove(p)
                tp += 1
            else:
                fp += 1
        fn += len(gold_remaining)

    result = _prf1(tp, fp, fn)
    return {
        "primary_metric": "entity_f1",
        "entity_precision": result["precision"],
        "entity_recall": result["recall"],
        "entity_f1": result["f1"],
    }


# ──────────────────────────────────────────────
# RE / QA: single-label multi-class metrics (Accuracy, Macro F1, Micro F1)
# ──────────────────────────────────────────────
def _multiclass_prf1(records: list, valid_labels: list, gold_key: str = "gold",
                      pred_key: str = "prediction") -> dict:
    labels = [_norm_label(l) for l in valid_labels]
    per_label = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}

    correct = 0
    total = 0
    for r in records:
        gold = _norm_label(r.get(gold_key, ""))
        pred = _norm_label(r.get(pred_key, ""))
        total += 1
        if gold == pred:
            correct += 1
        if gold in per_label:
            if pred == gold:
                per_label[gold]["tp"] += 1
            else:
                per_label[gold]["fn"] += 1
        if pred in per_label and pred != gold:
            per_label[pred]["fp"] += 1

    accuracy = correct / total if total else 0.0

    per_label_f1 = []
    micro_tp = micro_fp = micro_fn = 0
    for l, counts in per_label.items():
        stats = _prf1(counts["tp"], counts["fp"], counts["fn"])
        per_label_f1.append(stats["f1"])
        micro_tp += counts["tp"]; micro_fp += counts["fp"]; micro_fn += counts["fn"]

    macro_f1 = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
    micro_f1 = _prf1(micro_tp, micro_fp, micro_fn)["f1"]

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
    }


def compute_re_metrics(records: list, dataset_key: str) -> dict:
    valid_labels = list(CHEMPROT_RELATIONS.keys()) if dataset_key == "chemprot" else list(DDI_RELATIONS.keys())
    stats = _multiclass_prf1(records, valid_labels)
    return {
        "primary_metric": "macro_f1",
        "macro_f1": stats["macro_f1"],
        "micro_f1": stats["micro_f1"],
    }


def compute_qa_metrics(records: list, dataset_key: str) -> dict:
    valid_labels = MEDQA_LABELS if dataset_key == "medqa" else PUBMEDQA_LABELS
    stats = _multiclass_prf1(records, valid_labels)
    return {
        "primary_metric": "accuracy",
        "accuracy": stats["accuracy"],
        "macro_f1": stats["macro_f1"],
    }


# ──────────────────────────────────────────────
# MLC: multi-label macro/micro F1
# ──────────────────────────────────────────────
def compute_mlc_metrics(records: list, dataset_key: str) -> dict:
    valid_labels = [_norm_label(l) for l in (HOC_LABELS if dataset_key == "hoc" else LITCOVID_LABELS)]
    per_label = {l: {"tp": 0, "fp": 0, "fn": 0} for l in valid_labels}

    for r in records:
        gold = set(_norm_label(l) for l in (r.get("gold") or []))
        pred = set(_parse_multi_label(r.get("prediction", "")))
        for l in valid_labels:
            in_gold, in_pred = l in gold, l in pred
            if in_gold and in_pred:
                per_label[l]["tp"] += 1
            elif in_pred and not in_gold:
                per_label[l]["fp"] += 1
            elif in_gold and not in_pred:
                per_label[l]["fn"] += 1

    per_label_f1 = []
    micro_tp = micro_fp = micro_fn = 0
    for l, counts in per_label.items():
        stats = _prf1(counts["tp"], counts["fp"], counts["fn"])
        per_label_f1.append(stats["f1"])
        micro_tp += counts["tp"]; micro_fp += counts["fp"]; micro_fn += counts["fn"]

    macro_f1 = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
    micro_f1 = _prf1(micro_tp, micro_fp, micro_fn)["f1"]

    return {
        "primary_metric": "macro_f1",
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
    }


# ──────────────────────────────────────────────
# Summarization / Simplification: ROUGE-L (+ BERTScore / FKGL / DCRS)
# ──────────────────────────────────────────────
def _rouge_l(records: list, gold_key: str, pred_key: str) -> dict:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge_l": None, "rouge_note": "rouge-score not installed (pip install rouge-score)"}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for r in records:
        gold = r.get(gold_key, "") or ""
        pred = r.get(pred_key, "") or ""
        if not gold.strip() or not pred.strip():
            continue
        scores.append(scorer.score(gold, pred)["rougeL"].fmeasure)

    if not scores:
        return {"rouge_l": None, "rouge_note": "no non-empty gold/pred pairs to score"}
    return {"rouge_l": round(sum(scores) / len(scores), 4)}


def _bertscore(records: list, gold_key: str, pred_key: str) -> dict:
    try:
        from bert_score import score as bertscore_score
    except ImportError:
        return {"bertscore_f1": None, "note": "bert-score not installed (pip install bert-score)"}

    cands, refs = [], []
    for r in records:
        gold = (r.get(gold_key, "") or "").strip()
        pred = (r.get(pred_key, "") or "").strip()
        if gold and pred:
            cands.append(pred)
            refs.append(gold)

    if not cands:
        return {"bertscore_f1": None, "note": "no non-empty gold/pred pairs to score"}

    try:
        _, _, f1 = bertscore_score(cands, refs, lang="en", verbose=False)
        return {"bertscore_f1": round(float(f1.mean()), 4)}
    except Exception as e:
        return {"bertscore_f1": None, "note": f"bert-score failed: {e}"}


def _readability(records: list, pred_key: str) -> dict:
    try:
        import textstat
    except ImportError:
        return {
            "fkgl": None, "dcrs": None,
            "readability_note": "textstat not installed (pip install textstat)",
        }

    fkgl_scores, dcrs_scores = [], []
    for r in records:
        pred = (r.get(pred_key, "") or "").strip()
        if not pred:
            continue
        fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
        dcrs_scores.append(textstat.dale_chall_readability_score(pred))

    if not fkgl_scores:
        return {"fkgl": None, "dcrs": None, "readability_note": "no non-empty predictions to score"}
    return {
        "fkgl": round(sum(fkgl_scores) / len(fkgl_scores), 4),
        "dcrs": round(sum(dcrs_scores) / len(dcrs_scores), 4),
    }


def compute_summarization_metrics(records: list) -> dict:
    out = {"primary_metric": "rouge_l"}
    out.update(_rouge_l(records, gold_key="gold", pred_key="prediction"))
    bert = _bertscore(records, gold_key="gold", pred_key="prediction")
    out["bertscore_f1"] = bert.get("bertscore_f1")
    if "note" in bert:
        out["bertscore_note"] = bert["note"]
    return out


def compute_simplification_metrics(records: list) -> dict:
    out = {"primary_metric": "rouge_l"}
    out.update(_rouge_l(records, gold_key="gold", pred_key="prediction"))
    out.update(_readability(records, pred_key="prediction"))
    return out


# ──────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────
def compute_metrics(task: str, dataset_key: str, records: list) -> dict:
    if not records:
        return {"note": "no instances to score"}
    if task == "ner":
        return compute_ner_metrics(records)
    if task == "re":
        return compute_re_metrics(records, dataset_key)
    if task == "mlc":
        return compute_mlc_metrics(records, dataset_key)
    if task in ("qa_medqa", "qa_pubmedqa"):
        return compute_qa_metrics(records, "medqa" if task == "qa_medqa" else "pubmedqa")
    if task == "summarization":
        return compute_summarization_metrics(records)
    if task == "simplification":
        return compute_simplification_metrics(records)
    return {"note": f"no metric defined for task '{task}'"}
