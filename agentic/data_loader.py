"""
data_loader.py
Loads test instances from the BIDS-Xu-Lab benchmark datasets.
Reads the same JSON format used by the original GPT evaluation scripts.

Expected path (relative to BIOMEDICAL-NLP-NEXT root):
  benchmarks/Biomedical-NLP-Benchmarks/benchmarks/{dataset_key}/datasets/full_set/test.json

Also loads the original few-shot prompt examples from:
  benchmarks/Biomedical-NLP-Benchmarks/benchmarks/{dataset_key}/*.json  (prompt files)
"""

import json
import os
from pathlib import Path

# ── Dataset metadata ──────────────────────────────────────────────────────────
DATASET_CONFIG = {
    # key               folder name in repo                task type
    "bc5cdr_chem":   {"folder": "[NER]BC5CDR_Chemical",   "task": "ner",    "entity_type": "chemical"},
    "ncbi_disease":  {"folder": "[NER]NCBI_Disease",       "task": "ner",    "entity_type": "disease"},
    "chemprot":      {"folder": "[RE]Chemprot",            "task": "re",     "dataset": "chemprot"},
    "ddi":           {"folder": "[RE]DDI",                 "task": "re",     "dataset": "ddi"},
    "hoc":           {"folder": "[MLC]Hoc",                "task": "mlc",    "dataset": "hoc"},
    "litcovid":      {"folder": "[MLC]LitCovid",           "task": "mlc",    "dataset": "litcovid"},
    "medqa":         {"folder": "[QA]MedQA",               "task": "qa_medqa"},
    "pubmedqa":      {"folder": "[QA]PubMedQA",            "task": "qa_pubmedqa"},
    "pubmed_summ":   {"folder": "[Summarization]PubMed",   "task": "summarization", "dataset": "pubmed"},
    "ms2":           {"folder": "[Summarization]MS2",      "task": "summarization", "dataset": "ms2"},
    "cochrane":      {"folder": "[Simplification]CochranePLS", "task": "simplification"},
    "plos":          {"folder": "[Simplification]PLOS",    "task": "simplification"},
}


def get_benchmark_root(project_root: str = ".") -> Path:
    """Find the benchmark root directory."""
    candidates = [
        Path(project_root) / "benchmarks" / "Biomedical-NLP-Benchmarks" / "benchmarks",
        Path(project_root) / "benchmarks" / "benchmarks",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Benchmark directory not found. "
        f"Expected one of: {[str(c) for c in candidates]}\n"
        f"Make sure you have cloned the repo into benchmarks/"
    )


def load_test_data(dataset_key: str, project_root: str = ".") -> list[dict]:
    config     = _get_config(dataset_key)
    folder     = config["folder"]
    bench_root = get_benchmark_root(project_root)

    # MedQA uses TSV files (columns: meta_info, question, answer_idx, answer, options)
    if dataset_key == "medqa":
        tsv_candidates = [
            bench_root / folder / "datasets" / "full_set" / "test.tsv",
            bench_root / folder / "datasets" / "test.tsv",
            bench_root / folder / "test.tsv",
        ]
        for path in tsv_candidates:
            if path.exists():
                return _load_medqa_tsv(path)
        raise FileNotFoundError(
            f"test.tsv not found for medqa\nTried: {[str(p) for p in tsv_candidates]}"
        )

    # All other datasets: JSON
    json_candidates = [
        bench_root / folder / "datasets" / "full_set" / "test.json",
        bench_root / folder / "datasets" / "test.json",
        bench_root / folder / "test.json",
    ]
    for path in json_candidates:
        if path.exists():
            return _load_json(path)

    raise FileNotFoundError(
        f"test.json not found for '{dataset_key}'\n"
        f"Tried: {[str(p) for p in json_candidates]}"
    )


def _load_medqa_tsv(path: Path) -> list[dict]:
    import csv
    import ast

    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            options_raw = row.get("options", "")

            if i == 0:
                print(f"[DEBUG] options raw for row 0: {repr(options_raw)}")

            # Parse options → always produce a dict {"A": "...", "B": "...", ...}
            options = {}
            try:
                parsed = ast.literal_eval(options_raw)

                if isinstance(parsed, dict):
                    # Keys might be 'opa','opb',... or 'A','B',...
                    # Normalise to uppercase letters A,B,C,D,E
                    key_map = {}
                    for k in parsed:
                        k_str = str(k).strip()
                        if k_str.lower().startswith("op"):
                            # opa→A, opb→B, opc→C, opd→D, ope→E
                            letter = chr(ord("A") + ord(k_str[-1].lower()) - ord("a"))
                            key_map[letter] = parsed[k]
                        else:
                            key_map[k_str.upper()] = parsed[k]
                    options = key_map

                elif isinstance(parsed, list):
                    # Plain list → assign A, B, C, D, E
                    letters = "ABCDE"
                    options = {letters[j]: v for j, v in enumerate(parsed) if j < 5}

            except Exception:
                # Last resort: treat raw string as the only option
                options = {"A": options_raw}

            records.append({
                "id":         str(i),
                "meta_info":  row.get("meta_info", ""),
                "question":   row.get("question", "").strip(),
                "answer_idx": row.get("answer_idx", "").strip(),
                "answer":     row.get("answer", "").strip(),
                "options":    options,
            })
    return records

def load_few_shot_examples(dataset_key: str, n_shots: int = 1,
                           project_root: str = ".") -> list[dict]:
    config      = _get_config(dataset_key)
    folder      = config["folder"]
    bench_root  = get_benchmark_root(project_root)
    dataset_dir = bench_root / folder

    # Look for the original prompt JSON files first (works for all datasets)
    shot_name = {1: "one_shot", 5: "five_shot"}.get(n_shots, f"{n_shots}_shot")
    prompt_candidates = list(dataset_dir.glob(f"*{shot_name}*.json"))
    if prompt_candidates:
        examples = _load_json(prompt_candidates[0])
        if isinstance(examples, list):
            return examples[:n_shots]
        if isinstance(examples, dict):
            return [examples]

    # MedQA fallback: read from train.tsv
    if dataset_key == "medqa":
        tsv_path = dataset_dir / "datasets" / "full_set" / "train.tsv"
        if tsv_path.exists():
            data = _load_medqa_tsv(tsv_path)
            return data[:n_shots]
        return []

    # All other datasets: read from train.json
    train_candidates = [
        dataset_dir / "datasets" / "full_set" / "train.json",
        dataset_dir / "datasets" / "train.json",
    ]
    for path in train_candidates:
        if path.exists():
            data = _load_json(path)
            return data[:n_shots] if isinstance(data, list) else []

    return []



def parse_instance(instance: dict, dataset_key: str) -> dict:
    """
    Normalize a raw benchmark instance into a standard format for the agent.
    Returns a dict with task-relevant fields.
    """
    config = _get_config(dataset_key)
    task   = config["task"]

    if task == "ner":
        return {
            "task":        task,
            "dataset_key": dataset_key,
            "entity_type": config.get("entity_type", "any"),
            "sentence":    _get(instance, ["sentence", "text", "input"]),
            "gold_entities": _get_list(instance, ["entities", "labels", "gold"]),
            "id":          _get(instance, ["id", "pmid", "idx"], ""),
        }

    elif task == "re":
        return {
            "task":        task,
            "dataset_key": dataset_key,
            "dataset":     config.get("dataset", dataset_key),
            "sentence":    _get(instance, ["sentence", "text", "input"]),
            "entity1":     _get(instance, ["entity1", "arg1", "subject"]),
            "entity2":     _get(instance, ["entity2", "arg2", "object"]),
            "gold_label":  _get(instance, ["label", "relation", "gold"]),
            "id":          _get(instance, ["id", "idx"], ""),
        }

    elif task == "mlc":
        return {
            "task":        task,
            "dataset_key": dataset_key,
            "dataset":     config.get("dataset", dataset_key),
            "abstract":    _get(instance, ["abstract", "text", "input"]),
            "gold_labels": _get_list(instance, ["labels", "label", "gold"]),
            "id":          _get(instance, ["id", "pmid", "idx"], ""),
        }

    elif task == "qa_medqa":
        options = instance.get("options", {})

        # Safety: if options is still a list after loading, convert here too
        if isinstance(options, list):
            letters = "ABCDE"
            options = {letters[j]: v for j, v in enumerate(options) if j < 5}

        return {
            "task":        task,
            "dataset_key": dataset_key,
            "question":    _get(instance, ["question", "input"]),
            "options":     options,
            "gold_answer": _get(instance, ["answer_idx", "answer", "label", "gold"]),
            "id":          _get(instance, ["id", "idx"], ""),
        }

    elif task == "qa_pubmedqa":
        return {
            "task":        task,
            "dataset_key": dataset_key,
            "question":    _get(instance, ["question", "input"]),
            "context":     _get(instance, ["context", "abstract", "long_answer"]),
            "gold_answer": _get(instance, ["final_decision", "answer", "label"]),
            "id":          _get(instance, ["id", "pubid", "idx"], ""),
        }

    elif task == "summarization":
        return {
            "task":        task,
            "dataset_key": dataset_key,
            "dataset":     config.get("dataset", dataset_key),
            "text":        _get(instance, ["article", "text", "input", "src"]),
            "gold_summary":_get(instance, ["abstract", "summary", "tgt", "gold"]),
            "id":          _get(instance, ["id", "idx"], ""),
        }

    elif task == "simplification":
        return {
            "task":        task,
            "dataset_key": dataset_key,
            "text":        _get(instance, ["article", "text", "input", "src"]),
            "gold_simple": _get(instance, ["plain_language_summary", "simplified",
                                           "pls", "tgt", "gold"]),
            "id":          _get(instance, ["id", "idx"], ""),
        }

    return {"task": task, "dataset_key": dataset_key, "raw": instance}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_config(dataset_key: str) -> dict:
    if dataset_key not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset key: '{dataset_key}'. "
            f"Valid keys: {list(DATASET_CONFIG.keys())}"
        )
    return DATASET_CONFIG[dataset_key]


def _load_json(path: Path) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get(d: dict, keys: list[str], default: str = "") -> str:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _get_list(d: dict, keys: list[str]) -> list:
    for k in keys:
        if k in d and d[k] is not None:
            val = d[k]
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                return [v.strip() for v in val.split(";") if v.strip()]
    return []
