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
    "bc5cdr_chem":  {"folder": "[NER]BC5CDR_Chemical", "task": "ner", "entity_type": "chemical", "format": "conll"},
    "ncbi_disease": {"folder": "[NER]NCBI_Disease",     "task": "ner", "entity_type": "disease",  "format": "conll"},
    "chemprot": {"folder": "[RE]Chemprot", "task": "re", "dataset": "chemprot", "format": "tsv_re"},
    "ddi":      {"folder": "[RE]DDI",      "task": "re", "dataset": "ddi",      "format": "tsv_re"},
    "hoc":      {"folder": "[MLC]Hoc",      "task": "mlc", "dataset": "hoc",      "format": "tsv_mlc"},
    "litcovid": {"folder": "[MLC]LitCovid", "task": "mlc", "dataset": "litcovid", "format": "tsv_mlc"},
    "medqa":         {"folder": "[QA]MedQA",               "task": "qa_medqa"},
    "pubmedqa": {"folder": "[QA]PubMedQA", "task": "qa_pubmedqa", "format": "tsv_pubmedqa"},
    "pubmed_summ": {"folder": "[Summarization]PubMed", "task": "summarization", "dataset": "pubmed", "format": "tsv_gen"},
    "ms2":         {"folder": "[Summarization]MS2",     "task": "summarization", "dataset": "ms2",    "format": "tsv_gen"},
    "cochrane": {"folder": "[Simplification]CochranePLS", "task": "simplification", "format": "tsv_gen"},
    "plos":     {"folder": "[Simplification]PLOS",         "task": "simplification", "format": "tsv_gen"},
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


# New helper function:

# 1c. Add these two helpers alongside _load_medqa_tsv():

def _load_conll_tsv(path: Path) -> list[dict]:
    sentences, tokens, labels = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if tokens:
                    sentences.append(_conll_to_instance(tokens, labels, len(sentences)))
                    tokens, labels = [], []
            else:
                parts = line.split()
                tokens.append(parts[0])
                labels.append(parts[-1] if len(parts) >= 2 else "O")
    if tokens:
        sentences.append(_conll_to_instance(tokens, labels, len(sentences)))
    return sentences


def _conll_to_instance(tokens: list, labels: list, idx: int) -> dict:
    sentence, entities, span = " ".join(tokens), [], []
    for token, label in zip(tokens, labels):
        if label == "B":
            if span: entities.append(" ".join(span))
            span = [token]
        elif label == "I" and span:
            span.append(token)
        else:
            if span: entities.append(" ".join(span)); span = []
    if span: entities.append(" ".join(span))
    return {"id": str(idx), "sentence": sentence,
            "tokens": tokens, "labels": labels, "entities": entities}

def _load_re_tsv(path: Path) -> list[dict]:
    import csv
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            index    = row.get("index", str(i)).strip()
            sentence = row.get("sentence", "").strip()
            label    = row.get("label", "false").strip()
            pmid     = index.split(".")[0] if "." in index else str(i)

            # Detect dataset from placeholder type
            # ChemProt: @CHEMICAL$ and @GENE$
            # DDI:      @DRUG$ (both entities are drugs)
            if "@DRUG$" in sentence:
                entity1 = "@DRUG$"
                entity2 = "@DRUG$"
            elif "@CHEMICAL$" in sentence:
                entity1 = "@CHEMICAL$"
                entity2 = "@GENE$"
            else:
                entity1 = "entity1"
                entity2 = "entity2"

            records.append({
                "id":       index,
                "pmid":     pmid,
                "sentence": sentence,
                "entity1":  entity1,
                "entity2":  entity2,
                "label":    label,
            })
    return records

def _load_pubmedqa_tsv(path: Path) -> list[dict]:
    """
    Parse PubMedQA TSV format.
    Columns: QUESTION  CONTEXTS  LABELS  MESHES  YEAR
             reasoning_required_pred  reasoning_free_pred
             final_decision  LONG_ANSWER  pmid

    CONTEXTS is a Python list-as-string: ['sentence1', 'sentence2', ...]
    final_decision is the gold label: yes / no / maybe
    """
    import csv
    import ast

    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):

            # Parse CONTEXTS: stored as a Python list string
            contexts_raw = row.get("CONTEXTS", "[]")
            try:
                contexts = ast.literal_eval(contexts_raw)
                if isinstance(contexts, list):
                    context = " ".join(str(c) for c in contexts)
                else:
                    context = str(contexts_raw)
            except Exception:
                context = contexts_raw

            # Normalise final_decision to lowercase
            final_decision = row.get("final_decision", "").strip().lower()
            if final_decision not in ("yes", "no", "maybe"):
                final_decision = row.get("reasoning_free_pred", "maybe").strip().lower()

            records.append({
                "id":             row.get("pmid", str(i)).strip(),
                "question":       row.get("QUESTION", "").strip(),
                "context":        context.strip(),
                "long_answer":    row.get("LONG_ANSWER", "").strip(),
                "final_decision": final_decision,
                "meshes":         row.get("MESHES", ""),
                "year":           row.get("YEAR", ""),
            })
    return records

def _load_mlc_tsv(path: Path) -> list[dict]:
    import csv
    csv.field_size_limit(10_000_000)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            # labels column: "Label A;Label B;Label C"
            labels_raw = row.get("labels", row.get("label", "")).strip()
            labels = [l.strip() for l in labels_raw.split(";") if l.strip()]

            records.append({
                "id":     row.get("pmid", str(i)).strip(),
                "text":   row.get("text", "").strip(),
                "labels": labels,
            })
    return records

def _load_gen_tsv(path: Path) -> list[dict]:
    """
    Parse generation task TSV files (CochranePLS, PLOS, PubMed summarization).
    All share the same core pattern: source text → target text.

    CochranePLS columns: gem_id  gem_parent_id  source  target  doi  references
    PLOS columns:        likely  gem_id  source  target  (similar)
    PubMed columns:      may use article/abstract naming
    """
    import csv
    csv.field_size_limit(10_000_000)

    records = []


    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []

        # Detect source/target column names flexibly
        src_col = next(
            (c for c in fieldnames if c.lower() in ("source", "article", "text", "src", "input")),
            None
        )
        tgt_col = next(
            (c for c in fieldnames if c.lower() in ("target", "abstract", "summary",
                                                      "tgt", "output", "plain_language_summary")),
            None
        )
        id_col = next(
            (c for c in fieldnames if c.lower() in ("gem_id", "id", "pmid", "doi")),
            None
        )

        if not src_col or not tgt_col:
            raise ValueError(
                f"Cannot detect source/target columns in {path}\n"
                f"Found columns: {fieldnames}"
            )

        for i, row in enumerate(reader):
            records.append({
                "id":     row.get(id_col, str(i)).strip() if id_col else str(i),
                "text":   row.get(src_col, "").strip(),
                "target": row.get(tgt_col, "").strip(),
            })

    return records

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
    
    elif config.get("format") == "conll":
        tsv_candidates = [
            bench_root / folder / "datasets" / "full_set" / "test.tsv",
            bench_root / folder / "datasets" / "test.tsv",
            bench_root / folder / "test.tsv",
        ]
        for path in tsv_candidates:
            if path.exists():
                return _load_conll_tsv(path)
        raise FileNotFoundError(f"test.tsv not found for '{dataset_key}'")
    
    elif config.get("format") == "tsv_re":
        tsv_candidates = [
            bench_root / folder / "datasets" / "full_set" / "test.tsv",
            bench_root / folder / "datasets" / "test.tsv",
            bench_root / folder / "test.tsv",
        ]
        for path in tsv_candidates:
            if path.exists():
                return _load_re_tsv(path)
        raise FileNotFoundError(f"test.tsv not found for '{dataset_key}'")
    
    elif config.get("format") == "tsv_pubmedqa":
        tsv_candidates = [
            bench_root / folder / "datasets" / "full_set" / "test.tsv",
            bench_root / folder / "datasets" / "test.tsv",
            bench_root / folder / "test.tsv",
        ]
        for path in tsv_candidates:
            if path.exists():
                return _load_pubmedqa_tsv(path)
        raise FileNotFoundError(f"test.tsv not found for '{dataset_key}'")
    
    elif config.get("format") == "tsv_gen":
        tsv_candidates = [
            bench_root / folder / "datasets" / "full_set" / "test.tsv",
            bench_root / folder / "datasets" / "test.tsv",
            bench_root / folder / "test.tsv",
        ]

        for path in tsv_candidates:
            if path.exists():
                return _load_gen_tsv(path)
        raise FileNotFoundError(f"test.tsv not found for '{dataset_key}'")
    
    elif config.get("format") == "tsv_mlc":
        tsv_candidates = [
            bench_root / folder / "datasets" / "full_set" / "test.tsv",
            bench_root / folder / "datasets" / "test.tsv",
            bench_root / folder / "test.tsv",
        ]
        for path in tsv_candidates:
            if path.exists():
                return _load_mlc_tsv(path)
        raise FileNotFoundError(f"test.tsv not found for '{dataset_key}'")

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
    
    elif config.get("format") == "conll":
        tsv_path = dataset_dir / "datasets" / "full_set" / "train.tsv"
        if tsv_path.exists():
            data = _load_conll_tsv(tsv_path)
            return data[:n_shots]
        # Also check prompt_oneshot.txt / prompt_fiveshot.txt
        txt_name = {1: "prompt_oneshot.txt", 5: "prompt_fiveshot.txt"}.get(n_shots)
        if txt_name:
            txt_path = dataset_dir / txt_name
            if txt_path.exists():
                return []  # txt prompts handled directly — return empty, harness skips
        return []
    
    elif config.get("format") == "tsv_re":
        tsv_path = dataset_dir / "datasets" / "full_set" / "train.tsv"
        if tsv_path.exists():
            data = _load_re_tsv(tsv_path)
            return data[:n_shots]
        return []
    
    elif config.get("format") == "tsv_pubmedqa":
        tsv_path = dataset_dir / "datasets" / "full_set" / "train.tsv"
        if tsv_path.exists():
            data = _load_pubmedqa_tsv(tsv_path)
            return data[:n_shots]
        return []
    
    elif config.get("format") == "tsv_gen":
        tsv_path = dataset_dir / "datasets" / "full_set" / "train.tsv"
        if tsv_path.exists():
            data = _load_gen_tsv(tsv_path)
            return data[:n_shots]
        return []

    elif config.get("format") == "tsv_mlc":
        tsv_path = dataset_dir / "datasets" / "full_set" / "train.tsv"
        if tsv_path.exists():
            data = _load_mlc_tsv(tsv_path)
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
            "task":          task,
            "dataset_key":   dataset_key,
            "entity_type":   config.get("entity_type", "any"),
            "sentence":      _get(instance, ["sentence", "text", "input"]),
            "gold_entities": _get_list(instance, ["entities", "labels", "gold"]),
            "id":            _get(instance, ["id", "pmid", "idx"], ""),
        }

    elif task == "re":
        return {
            "task":        task,
            "dataset_key": dataset_key,
            "dataset":     config.get("dataset", dataset_key),
            "sentence":    _get(instance, ["sentence", "text", "input"]),
            "entity1":     _get(instance, ["entity1", "arg1", "subject"], "@CHEMICAL$"),
            "entity2":     _get(instance, ["entity2", "arg2", "object"],  "@GENE$"),
            "gold_label":  _get(instance, ["label", "relation", "gold"]),
            "id":          _get(instance, ["id", "index", "idx"], ""),
        }

    elif task == "mlc":
        return {
            "task":        task,
            "dataset_key": dataset_key,
            "dataset":     config.get("dataset", dataset_key),
            "abstract":    _get(instance, ["abstract", "text", "source", "input"]),  # ✓ "text" covered
            "gold_labels": _get_list(instance, ["labels", "label", "target", "gold"]),  # ✓ "labels" covered
            "id":          _get(instance, ["id", "pmid", "idx"], ""),  # ✓ "pmid" covered
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
            "question":    _get(instance, ["question", "QUESTION", "input"]),
            "context":     _get(instance, ["context", "CONTEXTS", "abstract", "long_answer"]),
            "gold_answer": _get(instance, ["final_decision", "answer", "label"]),
            "id":          _get(instance, ["id", "pmid", "pubid", "idx"], ""),
        }

    elif task == "summarization":
        return {
            "task":         task,
            "dataset_key":  dataset_key,
            "dataset":      config.get("dataset", dataset_key),
            "text":         _get(instance, ["text", "source", "article", "src", "input"]),
            "gold_summary": _get(instance, ["target", "abstract", "summary", "tgt", "gold"]),
            "id":           _get(instance, ["id", "gem_id", "idx"], ""),
        }

    elif task == "simplification":
        return {
            "task":        task,
            "dataset_key": dataset_key,
            "text":        _get(instance, ["text", "source", "article", "src", "input"]),
            "gold_simple": _get(instance, ["target", "plain_language_summary",
                                            "simplified", "pls", "tgt", "gold"]),
            "id":          _get(instance, ["id", "gem_id", "idx"], ""),
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
