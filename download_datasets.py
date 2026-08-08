"""
download_datasets.py
Downloads all BioNLP benchmark datasets that are not included in the repo
(stored as HuggingFace links in 'data path.txt') and saves them as TSV
in the expected directory structure.

Usage:
    python download_datasets.py

Requirements:
    pip install datasets huggingface_hub

Datasets downloaded:
    - PubMed Summarization   → ccdv/pubmed-summarization
    - MS²                    → allenai/ms2
    - PLOS Simplification    → BioLaySumm/PLOS (or compatible)
    - HoC                    → if missing
    - LitCovid               → if missing
"""

import json
import csv
import os
from pathlib import Path

# ── Config: HuggingFace dataset IDs and their target folders ─────────────────
HF_DATASETS = {
    "pubmed_summ": {
        "hf_id":   "ccdv/pubmed-summarization",
        "folder":  "[Summarization]PubMed",
        "splits":  {"train": "train", "dev": "validation", "test": "test"},
        "src_col": "article",
        "tgt_col": "abstract",
    },
    "ms2": {
        "hf_id":       "allenai/mslr2022",
        "hf_name":     "ms2",              # config name within mslr2022
        "folder":      "[Summarization]MS2",
        "loader":      "parquet",          # fetch via HF's auto-converted parquet
                                            # files instead of `datasets.load_dataset`
                                            # (mslr2022 uses a legacy loading script
                                            # that needs trust_remote_code, which
                                            # newer `datasets` releases no longer accept)
        # NOTE: the official MS^2 test split has empty `target` values for every
        # row (gold summaries are held out). Per the benchmark paper (Table 5,
        # footnote b): "The gold standard of the testing set of MS^2 is not
        # publicly available; we used the validation set instead." We do the same:
        # train.tsv <- HF train split, test.tsv <- HF validation split.
        "splits":      {"train": "train", "test": "validation"},
        "src_col":     "abstract",         # list of abstracts → join into one string
        "tgt_col":     "target",           # gold summary string
        "src_is_list": True,               # tells _write_tsv to join the list
        "id_col":      "review_id",        # use review_id as the instance ID
    },
}

BENCH_ROOT = Path("benchmarks/Biomedical-NLP-Benchmarks/benchmarks")


def download_all():
    for key, cfg in HF_DATASETS.items():
        folder = BENCH_ROOT / cfg["folder"] / "datasets" / "full_set"
        folder.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if (folder / "test.tsv").exists():
            print(f"[SKIP] {key} — test.tsv already exists")
            continue

        print(f"\n[DOWNLOADING] {key} from {cfg['hf_id']} ...")

        if cfg.get("loader") == "parquet":
            try:
                for split_file, split_name in cfg["splits"].items():
                    out_path = folder / f"{split_file}.tsv"
                    if out_path.exists():
                        print(f"  [SKIP] {split_file}.tsv exists")
                        continue
                    print(f"  Fetching '{split_name}' split via HF parquet API...")
                    records = _fetch_parquet_records(cfg["hf_id"], cfg["hf_name"], split_name)
                    print(f"  Writing {split_file}.tsv ({len(records)} instances)...")
                    _write_tsv(records, out_path, cfg)
                print(f"  [DONE] {key}")
            except Exception as e:
                print(f"  [ERROR] {key}: {e}")
                print(f"  → Try manually: see instructions below for {key}")
                _print_manual_instructions(key, cfg)
            continue

        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: 'datasets' library not installed.")
            print("Run: pip install datasets huggingface_hub")
            return

        try:
            # Load from HuggingFace
            load_kwargs = {"path": cfg["hf_id"], "trust_remote_code": True}
            if cfg.get("hf_name"):
                load_kwargs["name"] = cfg["hf_name"]

            ds = load_dataset(**load_kwargs)

            for split_file, split_name in cfg["splits"].items():
                out_path = folder / f"{split_file}.tsv"
                if out_path.exists():
                    print(f"  [SKIP] {split_file}.tsv exists")
                    continue

                if split_name not in ds:
                    print(f"  [WARN] split '{split_name}' not in dataset, skipping {split_file}")
                    continue

                split_data = ds[split_name]
                print(f"  Writing {split_file}.tsv ({len(split_data)} instances)...")

                _write_tsv(split_data, out_path, cfg)

            print(f"  [DONE] {key}")

        except Exception as e:
            print(f"  [ERROR] {key}: {e}")
            print(f"  → Try manually: see instructions below for {key}")
            _print_manual_instructions(key, cfg)



def _fetch_parquet_records(hf_id: str, hf_name: str, split: str) -> list:
    """
    Fetch all rows of a dataset split as a list of dicts, via HuggingFace's
    auto-converted parquet files (works even for datasets whose original
    loading script is no longer supported by the installed `datasets` version).
    """
    import tempfile
    import pandas as pd
    import requests

    api_url = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}&config={hf_name}"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = [
        f for f in resp.json()["parquet_files"]
        if f["config"] == hf_name and f["split"] == split
    ]
    if not files:
        raise ValueError(f"No parquet files found for {hf_id}/{hf_name}/{split}")

    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted(files, key=lambda x: x["filename"]):
            local_path = os.path.join(tmp, f["filename"])
            r = requests.get(f["url"], timeout=300)
            r.raise_for_status()
            with open(local_path, "wb") as out:
                out.write(r.content)
            frames.append(pd.read_parquet(local_path))
    df = pd.concat(frames, ignore_index=True)
    records = df.to_dict("records")

    # Parquet list-columns come back as numpy arrays, not plain Python lists,
    # so `isinstance(x, list)` checks downstream (e.g. in _write_tsv) would
    # silently miss them and fall back to str()'ing the raw array repr.
    for record in records:
        for k, v in record.items():
            if hasattr(v, "tolist"):
                record[k] = v.tolist()
    return records


def _write_tsv(split_data, out_path: Path, cfg: dict):
    src_col    = cfg["src_col"]
    tgt_col    = cfg["tgt_col"]
    src_is_list = cfg.get("src_is_list", False)
    is_classification = cfg.get("is_classification", False)
    id_col     = cfg.get("id_col", None)   # ← new

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        if is_classification:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "label"],
                                    delimiter="\t")
            writer.writeheader()
            for i, row in enumerate(split_data):
                rid   = str(row.get(id_col, i)) if id_col else str(i)
                text  = str(row.get(src_col, "")).replace("\t", " ").replace("\n", " ")
                label = row.get(tgt_col, "")
                if isinstance(label, list):
                    label = ";".join(str(l) for l in label)
                writer.writerow({"id": rid, "text": text, "label": str(label)})
        else:
            writer = csv.DictWriter(f, fieldnames=["id", "source", "target"],
                                    delimiter="\t")
            writer.writeheader()
            for i, row in enumerate(split_data):
                rid = str(row.get(id_col, i)) if id_col else str(i)
                src = row.get(src_col, "")
                if src_is_list and isinstance(src, list):
                    src = " ".join(str(s) for s in src)  # join abstract list
                src = str(src).replace("\t", " ").replace("\n", " ")
                tgt = str(row.get(tgt_col, "")).replace("\t", " ").replace("\n", " ")
                writer.writerow({"id": rid, "source": src, "target": tgt})




def _print_manual_instructions(key: str, cfg: dict):
    """Print manual download instructions for datasets that fail automatic download."""
    folder = BENCH_ROOT / cfg["folder"] / "datasets" / "full_set"
    print(f"""
  MANUAL DOWNLOAD for {key}:
  1. Go to: https://huggingface.co/datasets/{cfg['hf_id']}
  2. Download the dataset files
  3. Run the conversion:
       python download_datasets.py --convert {key} /path/to/downloaded/file
  Or use the HuggingFace CLI:
       huggingface-cli download {cfg['hf_id']} --repo-type dataset
  Target directory: {folder}
  Expected file:    {folder}/test.tsv
  Expected columns: id  source  target  (tab-separated)
""")


# ── Manual conversion mode ────────────────────────────────────────────────────
def convert_manual(key: str, input_path: str):
    """
    Convert a manually downloaded file to the expected TSV format.
    Usage: python download_datasets.py --convert pubmed_summ /path/to/file.json
    """
    if key not in HF_DATASETS:
        print(f"Unknown key: {key}. Valid: {list(HF_DATASETS.keys())}")
        return

    cfg    = HF_DATASETS[key]
    folder = BENCH_ROOT / cfg["folder"] / "datasets" / "full_set"
    folder.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_path)
    suffix     = input_path.suffix.lower()

    print(f"Converting {input_path} → {folder}/test.tsv ...")

    if suffix == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
    elif suffix == ".jsonl":
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif suffix == ".tsv":
        # Already TSV — just copy with column rename if needed
        import shutil
        shutil.copy(input_path, folder / "test.tsv")
        print(f"Copied to {folder}/test.tsv")
        return
    else:
        print(f"Unsupported format: {suffix}")
        return

    out_path = folder / "test.tsv"
    src_col  = cfg["src_col"]
    tgt_col  = cfg["tgt_col"]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "target"], delimiter="\t")
        writer.writeheader()
        for i, row in enumerate(data):
            src = str(row.get(src_col, "")).replace("\t", " ").replace("\n", " ")
            tgt = str(row.get(tgt_col, "")).replace("\t", " ").replace("\n", " ")
            writer.writerow({"id": str(i), "source": src, "target": tgt})

    print(f"Done: {out_path} ({len(data)} instances)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "--convert":
        key        = sys.argv[2]
        input_file = sys.argv[3] if len(sys.argv) > 3 else None
        if not input_file:
            print("Usage: python download_datasets.py --convert <key> <input_file>")
        else:
            convert_manual(key, input_file)
    else:
        download_all()