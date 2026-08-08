# agentic/ — BioNLP Agentic AI Evaluation

Evaluates a multi-step agentic pipeline on all 12 BioNLP benchmarks
from Chen et al. (2024), extended with chain-of-thought reasoning,
tool use (PubMed search, entity lookup), and self-verification.

**See [DESIGN_REVIEW.md](DESIGN_REVIEW.md)** for a full quantitative +
qualitative analysis of a run, a critique of the agent design, and the
per-dataset redesign this codebase implements (tool policy, output
validation/repair, dynamic few-shot, self-consistency, etc.) — including
what was smoke-tested and confirmed vs. what's still opt-in or deferred.

## File structure

```
agentic/
├── __init__.py
├── agent_harness.py        ← core agent loop (tool use + CoT + validate/repair)
├── data_loader.py          ← reads all 12 benchmark datasets
├── dynamic_fewshot.py      ← TF-IDF nearest-neighbor few-shot retrieval
├── metrics.py              ← official per-task metrics (entity F1, macro/micro F1,
│                              accuracy, ROUGE-L, BERTScore, FKGL/DCRS)
├── run_agentic_eval.py     ← main evaluation script
├── requirements.txt
├── DESIGN_REVIEW.md        ← analysis + redesign write-up
├── tools/
│   ├── __init__.py
│   ├── pubmed_search.py    ← NCBI E-utilities PubMed search
│   └── entity_lookup.py    ← MeSH / gene database entity lookup
└── prompts/
    ├── __init__.py
    └── task_prompts.py     ← one prompt template per task type
```

## Setup

```bash
pip install -r agentic/requirements.txt
export OPENAI_API_KEY=your_key_here
# Optional: for Azure OpenAI
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# Optional: polite PubMed usage
export PUBMED_EMAIL=your@email.com
```

### Local models via Ollama

Pass `--host` to route requests to a local Ollama server's OpenAI-compatible
API instead of OpenAI/Azure — no API key needed:

```bash
python -m agentic.run_agentic_eval \
  --model llama3.2:3b-instruct-fp16 \
  --host http://127.0.0.1:11435 \
  --setting zero_shot --datasets medqa --max_instances 5
```

## Quick start

```bash
# From BIOMEDICAL-NLP-NEXT root

# Smoke test: 5 instances on MedQA, zero-shot
python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting zero_shot \
  --datasets medqa \
  --max_instances 5

# Full run on QA datasets, one-shot with tools
python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting one_shot \
  --datasets medqa pubmedqa

# All 12 datasets, zero-shot, tools disabled (plain CoT)
python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting zero_shot \
  --no_tools

# All 12 datasets, one-shot
python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting one_shot \
  --output_dir agentic/results

# Redesign flags (see DESIGN_REVIEW.md §4 for what each targets):
python -m agentic.run_agentic_eval \
  --model gpt-4 --setting zero_shot --datasets medqa pubmedqa \
  --self_consistency_n 5 --self_consistency_temperature 0.7

python -m agentic.run_agentic_eval \
  --model gpt-4 --setting one_shot --datasets ncbi_disease bc5cdr_chem hoc litcovid \
  --dynamic_fewshot
```

## Per-dataset tool policy

Tool use is no longer uniform across tasks — `DATASET_TOOL_POLICY` in
`run_agentic_eval.py` restricts each dataset to the tools evidence showed
were actually useful (see DESIGN_REVIEW.md §4): only NER keeps a tool
(`entity_lookup`, scoped — `pubmed_search` dropped), everything else runs
tool-free by default. `--no_tools` still overrides everything to off.
`--max_tool_calls_per_turn` (default 3) caps how many tool calls a single
LLM turn can actually execute — extra calls in the same turn get a canned
"budget exceeded" response instead of hitting the network, guarding against
the 80-calls-in-one-turn blowup observed on MedQA pre-redesign.

## Other new flags

| Flag | Default | What it does |
|------|---------|---------------|
| `--self_consistency_n` | `1` (off) | For medqa/pubmedqa: sample N times, majority-vote the answer |
| `--self_consistency_temperature` | `0.7` | Sampling temperature when N > 1 |
| `--dynamic_fewshot` | off | For ncbi_disease/bc5cdr_chem/hoc/litcovid one/five-shot: TF-IDF nearest-neighbor exemplar retrieval per instance instead of a fixed static set |
| `--mlc_gate` | off | For hoc/litcovid: cheap yes/no presence gate before the full multi-label call — **off by default**, a smoke test found the gate itself produces false negatives about as often as it saves a call; see DESIGN_REVIEW.md §4 |
| `--max_tool_calls_per_turn` | `3` | Cap on tool calls executed per LLM turn |

## Output format

Each dataset produces a JSON file:
`agentic/results/{dataset}_{model}_{setting}.json`

Each record contains:
```json
{
  "id": "...",
  "dataset": "medqa",
  "setting": "zero_shot",
  "model": "gpt-4",
  "gold": "C",
  "prediction": "C",
  "raw_response": "Step 1: ... Final answer: C",
  "num_steps": 2,
  "tool_calls": [
    {"step": 1, "tool": "entity_lookup", "args": {...}, "result_len": 312, "executed": true}
  ],
  "input_tokens": 847,
  "output_tokens": 213,
  "total_tokens": 1060,
  "error": null
}
```

## Metrics

Each dataset's summary (in `summary_{model}_{setting}_{timestamp}.json`)
includes a `metrics` dict computed by `agentic/metrics.py` — the paper's own
official metric per task (entity-level F1 for NER, macro/micro F1 for RE
and MLC, accuracy + macro F1 for QA, ROUGE-L + BERTScore for summarization,
ROUGE-L + FKGL/DCRS for simplification) — plus these cost/efficiency stats:

| Metric | Description |
|--------|-------------|
| `avg_steps` | Average LLM calls per instance |
| `avg_tool_calls` | Average tool invocations per instance |
| `total_tokens` | Total token cost for the dataset |
| `avg_tokens_per_inst` | Average tokens per instance |
| `allowed_tools` | Which tools were actually offered for this dataset (see policy below) |

## Datasets supported

| Key | Task | Dataset |
|-----|------|---------|
| `bc5cdr_chem` | NER | BC5CDR Chemical |
| `ncbi_disease` | NER | NCBI Disease |
| `chemprot` | RE | ChemProt |
| `ddi` | RE | DDI2013 |
| `hoc` | MLC | HoC |
| `litcovid` | MLC | LitCovid |
| `medqa` | QA | MedQA (5-option) |
| `pubmedqa` | QA | PubMedQA |
| `pubmed_summ` | Summarization | PubMed |
| `ms2` | Summarization | MS² |
| `cochrane` | Simplification | Cochrane PLS |
| `plos` | Simplification | PLOS |
