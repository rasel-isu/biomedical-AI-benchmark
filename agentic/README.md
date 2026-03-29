# agentic/ — BioNLP Agentic AI Evaluation

Evaluates a multi-step agentic pipeline on all 12 BioNLP benchmarks
from Chen et al. (2024), extended with chain-of-thought reasoning,
tool use (PubMed search, entity lookup), and self-verification.

## File structure

```
agentic/
├── __init__.py
├── agent_harness.py        ← core agent loop (tool use + CoT)
├── data_loader.py          ← reads all 12 benchmark datasets
├── run_agentic_eval.py     ← main evaluation script
├── requirements.txt
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
```

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
    {"step": 1, "tool": "entity_lookup", "args": {...}, "result_len": 312}
  ],
  "input_tokens": 847,
  "output_tokens": 213,
  "total_tokens": 1060,
  "error": null
}
```

## New metrics tracked (vs. original paper)

| Metric | Description |
|--------|-------------|
| `avg_steps` | Average LLM calls per instance |
| `avg_tool_calls` | Average tool invocations per instance |
| `total_tokens` | Total token cost for the dataset |
| `avg_tokens_per_inst` | Average tokens per instance |

Run `evaluation/run_eval.py` to compute F1/accuracy on the output JSONs.

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
