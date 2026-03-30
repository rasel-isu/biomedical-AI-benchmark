# Biomedical-NLP-Next

**A Unified Evaluation Framework for Next-Generation AI Architectures on Biomedical NLP**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmarks: 12](https://img.shields.io/badge/benchmarks-12-orange.svg)]()
[![Tasks: 6](https://img.shields.io/badge/tasks-6-purple.svg)]()

This repository benchmarks **five emerging AI paradigms** — Agentic AI, Large Action Models (LAM), Large Concept Models (LCM), Mixture-of-Experts (MoE), and fine-tuned baselines — across the full 12-task BioNLP evaluation suite from [Chen et al. (2025), *Nature Communications*](https://www.nature.com/articles/s41467-025-56989-2), co-authored with Dr. Zhiyong Lu (NIH/NLM).

> **Core question:** Beyond standard prompting, do tool-augmented agentic pipelines, action-oriented models, concept-level reasoning, and sparse expert routing meaningfully improve LLM performance on structured biomedical NLP — and at what computational cost?

---

## Why this benchmark matters

The Chen et al. (2025) study established that fine-tuned BERT-scale models still outperform zero/few-shot LLMs on most structured BioNLP tasks, while GPT-4 leads on reasoning-heavy QA. That study, however, evaluated models in a **passive single-turn setting**. The field has since moved toward:

- **Agentic pipelines** that retrieve evidence and verify answers mid-reasoning
- **Large Action Models** that take structured actions over biomedical APIs and tools
- **Large Concept Models** that reason over semantic concept embeddings rather than raw tokens
- **Mixture-of-Experts** architectures that activate task-specialized sub-networks

This framework evaluates all of these paradigms on the same 12 benchmarks, enabling direct apples-to-apples comparison with the fine-tuned BERT and GPT-4 results from Chen et al.

---

## Benchmark suite

All evaluations run on the 12 BioNLP tasks from Chen et al. (2025):

| Task group | Datasets | Primary metric |
|---|---|---|
| Named entity recognition (NER) | BC5CDR-Chemical, NCBI Disease | Entity F1 |
| Relation extraction (RE) | ChemProt, DDI 2013 | Micro F1 |
| Multi-label classification (MLC) | HoC, LitCovid | Macro F1 |
| Question answering (QA) | MedQA (5-option), PubMedQA | Accuracy |
| Summarization | PubMed, MS² | ROUGE-L |
| Text simplification | Cochrane PLS, PLOS | ROUGE-L / BERTScore |

---

## Repository structure

```
Biomedical-NLP-Next/
│
├── agentic/                    ← Agentic AI evaluation
│   ├── agent_harness.py        ← CoT + tool-use + self-verification loop
│   ├── data_loader.py          ← unified loader for all 12 datasets
│   ├── run_agentic_eval.py     ← evaluation entry point
│   ├── tools/
│   │   ├── pubmed_search.py    ← live PubMed retrieval (NCBI E-utilities)
│   │   └── entity_lookup.py    ← MeSH / NCBI Gene ontology lookup
│   └── prompts/
│       └── task_prompts.py     ← per-task prompt templates
│
├── lam/                        ← Large Action Model evaluation
│   └── ...
│
├── lcm/                        ← Large Concept Model evaluation
│   └── ...
│
├── moe/                        ← Mixture-of-Experts evaluation
│   └── ...
│
├── baselines/                  ← Fine-tuned and prompted baselines
│   └── ...
│
├── evaluation/                 ← Unified scoring scripts
│   └── ...
│
├── analysis/                   ← Cross-paradigm comparison and cost analysis
│   └── ...
│
├── benchmarks/
│   └── Biomedical-NLP-Benchmarks/   ← data submodule (Chen et al. 2025)
│
├── download_datasets.py        ← auto-downloads HuggingFace datasets
├── run.sh                      ← end-to-end run examples and smoke tests
└── README.md
```

---

## Quickstart

### 1. Environment setup

```bash
conda create -n bio-nlp-next python=3.11 -y
conda activate bio-nlp-next

pip install torch transformers datasets accelerate peft
pip install langchain openai anthropic
pip install evaluate rouge-score bert-score
pip install pandas numpy scikit-learn jupyter
```

### 2. Download datasets

```bash
https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/v1.0.0/benchmarks

```

### 3. Configure API access

```bash
export OPENAI_API_KEY=your_key_here

# Optional: Azure OpenAI
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Optional: polite PubMed rate limits
export PUBMED_EMAIL=your@email.com
```

### 4. Run evaluations

```bash
# Agentic — zero-shot smoke test
python -m agentic.run_agentic_eval \
  --model gpt-4 --setting zero_shot --datasets medqa --max_instances 5

# Agentic — NER tasks, one-shot with tools
python -m agentic.run_agentic_eval \
  --model gpt-4 --setting one_shot \
  --datasets ncbi_disease bc5cdr_chem

# Plain CoT baseline (no tools)
python -m agentic.run_agentic_eval \
  --model gpt-4 --setting zero_shot --no_tools

# Full 12-dataset agentic run
python -m agentic.run_agentic_eval \
  --model gpt-4 --setting zero_shot \
  --output_dir agentic/results
```

See `run.sh` for LAM, LCM, MoE, and baseline run commands.

---

## Module overview

### `agentic/` — Agentic AI

A multi-step agent loop that wraps every BioNLP instance in chain-of-thought reasoning with optional live tool use:

```
Prompt
  └─► [LLM] Chain-of-thought reasoning
              │ (if uncertain)
              ├─► pubmed_search(query)  → top-3 PubMed abstracts
              └─► entity_lookup(name)   → MeSH/gene synonyms + definitions
                          │
                    [LLM] Evidence-grounded reasoning
                          │
                    [LLM] Self-verification → final answer
```

Every run logs step count, tool calls, and token cost per instance — enabling direct measurement of agentic overhead vs. accuracy gain across all 12 tasks.

---

### `lam/` — Large Action Models

Evaluates models that produce **structured actions** over biomedical APIs rather than free-text answers. For BioNLP, this means:

- Structured entity normalization via UMLS/MeSH API calls
- Relation assertion over knowledge graph endpoints
- Clinical trial lookup and patient matching via structured API actions

---

### `lcm/` — Large Concept Models

Evaluates concept-level reasoning where models operate over **semantic embeddings of biomedical concepts** (UMLS CUIs, MeSH descriptors) rather than raw token sequences. Especially relevant for tasks with high lexical divergence between mention surface forms and ontology entries — the core challenge studied in biomedical entity linking.

---

### `moe/` — Mixture of Experts

Evaluates sparse MoE architectures on BioNLP tasks. Core hypothesis: task-specialized expert sub-networks should outperform dense models of equivalent parameter count on multi-task biomedical benchmarks, while reducing per-task inference cost through sparse routing.

---

### `baselines/`

Fine-tuned and prompted baselines for direct comparison against Chen et al. (2025) reported numbers:

| Model | Setting | Notes |
|---|---|---|
| PubMedBERT | Fine-tuned | Domain-specific pretraining on PubMed |
| BioMedBERT | Fine-tuned | Full-text biomedical pretraining |
| GPT-3.5 | Zero-shot | Chen et al. baseline |
| GPT-4 | Zero-shot / few-shot | Chen et al. baseline |
| LLaMA 3 | Zero-shot | Open-source reference |

---

### `evaluation/`

Unified scoring across all task types:

| Task type | Metrics |
|---|---|
| NER | Exact-match entity F1 (span + type) |
| RE | Micro F1 per relation class |
| MLC | Macro F1, per-label F1 |
| QA | Accuracy (exact match for MedQA; yes/no/maybe for PubMedQA) |
| Summarization | ROUGE-1/2/L, BERTScore |
| Simplification | ROUGE-L, BERTScore, SARI |

---

### `analysis/`

Cross-paradigm comparison tools:

- **Performance delta tables** — agentic vs. plain CoT vs. fine-tuned baseline per task and dataset
- **Token cost analysis** — total tokens, cost per instance, cost per accuracy point gained
- **Tool-use analysis** — when the agent invokes retrieval, what it retrieves, and whether it helps
- **Error analysis** — breakdown by category: missing entity, boundary issue, hallucination, wrong label

---

## Output format

Each module writes prediction records to JSON:

```json
{
  "id": "...",
  "dataset": "ncbi_disease",
  "setting": "zero_shot",
  "model": "gpt-4",
  "gold": "Neuroblastoma",
  "prediction": "Neuroblastoma",
  "raw_response": "Step 1: The mention 'NB' … Final answer: Neuroblastoma",
  "num_steps": 2,
  "tool_calls": [
    {
      "step": 1,
      "tool": "entity_lookup",
      "args": {"entity": "NB", "entity_type": "disease"},
      "result_len": 312
    }
  ],
  "input_tokens": 847,
  "output_tokens": 213,
  "total_tokens": 1060,
  "error": null
}
```

---

## Roadmap

- [ ] Full results table: all five paradigms × 12 tasks
- [ ] Open-source model support (Llama 3, Mistral, BioMedGPT) via local inference
- [ ] Integration of Bio-ZSEL zero-shot entity linking as an agentic tool for NER tasks
- [ ] Federated learning evaluation module
- [ ] HuggingFace leaderboard submission

---

## Citation

If you use this framework, please cite the benchmark it extends:

```bibtex
@article{chen2025benchmarking,
  title   = {Benchmarking large language models for biomedical natural language
             processing applications and recommendations},
  author  = {Chen, Qingyu and Hu, Yan and Peng, Xueqing and Xie, Qianqian
             and Jin, Qiao and Gilson, Aidan and Singer, Maxwell B and
             Ai, Xuguang and Lai, Po-Ting and Wang, Zhizheng and
             Keloth, Vipina K and Raja, Kalpana and Huang, Jimin and
             He, Huan and Lin, Fongci and Du, Jingcheng and Zhang, Rui and
             Zheng, W Jim and Adelman, Ron A and Lu, Zhiyong and Xu, Hua},
  journal = {Nature Communications},
  volume  = {16},
  pages   = {3280},
  year    = {2025},
  doi     = {10.1038/s41467-025-56989-2}
}
```

---

## License

MIT
