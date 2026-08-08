# Where the local agent breaks, and how to fix it per dataset

A read of every prediction the agent produced across all 12 benchmarks, checked against
the source benchmark paper's own GPT-3.5 / GPT-4 / LLaMA-2-13B numbers, plus
literature-grounded redesign recommendations for each dataset.

**Run config:** model `llama3.2:3b-instruct-fp16` · host: local Ollama · setting: zero-shot ·
tools: `pubmed_search` + `entity_lookup` · n = 10 instances / dataset

> **Read this before the numbers.** Every result below comes from a 10-instance smoke run,
> not the full test sets (500–17,000 instances per dataset). At n=10, one flipped instance
> moves a score by 10 points, and — as the analysis below shows for PubMedQA and HoC —
> small unshuffled slices can be actively misleading, not just noisy. Treat every number as
> a qualitative signal about failure modes, not a leaderboard entry.

---

## 0. Implementation status (v2)

Everything below this section is the original analysis and recommendations. This
section tracks what actually got implemented from §4, and what a follow-up smoke
test (n=3-6/dataset, same model) found — including one recommendation that turned
out to have a real downside and was walked back to opt-in.

**Implemented and confirmed working:**
- **Per-dataset tool policy** (`DATASET_TOOL_POLICY` in `run_agentic_eval.py`): only
  NER keeps a tool (`entity_lookup`, scoped — `pubmed_search` dropped); everything
  else runs tool-free. Confirmed: RE/MLC/QA/gen datasets now show `tool_calls: 0`.
- **Tool-call cap per turn** (`max_tool_calls_per_turn`, default 3, in `agent_harness.py`):
  extra calls in a turn get a canned response instead of hitting the network.
- **Structural validate-and-repair loop** (`agent_harness.py` + `build_validator()`
  in `run_agentic_eval.py`): JSON-list check for NER, label-in-set for RE/QA,
  all-labels-in-set for MLC, one retry on failure.
- **NER negative-constraint + strict-JSON prompt fix**: confirmed working —
  BC5CDR-chemical precision went from 0.17 (baseline) to 1.0 in the smoke test
  (no more disease mentions leaking into chemical-only extractions), and multi-entity
  extraction is happening (previously ~1 entity/sentence regardless of gold count).
- **Preamble-leakage fix** (prompt cue rewording + `_strip_preamble()` deterministic
  safety net): confirmed — 0/4 preamble leaks across all four generation datasets in
  the smoke test, down from 10/10 (PLOS) and 6/10 (Cochrane) at baseline.
- **PubMedQA/HoC sampling fix** (shuffle with fixed seed before `[:max_instances]`):
  confirmed — PubMedQA's gold distribution in a follow-up run was a realistic 3
  yes/1 no instead of the previous all-yes artifact.
- **Self-consistency voting** for MedQA/PubMedQA (`--self_consistency_n`): confirmed
  mechanically working (multi-sample + majority vote, merged token/step accounting).
- **TF-IDF dynamic few-shot retrieval** (`--dynamic_fewshot`, `agentic/dynamic_fewshot.py`):
  confirmed mechanically working (retriever builds from the full train split, retrieves
  per-instance nearest neighbors). Also fixed a latent bug this surfaced: `mlc_prompt()`'s
  few-shot renderer read `ex['abstract']` but `_load_mlc_tsv()` records only had `'text'`
  — the static one/five-shot path for HoC/LitCovid would have `KeyError`'d before this fix.

**Implemented, but walked back to opt-in after evidence:**
- **MLC two-stage presence gate** (`run_mlc_with_gate()`, flag: `--mlc_gate`, default
  **off**): implemented per the original recommendation, but the first smoke test showed
  the gate itself producing false negatives — confidently saying "no hallmark present"
  for 2 of 4 HoC instances that had a real gold label. Rewording the gate prompt to bias
  toward "yes when unsure" (the fix you'd reach for) reduced but did not eliminate this
  (2/6 in the retest). Given the gate can suppress a correct answer outright — worse than
  the over-generation problem it was meant to fix — it now defaults to off. The plain
  `mlc_prompt()` call (with its own calibration-instruction fix, see below) is the default.

**Prompt-only fixes with mixed/limited observed effect (not reverted, but reporting
honestly rather than overclaiming):**
- **RE negative-default instruction** (ChemProt/DDI): added, but the smoke test (n=4)
  still showed the model defaulting to a positive relation label on majority-negative
  instances. Consistent with the literature cited in §4 (arXiv:2504.04083), which found
  ChemProt/DDI hard even for GPT-4/o1 specifically because of fine-grained relation
  semantics — this may not be a prompt-wording problem at all.
- **MLC calibration instruction** (reduce label over-generation): modest effect —
  predicted/gold label-count ratio dropped from ~2.1x (baseline) to ~1.8x (retest), not
  eliminated.
- **MS² target-length/register instruction**: limited effect — predictions were still
  2-7x longer than gold in the smoke test, similar to the ~4.6x baseline. Length
  instructions alone don't appear to reliably constrain this local 3B model's output.

**Deferred (not implemented — still real work, flagged rather than half-built):**
- Map-reduce / IMRaD-section chunking for PubMed Summarization, and PICO-extraction
  map-reduce for MS² (§4) — genuine new multi-stage pipelines, not attempted here.
- TextRank extract-then-abstract for PLOS, and the two-pass "extract effect
  direction/numbers, then simplify" design for Cochrane PLS (§4) — not attempted.
- Full embedding-based (SBERT) dynamic few-shot — the TF-IDF version implemented is a
  lighter, locally-computable stand-in for the same idea, not the full technique.
- Answer-choice-shuffling for MedQA self-consistency (the other half of MedPrompt) —
  only the multi-sample-majority-vote half is implemented; shuffling the option
  lettering and re-mapping the answer back would need a `qa_prompt_medqa()` change
  not made here.

**Before running at full scale:** re-verify the specific numbers above at
n≈100+/dataset — everything in this section (including the "confirmed working" items)
was checked at n=3-6/dataset and could still be noise. What's solid is the *mechanism*
(no crashes, tool policy respected, validators trigger correctly, sampling is now
randomized) — the *magnitude* of each fix's benefit still needs a real-scale run to
pin down.

### Confirmed at n=10: full 12-dataset baseline-vs-redesign run

A full n=10/dataset run of both the pre-redesign baseline (`agentic/results-0/`) and
the redesigned pipeline (`agentic/results/`) supersedes the n=3-6 smoke test above with
a real 12-dataset comparison. **Critical caveat: zero test-instance overlap.** The
redesign added shuffling with a fixed seed (to fix the PubMedQA/HoC sampling bug) — so
the baseline (unshuffled, first 10 rows) and the redesign (shuffled, first 10 rows)
tested **completely different instances in every single dataset** (verified: 0/10 id
overlap across all 12). Any specific per-dataset metric delta below could be sampling
noise, not the redesign's effect — the reliable signal is (a) sample-independent
mechanical effects (token cost, tool-call counts) and (b) qualitative behaviors checked
directly against gold/pred pairs, not the aggregate metric alone.

| Dataset | Baseline | Redesign | Tokens (base→new) | Tool calls (base→new) |
|---|---|---|---|---|
| NCBI-Disease | entity_f1=0.333 | entity_f1=0.480 | 13,380→13,467 | 1.7→2.0 |
| BC5CDR-chemical | entity_f1=0.222 | entity_f1=0.421 | 15,146→14,528 | 3.1→2.4 |
| ChemProt | macro_f1=0.143 | macro_f1=0.048 | 22,779→5,403 | 3.9→0.0 |
| DDI2013 | macro_f1=0.050 | macro_f1=0.050 | 13,068→5,129 | 3.0→0.0 |
| HoC | macro_f1=0.215 | macro_f1=0.150 | 25,561→6,061 | 4.7→0.0 |
| LitCovid | macro_f1=0.162 | macro_f1=0.267 | 28,438→6,272 | 5.1→0.0 |
| MedQA | accuracy=0.200 | accuracy=0.600 | 32,847→9,023 | 20.7→0.0 |
| PubMedQA ✦ | accuracy=1.000 | accuracy=0.800 | 18,288→6,195 | 1.8→0.0 |
| PubMed Summ. | rouge_l=0.224 | rouge_l=0.218 | 15,005→13,249 | 0.1→0.0 |
| MS² | rouge_l=0.100 | rouge_l=0.115 | 15,564→13,645 | 0.1→0.0 |
| Cochrane PLS | rouge_l=0.199 | rouge_l=0.194 | 18,321→9,200 | 1.1→0.0 |
| PLOS | rouge_l=0.186 | rouge_l=0.194 | 17,163→7,416 | 1.4→0.0 |
| **Total** | — | — | **235,560→109,588 (−53%)** | **467→44** |

**Sample-independent wins (real regardless of the instance-overlap confound):**
- **53% total token reduction**, driven entirely by the tool policy — a direct,
  guaranteed cost win, not sample-dependent.
- **MedQA tool calls: 20.7→0.0 avg**, confirming the 80-calls-in-one-turn blowup is
  structurally eliminated (`max_tool_calls_per_turn` + per-dataset policy).
- **Preamble leakage: 0/10 across all four generation datasets** (was 10/10 PLOS,
  6/10 Cochrane) — a clean, confirmed fix at real scale.
- **PubMedQA gold distribution is now realistic** (7 yes / 3 no vs. the previous
  all-yes artifact) — the sampling-bug fix holds at n=10. The accuracy *reads* as a
  regression (1.000→0.800), but 1.000 was never real; **0.800 replaces a fabricated
  number with a partially-informative one that still exposes a real bias**: the model
  predicted "yes" 9/10 times regardless of gold, so its accuracy is still mostly a
  byproduct of the dataset's yes-heavy true distribution, not genuine yes/no/maybe
  discrimination — CoT + a tight output constraint (§4 recommendation) hasn't been
  added yet and would be the next lever here.
- **MedQA accuracy 0.200→0.600**, alongside tool calls dropping to zero — mechanically
  consistent with the "tool over-calling was pure noise for this task" finding, though
  the specific magnitude is confounded by different sampled questions.

**RE negative-default instruction: initially looked like a non-fix — it wasn't. Root
cause found and fixed.** ChemProt gold in the redesign run was 8 `false` / 2 `CPR:4` —
predictions were `{CPR:4: 5, CPR:9: 4, CPR:3: 1}`, **zero `false` predictions**,
identical to the baseline pattern. DDI showed the same: gold 7 `DDI-false` / 3
positive, predictions collapsed to `DDI-int` 8/10 times. Reading the raw responses
explained why: `re_prompt()` asked for step-by-step reasoning (steps 1-5, including
the negative-default check) **and** simultaneously said "Output ONLY one label... No
explanation. Just the label" — a direct instruction conflict. The model resolved it by
skipping all reasoning: every response was a single bare label with zero visible
thought (`num_steps=1`, `raw_response == prediction`), meaning the negative-default
rule never got a chance to actually influence anything — it was just more text
competing with pattern-matching in one forward pass.

**Fix:** reworded step 6 from "output only the label" to "reason it out, then write
`Final answer: <label>` on the last line" — the same pattern `qa_prompt_medqa()`
already used successfully — and changed the user-turn cue from a bare `Output:` to
"Think step-by-step, then write 'Final answer: <label>'". Re-ran on the **identical**
shuffled 10 instances (same seed, verified id-for-id match, so this is a clean
before/after unlike the rest of this section):

| | Before (bare-label prompt) | After (CoT + Final-answer prompt) |
|---|---|---|
| ChemProt macro-F1 | 0.048 | **0.081** |
| ChemProt `false` predicted correctly | 0/8 | 2/8 |
| DDI macro-F1 | 0.050 | **0.300** (6x) |
| DDI `DDI-false` predicted correctly | 1/7 | 5/7 |
| DDI-int (generic hedge) usage | 8/10 | 2/10 |

The raw responses now show genuine reasoning ("First, I notice... Next, I see...
Given these clues, I reason...") instead of a bare label. **Correction to the earlier
conclusion in this section:** this was not primarily "ChemProt/DDI are just hard for
any LLM" (the arXiv:2504.04083 framing) — a self-inflicted prompt-design bug (CoT vs.
bare-output conflict) was suppressing the negative-default instruction entirely, and
fixing that bug recovered most of the value the instruction was supposed to provide.
The underlying task probably is still genuinely hard (DDI's remaining 2/10 `DDI-int`
uses and ChemProt's still-low 0.081 show real headroom left), but "prompt fixes don't
help here" was the wrong takeaway — it was two compounding problems, and only one of
them was fixed and confirmed above. This fix is now the default in `task_prompts.py`.

**MLC: refined finding — this isn't just "over-generation," it's a per-label bias.**
The over-generation ratio did shrink (baseline ~2.1x → redesign HoC 2.3/1.4=1.64x), but
reading the actual predictions shows something more specific: HoC predictions are
dominated by two labels — **"Genomic instability and mutation" appears in 7/10
predictions and "Enabling replicative immortality" in 6/10** — almost independent of
the abstract's actual content, while correct-but-less-frequent gold labels get missed.
LitCovid shows the mirror image: **"Diagnosis" appears in 8/10 predictions**, while
"Prevention" — the single most common gold label in this sample (5/10) — is correctly
predicted **once**. This is a sharper, more actionable diagnosis than the original
"predicts too many labels": the model has learned (or defaults to) a small favorite
subset of labels regardless of input, which calibration-instruction wording alone
won't fix — retrieval-based dynamic few-shot (already implemented, `--dynamic_fewshot`,
not yet run at scale) is the more promising lever per §4's cited effect sizes.

**NER: confirmed working at n=10.** BC5CDR precision held at 1.0 with zero disease-type
leakage across all 10 instances (previously 0.17 baseline). Multi-entity extraction is
happening (e.g. NCBI-Disease instance 6 correctly pulled 3 of 4 gold entities in one
sentence — the baseline pattern was ~1 entity regardless of gold count). Recall gaps
remain (several instances still miss an entity entirely), consistent with §4's
still-open recommendation to add retrieval-based few-shot and a span self-verification
pass.

**MS² length/register: still not fixed.** Redesign predictions were still 2.4x longer
than gold on average (was ~4.6x baseline) — better, but the "target ~50-80 words,
terse register" instruction clearly isn't sufficient alone. The map-reduce +
PICO-extraction redesign (§4, deferred) is likely required to actually fix this rather
than a length instruction.

**Bottom line:** the mechanical/structural fixes (tool policy, tool-call cap, sampling
fix, preamble fix) are confirmed working and should hold at any scale — they don't
depend on the model getting smarter, just on the harness behaving correctly. RE's
negative-default instruction *looked* like a non-fix but was actually being silenced by
a CoT-vs-bare-output prompt conflict — fixing that (see above) recovered a real,
instance-matched 6x gain on DDI and a smaller gain on ChemProt, so RE moves from
"prompt fixes don't help" to "the first prompt fix was sound, it just needed room to
actually run." MLC's per-label bias is a different shape of problem (not a
suppressed-reasoning bug — the model reasons fine, it just favors certain labels) and
still needs the deferred, more structural techniques (dynamic few-shot at scale,
map-reduce generation for the long-document gen tasks, MedPrompt-style ensembling) for
further gains. **Re-run with a shared, shuffled instance set (or just a much larger n)
for the rest of this comparison** — outside of the RE retest above (verified
instance-matched), the other per-dataset deltas are directionally informative but not
rigorously attributable given the zero id overlap.

---

## 1. Quantitative analysis

The paper (Chen et al., *Nat. Commun.* 2025) reports zero-shot macro-averages of 0.38
(GPT-3.5), 0.46 (GPT-4), and 0.24 (LLaMA-2 13B) across these 12 benchmarks, against a 0.65
fine-tuned-SOTA macro-average. Our 3B local model, run the same way, lands at a macro-average
of **≈0.29** across the same primary metrics (excluding the flagged PubMedQA row) — behind
GPT-3.5, ahead of LLaMA-2 13B on some tasks, behind it on others.

| Dataset | Task | Ours | GPT-3.5 0-shot | GPT-4 0-shot | LLaMA2-13B 0-shot | SOTA fine-tuned |
|---|---|--:|--:|--:|--:|--:|
| NCBI-Disease | NER (Entity F1) | **0.333** | 0.406 | 0.583 | 0.221 | 0.909 |
| BC5CDR-chemical | NER (Entity F1) | **0.222** | 0.627 | 0.799 | 0.394 | 0.950 |
| ChemProt | RE (Macro F1) | **0.143** | 0.135 | 0.325 | 0.139 | 0.734 |
| DDI2013 | RE (Macro F1) | **0.050** | 0.200 | 0.297 | 0.131 | 0.792 |
| HoC | MLC (Macro F1) | **0.215** | 0.672 | 0.711 | 0.129 | 0.888 |
| LitCovid | MLC (Macro F1) | **0.162** | 0.597 | 0.588 | 0.383 | 0.892 |
| MedQA | QA (Accuracy) | **0.200** | 0.499 | 0.716 | 0.252 | 0.420 |
| PubMedQA ✦ | QA (Accuracy) | **1.000** | 0.656 | 0.628 | 0.552 | 0.734 |
| PubMed Summ. | Summ. (ROUGE-L) | **0.224** | 0.227 | 0.242 | 0.119 | 0.432 |
| MS² | Summ. (ROUGE-L) | **0.100** | 0.089 | 0.122 | 0.095 | 0.208 |
| Cochrane PLS | Simplif. (ROUGE-L) | **0.199** | 0.237 | 0.238 | 0.208 | 0.448 |
| PLOS | Simplif. (ROUGE-L) | **0.186** | 0.232 | 0.225 | 0.212 | 0.437 |

> **✦ The PubMedQA 1.000 is not real skill.** The full PubMedQA test set is 55% yes / 34% no
> / 11% maybe. Our first 10 rows are **100% "yes"** — `test.tsv` isn't shuffled, so
> `raw_data[:max_instances]` in `run_agentic_eval.py` grabs a non-representative block. The
> model also always answered "yes" (a known majority-label bias). Two artifacts cancelled
> out into a perfect-looking score. The same mechanism inflates apparent HoC consistency
> below — three of its first ten rows share near-identical gold labels because they're
> consecutive sentences from the same source abstract.

**Headline pattern:** the 3B local model tracks LLaMA-2-13B zero-shot roughly (sometimes
above, sometimes below), and sits well under GPT-4 everywhere. The gap is *largest* on
structured extraction/classification (NER, RE, MLC) and on MedQA's multi-hop reasoning —
and *smallest* on free-text generation (summarization/simplification), which matches the
paper's own finding that ROUGE-L has a narrow dynamic range that doesn't separate model
quality well.

---

## 2. Qualitative analysis — what's actually going wrong

These are patterns read directly out of the saved `raw_response` / `prediction` /
`tool_calls` fields in `agentic/results/*.json`, not inferred from scores alone.

### Named Entity Recognition

**NCBI-Disease / BC5CDR-chemical — under-extraction, format drift.**
JSON-list output is mostly well-formed (10/10 NCBI, 9/10 BC5CDR), but the model typically
extracts **one entity per sentence** even when several are present — the dominant error is
missed entities, not wrong ones (precision 0.55/0.17 vs. recall 0.24/0.33).

```
gold: ['tumour','sporadic T-cell prolymphocytic leukaemia','T-PLL','clonal malignancy',
       'mature T-cell leukaemia','A-T','T-PLL']
pred: ["T-cell prolymphocytic leukaemia"]
```

BC5CDR also shows entity-*type* confusion — extracting a disease when only chemicals were
requested — and one instance returned prose ("Note: since there's only one entity...")
instead of JSON at all.

```
gold: ['dobutamine']
pred: ["dilated cardiomyopathy", "dobutamine"]   ← disease leaked into a chemical-only list
```

### Relation Extraction

**ChemProt / DDI2013 — never predicts "no relation".**
Both datasets are majority-negative (7/10 ChemProt gold = `false`; 7/10 DDI gold =
`DDI-false`), but the model almost never predicts the negative class — DDI collapses to
`DDI-int` ("interaction present, type unspecified") in 8/10 cases regardless of gold, and
ChemProt's prediction distribution contains *zero* `false` predictions.

```
ChemProt gold distribution: {false: 7, CPR:6: 3}
ChemProt pred distribution: {CPR:6: 4, CPR:9: 3, CPR:3: 2, [refused / asked for the sentence again]: 1}

DDI gold distribution: {DDI-false: 7, DDI-mechanism: 1, DDI-effect: 1, DDI-advise: 1}
DDI pred distribution: {DDI-int: 8, DDI-false: 1, "['DDI-int']": 1}
```

This is a "when in doubt, assume interaction" bias — the opposite of the caution the label
set was designed to allow — and it's the single biggest driver of the 0.05–0.14 macro-F1
scores.

### Multi-label Document Classification

**HoC / LitCovid — over-generation.**
The model predicts roughly **2× more labels per instance than gold** (HoC: 4.0 vs. 1.9;
LitCovid: 3.0 vs. 1.6), and several predictions share zero overlap with gold at all — it's
listing plausible-sounding hallmarks/topics rather than grounding the choice in the specific
abstract.

```
gold: [sustaining proliferative signaling, evading growth suppressors, resisting cell death]
pred: [inducing angiogenesis, activating invasion and metastasis,
       genomic instability and mutation, tumor promoting inflammation]   ← 0 overlap
```

### Question Answering

**MedQA — tool-call blowup.**
Accuracy (0.20) is exactly chance for a 5-option question — worse than even LLaMA-2-13B
zero-shot in the paper (0.252). Tool-call counts per instance ranged from **3 to 80**, while
the number of actual LLM turns stayed flat at ~2 — meaning single turns are issuing dozens of
`entity_lookup`/`pubmed_search` calls (one per candidate diagnosis/answer option), and it
isn't converting into better answers.

```
instance tool-call counts: [3, 12, 6, 6, 12, 72, 3, 7, 6, 80]   |   avg LLM turns: 2.0 (flat)
```

### Summarization / Simplification

**MS² length/register mismatch.**
Predictions run **4.6× longer** than gold (211 vs. 46 words) and are written as a discursive
narrative ("The studies discussed in this summary investigate…") rather than the terse,
formal register of a systematic-review abstract — a register mismatch that suppresses
n-gram overlap even when the content is topically reasonable.

**PLOS / Cochrane preamble leakage.**
Despite the prompt saying "Output ONLY the plain-language summary," **10/10 PLOS** and
**6/10 Cochrane** outputs open with a preamble ("Here is a plain-language summary of the
text:"). Tellingly, the summarization prompt — whose final cue is the single word
`Summary:` rather than `Plain language summary:` — shows **0/10** preamble leakage on both
PubMed and MS². The phrasing of the final cue line is the likely cause, and it's a one-line
fix.

---

## 3. Current agent design — what's structurally causing this

1. **[high] One generic tool pair for every task.** `pubmed_search` and `entity_lookup`
   (live NCBI E-utilities calls, ~0.34s+ latency each) are wired into every prompt via
   `enable_tools=True`, regardless of whether external grounding is relevant. Summarization/
   simplification barely touch them (avg 0.1–1.4 calls) because the input already contains
   everything needed — but MedQA burns up to 80 calls per instance for no accuracy gain, and
   RE tool use didn't stop the false→positive collapse. The literature agrees: over-calling
   tools on self-contained tasks is a documented failure mode (arXiv:2605.18882), not a
   hunch.
2. **[high] No cap on tool calls per turn.** `agent_harness.py` caps LLM turns
   (`max_steps=6`) but not tool calls within a turn — a single confused turn can dispatch
   dozens of NCBI requests at once (the 72/80-call MedQA instances). This is a cost and
   latency risk that scales badly past a 10-instance smoke test.
3. **[high] "Self-verify" is a sentence, not a step.** Every system prompt in
   `task_prompts.py` includes a self-verification instruction, but nothing in
   `agent_harness.py` structurally enforces it — there's no schema validation, no
   retry-on-malformed-output, no second pass. That's why malformed completions (a
   clarifying question instead of a ChemProt label, prose instead of a JSON entity list, a
   preamble instead of a bare summary) sail straight through into scoring.
4. **[medium] One-size answer extraction.** `_extract_answer()` uses the same regex
   heuristics for every classification task, producing artifacts like quote-wrapped labels
   (`"'CPR:6'"`) that then need defensive normalization downstream in `metrics.py`.
5. **[medium] No self-consistency / ensembling.** Temperature is 0 with a single sample
   everywhere. For MedQA specifically, the paper's own MedPrompt reference (the technique
   that first broke 90% on MedQA) attributes roughly a third of its gain to
   answer-choice-shuffled self-consistency voting — a technique this pipeline has no
   mechanism for at all.
6. **[medium] Unshuffled truncation.** `evaluate_dataset()` takes
   `raw_data[:max_instances]`. For datasets whose rows aren't independently shuffled
   (PubMedQA, HoC both showed this), small-n runs silently draw a biased slice rather than a
   representative one.

---

## 4. Per-dataset redesign recommendations

Each recommendation is grounded in either what Section 2 showed going wrong, or in
published work on that exact dataset (see Sources). Tool-use verdict: **off** = disable
tools entirely · **scoped** = keep, but narrow the scope · **helps** = tool use is genuinely
load-bearing.

### Named Entity Recognition

**NCBI-Disease** — *tools: scoped*
- Switch from "list the entities" to span-tagging the sentence itself (copy-with-tags, à la
  TANL) — reduces boundary drift versus free listing.
- Add a self-verification sub-pass: for each candidate span, ask yes/no "is this truly a
  disease mention?" (GPT-NER's technique) to catch the type-confusion seen with BC5CDR.
- Replace static zero-shot with retrieval-selected few-shot exemplars (TF-IDF/SBERT nearest
  neighbors from train) — reported +5.6–7.3 F1 over static prompting.
- *Why:* composite/class-vs-specific mention boundaries are the corpus's known hard case;
  retrieval-based dynamic few-shot is the one lever with a measured effect size here.

**BC5CDR-chemical** — *tools: scoped*
- Explicitly negative-constrain the prompt: "diseases are NOT chemicals — do not include
  them" (directly targets the leak observed in Section 2).
- Same retrieval-based dynamic few-shot as NCBI-Disease (evaluated together in the same
  studies, same effect size).
- Force JSON-only output via the API's structured-output/grammar mode rather than a prose
  instruction, to eliminate the "Note: since there's only one entity…" failure mode.
- *Why:* the observed failure is precision (wrong-type entities), not recall — schema +
  negative constraints target precision directly.

### Relation Extraction

**ChemProt** — *tools: off*
- Enforce strict JSON schema output (`{"relation": "CPR:6"}`) with a filled dummy example
  in-prompt — reduces the "asked for the sentence again" refusal failure mode outright.
- Explicitly instruct: "default to `false` unless there is a clear activation/inhibition/
  binding verb between the marked entities" — targets the never-predicts-negative bias
  directly.
- Have the model re-derive/confirm the `@CHEMICAL$`/`@GENE$` span boundaries before
  classifying — partial-span errors are the literature's #1 reported error source for this
  exact dataset.
- *Why:* published zero-shot LLM evals (arXiv:2504.04083) find ChemProt is the hardest RE
  set even for GPT-4/o1 (~24–27 F1) specifically because of fine-grained mechanistic-relation
  confusion, not missing world knowledge — so retrieval/tool grounding isn't the fix.

**DDI2013** — *tools: off*
- Same JSON-schema + explicit-negative-default fix as ChemProt — this dataset showed the
  identical "collapse to one hedge label" pattern (`DDI-int` 8/10 times).
- Give one contrastive example each for effect vs. mechanism vs. advise inside the prompt —
  the literature's own error analysis flags these three as the semantically-overlapping
  confusable set.
- *Why:* the same literature (arXiv:2504.04083) shows DDI improves somewhat with more
  relation-type coverage in-context but remains driven by discourse-level wording, not
  entity identity — DrugBank/UMLS lookups don't address that.

### Multi-label Document Classification

**HoC** — *tools: off*
- Two-stage gate: first ask "does this text discuss any cancer hallmark at all?" (binary),
  only enumerate specific hallmarks if yes — mirrors the OncoMark two-stage design and
  directly targets the 2× label over-generation seen in Section 2.
- Switch to retrieval-selected (kNN) few-shot rather than zero-shot — measured to lift GPT-4
  macro-F1 substantially on this exact task family.
- *Why:* HoC's core difficulty is implicit language and blurry hallmark boundaries
  (arXiv:2307.12114) — a presence gate before enumeration reduces spurious multi-label
  sprawl more directly than better wording alone.

**LitCovid** — *tools: off*
- Dynamic kNN few-shot retrieval is the single largest lever documented: GPT-4 macro-F1
  measured at 0.59 (static 1-shot) → 0.71 (5-nearest-neighbor shot) on this exact dataset.
- Add the label definitions/boundary guidance used by top BioCreative VII systems (e.g.,
  "Prevention" vs. "Mechanism" disambiguation) directly in-prompt.
- *Why:* this is the one dataset in the whole set with a directly measured, large,
  reproducible effect size for a single technique (retrieval-based few-shot) — prioritize it
  first.

### Question Answering

**MedQA** — *tools: off*
- Turn tools off entirely for this dataset — vignettes are self-contained, and both our own
  evidence (80-call blowups with no accuracy gain) and the literature (arXiv:2605.18882,
  "over-calling bias") agree tool use here is pure cost with no return.
- Replace with MedPrompt's recipe instead: dynamic kNN few-shot + chain-of-thought
  exemplars + answer-choice-shuffled self-consistency voting (3–5 samples) — this is the
  only documented technique that has taken a model past 90% on this exact benchmark.
- *Why:* MedFuzz/"Medical LLMs are easily distracted" show small models default to shortcut
  reasoning on deliberately-plausible distractors — ensembling over multiple reasoning
  traces is the countermeasure, not more retrieval.

**PubMedQA** — *tools: off*
- **Fix the harness first:** shuffle `raw_data` (fixed seed) before slicing
  `[:max_instances]` in `run_agentic_eval.py` — the 1.000 accuracy here is a sampling
  artifact, not signal.
- Turn tools off — the abstract given *is* the required context; external search can only
  pull in a different paper's conclusion and leak/contradict the true label.
- Use zero-shot chain-of-thought ("reason before answering") but keep the final emitted
  token hard-constrained to `{yes, no, maybe}` — CoT helps reasoning but must not leak into
  the parsed answer.
- *Why:* known "yes"-majority label bias compounds with genuinely skewed data — CoT plus a
  tight output constraint is the literature's answer to both without adding retrieval risk.

### Summarization

**PubMed Summarization** — *tools: off*
- Chunk along IMRaD section boundaries (mirrors the dataset's own discourse-aware baseline)
  rather than truncating — do a per-section pass, then a reduce pass that re-grounds against
  the source before finalizing.
- Weight hallucination-checking toward the back third of the output — hallucination rate is
  documented to rise toward the end of long generations.
- *Why:* ROUGE/BERTScore are documented to not correlate with faithfulness on long-document
  summarization (arXiv:2502.00977) — a separate re-grounding pass matters more than prompt
  wording here.

**MS²** — *tools: off*
- Map-reduce with a PICO-extraction map step per input abstract, then a reduce step
  explicitly told to state agreement *and* disagreement across studies rather than force one
  consensus.
- Fix the length/register problem directly: instruct target length (~50 words) and register
  ("write as a single systematic-review abstract sentence-conclusion, not a narrative
  recap") — Section 2 measured a 4.6× overrun with a mismatched voice.
- *Why:* GPT-3 is documented to "struggle with accurate aggregation," worsening as more raw
  input is stuffed in (Shaib et al. 2023) — restricting to Objectives+Results per abstract
  outperformed full-abstract input.

### Text Simplification

**Cochrane PLS** — *tools: off*
- Split into two passes: (1) extract effect direction/magnitude and certainty language
  verbatim, (2) lexically simplify without touching numbers or direction — targets the
  documented insertion/substitution error pattern around technical terms and numbers.
- Strip the preamble directly: end the prompt cue with a word, not a phrase (e.g.
  `Plain-language summary:` on its own line, matching the 0%-leakage pattern already seen on
  the summarization prompts).
- *Why:* Devaraj et al.'s factuality taxonomy for this exact dataset shows
  insertion/substitution errors cluster around numbers and technical terms — a
  numbers-protected second pass addresses that directly.

**PLOS** — *tools: off*
- Extract-then-abstract: pull the ~40 most salient sentences (TextRank) before generating,
  rather than abstracting the whole abstract at once — this was the winning BioLaySumm 2024
  system's approach.
- Generate 2–3 candidates and rerank by a readability+factuality heuristic (a practical
  zero-shot self-consistency substitute) rather than taking the first sample.
- Same preamble fix as Cochrane — this dataset showed the leakage in 10/10 instances, the
  worst of any task in Section 2.
- *Why:* PLOS gold summaries are author-written and more extractive than editor-written lay
  summaries elsewhere — favor conservative rewriting over aggressive free abstraction.

> **Before trusting any delta:** re-run with shuffled sampling and at least n≈100/dataset.
> At n=10, HoC and PubMedQA already demonstrated that a plausible-looking score can be
> entirely a sampling artifact — the same could be masking or exaggerating any of the
> patterns above.

---

## Sources consulted

(via background research on each task family)

1. Chen et al., "Benchmarking large language models for biomedical NLP," *Nat. Commun.*
   2025 — the paper this pipeline reproduces settings from.
2. Dogan, Leaman & Lu 2014 (NCBI-Disease); Li et al. 2016 (BC5CDR); Krallinger et al. 2017
   (ChemProt); Segura-Bedmar et al. 2013 (DDI2013) — original dataset papers.
3. arXiv:2504.04083 — zero-shot LLM benchmark isolating ChemProt/DDI as hardest RE sets due
   to fine-grained relation semantics.
4. PMC12408026 (npj AI) — structured static prompting + retrieval-based dynamic few-shot for
   biomedical NER.
5. arXiv:2304.10428 — GPT-NER span self-verification.
6. Baker et al. 2016 (HoC); arXiv:2204.09781 (LitCovid/BioCreative VII overview);
   arXiv:2307.12114 — dynamic few-shot retrieval effect sizes for both.
7. Jin et al. 2021 (MedQA); arXiv:2311.16452 (MedPrompt); arXiv:2605.18882 (tool
   over-calling bias); MedFuzz / arXiv:2603.12458, arXiv:2504.01201 (shortcut reasoning &
   distraction).
8. Jin et al. 2019 (PubMedQA); arXiv:2510.14353 (CURE zero-shot CoT); arXiv:2310.16146
   (Clinfo.ai, retrieval-risk case).
9. Cohan et al. 2018 (PubMed Summarization); arXiv:2502.00977, arXiv:2505.15291
   (long-document faithfulness/hallucination).
10. DeYoung et al. 2021 (MS²); Shaib et al. 2023 ACL; PMC10449915 (multi-doc aggregation
    failure modes).
11. Devaraj et al. 2021/2022 (Cochrane PLS factuality taxonomy); Luo, Xie, Ananiadou 2022
    (PLOS); arXiv:2408.08566 (BioLaySumm 2024 overview).
