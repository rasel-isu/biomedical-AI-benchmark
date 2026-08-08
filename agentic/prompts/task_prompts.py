"""
prompts/task_prompts.py

Agentic prompt templates for all 6 BioNLP task types.
Aligned to the BIDS-Xu-Lab benchmark prompt format, extended with:
  - explicit chain-of-thought instructions
  - tool use guidance
  - self-verification step

Each function returns (system_prompt, user_prompt) for a single instance.
"""

from typing import Optional

# ═══════════════════════════════════════════════════
# 1. Named Entity Recognition  (BC5CDR-chemical, NCBI Disease)
# ═══════════════════════════════════════════════════

def ner_prompt(
    sentence: str,
    entity_type: str,                 # "chemical" or "disease"
    few_shot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    entity_desc = {
        "chemical": "chemicals, drugs, and chemical compounds",
        "disease":  "diseases, disorders, and medical conditions",
    }.get(entity_type, entity_type)
    # The negative type is whichever of the two label sets this call is NOT
    # extracting — spelled out because the model has been observed to leak
    # the other type into its answer (e.g. a disease mention inside a
    # chemical-only extraction).
    negative_desc = {
        "chemical": "diseases, disorders, symptoms, or medical conditions",
        "disease":  "chemicals, drugs, or chemical compounds",
    }.get(entity_type)

    system = (
        f"You are an expert biomedical named entity recognition (NER) system. "
        f"Your task is to identify and extract all {entity_desc} mentioned in "
        f"biomedical text.\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Read the input sentence carefully.\n"
        f"2. Extract EVERY mention of {entity_desc} — sentences often contain "
        f"more than one; do not stop after the first.\n"
        + (f"3. Do NOT include {negative_desc} — those are a different entity type.\n"
           if negative_desc else "")
        + f"4. If needed, use the entity_lookup tool to confirm ambiguous entity names.\n"
        f"5. Self-verify: re-read the sentence and confirm you haven't missed any entities "
        f"and haven't included any entity of the wrong type.\n"
        f"6. Output ONLY a compact, single-line JSON list of entity strings, exactly as "
        f"they appear in the text. Example: [\"aspirin\", \"headache\"]\n"
        f"   If no entities are found, output: []\n"
        f"   Output nothing else — no notes, no explanation, no markdown fences."
    )

    examples_text = ""
    if few_shot_examples:
        examples_text = "\n\nEXAMPLES:\n"
        for ex in few_shot_examples:
            examples_text += (
                f"Input: {ex['sentence']}\n"
                f"Output: {json_list(ex['entities'])}\n\n"
            )

    user = (
        f"{examples_text}"
        f"Now extract all {entity_desc} from the following sentence.\n\n"
        f"Input: {sentence}\n"
        f"Output:"
    )
    return system, user


# ═══════════════════════════════════════════════════
# 2. Relation Extraction  (ChemProt, DDI2013)
# ═══════════════════════════════════════════════════

CHEMPROT_RELATIONS = {
    "CPR:3": "UPREGULATOR/ACTIVATOR – the chemical activates/upregulates the protein",
    "CPR:4": "DOWNREGULATOR/INHIBITOR – the chemical inhibits/downregulates the protein",
    "CPR:5": "AGONIST – the chemical is an agonist of the protein",
    "CPR:6": "ANTAGONIST – the chemical is an antagonist of the protein",
    "CPR:9": "SUBSTRATE/PRODUCT – the protein acts on the chemical as substrate or product",
    "false": "No direct relation between the chemical and protein in this sentence",
}

DDI_RELATIONS = {
    "DDI-effect":    "The interaction results in a pharmacological effect",
    "DDI-mechanism": "A pharmacokinetic mechanism of interaction is described",
    "DDI-advise":    "A recommendation or advice about the interaction is given",
    "DDI-int":       "A drug-drug interaction is mentioned without specifying type",
    "DDI-false":     "No interaction between the drugs is described",
}

def re_prompt(
    sentence: str,
    entity1: str,
    entity2: str,
    dataset: str,                     # "chemprot" or "ddi"
    few_shot_examples: list[dict] | None = None,
) -> tuple[str, str]:

    if dataset.lower() == "chemprot":
        rel_map      = CHEMPROT_RELATIONS
        negative_label = "false"
        task_desc = (
            "The sentence uses @CHEMICAL$ to mark the chemical "
            "and @GENE$ to mark the protein/gene.\n"
            "Classify the relationship between @CHEMICAL$ and @GENE$."
        )
        valid_labels = list(CHEMPROT_RELATIONS.keys())

    else:  # ddi
        rel_map      = DDI_RELATIONS
        negative_label = "DDI-false"
        task_desc = (
            "The sentence uses @DRUG$ to mark both drug entities "
            "(there may be multiple @DRUG$ mentions).\n"
            "Classify the drug-drug interaction between the @DRUG$ entities."
        )
        valid_labels = list(DDI_RELATIONS.keys())

    relation_guide = "\n".join(
        f"  {k}: {v}" for k, v in rel_map.items()
    )

    # Most sentences in this dataset describe NO relation between the marked
    # entities — the model has been observed to almost never predict the
    # negative label, defaulting to "an interaction must be present" instead.
    # State the negative-default rule explicitly and up front to counter that.
    system = (
        f"You are an expert biomedical relation extraction system.\n\n"
        f"RELATION TYPES:\n{relation_guide}\n\n"
        f"IMPORTANT: most sentences describe NO relation at all between the marked "
        f"entities. Only choose a positive relation type ({[k for k in valid_labels if k != negative_label]}) "
        f"if the sentence contains a clear, explicit verb or phrase directly connecting "
        f"the two marked entities (e.g. inhibits, activates, binds, induces, causes an "
        f"effect/interaction). If that clear textual cue is absent, or you are unsure, "
        f"output \"{negative_label}\" — do not guess a positive label just because both "
        f"entities are mentioned in the same sentence.\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Read the sentence and identify relevant context around the two entities.\n"
        f"2. If the entity meaning is unclear, use entity_lookup to disambiguate.\n"
        f"3. If the sentence context is insufficient, use pubmed_search for background.\n"
        f"4. Reason step-by-step about which relation best fits — explicitly check the "
        f"negative-default rule above first: is there really an explicit textual cue, or "
        f"are you about to guess? Write this reasoning out; do not skip straight to a label.\n"
        f"5. Self-verify your choice against the relation definitions.\n"
        f"6. On the LAST line, write exactly: Final answer: <label>\n"
        f"   where <label> is one of: {valid_labels}"
    )

    examples_text = ""
    if few_shot_examples:
        examples_text = "\n\nEXAMPLES:\n"
        for ex in few_shot_examples:
            examples_text += (
                f"Sentence: {ex['sentence']}\n"
                f"Entity 1: {ex['entity1']}\nEntity 2: {ex['entity2']}\n"
                f"Final answer: {ex['label']}\n\n"
            )

    user = (
        f"{examples_text}"
        f"{task_desc}\n\n"
        f"Sentence: {sentence}\n"
        f"Entity 1: {entity1}\n"
        f"Entity 2: {entity2}\n\n"
        f"Think step-by-step, then write 'Final answer: <label>' on the last line."
    )
    return system, user


# ═══════════════════════════════════════════════════
# 3. Multi-label Document Classification  (HoC, LitCovid)
# ═══════════════════════════════════════════════════

HOC_LABELS = [
    "Sustaining proliferative signaling",
    "Evading growth suppressors",
    "Resisting cell death",
    "Enabling replicative immortality",
    "Inducing angiogenesis",
    "Activating invasion and metastasis",
    "Genomic instability and mutation",
    "Tumor promoting inflammation",
    "Cellular energetics",
    "Avoiding immune destruction",
]

LITCOVID_LABELS = [
    "Mechanism",
    "Transmission",
    "Diagnosis",
    "Treatment",
    "Prevention",
    "Case Report",
    "Epidemic Forecasting",
]

def mlc_prompt(
    abstract: str,
    dataset: str,                     # "hoc" or "litcovid"
    few_shot_examples: list[dict] | None = None,
) -> tuple[str, str]:

    if dataset.lower() == "hoc":
        label_list = HOC_LABELS
        task_desc  = "hallmarks of cancer"
    else:
        label_list = LITCOVID_LABELS
        task_desc  = "COVID-19 research topics"

    labels_formatted = "\n".join(f"  - {l}" for l in label_list)

    system = (
        f"You are an expert biomedical document classification system.\n"
        f"Assign all applicable {task_desc} labels to the given abstract.\n\n"
        f"VALID LABELS:\n{labels_formatted}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Read the abstract carefully.\n"
        f"2. Use pubmed_search if you need background on a specific topic.\n"
        f"3. Identify ONLY labels that are clearly and explicitly supported by the text — "
        f"most abstracts have just 1-3 applicable labels. Do not include a label out of "
        f"caution or because it seems generically related; every label you output must be "
        f"directly grounded in something the abstract actually says.\n"
        f"4. Self-verify: for each label you are about to output, point to the specific "
        f"phrase in the abstract that justifies it. If you can't, drop that label.\n"
        f"5. Output ONLY a semicolon-separated list of applicable labels (or a single "
        f"label, or none at all if nothing applies).\n"
        f"   Example: Label A;Label B\n"
        f"   Use exact label names from the list above. No explanation."
    )

    examples_text = ""
    if few_shot_examples:
        examples_text = "\n\nEXAMPLES:\n"
        for ex in few_shot_examples:
            examples_text += (
                f"Abstract: {ex['abstract'][:300]}...\n"
                f"Output: {';'.join(ex['labels'])}\n\n"
            )

    user = (
        f"{examples_text}"
        f"Classify the following abstract.\n\n"
        f"Abstract: {abstract}\n"
        f"Output:"
    )
    return system, user


def mlc_gate_prompt(abstract: str, dataset: str) -> tuple[str, str]:
    """
    Cheap binary pre-check used before the full mlc_prompt() call: does this
    abstract discuss the topic family at all? A "no" here lets the caller
    skip the full multi-label call and predict an empty label set directly,
    which is both faster and — per the observed 2x label over-generation —
    a stronger prior than asking the full prompt to also decide "none apply".
    """
    task_desc = "hallmarks of cancer" if dataset.lower() == "hoc" else "COVID-19 research topics"

    system = (
        f"You are a biomedical document triage system. Decide whether the given "
        f"abstract discusses ANY {task_desc} at all — do not judge which specific "
        f"one, only whether at least one applies.\n"
        f"If you are at all unsure, or it's a borderline/partial case, answer yes: "
        f"a wrong 'yes' only costs one extra classification step, but a wrong 'no' "
        f"means a real label gets silently skipped and is much more costly.\n"
        f"Output ONLY one word: yes or no."
    )
    user = (
        f"Abstract: {abstract}\n\n"
        f"Does this abstract discuss any {task_desc}? Answer yes or no:"
    )
    return system, user


# ═══════════════════════════════════════════════════
# 4. Question Answering  (MedQA, PubMedQA)
# ═══════════════════════════════════════════════════

def qa_prompt_medqa(
    question: str,
    options: dict[str, str],          # {"A": "...", "B": "...", ...}
    few_shot_examples: list[dict] | None = None,
) -> tuple[str, str]:

    options_text = "\n".join(f"  {k}. {v}" for k, v in options.items())

    system = (
        "You are a highly knowledgeable medical expert answering USMLE-style questions.\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the question and all answer options carefully.\n"
        "2. Use pubmed_search or entity_lookup if you need to verify clinical facts.\n"
        "3. Reason step-by-step through the differential.\n"
        "4. Eliminate clearly wrong options first.\n"
        "5. Self-verify: confirm your chosen answer is consistent with the clinical scenario.\n"
        "6. Output your final answer as ONLY a single letter (A, B, C, D, or E).\n"
        "   Write 'Final answer: X' on the last line."
    )

    examples_text = ""
    if few_shot_examples:
        examples_text = "\n\nEXAMPLES:\n"
        for ex in few_shot_examples:
            opts = "\n".join(f"  {k}. {v}" for k, v in ex["options"].items())
            examples_text += (
                f"Question: {ex['question']}\n{opts}\n"
                f"Final answer: {ex['answer']}\n\n"
            )

    user = (
        f"{examples_text}"
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        f"Think step-by-step, then write 'Final answer: X' on the last line."
    )
    return system, user


def qa_prompt_pubmedqa(
    question: str,
    context: str,
    few_shot_examples: list[dict] | None = None,
) -> tuple[str, str]:

    system = (
        "You are a biomedical research expert answering questions about scientific studies.\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the question and the provided abstract carefully.\n"
        "2. Use pubmed_search if additional context would help.\n"
        "3. Reason step-by-step based on the evidence.\n"
        "4. Self-verify your answer against the abstract.\n"
        "5. Output ONLY: yes, no, or maybe\n"
        "   Write 'Final answer: yes/no/maybe' on the last line."
    )

    examples_text = ""
    if few_shot_examples:
        examples_text = "\n\nEXAMPLES:\n"
        for ex in few_shot_examples:
            examples_text += (
                f"Question: {ex['question']}\n"
                f"Abstract: {ex['context'][:300]}...\n"
                f"Final answer: {ex['answer']}\n\n"
            )

    user = (
        f"{examples_text}"
        f"Question: {question}\n\n"
        f"Abstract:\n{context}\n\n"
        f"Think step-by-step, then write 'Final answer: yes/no/maybe'."
    )
    return system, user


# ═══════════════════════════════════════════════════
# 5. Text Summarization  (PubMed, MS²)
# ═══════════════════════════════════════════════════

def summarization_prompt(
    text: str,
    dataset: str = "pubmed",          # "pubmed" or "ms2"
    few_shot_examples: list[dict] | None = None,
) -> tuple[str, str]:

    if dataset == "ms2":
        task_desc = (
            "Summarize the following collection of biomedical research abstracts "
            "into a single coherent systematic review abstract."
        )
        # MS^2 gold summaries are terse systematic-review conclusions (~40-60 words),
        # not a narrative recap of each input study — the model has been observed to
        # write ~4-5x longer discursive summaries ("The studies discussed..."), which
        # tanks n-gram-overlap metrics even when the content is topically reasonable.
        length_guide = (
            f"6. Target length: about 50-80 words — write it as a single systematic-"
            f"review abstract's conclusion (formal, terse register), NOT a narrative "
            f"recap of each individual study.\n"
        )
    else:
        task_desc = (
            "Write a concise, accurate abstract for the following biomedical article."
        )
        length_guide = ""

    system = (
        f"You are an expert biomedical writer producing high-quality summaries.\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Read all the provided text carefully.\n"
        f"2. Identify the key findings, methods, and conclusions.\n"
        f"3. Write a summary that is accurate, complete, and readable.\n"
        f"4. Self-verify: ensure all key findings are captured and no misinformation "
        f"is introduced.\n"
        f"5. Output ONLY the summary text — no preface such as \"Here is a summary:\", "
        f"no heading, no bullet points. Start directly with the summary's first word.\n"
        f"{length_guide}"
    )

    examples_text = ""
    if few_shot_examples:
        examples_text = "\n\nEXAMPLE:\n"
        ex = few_shot_examples[0]
        examples_text += f"Input:\n{ex['text'][:500]}...\n\nOutput:\n{ex['summary']}\n\n"

    user = (
        f"{examples_text}"
        f"{task_desc}\n\n"
        f"Input:\n{text}\n\n"
        f"Summary:"
    )
    return system, user


# ═══════════════════════════════════════════════════
# 6. Text Simplification  (Cochrane PLS, PLOS)
# ═══════════════════════════════════════════════════

def simplification_prompt(
    text: str,
    few_shot_examples: list[dict] | None = None,
) -> tuple[str, str]:

    system = (
        "You are a science communicator simplifying biomedical research for general audiences.\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the technical biomedical text.\n"
        "2. Identify the core message, methods, and findings.\n"
        "3. Rewrite in plain language (no jargon; accessible to non-specialists).\n"
        "4. Keep all key facts, numbers, and effect directions accurate — do not add, "
        "remove, or soften findings.\n"
        "5. Self-verify: is every sentence understandable to a non-scientist, and did "
        "you change any number or direction of effect? (You must not.)\n"
        "6. Output ONLY the plain-language summary text — no preface such as \"Here is "
        "a plain-language summary:\", no heading, no bullet points. Start directly with "
        "the summary's first word."
    )

    examples_text = ""
    if few_shot_examples:
        examples_text = "\n\nEXAMPLE:\n"
        ex = few_shot_examples[0]
        examples_text += (
            f"Technical:\n{ex['text'][:500]}...\n\n"
            f"Plain language:\n{ex['simplified']}\n\n"
        )

    # Cue kept to a single word ("Summary:") rather than a descriptive phrase
    # ("Plain language summary:") — the descriptive phrase was empirically
    # linked to preamble leakage ("Here is a plain-language summary...") in
    # 6-10/10 sampled outputs, while the single-word cue used in
    # summarization_prompt() showed 0/10 leakage on the same model.
    user = (
        f"{examples_text}"
        f"Simplify the following biomedical text for a general audience.\n\n"
        f"Technical text:\n{text}\n\n"
        f"Summary:"
    )
    return system, user


# ═══════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════

def json_list(items: list) -> str:
    import json
    return json.dumps(items)
