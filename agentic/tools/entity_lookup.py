"""
tools/entity_lookup.py
Biomedical entity lookup tool using the NCBI E-utilities (no API key required).
Queries MeSH for diseases/chemicals and gene databases for genes.
Falls back to a simple PubMed title search when specific DB lookup fails.
"""

import urllib.request
import urllib.parse
import json
import os
import time

EMAIL = os.getenv("PUBMED_EMAIL", "researcher@example.com")
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY= "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


# Map entity types to NCBI databases
_DB_MAP = {
    "disease":  "mesh",
    "chemical": "mesh",
    "gene":     "gene",
    "drug":     "mesh",
    "any":      "mesh",
}


def entity_lookup(entity: str, entity_type: str = "any") -> dict:
    """
    Look up a biomedical entity in NCBI databases.

    Args:
        entity:      Entity name or synonym (e.g. "metformin", "type 2 diabetes")
        entity_type: One of disease | chemical | gene | drug | any

    Returns:
        {
          "entity": str,
          "entity_type": str,
          "results": [
            {
              "id": str,
              "name": str,
              "description": str,
              "synonyms": list[str],
              "source": str
            }
          ]
        }
    """
    db = _DB_MAP.get(entity_type, "mesh")

    # ── Search for entity IDs ──
    search_params = urllib.parse.urlencode({
        "db":     db,
        "term":   entity,
        "retmax": 3,
        "retmode":"json",
        "email":  EMAIL,
    })
    try:
        with urllib.request.urlopen(
            f"{ESEARCH}?{search_params}", timeout=10
        ) as resp:
            search_data = json.loads(resp.read().decode())
        ids = search_data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        return {"entity": entity, "entity_type": entity_type,
                "results": [], "error": str(e)}

    if not ids:
        return {"entity": entity, "entity_type": entity_type, "results": []}

    # ── Fetch summaries ──
    time.sleep(0.34)
    summary_params = urllib.parse.urlencode({
        "db":     db,
        "id":     ",".join(ids),
        "retmode":"json",
        "email":  EMAIL,
    })
    try:
        with urllib.request.urlopen(
            f"{ESUMMARY}?{summary_params}", timeout=10
        ) as resp:
            summary_data = json.loads(resp.read().decode())
    except Exception as e:
        return {"entity": entity, "entity_type": entity_type,
                "results": [], "error": str(e)}

    results = _parse_summary(summary_data, db, ids)
    return {"entity": entity, "entity_type": entity_type, "results": results}


def _parse_summary(data: dict, db: str, ids: list[str]) -> list[dict]:
    result_list = []
    uids = data.get("result", {}).get("uids", ids)

    for uid in uids:
        entry = data.get("result", {}).get(str(uid), {})
        if not entry:
            continue

        if db == "mesh":
            name  = entry.get("ds_meshterms", [entry.get("name", "")])[0] \
                    if entry.get("ds_meshterms") else entry.get("name", "")
            desc  = entry.get("ds_scopenote", entry.get("description", ""))
            synon = entry.get("ds_meshterms", [])
        elif db == "gene":
            name  = entry.get("name", "")
            desc  = entry.get("summary", "")
            synon = entry.get("otheraliases", "").split(", ") \
                    if entry.get("otheraliases") else []
        else:
            name  = entry.get("name", str(uid))
            desc  = entry.get("description", "")
            synon = []

        result_list.append({
            "id":          uid,
            "name":        name,
            "description": str(desc)[:500],  # keep tokens manageable
            "synonyms":    synon[:10],
            "source":      db.upper(),
        })

    return result_list
