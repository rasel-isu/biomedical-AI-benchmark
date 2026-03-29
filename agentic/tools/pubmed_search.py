"""
tools/pubmed_search.py
PubMed E-utilities search tool for the agentic BioNLP pipeline.
Returns titles + abstracts for the top-N results.
No API key required (E-utilities are free; add email for higher rate limits).
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import time

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EMAIL = os.getenv("PUBMED_EMAIL", "researcher@example.com")  # set env var for polite use


def pubmed_search(query: str, max_results: int = 3) -> dict:
    """
    Search PubMed and return abstracts for the top `max_results` articles.

    Args:
        query:       PubMed search query string
        max_results: Number of results (1–5)

    Returns:
        {
          "query": str,
          "results": [
            {"pmid": str, "title": str, "abstract": str},
            ...
          ],
          "total_found": int
        }
    """
    max_results = min(max(1, max_results), 5)  # clamp 1–5

    # ── Step 1: search for PMIDs ──
    search_params = urllib.parse.urlencode({
        "db":     "pubmed",
        "term":   query,
        "retmax": max_results,
        "retmode":"json",
        "email":  EMAIL,
    })
    try:
        with urllib.request.urlopen(
            f"{PUBMED_SEARCH_URL}?{search_params}", timeout=10
        ) as resp:
            import json
            search_data = json.loads(resp.read().decode())
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        total = int(search_data.get("esearchresult", {}).get("count", 0))
    except Exception as e:
        return {"query": query, "results": [], "total_found": 0, "error": str(e)}

    if not pmids:
        return {"query": query, "results": [], "total_found": 0}

    # ── Step 2: fetch abstracts ──
    time.sleep(0.34)  # NCBI rate limit: ≤3 req/sec without API key
    fetch_params = urllib.parse.urlencode({
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email":   EMAIL,
    })
    try:
        with urllib.request.urlopen(
            f"{PUBMED_FETCH_URL}?{fetch_params}", timeout=15
        ) as resp:
            xml_data = resp.read().decode()
    except Exception as e:
        return {"query": query, "results": [], "total_found": total, "error": str(e)}

    results = _parse_pubmed_xml(xml_data)
    return {"query": query, "results": results, "total_found": total}


def _parse_pubmed_xml(xml_str: str) -> list[dict]:
    """Parse PubMed efetch XML into a list of {pmid, title, abstract} dicts."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return []

    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el    = article.find(".//PMID")
        title_el   = article.find(".//ArticleTitle")
        abstract_el= article.find(".//AbstractText")

        pmid     = pmid_el.text.strip()     if pmid_el     is not None else ""
        title    = _elem_text(title_el)
        abstract = _elem_text(abstract_el)

        if title or abstract:
            articles.append({
                "pmid":     pmid,
                "title":    title,
                "abstract": abstract[:1000],  # truncate to keep token cost manageable
            })
    return articles


def _elem_text(elem) -> str:
    """Recursively collect all text (handles mixed-content elements like ArticleTitle)."""
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_elem_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts).strip()
