"""
dynamic_fewshot.py
TF-IDF-based dynamic few-shot exemplar retrieval.

Literature basis (see agentic/DESIGN_REVIEW.md §4): retrieval-selected
few-shot examples (nearest neighbors of the current instance, by TF-IDF or
SBERT similarity) measurably beat a fixed static few-shot set for biomedical
NER and multi-label classification — e.g. GPT-4 macro-F1 on LitCovid rising
from 0.59 (static 1-shot) to 0.71 (5-nearest-neighbor shot) in the cited
studies. This implements the TF-IDF half of that techique: no network
calls, no embedding-model download, just scikit-learn (already a project
dependency) over the dataset's own train split.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, pool: list, text_fn):
        """
        pool:    list of example dicts to retrieve from (e.g. from
                 data_loader.load_train_pool()).
        text_fn: callable(example_dict) -> str, the text to match on.
        """
        self.pool = pool
        texts = [text_fn(ex) for ex in pool]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def top_k(self, query_text: str, k: int) -> list:
        if not self.pool or self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [self.pool[i] for i in top_idx]
