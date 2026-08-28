# Chunk by section
import re


def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)


# BM25Index implementation
import math
from collections import Counter
from typing import Callable, Optional, Any, List, Dict, Tuple


class BM25Index:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._index_built: bool = False

        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [token for token in tokens if token]

    def _update_stats_add(self, doc_tokens: List[str]):
        self._doc_len.append(len(doc_tokens))

        seen_in_doc = set()
        for token in doc_tokens:
            if token not in seen_in_doc:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen_in_doc.add(token)

        self._index_built = False

    def _calculate_idf(self):
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def _build_index(self):
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return

        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        self._calculate_idf()
        self._index_built = True

    def add_document(self, document: Dict[str, Any]):
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError(
                "Document dictionary must contain a 'content' key."
            )

        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        doc_tokens = self._tokenizer(content)

        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        self._update_stats_add(doc_tokens)

    def _compute_bm25_score(
        self, query_tokens: List[str], doc_index: int
    ) -> float:
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_index])
        doc_length = self._doc_len[doc_index]

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self._avg_doc_len)
            )
            score += numerator / (denominator + 1e-9)

        return score

    def search(
        self,
        query_text: str,
        k: int = 1,
        score_normalization_factor: float = 0.1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents:
            return []

        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        query_tokens = self._tokenizer(query_text)
        if not query_tokens:
            return []

        raw_scores = []
        for i in range(len(self.documents)):
            raw_score = self._compute_bm25_score(query_tokens, i)
            if raw_score > 1e-9:
                raw_scores.append((raw_score, self.documents[i]))

        raw_scores.sort(key=lambda item: item[0], reverse=True)

        normalized_results = []
        for raw_score, doc in raw_scores[:k]:
            normalized_score = math.exp(-score_normalization_factor * raw_score)
            normalized_results.append((doc, normalized_score))

        normalized_results.sort(key=lambda item: item[1])

        return normalized_results

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"BM25VectorStore(count={len(self)}, k1={self.k1}, b={self.b}, index_built={self._index_built})"

with open("./report.md", "r") as f:
    text = f.read()

# 1. Chunk the text by section
chunks = chunk_by_section(text)



# 2. Create a BM25 store and add each chunk to it

store = BM25Index()
for chunk in chunks:
    store.add_document({"content": chunk})


# 3. Search the store with a query

results = store.search("What happens with INC-2023-Q4-011?",3)
for doc,distance in results:
    print(distance)
    print(doc["content"])