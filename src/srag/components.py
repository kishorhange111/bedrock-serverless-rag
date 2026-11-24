"""Pluggable pieces: chunker, embedders, vector stores and generators.

Each AWS-backed class has an offline twin with the same interface, so the Lambda
handlers run unchanged in unit tests (with moto for S3/DynamoDB) and in production.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# --- Chunking -----------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    tenant: str
    ordinal: int
    text: str
    source: str

    @property
    def chunk_id(self) -> str:
        # Tenant-qualified: two tenants uploading "policy.txt" must never collide.
        return f"{self.tenant}/{self.doc_id}#{self.ordinal}"


def chunk_text(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    """Paragraph-aware chunking: pack whole paragraphs up to ``max_words``, split
    oversized paragraphs with a word overlap so no sentence is cut without context."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    cur: list[str] = []
    for p in paras:
        words = p.split()
        if len(words) > max_words:
            if cur:
                out.append(" ".join(cur))
                cur = []
            step = max_words - overlap
            for i in range(0, len(words), step):
                out.append(" ".join(words[i : i + max_words]))
                if i + max_words >= len(words):
                    break
            continue
        if len(cur) + len(words) > max_words and cur:
            out.append(" ".join(cur))
            cur = []
        cur += words
    if cur:
        out.append(" ".join(cur))
    return out


# --- Embeddings ---------------------------------------------------------------


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in on for at by with and or is are was were be been do does did how "
    "what when where which who why many much this that it its as from take".split()
)


def _stem(w: str) -> str:
    """Crude suffix stripping (plans→plan, refunds→refund); enough for a lexical baseline."""
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


class HashingEmbedder:
    """Offline, deterministic: signed feature hashing of unigrams + bigrams, L2-normalised.
    Lexical only (no synonyms), which makes it a conservative stand-in for evals."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            toks = [_stem(w) for w in _TOKEN.findall(t.lower()) if w not in _STOP]
            for f in toks + [a + "_" + b for a, b in zip(toks, toks[1:], strict=False)]:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                m[r, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(n, 1e-9)


class BedrockTitanEmbedder:
    """Amazon Titan Text Embeddings V2 via bedrock-runtime InvokeModel (one text per call)."""

    def __init__(
        self,
        client: Any,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dim: int = 512,
    ):
        self.client, self.model_id, self.dim = client, model_id, dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            body = json.dumps({"inputText": t[:30_000], "dimensions": self.dim, "normalize": True})
            resp = self.client.invoke_model(modelId=self.model_id, body=body)
            rows.append(json.loads(resp["body"].read())["embedding"])
        return np.asarray(rows, dtype=np.float32)


# --- Vector stores ------------------------------------------------------------


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def delete_doc(self, tenant: str, doc_id: str) -> int: ...

    def search(self, tenant: str, vector: np.ndarray, k: int) -> list[Hit]: ...


@dataclass
class InMemoryVectorStore:
    rows: dict[str, tuple[Chunk, np.ndarray]] = field(default_factory=dict)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        for c, v in zip(chunks, vectors, strict=True):
            self.rows[c.chunk_id] = (c, v)

    def delete_doc(self, tenant: str, doc_id: str) -> int:
        dead = [k for k, (c, _) in self.rows.items() if c.doc_id == doc_id and c.tenant == tenant]
        for k in dead:
            del self.rows[k]
        return len(dead)

    def search(self, tenant: str, vector: np.ndarray, k: int) -> list[Hit]:
        # The tenant filter is applied *before* ranking, never after (no cross-tenant top-k).
        cand = [(c, v) for c, v in self.rows.values() if c.tenant == tenant]
        scored = sorted(((float(v @ vector), c) for c, v in cand), key=lambda x: -x[0])[:k]
        return [Hit(c.chunk_id, c.doc_id, c.text, c.source, s) for s, c in scored]


def opensearch_knn_query(tenant: str, vector: np.ndarray, k: int) -> dict:
    """k-NN query with an *efficient filter* (pre-filtering inside the HNSW search) on tenant."""
    return {
        "size": k,
        "_source": ["chunk_id", "doc_id", "text", "source"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": [float(x) for x in vector],
                    "k": k,
                    "filter": {"term": {"tenant": tenant}},
                }
            }
        },
    }


def opensearch_index_body(dim: int) -> dict:
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "innerproduct",
                    },
                },
                "tenant": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "text": {"type": "text"},
            }
        },
    }


class OpenSearchVectorStore:
    """OpenSearch Serverless (vector search collection) via opensearch-py + SigV4."""

    def __init__(self, client: Any, index: str):
        self.client, self.index = client, index

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from opensearchpy import helpers

        actions = [
            {
                "_op_type": "index",
                "_index": self.index,
                "_source": {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "tenant": c.tenant,
                    "source": c.source,
                    "text": c.text,
                    "embedding": [float(x) for x in v],
                },
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        helpers.bulk(self.client, actions)

    def delete_doc(self, tenant: str, doc_id: str) -> int:
        q = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"tenant": tenant}},
                        {"term": {"doc_id": doc_id}},
                    ]
                }
            }
        }
        return int(self.client.delete_by_query(index=self.index, body=q).get("deleted", 0))

    def search(self, tenant: str, vector: np.ndarray, k: int) -> list[Hit]:
        res = self.client.search(index=self.index, body=opensearch_knn_query(tenant, vector, k))
        return [
            Hit(
                h["_source"]["chunk_id"],
                h["_source"]["doc_id"],
                h["_source"]["text"],
                h["_source"]["source"],
                float(h["_score"]),
            )
            for h in res["hits"]["hits"]
        ]


# --- Generators ---------------------------------------------------------------


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class Generator(Protocol):
    def generate(self, system: str, prompt: str, max_tokens: int) -> Generation: ...


class ClaudeOnBedrock:
    """Claude via the Anthropic SDK's Bedrock client (IAM auth, no API key in the Lambda)."""

    def __init__(self, model_id: str, region: str | None = None):
        from anthropic import AnthropicBedrock

        self.client = AnthropicBedrock(aws_region=region) if region else AnthropicBedrock()
        self.model_id = model_id

    def generate(self, system: str, prompt: str, max_tokens: int) -> Generation:
        msg = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Generation(text, msg.usage.input_tokens, msg.usage.output_tokens, self.model_id)


_SENT = re.compile(r"(?<=[.!?])\s+")


class ExtractiveGenerator:
    """Offline stand-in: answers with the source sentence that best overlaps the question,
    cited as [n]. Deterministic, so handler tests and the eval need no model access."""

    model = "extractive-offline"

    def generate(self, system: str, prompt: str, max_tokens: int) -> Generation:
        q = set(_TOKEN.findall(prompt.split("QUESTION:")[-1].lower()))
        best, best_s = None, 0.0
        for m in re.finditer(r"^\[(\d+)\] (.+)$", prompt, flags=re.M):
            for s in _SENT.split(m.group(2)):
                w = set(_TOKEN.findall(s.lower()))
                score = len(q & w) / math.sqrt(len(w) + 1)
                if score > best_s:
                    best, best_s = f"{s.strip()} [{m.group(1)}]", score
        text = best or "I don't know based on the provided documents."
        return Generation(text, len(prompt) // 4, len(text) // 4, self.model)
