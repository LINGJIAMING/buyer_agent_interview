# -*- coding: utf-8 -*-
"""混合检索：BM25 + Dense + RRF + 可选 Rerank。"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from rag.constants import CHUNKS_JSONL, INDEX_DIR


def _tokenize(text: str) -> list:
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return list(text)


def rrf_merge(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class SopHybridRetriever:
    def __init__(
        self,
        index_dir: Optional[Path] = None,
        chunks_path: Optional[Path] = None,
        bm25_only: Optional[bool] = None,
    ):
        self.index_dir = Path(index_dir or INDEX_DIR)
        self.chunks_path = Path(chunks_path or CHUNKS_JSONL)
        if bm25_only is None:
            bm25_only = os.getenv("RAG_BM25_ONLY", "").lower() in ("1", "true", "yes")
        self.bm25_only = bm25_only
        self.chunks: list[dict] = []
        self.chunk_by_id: dict[str, dict] = {}
        self.bm25 = None
        self.bm25_texts: list[str] = []
        self.embed_model = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_ids: list[str] = []
        self.faiss_index = None
        self._load()

    def _load(self):
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                self.chunks.append(c)
                self.chunk_by_id[c["chunk_id"]] = c

        with (self.index_dir / "bm25_model.pkl").open("rb") as f:
            self.bm25 = pickle.load(f)
        with (self.index_dir / "bm25_corpus.pkl").open("rb") as f:
            data = pickle.load(f)
            self.bm25_texts = data["texts"]

        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        map_path = self.index_dir / "chunk_id_map.json"
        if map_path.exists():
            with map_path.open("r", encoding="utf-8") as f:
                self.chunk_ids = json.load(f)

        if self.bm25_only:
            return

        emb_path = self.index_dir / "embeddings.npy"
        if not emb_path.exists():
            return

        self.embeddings = np.load(emb_path)
        faiss_path = self.index_dir / "faiss.index"
        if faiss_path.exists():
            import faiss
            self.faiss_index = faiss.read_index(str(faiss_path))

        local_model = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
        meta_path = self.index_dir / "embedding_meta.json"
        model_name = "BAAI/bge-small-zh-v1.5"
        if meta_path.exists():
            model_name = json.loads(meta_path.read_text(encoding="utf-8")).get("model", model_name)

        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(local_model or model_name)
        except Exception as e:
            print(f"[WARN] Dense 检索不可用（无法加载 embedding 模型）: {e}")
            self.embeddings = None
            self.faiss_index = None

    def _bm25_search(self, query: str, top_k: int) -> list[str]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunk_ids[i] for i in idx if scores[i] > 0]

    def _dense_search(self, query: str, top_k: int) -> list[str]:
        if self.embed_model is None or self.embeddings is None:
            return []
        q = self.embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        if self.faiss_index is not None:
            _, I = self.faiss_index.search(q, top_k)
            return [self.chunk_ids[i] for i in I[0] if i >= 0]
        sims = (self.embeddings @ q.T).ravel()
        idx = np.argsort(sims)[::-1][:top_k]
        return [self.chunk_ids[i] for i in idx]

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
        kb_lane: Optional[str] = None,
        scene: Optional[str] = None,
        use_rerank: bool = True,
    ) -> list[dict]:
        bm25_ids = self._bm25_search(query, candidate_k)
        dense_ids = self._dense_search(query, candidate_k)
        merged = rrf_merge([bm25_ids, dense_ids])
        candidates = [cid for cid, _ in merged[:candidate_k]]

        results = []
        for cid in candidates:
            c = dict(self.chunk_by_id[cid])
            if kb_lane and c.get("kb_lane") != kb_lane:
                c["_lane_penalty"] = True
            if scene and c.get("scene") != scene:
                c["_scene_penalty"] = True
            results.append(c)

        if use_rerank and results:
            results = self._rerank(query, results, top_k)
        else:
            results = results[:top_k]

        return results

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return candidates[:top_k]

        model_name = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")
        try:
            reranker = CrossEncoder(model_name)
        except Exception:
            return candidates[:top_k]

        pairs = [(query, c.get("search_text") or c.get("content", "")) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
        out = []
        for c, s in ranked[:top_k]:
            c = dict(c)
            c["rerank_score"] = float(s)
            out.append(c)
        return out
