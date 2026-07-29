# -*- coding: utf-8 -*-
"""L1 精确 + L2 语义缓存（仅缓存 policy_direct 类回复，见第 28 章）。"""
from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_FILE = PROJECT_ROOT / "data" / "response_cache.jsonl"


def normalize_query(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


class ExactMatchCache:
    def __init__(self, max_size: int = 800):
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Optional[dict[str, Any]]:
        key = hashlib.md5(normalize_query(query).encode("utf-8")).hexdigest()
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def set(self, query: str, payload: dict[str, Any]) -> None:
        key = hashlib.md5(normalize_query(query).encode("utf-8")).hexdigest()
        if len(self._store) >= self.max_size:
            self._store.popitem(last=False)
        self._store[key] = payload


class SemanticCache:
    """字符哈希向量 + 余弦相似度（与课程 demo 一致，无额外 API）。"""

    def __init__(self, similarity_threshold: float = 0.88, max_size: int = 1500):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self._entries: list[tuple[np.ndarray, str, dict[str, Any]]] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _embed(text: str, dim: int = 128) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float64)
        for ch in normalize_query(text):
            vec[hash(ch) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def get(self, query: str) -> Optional[dict[str, Any]]:
        q_vec = self._embed(query)
        best_sim = 0.0
        best_payload: Optional[dict[str, Any]] = None
        for emb, _cached_q, payload in self._entries:
            sim = float(np.dot(q_vec, emb))
            if sim > best_sim:
                best_sim = sim
                best_payload = payload
        if best_payload is not None and best_sim >= self.threshold:
            self.hits += 1
            out = dict(best_payload)
            out["cache_sim"] = round(best_sim, 4)
            return out
        self.misses += 1
        return None

    def set(self, query: str, payload: dict[str, Any]) -> None:
        emb = self._embed(query)
        self._entries.append((emb, normalize_query(query), payload))
        if len(self._entries) > self.max_size:
            self._entries = self._entries[-self.max_size :]


class BuyerResponseCache:
    """L1 → L2；仅 policy_direct 写入缓存。"""

    def __init__(
        self,
        *,
        semantic_threshold: float = 0.88,
        persist_path: Optional[Path] = None,
        load_persist: bool = True,
    ):
        self.l1 = ExactMatchCache()
        self.l2 = SemanticCache(similarity_threshold=semantic_threshold)
        self.persist_path = persist_path or DEFAULT_CACHE_FILE
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0, "writes": 0}
        if load_persist:
            self._load_persist()

    def lookup(self, query: str) -> Optional[dict[str, Any]]:
        hit = self.l1.get(query)
        if hit:
            self.stats["l1_hits"] += 1
            hit = dict(hit)
            hit["cache_level"] = 1
            return hit
        hit = self.l2.get(query)
        if hit:
            self.stats["l2_hits"] += 1
            hit = dict(hit)
            hit["cache_level"] = 2
            return hit
        self.stats["misses"] += 1
        return None

    def remember_policy_direct(
        self,
        query: str,
        response: str,
        *,
        faq_id: str = "",
        source: str = "faq",
    ) -> None:
        payload = {
            "response": response,
            "response_mode": "policy_direct",
            "faq_id": faq_id,
            "source": source,
        }
        self.l1.set(query, payload)
        self.l2.set(query, payload)
        self.stats["writes"] += 1
        self._append_persist(query, payload)

    def _append_persist(self, query: str, payload: dict[str, Any]) -> None:
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "query_norm": normalize_query(query),
                "query": query[:500],
                **payload,
            }
            with self.persist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _load_persist(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with self.persist_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    q = row.get("query") or row.get("query_norm") or ""
                    if not q or not row.get("response"):
                        continue
                    payload = {
                        "response": row["response"],
                        "response_mode": row.get("response_mode", "policy_direct"),
                        "faq_id": row.get("faq_id", ""),
                        "source": row.get("source", "persist"),
                    }
                    self.l1.set(q, payload)
                    self.l2.set(q, payload)
        except (OSError, json.JSONDecodeError):
            pass

    def report(self) -> dict[str, Any]:
        total_hits = self.stats["l1_hits"] + self.stats["l2_hits"]
        total = total_hits + self.stats["misses"]
        return {
            **self.stats,
            "lookup_total": total,
            "hit_rate": round(total_hits / total, 4) if total else 0.0,
            "l1_size": len(self.l1._store),
            "l2_size": len(self.l2._entries),
        }

    def invalidate_all(self) -> None:
        self.l1 = ExactMatchCache()
        self.l2 = SemanticCache(similarity_threshold=self.l2.threshold)
        if self.persist_path.exists():
            self.persist_path.unlink(missing_ok=True)
