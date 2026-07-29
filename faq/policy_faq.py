# -*- coding: utf-8 -*-
"""Policy KB 衍生的 FAQ 索引（L3：结构化知识命中，政策直出）。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    try:
        import jieba

        return [t for t in jieba.cut(text) if t.strip()]
    except ImportError:
        return list(text)


def format_policy_direct(entry: dict[str, Any]) -> str:
    """政策原文口径回复，不走买手润色。"""
    title = entry.get("title") or entry.get("standard_q") or "平台说明"
    answer = (entry.get("answer") or entry.get("content") or "").strip()
    lines = [f"【{title}】", answer]
    actions = entry.get("actions") or []
    if actions:
        lines.append("操作建议：" + "；".join(str(a) for a in actions[:6]))
    urls = entry.get("urls") or entry.get("links") or []
    if urls:
        dedup = []
        for u in urls:
            if u and u not in dedup:
                dedup.append(u)
        if dedup:
            lines.append("相关链接：" + "；".join(dedup[:5]))
    deadlines = entry.get("deadlines") or []
    if deadlines:
        lines.append("时效：" + "；".join(str(d) for d in deadlines[:4]))
    return "\n".join(lines)


@dataclass
class FaqHit:
    faq_id: str
    score: float
    entry: dict[str, Any]
    matched_query: str


class PolicyFaqIndex:
    """published FAQ：标准问 + 同义问 BM25 检索。"""

    def __init__(
        self,
        faq_path: Path | str,
        *,
        min_score: float = 4.0,
    ):
        self.faq_path = Path(faq_path)
        self.min_score = min_score
        self.entries: list[dict[str, Any]] = []
        self._corpus: list[str] = []
        self._bm25: Any = None
        self._load()

    def _load(self) -> None:
        if not self.faq_path.exists():
            return
        with self.faq_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status", "published") != "published":
                    continue
                self.entries.append(row)
        self._corpus = [self._search_text(e) for e in self.entries]
        if self.entries and BM25_AVAILABLE:
            tokenized = [_tokenize(t) for t in self._corpus]
            self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _search_text(entry: dict[str, Any]) -> str:
        parts = [
            entry.get("standard_q", ""),
            " ".join(entry.get("paraphrases") or []),
            entry.get("title", ""),
            " ".join(entry.get("keywords") or []),
        ]
        return " ".join(p for p in parts if p)

    def match(self, query: str, top_k: int = 3) -> Optional[FaqHit]:
        if not self.entries or not query.strip():
            return None
        q = query.strip()
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(q))
            if len(scores) == 0:
                return None
            idx = int(scores.argmax())
            best = float(scores[idx])
            if best < self.min_score:
                return None
            entry = self.entries[idx]
            return FaqHit(
                faq_id=str(entry.get("faq_id", "")),
                score=best,
                entry=entry,
                matched_query=q,
            )
        # fallback: 关键词重叠
        q_lower = q.lower()
        best_i, best_s = -1, 0.0
        for i, e in enumerate(self.entries):
            text = self._search_text(e).lower()
            s = sum(1 for w in _tokenize(q_lower) if w in text)
            if s > best_s:
                best_s = s
                best_i = i
        if best_i < 0 or best_s < 2:
            return None
        return FaqHit(
            faq_id=str(self.entries[best_i].get("faq_id", "")),
            score=best_s,
            entry=self.entries[best_i],
            matched_query=q,
        )

    def count(self) -> int:
        return len(self.entries)
