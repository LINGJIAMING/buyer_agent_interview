# -*- coding: utf-8 -*-
"""
双库检索：policy_kb（Retriever）+ SOP chunk（SopHybridRetriever）。

返回与 retriever.Retriever.retrieve_context 相同结构，供 app / app_api 直接替换。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from retriever import Retriever
from rag.constants import INDEX_DIR, PARENTS_JSONL
from rag.hybrid_search import SopHybridRetriever

# 走 SOP 双库 merge 的场景
DEFAULT_SOP_SCENES = frozenset({
    "policy", "product", "activity", "inventory", "general",
})


def _load_parents(path: Path) -> dict[str, dict]:
    parents: dict[str, dict] = {}
    if not path.exists():
        return parents
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                parents[p["parent_id"]] = p
    return parents


def _format_sop_evidence(hit: dict, parent: Optional[dict], idx: int) -> str:
    src = hit.get("source_file", "")
    page = hit.get("page_or_slide", "")
    title = hit.get("title", "")
    body = (parent or hit).get("content", hit.get("content", ""))
    max_len = 900 if parent else 500
    snippet = body[:max_len] + ("..." if len(body) > max_len else "")
    return (
        f"【检索证据 S{idx}】来源: SOP/{src} (页{page})\n"
        f"标题: {title}\n"
        f"{snippet}"
    )


class MergedRetriever:
    """policy_kb + SOP 混合检索，Evidence 用 parent 正文。"""

    def __init__(
        self,
        policy_kb_path: str,
        sop_scenes: Optional[frozenset] = None,
        sop_top_k: int = 3,
        use_rerank: Optional[bool] = None,
        bm25_only: Optional[bool] = None,
    ):
        self.policy = Retriever(policy_kb_path)
        self.sop_scenes = sop_scenes or DEFAULT_SOP_SCENES
        self.sop_top_k = sop_top_k
        self.use_rerank = (
            use_rerank
            if use_rerank is not None
            else os.getenv("RAG_USE_RERANK", "false").lower() in ("1", "true", "yes")
        )
        self.parents = _load_parents(PARENTS_JSONL)
        self.sop: Optional[SopHybridRetriever] = None
        if (INDEX_DIR / "bm25_model.pkl").exists():
            try:
                if bm25_only is None:
                    bm25_only = os.getenv("RAG_BM25_ONLY", "true").lower() in (
                        "1", "true", "yes",
                    )
                self.sop = SopHybridRetriever(bm25_only=bm25_only)
            except Exception as e:
                print(f"[MergedRetriever] SOP 索引不可用: {e}")

    def retrieve_context(
        self,
        user_input: str,
        scene: Optional[str] = None,
        top_k: int = 3,
    ) -> dict:
        policy_result = self.policy.retrieve_context(
            user_input, scene=scene, top_k=top_k,
        )

        use_sop = (
            self.sop is not None
            and (scene or "general") in self.sop_scenes
        )
        sop_hits: list[dict] = []
        if use_sop:
            sop_hits = self.sop.search(
                user_input,
                top_k=self.sop_top_k,
                kb_lane=None,
                scene=None,
                use_rerank=self.use_rerank,
            )

        if not sop_hits:
            policy_result["retrieval_sources"] = ["policy_kb"]
            return policy_result

        evidence_blocks: list[str] = []
        if (
            policy_result.get("context")
            and policy_result["context"] != "未检索到明确相关的政策资料。"
        ):
            evidence_blocks.append(
                f"【检索证据 P1】来源: policy_kb\n{policy_result['context']}"
            )

        for i, hit in enumerate(sop_hits, start=1):
            parent = self.parents.get(hit.get("parent_id", ""))
            evidence_blocks.append(_format_sop_evidence(hit, parent, i))

        merged_context = "\n\n".join(evidence_blocks)

        sop_strong = bool(sop_hits) and (
            sop_hits[0].get("rerank_score", 0) > 0.5
            or not self.use_rerank
        )
        strong_hit = policy_result.get("strong_hit", False) or sop_strong
        low_confidence = (
            not strong_hit
            and not policy_result.get("strong_hit")
            and policy_result.get("low_confidence", True)
            and len(sop_hits) < 1
        )

        score = max(
            float(policy_result.get("score", 0)),
            float(sop_hits[0].get("rerank_score", 0)) if sop_hits else 0.0,
        )

        return {
            "context": merged_context,
            "score": score,
            "title": sop_hits[0].get("title") if sop_hits else policy_result.get("title", ""),
            "title_exact_match": policy_result.get("title_exact_match", False),
            "strong_hit": strong_hit,
            "retrieval_method": "merged_policy_sop",
            "candidates": policy_result.get("candidates", [])
            + [
                {
                    "title": h.get("source_file", ""),
                    "score": float(h.get("rerank_score", 0)),
                    "scene": h.get("scene", ""),
                    "chunk_id": h.get("chunk_id", ""),
                }
                for h in sop_hits
            ],
            "low_confidence": low_confidence,
            "follow_up_question": policy_result.get("follow_up_question", ""),
            "retrieval_sources": ["policy_kb", "sop_rag"],
            "sop_chunk_ids": [h.get("chunk_id") for h in sop_hits],
        }
