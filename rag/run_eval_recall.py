# -*- coding: utf-8 -*-
"""离线 Recall@K：用 expected_doc_hints 匹配 source_file / title。"""
from __future__ import annotations

import json
from pathlib import Path

from rag.constants import EVAL_DIR
from rag.hybrid_search import SopHybridRetriever

EVAL_PATH = EVAL_DIR / "rag_eval_50.jsonl"


def _hit_hints(result: dict, hints: list[str]) -> bool:
    blob = " ".join([
        result.get("source_file", ""),
        result.get("title", ""),
        result.get("content", "")[:200],
    ]).lower()
    for h in hints:
        if h.lower() in blob or h in blob:
            return True
    return False


def _hit_gold(result: dict, gold_ids: list[str]) -> bool:
    cid = result.get("chunk_id", "")
    return cid in gold_ids


def evaluate(
    top_k: int = 5,
    use_rerank: bool = True,
    bm25_only: bool = False,
    use_gold: bool = True,
) -> dict:
    retriever = SopHybridRetriever(bm25_only=bm25_only)
    cases = []
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    gold_labeled = sum(1 for c in cases if c.get("gold_chunk_ids"))
    mode = "gold_chunk_ids" if (use_gold and gold_labeled > 0) else "hints"

    hit = 0
    misses = []
    skipped_no_gold = 0
    for c in cases:
        gold_ids = c.get("gold_chunk_ids") or []
        hints = c.get("expected_doc_hints", [])
        results = retriever.search(
            c["question"],
            top_k=top_k,
            kb_lane=c.get("kb_lane"),
            scene=None,
            use_rerank=use_rerank,
        )
        if mode == "gold_chunk_ids":
            if not gold_ids:
                skipped_no_gold += 1
                continue
            ok = any(_hit_gold(r, gold_ids) for r in results)
        else:
            ok = any(_hit_hints(r, hints) for r in results)

        if ok:
            hit += 1
        else:
            misses.append({
                "id": c["id"],
                "question": c["question"],
                "gold_chunk_ids": gold_ids,
                "top_chunk_id": results[0].get("chunk_id") if results else None,
                "top_source": results[0].get("source_file") if results else None,
            })

    denom = len(cases) if mode == "hints" else (len(cases) - skipped_no_gold)
    return {
        "total": len(cases),
        "eval_mode": mode,
        "gold_labeled_count": gold_labeled,
        "skipped_no_gold": skipped_no_gold,
        "hit": hit,
        f"recall_at_{top_k}": round(hit / denom, 4) if denom else 0,
        "use_rerank": use_rerank,
        "bm25_only": bm25_only,
        "misses_sample": misses[:10],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--hints-only", action="store_true", help="强制用 expected_doc_hints，忽略 gold")
    args = ap.parse_args()
    r = evaluate(
        top_k=args.top_k,
        use_rerank=not args.no_rerank,
        bm25_only=args.bm25_only,
        use_gold=not args.hints_only,
    )
    print(json.dumps(r, ensure_ascii=False, indent=2))
