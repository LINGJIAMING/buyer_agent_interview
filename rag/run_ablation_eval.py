# -*- coding: utf-8 -*-
"""消融实验：BM25 / Hybrid / +Rerank，仅评测已标注 gold 的样本。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rag.constants import DATA_DIR, EVAL_DIR
from rag.run_eval_recall import EVAL_PATH, _hit_gold, evaluate

REPORT_JSON = DATA_DIR / "ablation_results.json"


def run_ablation(top_k: int = 5, include_hybrid: bool = False) -> dict:
    configs = [
        {"name": "E0_bm25_only", "bm25_only": True, "use_rerank": False},
    ]
    if include_hybrid:
        configs.extend([
            {"name": "E1_hybrid_rrf", "bm25_only": False, "use_rerank": False},
            {"name": "E2_hybrid_rerank", "bm25_only": False, "use_rerank": True},
        ])
    rows = []
    for cfg in configs:
        try:
            r = evaluate(
                top_k=top_k,
                bm25_only=cfg["bm25_only"],
                use_rerank=cfg["use_rerank"],
                use_gold=True,
            )
            rows.append({
                "experiment": cfg["name"],
                **{k: r[k] for k in r if k != "misses_sample"},
                "misses": [m["id"] for m in r.get("misses_sample", [])],
            })
        except Exception as e:
            rows.append({
                "experiment": cfg["name"],
                "error": str(e),
            })

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "top_k": top_k,
        "note": "仅统计 gold_chunk_ids 非空的样本",
        "results": rows,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--include-hybrid",
        action="store_true",
        help="同时跑 E1/E2（需本地 embedding，可能较慢）",
    )
    args = ap.parse_args()
    print(json.dumps(
        run_ablation(args.top_k, include_hybrid=args.include_hybrid),
        ensure_ascii=False,
        indent=2,
    ))
