# -*- coding: utf-8 -*-
"""从 api_chat 日志做简单频次聚类，供买手审核补 FAQ（第 35 章飞轮 · 采集/标注）。"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = PROJECT_ROOT / "log" / "api_chat" / "api_chat.jsonl"


def _norm(q: str) -> str:
    t = (q or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\u4e00-\u9fff]+", "", t)
    return t[:80]


def cluster_frequent_queries(
    log_path: Path | str = DEFAULT_LOG,
    *,
    min_count: int = 2,
    top_n: int = 30,
    paths_no_hit: tuple[str, ...] = ("llm_api", "llm_api_skill"),
) -> list[dict[str, Any]]:
    """按归一化问法聚类，输出高频候选（待买手审核）。"""
    path = Path(log_path)
    if not path.exists():
        return []
    buckets: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = row.get("raw_input") or row.get("working_query") or ""
            key = _norm(raw)
            if len(key) < 4:
                continue
            buckets[key].append(row)

    out = []
    for key, rows in buckets.items():
        if len(rows) < min_count:
            continue
        sample = rows[-1]
        paths = Counter(r.get("path", "") for r in rows)
        out.append(
            {
                "norm_query": key,
                "count": len(rows),
                "sample_input": sample.get("raw_input", ""),
                "paths": dict(paths),
                "suggest": "faq_candidate",
            }
        )
    out.sort(key=lambda x: x["count"], reverse=True)
    return out[:top_n]


def export_flywheel_report(
    log_path: Path | str = DEFAULT_LOG,
    output_md: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(log_path)
    total = 0
    by_path: Counter = Counter()
    low_rated_proxy = 0  # path=business_api 误触发可作改进信号
    cache_hits = 0
    faq_hits = 0

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                by_path[row.get("path", "unknown")] += 1
                if row.get("path") == "business_api":
                    low_rated_proxy += 1
                if row.get("cache_level"):
                    cache_hits += 1
                if row.get("path") in ("faq_direct", "cache_policy_direct"):
                    faq_hits += 1

    clusters = cluster_frequent_queries(log_path)
    report = {
        "total_turns": total,
        "by_path": dict(by_path),
        "cache_or_faq_hits": cache_hits + faq_hits,
        "business_api_turns": low_rated_proxy,
        "top_faq_candidates": clusters[:15],
    }

    if output_md:
        out = Path(output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# 数据飞轮周报（自动生成）",
            "",
            f"- 会话轮次：{total}",
            f"- 路径分布：{json.dumps(dict(by_path), ensure_ascii=False)}",
            f"- 缓存/FAQ 直出命中：{cache_hits + faq_hits}",
            "",
            "## 待买手审核的高频问法",
            "",
        ]
        for c in clusters[:15]:
            lines.append(
                f"- （{c['count']}次）`{c['sample_input'][:60]}`"
            )
        lines.append("")
        lines.append("> 审核通过后写入 `data/faq_draft.jsonl`，再发布到 `faq_published.jsonl`。")
        out.write_text("\n".join(lines), encoding="utf-8")

    return report
