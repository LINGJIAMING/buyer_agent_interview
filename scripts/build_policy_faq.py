#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 kb/policy_kb.jsonl 生成 FAQ（买手已审 = published）。

  py -3.9 scripts/build_policy_faq.py
  py -3.9 scripts/build_policy_faq.py --out data/faq_published.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = PROJECT_ROOT / "kb" / "policy_kb.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "faq_published.jsonl"


def _slug(title: str) -> str:
    s = re.sub(r"\s+", "_", title.strip())[:40]
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", s) or "faq"


def _standard_questions(title: str) -> list[str]:
    t = title.strip()
    return list(
        dict.fromkeys(
            [
                f"{t}是什么",
                f"什么是{t}",
                f"{t}有什么要求",
                f"有没有关于{t}的政策",
                t,
            ]
        )
    )


def _rule_paraphrases(title: str, keywords: list[str]) -> list[str]:
    """规则同义问（B 的基线；LLM 扩写见 expand_faq_paraphrases.py）。"""
    out: list[str] = []
    for kw in keywords[:8]:
        kw = kw.strip()
        if len(kw) < 2:
            continue
        out.extend(
            [
                kw,
                f"{kw}怎么弄",
                f"{kw}有什么规定",
                f"咨询{kw}",
            ]
        )
    if "问卷" in title:
        out.append(f"{title}入口")
        out.append(f"{title}链接")
    if "水洗" in title or "洗水" in title:
        out.extend(["水洗唛要求", "洗水唛怎么拍", "洗标合规"])
    if "JIT" in title.upper() or "jit" in title.lower():
        out.extend(["jit转备货", "发货率10%", "jit发货率要求"])
    return list(dict.fromkeys(out))[:20]


def policy_row_to_faq(row: dict, *, status: str = "published") -> dict:
    title = row.get("title", "").strip()
    keywords = row.get("keywords") or []
    std = _standard_questions(title)
    paraphrases = _rule_paraphrases(title, keywords)
    # 去掉与 standard_q 重复
    paraphrases = [p for p in paraphrases if p not in std]

    urls = row.get("urls") or row.get("links") or []
    return {
        "faq_id": f"faq_{_slug(title)}",
        "type": "faq",
        "status": status,
        "version": 1,
        "policy_source_id": title,
        "title": title,
        "category": row.get("category", ""),
        "standard_q": std[0],
        "paraphrases": paraphrases,
        "keywords": keywords,
        "answer": row.get("content", ""),
        "actions": row.get("actions") or [],
        "urls": urls,
        "deadlines": row.get("deadlines") or [],
        "response_mode": "policy_direct",
        "reviewed_by": "buyer_bootstrap",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--status", default="published")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"源文件不存在: {args.src}")

    faqs = []
    with args.src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                faqs.append(policy_row_to_faq(json.loads(line), status=args.status))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in faqs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(faqs)} FAQ rows -> {args.out}")


if __name__ == "__main__":
    main()
