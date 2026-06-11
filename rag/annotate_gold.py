#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式标注 gold_chunk_ids。

用法:
  python -m rag.annotate_gold
  python -m rag.annotate_gold --only-empty
  python -m rag.annotate_gold --id rag_eval_001
  python -m rag.annotate_gold --top 15 --bm25-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from rag.constants import CHUNKS_JSONL, EVAL_DIR
from rag.hybrid_search import SopHybridRetriever

EVAL_PATH = EVAL_DIR / "rag_eval_50.jsonl"


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def save_cases(path: Path, cases: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def _preview(text: str, n: int = 120) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return t if len(t) <= n else t[: n - 3] + "..."


def _parse_selection(raw: str, max_n: int) -> list[int]:
    raw = raw.strip().lower()
    if not raw:
        return []
    parts = re.split(r"[,，\s]+", raw)
    out = []
    for p in parts:
        if not p:
            continue
        if not p.isdigit():
            raise ValueError(f"无效编号: {p}")
        i = int(p)
        if i < 1 or i > max_n:
            raise ValueError(f"编号越界: {i} (1-{max_n})")
        out.append(i)
    return sorted(set(out))


def keyword_search_chunks(keyword: str, chunks_path: Path, limit: int = 10) -> list[dict]:
    kw = keyword.lower()
    hits = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            blob = " ".join([
                c.get("source_file", ""),
                c.get("title", ""),
                c.get("content", ""),
            ]).lower()
            if kw in blob:
                hits.append(c)
                if len(hits) >= limit:
                    break
    return hits


def _print_candidates(candidates: list[dict]) -> None:
    for i, c in enumerate(candidates, start=1):
        print(f"\n  [{i}] chunk_id: {c.get('chunk_id')}")
        print(f"      文件: {c.get('source_file')}  页/Slide: {c.get('page_or_slide')}")
        print(f"      标题: {c.get('title', '')[:60]}")
        print(f"      片段: {_preview(c.get('content', ''), 160)}")


def annotate_interactive(
    eval_path: Path = EVAL_PATH,
    only_empty: bool = False,
    case_id: Optional[str] = None,
    top_k: int = 10,
    bm25_only: bool = False,
    use_rerank: bool = False,
) -> dict:
    cases = load_cases(eval_path)
    retriever = SopHybridRetriever(bm25_only=bm25_only)

    annotated = 0
    skipped = 0

    print("=" * 60)
    print("RAG gold_chunk_ids 交互标注")
    print("=" * 60)
    print("命令: 输入编号如 1,2 确认 | s 跳过 | k 关键词搜索 | q 保存并退出")
    print(f"评测集: {eval_path}  共 {len(cases)} 题\n")

    for idx, case in enumerate(cases):
        cid = case.get("id", f"case_{idx}")
        if case_id and cid != case_id:
            continue
        gold = case.get("gold_chunk_ids") or []
        if only_empty and gold:
            continue

        print("\n" + "=" * 60)
        print(f"[{cid}] ({idx + 1}/{len(cases)})  topic={case.get('topic', '')}")
        print(f"Q: {case.get('question', '')}")
        print(f"参考答案: {_preview(case.get('reference_answer', ''), 200)}")
        print(f"hints: {case.get('expected_doc_hints', [])}")
        if gold:
            print(f"当前 gold: {gold}")

        # 标注阶段不按 kb_lane 过滤，避免错分 lane 漏掉正确 chunk
        candidates = retriever.search(
            case["question"],
            top_k=top_k,
            kb_lane=None,
            scene=None,
            use_rerank=use_rerank,
        )

        if not candidates:
            print("\n  (检索无结果，可用 k 关键词搜索)")
        else:
            print(f"\n--- 检索 Top-{len(candidates)} ---")
            _print_candidates(candidates)

        while True:
            try:
                raw = input("\n选择 gold 编号 (s/k/q): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n中断，正在保存...")
                save_cases(eval_path, cases)
                return {"saved": True, "annotated": annotated, "skipped": skipped}

            cmd = raw.lower()
            if cmd in ("q", "quit", "exit"):
                save_cases(eval_path, cases)
                print(f"已保存到 {eval_path}")
                return {"saved": True, "annotated": annotated, "skipped": skipped}
            if cmd in ("s", "skip", ""):
                skipped += 1
                break
            if cmd.startswith("k"):
                kw = raw[1:].strip() or (case.get("expected_doc_hints") or [""])[0]
                if cmd == "k" or cmd == "k ":
                    kw = input("搜索关键词: ").strip()
                extra = keyword_search_chunks(kw, CHUNKS_JSONL, limit=top_k)
                if not extra:
                    print(f"  未找到含「{kw}」的 chunk")
                    continue
                candidates = extra
                print(f"\n--- 关键词「{kw}」命中 ---")
                _print_candidates(candidates)
                continue

            try:
                picks = _parse_selection(raw, len(candidates))
            except ValueError as e:
                print(f"  {e}")
                continue

            if not picks:
                print("  未选择任何编号")
                continue

            ids = []
            for p in picks:
                chunk_id = candidates[p - 1].get("chunk_id")
                if chunk_id and chunk_id not in ids:
                    ids.append(chunk_id)

            case["gold_chunk_ids"] = ids
            annotated += 1
            print(f"  ✅ 已标注: {ids}")
            break

    save_cases(eval_path, cases)
    print(f"\n全部完成，已保存 {eval_path}")
    return {"saved": True, "annotated": annotated, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser(description="交互标注 rag_eval gold_chunk_ids")
    ap.add_argument("--eval", default=str(EVAL_PATH), help="评测集 jsonl 路径")
    ap.add_argument("--only-empty", action="store_true", help="只标注 gold_chunk_ids 为空的题")
    ap.add_argument("--id", dest="case_id", default=None, help="只标注指定 id，如 rag_eval_001")
    ap.add_argument("--top", type=int, default=10, help="展示检索候选数")
    ap.add_argument("--bm25-only", action="store_true", help="仅用 BM25（无需下载 embedding 模型）")
    ap.add_argument("--rerank", action="store_true", help="启用 rerank")
    args = ap.parse_args()

    stats = annotate_interactive(
        eval_path=Path(args.eval),
        only_empty=args.only_empty,
        case_id=args.case_id,
        top_k=args.top,
        bm25_only=args.bm25_only,
        use_rerank=args.rerank,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
