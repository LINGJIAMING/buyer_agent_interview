#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 LLM 为 faq_published 补充同义问（B），输出 faq_draft 供买手审核。

  set DEEPSEEK_API_KEY=...
  py -3.9 scripts/expand_faq_paraphrases.py --limit 5

审核通过后合并 paraphrases 到 faq_published.jsonl。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def expand_one(client, model: str, entry: dict) -> list[str]:
    title = entry.get("title", "")
    std = entry.get("standard_q", "")
    existing = entry.get("paraphrases") or []
    prompt = (
        "你是 Temu 买手运营助手。根据下列标准问和关键词，生成 5 条商家可能用的口语化同义问。"
        "只输出 JSON 数组，不要其它文字。\n"
        f"标准问：{std}\n标题：{title}\n关键词：{', '.join(entry.get('keywords') or [])}\n"
        f"已有同义问（勿重复）：{existing}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "只输出 JSON 字符串数组。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    text = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    arr = json.loads(m.group(0))
    return [str(x).strip() for x in arr if str(x).strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=PROJECT_ROOT / "data" / "faq_published.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "faq_draft_paraphrases.jsonl",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    _load_dotenv()
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("需要 DEEPSEEK_API_KEY")

    from openai import OpenAI

    client = OpenAI(
        api_key=key,
        base_url=os.getenv("API_BASE_URL", "https://api.deepseek.com"),
    )
    model = os.getenv("API_MODEL", "deepseek-chat")

    rows_out = []
    with args.inp.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    n = len(lines) if args.limit is None else min(args.limit, len(lines))
    for i, line in enumerate(lines[:n]):
        entry = json.loads(line)
        try:
            extra = expand_one(client, model, entry)
        except Exception as exc:
            extra = []
            entry["expand_error"] = str(exc)
        merged = list(dict.fromkeys((entry.get("paraphrases") or []) + extra))
        draft = dict(entry)
        draft["status"] = "draft"
        draft["paraphrases"] = merged
        draft["paraphrases_llm"] = extra
        rows_out.append(draft)
        print(f"[{i+1}/{n}] {entry.get('faq_id')} +{len(extra)} paraphrases")
        time.sleep(args.sleep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
