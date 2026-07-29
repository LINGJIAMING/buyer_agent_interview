# -*- coding: utf-8 -*-
"""Extract prompt-text blocks from 全品类.html into prompts.yaml skeleton."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

SRC = Path(r"d:\xwechat_files\wxid_ye3di3uhib4u21_d9d2\msg\file\2026-07\全品类.html")
DST_ROOT = Path(r"c:\Users\admin\Desktop\mybuyer_agent-main\skills\menswear_full_category")
STATIC = DST_ROOT / "static"
YAML_PATH = DST_ROOT / "prompts.yaml"


def main() -> None:
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, STATIC / "全品类.html")

    html = SRC.read_text(encoding="utf-8")
    pattern = re.compile(
        r'id="(prompt-text-\d+)">(.*?)</div>',
        re.DOTALL,
    )
    items = []
    for m in pattern.finditer(html):
        pid = m.group(1)
        body = m.group(2).strip()
        # strip nested tags if any
        body = re.sub(r"<[^>]+>", "", body).strip()
        before = html[: m.start()]
        titles = re.findall(
            r'card-title[^>]*>.*?<span>([^<]+)</span>\s*</div>',
            before,
            re.DOTALL,
        )
        title = titles[-1].strip() if titles else pid
        needs_image = any(
            k in body for k in ("上传", "图片", "主图", "服装图片", "看图")
        )
        items.append(
            {
                "id": pid.replace("prompt-text-", "p"),
                "html_id": pid,
                "title": title,
                "category": _guess_category(title, body),
                "needs_image": needs_image,
                "body": body,
            }
        )

    lines = [
        "# TEMU 男装全品类买手分析 · Prompt 模板库",
        "# 真相源：本文件；static/全品类.html 为复制用 UI（内容应对齐本文件）",
        "# 状态：V0 骨架 — 未接入 Router，仅 Skill 包 + Web 侧链展示",
        "",
        "skill:",
        "  id: menswear_full_category",
        "  name: TEMU男装全品类买手AI分析",
        "  version: 0.1.0",
        "  description: 选款/爆款/标题/主图等 Prompt 模板（商家分析用）",
        "",
        "prompts:",
    ]
    for it in items:
        lines.append(f"  - id: {it['id']}")
        lines.append(f"    html_id: {it['html_id']}")
        lines.append(f"    title: {_yaml_str(it['title'])}")
        lines.append(f"    category: {it['category']}")
        lines.append(f"    needs_image: {'true' if it['needs_image'] else 'false'}")
        lines.append("    body: |")
        for row in it["body"].splitlines():
            lines.append(f"      {row}" if row.strip() else "      ")
        lines.append("")

    YAML_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"copied html -> {STATIC / '全品类.html'}")
    print(f"wrote {len(items)} prompts -> {YAML_PATH}")


def _guess_category(title: str, body: str) -> str:
    t = title + body[:80]
    if "主图" in t:
        return "main_image"
    if "标题" in t:
        return "title"
    if "选款" in t or "爆款" in t or "全品类" in t and "分析" in t:
        return "selection"
    if "关键词" in t:
        return "keywords"
    return "other"


def _yaml_str(s: str) -> str:
    if any(c in s for c in ":#{}[],&*?|>!%@`'\""):
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return s


if __name__ == "__main__":
    main()
