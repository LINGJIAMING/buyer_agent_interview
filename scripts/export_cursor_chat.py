#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 Cursor agent-transcripts JSONL 导出为可读 Markdown。"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

TRANSCRIPT_ROOT = Path(
    r"C:\Users\admin\.cursor\projects\c-Users-admin-Desktop-cursor-agent\agent-transcripts"
)
OUTPUT_PATHS = [
    Path(r"c:\Users\admin\Desktop\CURSOR_CHAT_EXPORT.md"),
    Path(r"c:\Users\admin\Desktop\mybuyer_agent-main\log\CURSOR_CHAT_EXPORT.md"),
]

SESSION_TITLES = {
    "aa5fb540-acc0-4387-89a5-e805c12dc065": "买手 Agent 主会话（数据清洗 / Router / RAG / 业务 API / GitHub 发布）",
    "6049fd67-269f-4383-9f84-fe5accafb4ce": "补充会话（schemas 讲解 / 跨设备同步）",
    "20e7f3dd-5051-4c98-9724-ac5f026aa74b": "其他短会话",
}


def _clean_user_text(text: str) -> str:
    text = re.sub(r"<timestamp>.*?</timestamp>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^<user_query>\s*", "", text)
    text = re.sub(r"\s*</user_query>\s*$", "", text)
    return text.strip()


def _format_tool_use(item: dict) -> str:
    name = item.get("name") or "tool"
    inp = item.get("input") or {}
    if isinstance(inp, dict):
        bits = []
        for k, v in list(inp.items())[:3]:
            vs = str(v).replace("\n", " ")
            if len(vs) > 120:
                vs = vs[:117] + "..."
            bits.append(f"{k}={vs}")
        detail = "; ".join(bits) if bits else ""
    else:
        detail = str(inp)[:200]
    return f"> 🔧 **{name}**{(': ' + detail) if detail else ''}"


def _extract_message(msg: dict, role: str) -> str:
    content = msg.get("content") or msg.get("message", {}).get("content") or []
    if isinstance(content, str):
        return _clean_user_text(content) if role == "user" else content

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ == "text":
            t = item.get("text", "")
            if t.strip() in ("", "[REDACTED]"):
                continue
            parts.append(_clean_user_text(t) if role == "user" else t)
        elif typ == "tool_use":
            parts.append(_format_tool_use(item))

    return "\n\n".join(parts).strip()


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None
    obj = json.loads(line)
    role = obj.get("role", "unknown")
    msg = obj.get("message") or obj
    text = _extract_message(msg, role)
    if not text:
        return None
    return role, text


def export_session(jsonl_path: Path) -> str:
    sid = jsonl_path.parent.name
    title = SESSION_TITLES.get(sid, sid)
    lines_out = [f"## 会话：{title}", "", f"- 原始 ID：`{sid}`", f"- 源文件：`{jsonl_path}`", ""]

    turn = 0
    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(raw)
        if not parsed:
            continue
        role, text = parsed
        turn += 1
        if role == "user":
            lines_out.extend([f"### 用户 #{turn}", "", text, ""])
        else:
            lines_out.extend([f"### 助手 #{turn}", "", text, ""])

    return "\n".join(lines_out)


def main() -> None:
    jsonl_files = sorted(TRANSCRIPT_ROOT.glob("*/*.jsonl"))
    # 主会话放最前
    jsonl_files.sort(
        key=lambda p: (0 if "aa5fb540" in p.name else 1, p.name),
    )

    header = [
        "# Cursor 对话导出（买手 Agent 项目）",
        "",
        f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "说明：由本地 `agent-transcripts/*.jsonl` 自动转换；工具调用仅保留摘要，"
        "不含完整 diff/终端输出。换电脑后可用任意 Markdown 阅读器打开。",
        "",
        "---",
        "",
    ]

    body = []
    for path in jsonl_files:
        body.append(export_session(path))
        body.append("\n---\n")

    full = "\n".join(header) + "\n".join(body)

    for out in OUTPUT_PATHS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(full, encoding="utf-8")
        print(f"Wrote {out} ({len(full):,} chars)")


if __name__ == "__main__":
    main()
