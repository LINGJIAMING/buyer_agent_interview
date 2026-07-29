# -*- coding: utf-8 -*-
"""Skill 选择器：用一次短 LLM 调用，从注册表中选 skill_id 或 null。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from skills.registry import catalog_for_prompt, get_skill, load_skill_catalog


@dataclass
class SkillSelection:
    skill_id: Optional[str]
    reason: str
    raw: str = ""


SELECT_SYSTEM = """你是买手 Agent 的 Skill 路由器。
根据用户问题，判断是否应启用「分析类 Skill」（标题生成/诊断、主图评分、选款爆款分析等）。

规则：
1. 仅当用户明确要求做标题生成/优化/诊断、主图评分、选款/爆款分析、套用分析模板时，才选择对应 skill_id。
2. 政策、限流申诉、审版、核价、库存、报名、普通问答 → skill_id 必须为 null。
3. 只输出一个 JSON 对象，不要 Markdown，不要其它文字：
{"skill_id": "p4" 或 null, "reason": "一句话原因"}
4. skill_id 必须是下列目录中的 id，禁止编造。
"""


def select_skill(llm_client: Any, user_query: str) -> SkillSelection:
    """调用 llm_client 底层 OpenAI 兼容接口做一次短选择。"""
    catalog = load_skill_catalog()
    if not catalog:
        return SkillSelection(None, "注册表为空", "")

    catalog_text = catalog_for_prompt()
    messages = [
        {"role": "system", "content": SELECT_SYSTEM + "\n\n可用 Skill 目录：\n" + catalog_text},
        {"role": "user", "content": user_query},
    ]

    # 直接走底层 client，避免 inject_links 污染 JSON
    raw = ""
    try:
        resp = llm_client._client.chat.completions.create(
            model=llm_client.config.resolved_model(),
            messages=messages,
            max_tokens=120,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        return SkillSelection(None, f"选择器调用失败: {exc}", "")

    skill_id, reason = _parse_selection(raw)
    if skill_id and get_skill(skill_id) is None:
        return SkillSelection(None, f"非法 skill_id={skill_id}，已忽略", raw)
    return SkillSelection(skill_id, reason, raw)


def _parse_selection(raw: str) -> tuple[Optional[str], str]:
    text = raw.strip()
    # 容忍 ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            return None, "无法解析 JSON"
    except json.JSONDecodeError:
        return None, "JSON 解析失败"

    sid = obj.get("skill_id", None)
    reason = str(obj.get("reason") or "").strip() or "无原因"
    if sid is None or sid == "" or str(sid).lower() in ("null", "none"):
        return None, reason
    return str(sid).strip(), reason
