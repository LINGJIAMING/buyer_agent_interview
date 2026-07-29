# -*- coding: utf-8 -*-
"""Skill 注册表：扫描 skills/*/prompts.yaml，暴露给选择器。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

SKILLS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SkillEntry:
    id: str
    title: str
    category: str
    needs_image: bool
    body: str
    pack_id: str
    description: str  # 给选择器用的短描述


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("需要 pyyaml：pip install pyyaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def load_skill_catalog() -> tuple[SkillEntry, ...]:
    """加载全部 Skill 条目（进程内缓存）。"""
    entries: list[SkillEntry] = []
    for yaml_path in sorted(SKILLS_ROOT.glob("*/prompts.yaml")):
        pack_dir = yaml_path.parent.name
        if pack_dir.startswith("_"):
            continue
        data = _load_yaml(yaml_path)
        pack_meta = data.get("skill") or {}
        pack_id = str(pack_meta.get("id") or pack_dir)
        for p in data.get("prompts") or []:
            sid = str(p.get("id") or "").strip()
            if not sid:
                continue
            title = str(p.get("title") or sid).strip()
            category = str(p.get("category") or "other").strip()
            body = str(p.get("body") or "").strip()
            needs_image = bool(p.get("needs_image", False))
            desc = f"{title}（品类={category}；{'需要图片' if needs_image else '纯文本即可'}）"
            entries.append(
                SkillEntry(
                    id=sid,
                    title=title,
                    category=category,
                    needs_image=needs_image,
                    body=body,
                    pack_id=pack_id,
                    description=desc,
                )
            )
    return tuple(entries)


def get_skill(skill_id: str) -> Optional[SkillEntry]:
    for e in load_skill_catalog():
        if e.id == skill_id:
            return e
    return None


def catalog_for_prompt() -> str:
    """选择器用：只暴露 id + 短描述，不塞全文 body。"""
    lines = []
    for e in load_skill_catalog():
        lines.append(f"- id={e.id} | {e.description}")
    return "\n".join(lines) if lines else "(暂无已注册 Skill)"


def clear_catalog_cache() -> None:
    load_skill_catalog.cache_clear()
