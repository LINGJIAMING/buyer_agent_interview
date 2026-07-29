# -*- coding: utf-8 -*-
"""
操作员 API Key：仅存本地 secrets/operator_keys.json。
禁止写入 SQLite 记忆表，禁止注入 LLM prompt / messages。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS_DIR = PROJECT_ROOT / "secrets"
KEYS_PATH = SECRETS_DIR / "operator_keys.json"


def _env_fallback_key() -> str:
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "API_KEY"):
        val = os.getenv(name, "").strip()
        if val:
            return val
    return ""


def _read_file() -> Dict[str, Any]:
    if not KEYS_PATH.exists():
        return {}
    try:
        data = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_file(data: Dict[str, Any]) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    KEYS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def mask_key(api_key: str) -> str:
    k = (api_key or "").strip()
    if not k:
        return ""
    if len(k) <= 8:
        return "*" * len(k)
    return k[:4] + "…" + k[-4:]


def list_operators() -> List[dict]:
    data = _read_file()
    out = []
    for op_id, meta in data.items():
        if not isinstance(meta, dict):
            continue
        key = str(meta.get("api_key") or "")
        out.append(
            {
                "operator_id": op_id,
                "provider": meta.get("provider") or "deepseek",
                "has_key": bool(key.strip()),
                "key_masked": mask_key(key),
            }
        )
    out.sort(key=lambda x: x["operator_id"])
    return out


def save_operator_key(
    operator_id: str,
    api_key: str,
    *,
    provider: str = "deepseek",
) -> dict:
    op = (operator_id or "").strip()
    key = (api_key or "").strip()
    if not op:
        raise ValueError("operator_id 不能为空")
    if not key:
        raise ValueError("api_key 不能为空")
    if any(ch in op for ch in "/\\.."):
        raise ValueError("operator_id 非法")
    data = _read_file()
    data[op] = {
        "provider": (provider or "deepseek").strip() or "deepseek",
        "api_key": key,
    }
    _write_file(data)
    return {
        "operator_id": op,
        "provider": data[op]["provider"],
        "has_key": True,
        "key_masked": mask_key(key),
    }


def resolve_api_key_for_operator(
    operator_id: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    返回 (api_key, provider, source)。
    source: operator_file | env | missing
    Key 仅用于构造 LLM 客户端，不得写入记忆或 prompt。
    """
    op = (operator_id or "").strip()
    if op:
        data = _read_file()
        meta = data.get(op)
        if isinstance(meta, dict):
            key = str(meta.get("api_key") or "").strip()
            if key:
                provider = str(meta.get("provider") or "deepseek").strip() or "deepseek"
                return key, provider, "operator_file"

    env_key = _env_fallback_key()
    if env_key:
        return (
            env_key,
            os.getenv("API_PROVIDER", "deepseek").strip() or "deepseek",
            "env",
        )
    return "", "deepseek", "missing"
