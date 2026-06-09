# -*- coding: utf-8 -*-
"""
OpenAI 兼容 API 统一封装（DeepSeek / OpenAI / 自定义 Base URL）。

文档：https://api-docs.deepseek.com/
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from response_utils import inject_links, postprocess_response

# 内置厂商预设
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "label": "DeepSeek",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "label": "OpenAI",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "label": "通义千问（兼容模式）",
    },
}


@dataclass
class LlmApiConfig:
    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.85
    timeout_sec: float = 120.0

    def resolved_base_url(self) -> str:
        if self.base_url.strip():
            return self.base_url.strip().rstrip("/")
        preset = PROVIDER_PRESETS.get(self.provider.lower(), PROVIDER_PRESETS["deepseek"])
        return preset["base_url"]

    def resolved_model(self) -> str:
        if self.model.strip():
            return self.model.strip()
        preset = PROVIDER_PRESETS.get(self.provider.lower(), PROVIDER_PRESETS["deepseek"])
        return preset["default_model"]

    def provider_label(self) -> str:
        preset = PROVIDER_PRESETS.get(self.provider.lower())
        return preset["label"] if preset else self.provider


class LlmApiClient:
    def __init__(self, config: LlmApiConfig):
        if not config.api_key.strip():
            raise ValueError("API Key 不能为空")
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key.strip(),
            base_url=config.resolved_base_url(),
            timeout=config.timeout_sec,
        )

    def health_check(self) -> dict[str, Any]:
        """轻量探测：列模型或发极短请求。"""
        out: dict[str, Any] = {
            "provider": self.config.provider,
            "base_url": self.config.resolved_base_url(),
            "model": self.config.resolved_model(),
        }
        try:
            models = self._client.models.list()
            out["ok"] = True
            out["model_ids"] = [m.id for m in models.data[:10]]
        except Exception as e:
            out["ok"] = False
            out["error"] = str(e)
        return out

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        retrieved_context: str = "",
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.resolved_model(),
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = postprocess_response(text)
        text = inject_links(text, retrieved_context)
        return text
