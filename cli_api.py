#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
买手 Agent — 远程大模型交互入口（DeepSeek / OpenAI 兼容）。

使用方式：
  python cli_api.py

在下方「用户配置区」填写 API Key，或留空则在启动时交互输入。
"""
from __future__ import annotations

import getpass
import sys

# =============================================================================
# 用户配置区（可在此直接填写，留空则启动时提示输入）
# =============================================================================

# 厂商：deepseek | openai | qwen | custom
API_PROVIDER = "deepseek"

# DeepSeek 控制台：https://platform.deepseek.com/api_keys
API_KEY = ""

# 留空则使用厂商默认；custom 时必须填写完整 base_url
API_BASE_URL = ""

# 留空则使用厂商默认模型，如 deepseek-chat
API_MODEL = ""

# 生成参数
API_MAX_TOKENS = 512
API_TEMPERATURE = 0.6
API_TOP_P = 0.85

# 启动时是否探测 API 连通性
RUN_HEALTH_CHECK = True

# 聊天时是否打印 Router / QueryOpt 调试信息
VERBOSE = True

# =============================================================================
# 以下一般无需修改
# =============================================================================

from llm_providers import LlmApiConfig, PROVIDER_PRESETS, LlmApiClient
from app_api import create_api_app


def _prompt_api_key() -> str:
    key = (API_KEY or "").strip()
    if key:
        return key
    print("\n未在文件顶部配置 API_KEY，请交互输入（输入不回显）：")
    key = getpass.getpass("API Key: ").strip()
    if not key:
        print("错误：API Key 不能为空。", file=sys.stderr)
        sys.exit(1)
    return key


def _print_banner(cfg: LlmApiConfig):
    preset = PROVIDER_PRESETS.get(cfg.provider.lower(), {})
    print()
    print("=" * 56)
    print("  买手 Agent · 远程大模型版")
    print("=" * 56)
    print(f"  厂商     : {cfg.provider_label()} ({cfg.provider})")
    print(f"  Base URL : {cfg.resolved_base_url()}")
    print(f"  模型     : {cfg.resolved_model()}")
    print("-" * 56)
    print("  命令：exit / quit 退出 | reset 清空对话")
    print("=" * 56)
    print()


def main():
    api_key = _prompt_api_key()
    cfg = LlmApiConfig(
        provider=API_PROVIDER,
        api_key=api_key,
        base_url=API_BASE_URL,
        model=API_MODEL,
        max_tokens=API_MAX_TOKENS,
        temperature=API_TEMPERATURE,
        top_p=API_TOP_P,
    )

    if RUN_HEALTH_CHECK:
        print("正在探测 API 连通性…")
        client = LlmApiClient(cfg)
        hc = client.health_check()
        if hc.get("ok"):
            print(f"  ✓ 连接成功 | model={hc.get('model')}")
            ids = hc.get("model_ids")
            if ids:
                print(f"  可用模型（前几条）: {', '.join(ids[:5])}")
        else:
            print(f"  ✗ 连接失败: {hc.get('error')}", file=sys.stderr)
            ans = input("仍要继续进入聊天吗？(y/N): ").strip().lower()
            if ans != "y":
                sys.exit(2)

    app = create_api_app(cfg, verbose=VERBOSE)
    _print_banner(cfg)

    while True:
        try:
            user_input = input("商家: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue

        low = user_input.lower()
        if low in {"exit", "quit", "q"}:
            print("已退出。")
            break

        if low == "reset":
            app.reset_history()
            print("（对话历史已清空）")
            continue

        try:
            response = app.chat(user_input)
            print(f"买手: {response}\n")
        except Exception as e:
            print(f"[错误] {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
