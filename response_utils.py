# -*- coding: utf-8 -*-
"""回复后处理（无 torch 依赖，供 API 版 Agent 使用）。"""
import re


def postprocess_response(text: str) -> str:
    bad_prefixes = [
        "1. 问题判断：",
        "2. 需要商家补充的信息：",
        "3. 下一步动作：",
        "问题判断：",
        "需要商家补充的信息：",
        "下一步动作：",
    ]
    for p in bad_prefixes:
        text = text.replace(p, "")
    text = text.replace("[URL]", "__URL_PLACEHOLDER__")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def inject_links(response: str, retrieved_context: str = "") -> str:
    if not retrieved_context:
        return response
    links = re.findall(r"https?://[^\s，,；;]+", retrieved_context)
    if not links:
        return response
    dedup_links = list(dict.fromkeys(links))
    joined_links = "；".join(dedup_links)
    if "__URL_PLACEHOLDER__" in response:
        return response.replace("__URL_PLACEHOLDER__", joined_links).strip()
    if any(link in response for link in dedup_links):
        return response.strip()
    response = response.rstrip("。.!！？?")
    return f"{response}，链接：{joined_links}"
