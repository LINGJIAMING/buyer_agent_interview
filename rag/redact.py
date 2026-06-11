# -*- coding: utf-8 -*-
"""入库前脱敏：内网链接、多余空白。"""
import re

INTERNAL_URL_PATTERN = re.compile(
    r"https?://(?:note\.pdd\.net|agentseller\.temu\.com|bdl\.cdnfe\.com)[^\s\])>\"']*",
    re.I,
)
GENERIC_URL_PATTERN = re.compile(r"https?://[^\s\])>\"']+", re.I)
MULTI_SPACE = re.compile(r"[ \t]{2,}")
MULTI_NL = re.compile(r"\n{3,}")


def redact_text(text: str, redact_all_urls: bool = False) -> str:
    if not text:
        return ""
    t = INTERNAL_URL_PATTERN.sub("[内部/平台链接已省略]", text)
    if redact_all_urls:
        t = GENERIC_URL_PATTERN.sub("[链接已省略]", t)
    t = MULTI_SPACE.sub(" ", t)
    t = MULTI_NL.sub("\n\n", t)
    return t.strip()
