# -*- coding: utf-8 -*-
"""递归切分 + Small-to-Big 父子块。"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

from rag.constants import CHILD_CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SIZE
from rag.redact import redact_text

SECTION_PATTERN = re.compile(
    r"(?:^|\n)(?:第[一二三四五六七八九十]+[章节部分]|[一二三四五六七八九十]+[、\.．]|#{1,3}\s+|\d+\.\s+)",
)


def _split_recursive(text: str, max_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_size:
        return [text] if text else []

    separators = ["\n\n", "\n", "。", "；", " ", ""]
    for sep in separators:
        if sep and sep not in text:
            continue
        parts = text.split(sep) if sep else list(text)
        chunks: List[str] = []
        current = ""
        for p in parts:
            piece = p if sep == "" else (p + sep if p else "")
            if len(current) + len(piece) <= max_size:
                current += piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                if overlap and chunks:
                    tail = chunks[-1][-overlap:]
                    current = tail + piece
                else:
                    current = piece
        if current.strip():
            chunks.append(current.strip())
        if len(chunks) > 1 or (chunks and len(chunks[0]) <= max_size):
            return [c for c in chunks if len(c.strip()) >= 30]

    return [text[:max_size]]


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """返回 (section_title, body) 列表。"""
    text = text.strip()
    if not text:
        return []
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [("", text)]
    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if len(block) < 30:
            continue
        first_line = block.split("\n", 1)[0].strip()
        body = block
        sections.append((first_line[:80], body))
    return sections or [("", text)]


def build_parent_child_chunks(
    file_id: str,
    source_file: str,
    blocks: List[dict],
    meta: dict,
) -> Tuple[List[dict], List[dict]]:
    """
  blocks: [{text, page_or_slide, section_title}]
  返回 (parents, children)
    """
    parents: List[dict] = []
    children: List[dict] = []
    parent_seq = 0
    child_seq = 0

    for block in blocks:
        text = redact_text(block["text"])
        if len(text) < 30:
            continue
        page = block.get("page_or_slide", "")
        sec_title = block.get("section_title", "")

        sections = split_into_sections(text)
        if not sections:
            sections = [(sec_title, text)]

        for sec_title, body in sections:
            parent_seq += 1
            parent_id = f"{file_id}_p{parent_seq:04d}"
            parent = {
                "parent_id": parent_id,
                "title": sec_title or meta.get("title_hint", source_file),
                "content": body,
                "source_file": source_file,
                "page_or_slide": page,
                **{k: v for k, v in meta.items() if k != "title_hint"},
            }
            parents.append(parent)

            sub_chunks = _split_recursive(body, CHUNK_SIZE, CHUNK_OVERLAP)
            for sub in sub_chunks:
                child_seq += 1
                child_id = f"{file_id}_c{child_seq:05d}"
                small_parts = _split_recursive(sub, CHILD_CHUNK_SIZE, 40)
                for sp in small_parts:
                    children.append({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "title": sec_title or meta.get("title_hint", source_file),
                        "content": sp,
                        "search_text": f"{sec_title}\n{sp}".strip(),
                        "source_file": source_file,
                        "page_or_slide": page,
                        **{k: v for k, v in meta.items() if k != "title_hint"},
                    })

    return parents, children


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
