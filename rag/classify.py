# -*- coding: utf-8 -*-
"""按文件名 + 正文片段推断 kb_lane / scene / doc_type。"""
from __future__ import annotations

from rag.constants import (
    LANE_CATEGORY_TRAINING,
    LANE_KEYWORDS,
    LANE_MERCHANT_GUIDE,
    LANE_POLICY_OPS,
    SCENE_BY_LANE,
)


def classify_filename(name: str) -> dict:
    lower = name.lower()
    scores = {lane: 0 for lane in LANE_KEYWORDS}
    for lane, kws in LANE_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in lower or kw in name:
                scores[lane] += 1

    # pptx 默认偏培训 lane
    if lower.endswith(".pptx") and scores[LANE_CATEGORY_TRAINING] == 0:
        scores[LANE_CATEGORY_TRAINING] += 1

    best_lane = max(scores, key=scores.get)
    if scores[best_lane] == 0:
        best_lane = LANE_POLICY_OPS

    ext = "." + lower.rsplit(".", 1)[-1] if "." in lower else ""
    doc_type = {
        ".pdf": "pdf_sop",
        ".pptx": "ppt_training",
        ".docx": "docx_sop",
    }.get(ext, "unknown")

    return {
        "kb_lane": best_lane,
        "scene": SCENE_BY_LANE.get(best_lane, "policy"),
        "doc_type": doc_type,
        "lane_scores": scores,
    }
