# -*- coding: utf-8 -*-
"""RAG 路径与分类规则。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_ROOT = Path(__file__).resolve().parent

# 原始资料目录（本地放置 PDF/PPTX/DOCX，不入 Git）
DEFAULT_SOURCE_DIR = Path(
    os.getenv("RAG_SOURCE_DIR", str(PROJECT_ROOT / "data" / "sop_raw"))
)

# 产物（可上传 GitHub 的结构化数据）
DATA_DIR = RAG_ROOT / "data"
INDEX_DIR = RAG_ROOT / "index"
EVAL_DIR = RAG_ROOT / "eval"

MANIFEST_CSV = DATA_DIR / "manifest.csv"
CHUNKS_JSONL = DATA_DIR / "sop_chunks.jsonl"
PARENTS_JSONL = DATA_DIR / "sop_parents.jsonl"
HYPO_QUESTIONS_JSONL = DATA_DIR / "hypo_questions.jsonl"
INGEST_REPORT_JSON = DATA_DIR / "ingest_report.json"

INCLUDE_EXTENSIONS = {".pdf", ".pptx", ".docx"}
EXCLUDE_EXTENSIONS = {
    ".xlsx", ".xls", ".jpg", ".jpeg", ".png", ".gif", ".mp4",
    ".ttf", ".prj", ".avif", ".doc", ".zip",
}

CHUNK_SIZE = 480
CHUNK_OVERLAP = 80
CHILD_CHUNK_SIZE = 240

# kb_lane：检索时分流，降低「开款 PPT」污染「合规政策」
LANE_POLICY_OPS = "policy_ops"
LANE_MERCHANT_GUIDE = "merchant_guide"
LANE_CATEGORY_TRAINING = "category_training"

LANE_KEYWORDS = {
    LANE_POLICY_OPS: [
        "合规", "GPSR", "EPR", "洗水", "水洗", "标签", "PFAS", "TRO",
        "知识产权", "欧代", "英代", "承诺", "GPSR", "undertaking", "资质",
        "实拍", "包装法", "回收", "侵权", "土耳其", "代理",
    ],
    LANE_MERCHANT_GUIDE: [
        "全托管", "发货", "报备", "运费", "质检", "收货", "合并", "转运",
        "直发", "时效", "入驻", "商家", "运营", "Y2", "模板",
    ],
    LANE_CATEGORY_TRAINING: [
        "开款", "培训", "趋势", "春上新", "运动", "泳衣", "品类",
        "非品牌", "企业店", "布局", "GMV", "pptx", "ppt",
    ],
}

SCENE_BY_LANE = {
    LANE_POLICY_OPS: "policy",
    LANE_MERCHANT_GUIDE: "inventory",
    LANE_CATEGORY_TRAINING: "activity",
}
