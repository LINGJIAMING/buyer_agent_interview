# config.py
import os

MODEL_ID = "/root/.cache/modelscope/hub/models/qwen/Qwen2___5-7B-Instruct"
ADAPTER_PATH = "/root/autodl-tmp/LLaMA-Factory/saves/buyer_agent_v2_1/checkpoint-1014"

# 避免本地无 torch 时 import 失败；model_loader 内再解析为 torch.dtype
DEVICE_DTYPE_NAME = os.getenv("DEVICE_DTYPE", "bfloat16")
MAX_HISTORY_ROUNDS = 4
MAX_NEW_TOKENS = 192
TEMPERATURE = 0.6
TOP_P = 0.85
REPETITION_PENALTY = 1.08

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
POLICY_KB_PATH = os.path.join(PROJECT_ROOT, "kb", "policy_kb.md")

# Query Optimizer
ENABLE_QUERY_OPTIMIZER = os.getenv("ENABLE_QUERY_OPTIMIZER", "true").lower() in (
    "1",
    "true",
    "yes",
)
QUERY_OPT_LOG_DIR = os.path.join(PROJECT_ROOT, "log", "query_opt")

# Business API（备货 / 核价）
ENABLE_BUSINESS_API = os.getenv("ENABLE_BUSINESS_API", "true").lower() in (
    "1",
    "true",
    "yes",
)
BUSINESS_API_MODE = os.getenv("BUSINESS_API_MODE", "mock")  # mock | http
BUSINESS_API_LOG_DIR = os.path.join(PROJECT_ROOT, "log", "business_api")

# SOP RAG 双库（policy_kb + sop_chunks）
ENABLE_SOP_RAG = os.getenv("ENABLE_SOP_RAG", "true").lower() in ("1", "true", "yes")
RAG_BM25_ONLY = os.getenv("RAG_BM25_ONLY", "true").lower() in ("1", "true", "yes")
RAG_USE_RERANK = os.getenv("RAG_USE_RERANK", "false").lower() in ("1", "true", "yes")
RAG_SOP_TOP_K = int(os.getenv("RAG_SOP_TOP_K", "3"))

# Skill 注册表 + 模型选择（分析类模板；不改 Router 场景表）
ENABLE_SKILL_SELECTOR = os.getenv("ENABLE_SKILL_SELECTOR", "true").lower() in (
    "1",
    "true",
    "yes",
)

# FAQ（policy_kb 衍生）+ L1/L2 响应缓存（第 28 章）
ENABLE_FAQ_LAYER = os.getenv("ENABLE_FAQ_LAYER", "true").lower() in (
    "1",
    "true",
    "yes",
)
ENABLE_RESPONSE_CACHE = os.getenv("ENABLE_RESPONSE_CACHE", "true").lower() in (
    "1",
    "true",
    "yes",
)
FAQ_PUBLISHED_PATH = os.path.join(PROJECT_ROOT, "data", "faq_published.jsonl")
FAQ_MIN_BM25_SCORE = float(os.getenv("FAQ_MIN_BM25_SCORE", "4.0"))
CACHE_SEMANTIC_THRESHOLD = float(os.getenv("CACHE_SEMANTIC_THRESHOLD", "0.88"))
RESPONSE_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "response_cache.jsonl")