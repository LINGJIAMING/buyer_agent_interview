# config.py — 通过环境变量配置，避免提交个人路径
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 本地 Qwen + LoRA（cli.py / app.py）
MODEL_ID = os.getenv(
    "MODEL_ID",
    "/path/to/Qwen2.5-7B-Instruct",
)
ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH",
    "/path/to/buyer_agent_lora_checkpoint",
)

DEVICE_DTYPE_NAME = os.getenv("DEVICE_DTYPE", "bfloat16")
MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "4"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
TOP_P = float(os.getenv("TOP_P", "0.85"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.08"))

POLICY_KB_PATH = os.getenv(
    "POLICY_KB_PATH",
    os.path.join(PROJECT_ROOT, "kb", "policy_kb.md"),
)

ENABLE_QUERY_OPTIMIZER = os.getenv("ENABLE_QUERY_OPTIMIZER", "true").lower() in (
    "1", "true", "yes",
)
QUERY_OPT_LOG_DIR = os.path.join(PROJECT_ROOT, "log", "query_opt")

ENABLE_BUSINESS_API = os.getenv("ENABLE_BUSINESS_API", "true").lower() in (
    "1", "true", "yes",
)
BUSINESS_API_MODE = os.getenv("BUSINESS_API_MODE", "mock")
BUSINESS_API_LOG_DIR = os.path.join(PROJECT_ROOT, "log", "business_api")

# SOP RAG 双库（policy_kb + sop_chunks）
ENABLE_SOP_RAG = os.getenv("ENABLE_SOP_RAG", "true").lower() in ("1", "true", "yes")
RAG_BM25_ONLY = os.getenv("RAG_BM25_ONLY", "true").lower() in ("1", "true", "yes")
RAG_USE_RERANK = os.getenv("RAG_USE_RERANK", "false").lower() in ("1", "true", "yes")
RAG_SOP_TOP_K = int(os.getenv("RAG_SOP_TOP_K", "3"))
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "")
