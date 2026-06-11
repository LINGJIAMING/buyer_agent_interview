# -*- coding: utf-8 -*-
"""
RAGAS 评测（DeepSeek Judge + 本地 bge embedding）+ 消融报告生成。

Judge 使用 DeepSeek API；Embedding 使用本地 models/bge-small-zh-v1.5，无需 OpenAI Key。
无 API Key 时仅输出消融 Recall。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from rag.constants import DATA_DIR, EVAL_DIR
from rag.run_ablation_eval import run_ablation

EVAL_PATH = EVAL_DIR / "rag_eval_50.jsonl"
RAGAS_JSON = DATA_DIR / "ragas_results.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_api_key(explicit: str = "") -> str:
    """从参数、.env、进程环境、Windows 用户变量读取 API Key。"""
    if explicit.strip():
        return explicit.strip()
    _load_dotenv()
    for name in ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY", "RAGAS_API_KEY"):
        val = os.getenv(name, "").strip()
        if val:
            return val
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                for name in ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY"):
                    try:
                        val = str(winreg.QueryValueEx(key, name)[0]).strip()
                        if val:
                            return val
                    except OSError:
                        continue
        except OSError:
            pass
    return ""


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def _resolve_embedding_path(explicit: str = "") -> Path:
    """本地 embedding 目录，默认 models/bge-small-zh-v1.5。"""
    if explicit.strip():
        return Path(explicit.strip())
    _load_dotenv()
    for name in ("EMBEDDING_MODEL_PATH", "RAGAS_EMBEDDING_PATH"):
        val = os.getenv(name, "").strip()
        if val:
            return Path(val)
    default = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
    return default


def _build_local_embeddings(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"本地 embedding 模型不存在: {model_path}")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    device = os.getenv("RAGAS_EMBEDDING_DEVICE", "cpu")
    wrapped = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    return wrapped, str(model_path), device


def _load_gold_cases():
    cases = []
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("gold_chunk_ids"):
                cases.append(c)
    return cases


def run_ragas_subset(api_key: str, max_cases: int = 5, embedding_path: str = "") -> dict:
    """对已标注 gold 的样本跑 RAGAS（需 ragas + openai 兼容 API）。"""
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as e:
        return {
            "skipped": True,
            "reason": f"缺少依赖: {e}，请 pip install ragas langchain-openai langchain-community datasets",
        }

    from rag.merged_retriever import MergedRetriever
    from config import POLICY_KB_PATH

    try:
        emb_path = _resolve_embedding_path(embedding_path)
        embeddings, emb_model, emb_device = _build_local_embeddings(emb_path)
    except Exception as e:
        return {"skipped": True, "reason": f"本地 embedding 初始化失败: {e}"}

    retriever = MergedRetriever(POLICY_KB_PATH)
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.getenv("RAGAS_MODEL", "deepseek-chat"),
            api_key=api_key,
            base_url=os.getenv("RAGAS_BASE_URL", "https://api.deepseek.com"),
            temperature=0,
        )
    )

    rows = []
    for c in _load_gold_cases()[:max_cases]:
        ret = retriever.retrieve_context(c["question"], scene=c.get("scene"))
        rows.append({
            "question": c["question"],
            "answer": c.get("reference_answer", ""),
            "contexts": [ret.get("context", "")],
            "ground_truth": c.get("reference_answer", ""),
        })

    if not rows:
        return {"skipped": True, "reason": "无 gold 样本"}

    ds = Dataset.from_list(rows)
    result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    scores = {}
    try:
        df = result.to_pandas()
        scores = {col: float(df[col].mean()) for col in df.columns if df[col].dtype.kind in "fiu"}
    except Exception:
        scores = dict(result) if hasattr(result, "items") else {"raw": str(result)}
    return {
        "skipped": False,
        "scores": scores,
        "cases": len(rows),
        "judge": {
            "llm": os.getenv("RAGAS_MODEL", "deepseek-chat"),
            "base_url": os.getenv("RAGAS_BASE_URL", "https://api.deepseek.com"),
        },
        "embeddings": {"model": emb_model, "device": emb_device},
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default="")
    ap.add_argument("--max-cases", type=int, default=5)
    ap.add_argument("--embedding-path", default="", help="本地 bge 模型目录")
    ap.add_argument("--skip-ragas", action="store_true")
    args = ap.parse_args()

    api_key = _resolve_api_key(args.api_key)

    ablation = run_ablation(top_k=5)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ablation": ablation,
        "ragas": None,
    }

    if not args.skip_ragas and api_key:
        report["ragas"] = run_ragas_subset(
            api_key, args.max_cases, embedding_path=args.embedding_path
        )
    else:
        report["ragas"] = {
            "skipped": True,
            "reason": (
                "未检测到 API Key（支持 DEEPSEEK_API_KEY / API_KEY / 项目根 .env）；"
                "消融 Recall 已写入 ablation_results.json"
            ),
        }

    RAGAS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RAGAS_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
