# -*- coding: utf-8 -*-
"""Phase 2：BM25 + Dense Embedding + FAISS 索引构建。"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from rag.constants import CHUNKS_JSONL, DATA_DIR, INDEX_DIR


def _tokenize(text: str) -> List[str]:
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return list(text)


def load_chunks(path: Optional[Path] = None) -> list[dict]:
    path = path or CHUNKS_JSONL
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_bm25(chunks: list[dict], out_dir: Path):
    from rank_bm25 import BM25Okapi

    corpus = [c.get("search_text") or c.get("content", "") for c in chunks]
    tokenized = [_tokenize(t) for t in corpus]
    bm25 = BM25Okapi(tokenized)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "bm25_corpus.pkl").open("wb") as f:
        pickle.dump({"tokenized": tokenized, "texts": corpus}, f)
    with (out_dir / "bm25_model.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    return bm25


def build_dense(chunks: list[dict], out_dir: Path, model_name: Optional[str] = None):
    from sentence_transformers import SentenceTransformer

    model_name = model_name or os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-small-zh-v1.5",
    )
    texts = [c.get("search_text") or c.get("content", "") for c in chunks]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    embeddings = embeddings.astype(np.float32)

    # L2 normalize for cosine via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = embeddings / norms

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "embedding_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"model": model_name, "dim": int(embeddings.shape[1]), "count": len(chunks)}, f)

    try:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(out_dir / "faiss.index"))
    except ImportError:
        pass

    chunk_ids = [c["chunk_id"] for c in chunks]
    with (out_dir / "chunk_id_map.json").open("w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)

    return embeddings


def build_all(
    chunks_path: Optional[Path] = None,
    index_dir: Optional[Path] = None,
    dense: bool = True,
) -> dict:
    chunks_path = chunks_path or CHUNKS_JSONL
    index_dir = index_dir or INDEX_DIR
    if not chunks_path.exists():
        raise FileNotFoundError(f"先运行 ingest: {chunks_path}")

    chunks = load_chunks(chunks_path)
    build_bm25(chunks, index_dir)
    report = {"chunk_count": len(chunks), "index_dir": str(index_dir), "bm25": "ok"}
    if dense:
        try:
            build_dense(chunks, index_dir)
            report["dense"] = "ok"
        except Exception as e:
            report["dense"] = f"failed: {e}"
            report["hint"] = (
                "可设置 HF_ENDPOINT=https://hf-mirror.com 后重试，"
                "或 python -m rag.build_index --bm25-only"
            )
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(CHUNKS_JSONL))
    ap.add_argument("--bm25-only", action="store_true")
    args = ap.parse_args()
    r = build_all(Path(args.chunks), dense=not args.bm25_only)
    print(json.dumps(r, ensure_ascii=False, indent=2))
