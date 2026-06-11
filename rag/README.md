# RAG 模块 — SOP 知识库

## 目录

```text
rag/
  data/           # manifest、chunks、评测结果（结构化，可公开）
  index/          # BM25 + FAISS 元数据（大文件可本地重建）
  eval/           # rag_eval_50.jsonl
  manifest.py     # Phase 0 清点
  ingest.py       # Phase 1 解析切分
  build_index.py  # Phase 2 索引
  hybrid_search.py
  merged_retriever.py  # policy_kb + SOP 双库 merge
  run_pipeline.py
```

## 命令

```bash
pip install -r requirements-rag.txt

# 本仓库已含 sop_chunks.jsonl，可直接建索引与评测
python -m rag.build_index --bm25-only
python -m rag.run_eval_recall --bm25-only --no-rerank
python -m rag.run_ablation_eval

# 从原始 PDF/PPTX/DOCX 重新入库（原件不入 Git）
export RAG_SOURCE_DIR=/path/to/your/sop_files
python -m rag.run_pipeline manifest
python -m rag.run_pipeline ingest
python -m rag.run_pipeline index

# gold 标注（可选）
python -m rag.annotate_gold --only-empty
```

## 评测口径

| 口径 | 说明 | 当前参考值 |
| --- | --- | --- |
| Hint Recall@5 | `expected_doc_hints` 文件名匹配 | ~0.88 / 50 |
| Gold Recall@5 | `gold_chunk_ids` 精确匹配 | **0.50 / 10**（已标注子集） |

详见 `rag/data/ablation_results.json`。
