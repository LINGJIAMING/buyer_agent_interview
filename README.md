# Buyer Agent — 电商买手智能助手

基于 **Qwen2.5-7B（QLoRA 微调）+ 双库 RAG + 规则 Router + 结构化业务 API** 的垂直领域 Agent，面向 Temu 全托管买手场景：政策咨询、核价、备货、审版、限流申诉等。

> 本仓库为**面试/开源展示版**：含核心代码、脱敏政策库与 **2,146 条 SOP chunk**，不含原始 PPT/PDF、群聊记录与内部研发日志。

## 系统架构

```text
User Input
  → Query Optimizer（上下文补全 / 去噪 / 术语对齐）
  → Router（scene + subtask）
  → Business API（备货 / 核价，Pydantic + Mock API）
  → MergedRetriever（policy_kb + SOP，BM25 / Hybrid + 可选 Rerank）
  → LLM（本地微调 Qwen 或 DeepSeek 等 OpenAI 兼容 API）
  → Evidence-bound 买手口吻回复
```

## 目录结构

```text
mybuyer_agent-github/
├── cli.py / cli_api.py     # 本地 GPU 版 / API 版
├── app.py / app_api.py
├── router.py               # 8 类场景路由
├── query_optimizer.py
├── retriever.py            # policy_kb 混合检索
├── rag/                    # SOP 入库 · 索引 · 评测 · 双库 merge
│   ├── data/sop_chunks.jsonl
│   ├── eval/rag_eval_50.jsonl
│   └── merged_retriever.py
├── business/               # 备货 / 核价结构化 API
├── kb/                     # 政策知识库（脱敏摘要）
├── data/                   # Router 评测集（50 条）
├── autodl_eval/            # 云端 SFT 对比脚本
└── docs/                   # 架构与面试要点
```

## 快速开始

### 1. 无 GPU、无 API Key（逻辑与评测）

```bash
pip install pydantic jieba rank-bm25
python test_business_api.py
python router_eval.py
python query_optimizer_test.py
python -m rag.run_eval_recall --bm25-only --no-rerank
```

### 2. DeepSeek API 聊天（含 SOP RAG）

```bash
pip install pydantic openai jieba rank-bm25
# 在 cli_api.py 顶部填写 API_KEY，或启动时交互输入
python cli_api.py
```

### 3. 本地微调 Qwen

```bash
pip install -r requirements.txt
cp .env.example .env   # 填写 MODEL_ID、ADAPTER_PATH
python cli.py
```

### 4. 完整 RAG 管线（可选）

```bash
pip install -r requirements-rag.txt
python -m rag.build_index --bm25-only
python -m rag.run_ablation_eval
```

## 核心能力

| 模块 | 说明 |
| --- | --- |
| **Router** | 关键词打分 + 优先级消歧，8 类业务场景 |
| **Query Optimizer** | SPU/SKC 补全、错别字、术语对齐、多意图标记 |
| **双库 RAG** | `policy_kb` + 84 份 SOP → 2,146 chunk；BM25 + 向量 RRF + Rerank |
| **Evidence Prompt** | 生成仅依据【检索证据】，降低政策幻觉 |
| **Business API** | `StockOrderRequest` / `PriceReviewRequest`，Mock 可换 HTTP |
| **评测** | Router 50 条 + RAG Recall@5 + 消融 E0～E2 脚本 |

## 评测数据

- `data/router_eval_50.jsonl` — Router 离线评测
- `rag/eval/rag_eval_50.jsonl` — RAG 评测（10 条 gold chunk 标注）
- `rag/data/ablation_results.json` — E0 BM25 Recall@5 = 0.50（gold 子集）
- `autodl_eval/test_dataset_100.jsonl` — 基座 vs SFT 对比用例

## 未包含内容（隐私与体积）

- 原始商家群聊 / 清洗前 JSONL
- 内部研发日志、API 调用日志、Cursor 导出脚本
- `rag_buyer_file/` 原始 PPT/PDF（仅保留结构化 chunk）
- 模型权重与 LoRA checkpoint


## 技术栈

Python 3.9+ · PyTorch · Transformers · PEFT · Pydantic · OpenAI SDK · jieba · rank-bm25 · sentence-transformers · FAISS · RAGAS（可选）

## License

MIT（知识库与模型授权请自行合规）
