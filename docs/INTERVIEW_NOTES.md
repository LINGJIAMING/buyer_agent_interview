# 面试讲解要点

## 30 秒电梯演讲

> 我做了一个 Temu 全托管买手 Agent：QLoRA 微调 Qwen2.5-7B，加上规则 Router、Query 优化层、双库 RAG（政策摘要 + 84 份 SOP 切成 2146 chunk），以及备货/核价的 Pydantic 结构化 API。离线评测 Router 50 条、RAG gold Recall@5 50%，并设计了消融与 RAGAS 脚本。

## 可展开的三条线

### 1. Agent 链路

- **Query Optimizer**：多轮指代、SPU/SKC 补全、口语去噪、术语对齐（如「直通车」→「开广告」）
- **Router**：8 类场景 + `general`，关键词打分 + 优先级消歧
- **Business API**：槽位齐全时走 Mock/HTTP，否则 LLM 澄清

### 2. RAG 工程

- **入库**：manifest 清点 → PyMuPDF/pptx/docx 解析 → 父子块（child 检索、parent 生成）
- **检索**：BM25 + bge-small-zh 向量 RRF；可选 Cross-Encoder Rerank
- **双库 merge**：`policy_kb` 优先短政策，SOP 补操作细节；`kb_lane` 降低开款 PPT 污染
- **评测**：hint 粗评 vs gold 严评两套口径；E0 消融已跑，Hybrid/Rerank 待本地 embedding

### 3. 工程化与合规

- GitHub 展示版与内网完整版分离：无群聊、无 API Key、无原始 SOP 原件
- Evidence-bound Prompt 约束生成，聊天日志记录 `sop_chunk_ids` 可审计
- 双推理后端：云端 SFT 与 DeepSeek API 共用同一套检索与 Prompt

## 常见追问

| 问题 | 要点 |
| --- | --- |
| 为什么 gold Recall 只有 50%？ | 10 条严口径；同文件多 chunk、关键词误召；Rerank/假设问可提升 |
| hint 0.88 vs gold 0.50？ | hint 看文件名，gold 看 chunk_id，面试要说清口径 |
| 如何防幻觉？ | 双库 Evidence + 低分拒答 + 业务 API 与聊天分离 |
| 为什么不用纯向量？ | 政策/SKU 号 BM25 更稳；RRF 融合互补 |

## 演示命令

```bash
python router_eval.py
python -m rag.run_eval_recall --bm25-only --no-rerank
python cli_api.py   # 需自行配置 API Key
```
