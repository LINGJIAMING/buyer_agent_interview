# 架构说明

## 推理模式

| 入口 | 类 | LLM |
| --- | --- | --- |
| `web_server.py` / `cli_api.py` | `BuyerAgentApiApp` | DeepSeek 等 HTTP API（**推荐展示**） |
| `cli.py` | `BuyerAgentApp` | 本地 Qwen2.5-7B + LoRA（可选，需自备权重） |

共享：QueryOptimizer、Router、FAQ/Cache 快路径、MergedRetriever、BusinessActionExecutor、`build_messages`。

## 请求路径（API 版）

```text
1. QueryOptimizer（可选）
2. Router → scene / subtask
3. L1 Exact cache → L2 Semantic cache → L3 FAQ BM25（policy_direct，不调 LLM）
4. BusinessActionExecutor（备货/核价 Mock）
5. Skill 选择器 + `skills/menswear_full_category`（分析模板，可选）
6. MergedRetriever → Evidence-bound LLM
```

`ENABLE_FAQ_LAYER` / `ENABLE_RESPONSE_CACHE` 见 `config.py`。

## 检索与知识分层

```text
PolicyFaqIndex(data/faq_published.jsonl)   # 问法层，政策直出
MergedRetriever
  ├─ Retriever(policy_kb)                  # 政策短卡
  └─ SopHybridRetriever(sop_chunks)        # 长 SOP，BM25 / Hybrid
```

## 网页与记忆

- `merchant_id` + `session_id` → SQLite `data/buyer_memory.db`（本地，gitignore）
- `operator_id` → `secrets/operator_keys.json`（gitignore，仅 example 入库）
- Key **不**写入模型记忆表与 system prompt

## Router 场景

`policy` · `price_negotiation` · `inventory` · `product` · `activity` · `price_limit` · `approval` · `general`

## 离线管线

```text
manifest → ingest → build_index → run_eval_recall / run_ablation_eval
scripts/build_policy_faq.py  # policy_kb.jsonl → faq_published
```

## 安全与评测

- `data/prompt_injection_eval_50.jsonl` + `run_prompt_injection_eval.py`
- 机制：Evidence Prompt + RAG 接地（非硬拦截器）
