# 架构说明

## 推理双模式

| 入口 | 类 | LLM |
| --- | --- | --- |
| `cli.py` | `BuyerAgentApp` | 本地 Qwen2.5-7B + LoRA |
| `cli_api.py` | `BuyerAgentApiApp` | DeepSeek 等 HTTP API |

二者共享：QueryOptimizer、Router、MergedRetriever（或 Retriever）、BusinessActionExecutor、`build_messages`。

## 检索双库

```text
MergedRetriever
  ├─ Retriever(policy_kb.md)     # 轻量政策摘要，响应快
  └─ SopHybridRetriever          # sop_chunks.jsonl，BM25 / Hybrid / Rerank
         └─ parent 块拼入 Evidence（Small-to-Big）
```

`ENABLE_SOP_RAG=false` 时回退为仅 `policy_kb`。

## Router 场景

`policy` · `price_negotiation` · `inventory` · `product` · `activity` · `price_limit` · `approval` · `general`

## 业务 API

- **备货**：`StockOrderRequest(skc, quantity, size?)`
- **核价**：`PriceReviewRequest(skc, target_price, currency, reason)`

规则槽位 + Pydantic 校验；生产环境在 `business/apis.py` 接 HTTP。

## RAG 管线（离线）

```text
manifest → ingest → build_index → run_eval_recall / run_ablation_eval
```

入库脚本支持 PDF/PPTX/DOCX；内网 URL 经 `redact.py` 脱敏。

## 生成约束

`prompts.EVIDENCE_BOUND_PROMPT`：仅依据【检索证据】作答，证据不足时澄清，禁止虚假后台操作承诺。
