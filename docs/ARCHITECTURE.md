# 架构说明（简版）

## 推理双模式

| 入口 | 类 | LLM |
| --- | --- | --- |
| `cli.py` | `BuyerAgentApp` | 本地 Qwen2.5-7B + LoRA |
| `cli_api.py` | `BuyerAgentApiApp` | DeepSeek 等 HTTP API |

二者共享：QueryOptimizer、Router、Retriever、BusinessActionExecutor、`build_messages`。

## Router 场景

`policy` · `price_negotiation` · `inventory` · `product` · `activity` · `price_limit` · `approval` · `general`

## 业务 API（可执行动作）

- **备货**：`StockOrderRequest(skc, quantity, size?)`
- **核价**：`PriceReviewRequest(skc, target_price, currency, reason)`

槽位抽取为规则引擎 + Pydantic 校验；真实环境在 `business/apis.py` 实现 HTTP 客户端。

## 知识库

仅使用 `kb/policy_kb.md`（及可选 `policy_kb.jsonl`），内容为政策要点摘要，**非**原始聊天导出。
