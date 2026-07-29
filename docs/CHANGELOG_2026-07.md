# 买手 Agent · 公开仓变更摘要（2026-07-22 ~ 2026-07-29）

> 完整研发日志在本地私有仓库 `mybuyer_agent-main/log/`，本文件为**脱敏摘要**。

## 时间线

| 日期 | 主题 |
|------|------|
| 07-22 | Prompt 注入评测集 50 条 + 自动化脚本 |
| 07-27 | 网页 + 商家记忆 SQLite + 多操作员 Key（Key 不进 prompt） |
| 07-29 | FAQ 问法层 + L1/L2 缓存 + 日志飞轮脚本 |

## 架构（API 版主链路）

```text
QueryOpt → Router
  → L1/L2 Cache → FAQ policy_direct
  → Business Mock → MergedRetriever → LLM（Evidence-bound）
```

## 新增模块（本仓库已含）

- `faq/`、`cache/`、`flywheel/`
- `web_server.py`、`memory/`、`web/static/`
- `scripts/build_policy_faq.py` 等
- `data/faq_published.jsonl`、`data/prompt_injection_eval_50.jsonl`

## 评测口径（面试可讲）

- Router 50 条标注集 V2：Scene/Joint **100%**
- RAG：50 条 hint Recall@5≈**0.88**；10 条 gold 严评 Recall@5=**0.50**
- 注入评测：域外 10/10 拒答（规则自动分 + 人工复核）

## 启动网页

```powershell
py -3.9 -m uvicorn web_server:app --host 127.0.0.1 --port 8001
# 浏览器 http://127.0.0.1:8001 ，填写商家 ID 后「进入会话」
```

详见根目录 `README.md`。
