# 本仓库未包含的内容

为面试展示与隐私合规，以下内容**故意不上传**（详见 `docs/PUBLIC_SYNC_CHECKLIST.md`）：

| 类别 | 说明 |
| --- | --- |
| 原始 SOP | `rag_buyer_file/` 下 PDF/PPTX/DOCX 原件 |
| 群聊知识库 | 原始问答 txt |
| 完整研发日志 | `log/*_DEVLOG.md`、Cursor 导出、终端实录 |
| 真实对话日志 | `log/api_chat/*.jsonl`、本地 `response_cache.jsonl` |
| API 与密钥 | `.env`、`secrets/operator_keys.json` |
| 本地记忆库 | `data/buyer_memory.db` |
| 模型权重 | Qwen 基座、LoRA checkpoint |
| 个人资料 | 简历、面试笔记、手机号邮箱 |

## 已脱敏上传

- `rag/data/sop_chunks.jsonl` — chunk 文本，内网链接经 redact
- `kb/policy_kb.md` / `policy_kb.jsonl` — 政策要点
- `data/faq_published.jsonl` — 由 policy 衍生的 FAQ（含公开平台链接）
- `data/router_eval_50.jsonl`、`data/prompt_injection_eval_50.jsonl` — 评测用例
- `skills/menswear_full_category/` — Prompt 模板 YAML + 静态工具页

## 复现索引

```bash
pip install -r requirements-rag.txt
python scripts/build_policy_faq.py
python -m rag.build_index --bm25-only
python router_eval.py
```
