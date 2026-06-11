# 本仓库未包含的内容

为面试展示与隐私合规，以下内容**故意不上传**：

| 类别 | 说明 |
| --- | --- |
| 原始 SOP | `rag_buyer_file/` 下 PDF/PPTX/DOCX 原件 |
| 群聊知识库 | `kb/` 下原始问答 txt |
| 研发日志 | `log/*_DEVLOG.md`、终端实录、优化轨迹 |
| API 与密钥 | `.env`、`cli_api.py` 内 Key、RAGAS 运行结果含路径的 json |
| 模型权重 | Qwen 基座、LoRA checkpoint |
| 个人环境 | 本机绝对路径、Cursor 导出脚本 |

## 已脱敏上传

- `rag/data/sop_chunks.jsonl` — 结构化 chunk，内网链接已 redact
- `kb/policy_kb.md` — 政策要点摘要
- `data/router_eval_50.jsonl` — 合成/标注 Router 用例

## 复现索引

大体积 BM25/FAISS 二进制可从 chunk 重建：

```bash
pip install -r requirements-rag.txt
python -m rag.build_index --bm25-only
```
