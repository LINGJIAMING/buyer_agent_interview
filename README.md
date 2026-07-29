# Buyer Agent — 电商买手智能助手（面试展示仓）

基于 **DeepSeek 等 API + Policy/SOP 双库 RAG + FAQ 缓存 + Router + 结构化业务 API** 的买手场景 Agent：政策咨询、问卷入口、核价/备货协同、审版与限流申诉。

> **展示版**：含核心代码、脱敏政策库与 **2,146 条 SOP chunk**、27 条 FAQ；不含原始文档、真实对话日志与密钥。  
> 上传范围说明：`docs/PUBLIC_SYNC_CHECKLIST.md`

## 怎么打开网页（本机演示）

```powershell
cd mybuyer_agent-github
copy .env.example .env   # 可选：填 DEEPSEEK_API_KEY
py -3.9 -m pip install fastapi uvicorn pyyaml pydantic openai jieba rank-bm25
py -3.9 -m uvicorn web_server:app --reload --host 127.0.0.1 --port 8001
```

浏览器打开 **http://127.0.0.1:8001** → 填写 **商家 ID** → 点 **「进入会话」** → 发消息。

| 地址 | 说明 |
|------|------|
| http://127.0.0.1:8001 | 聊天 |
| http://127.0.0.1:8001/health | Key / 记忆库状态 |
| http://127.0.0.1:8001/tools/menswear | 男装全品类 Prompt 工具页（侧链） |

Windows 可双击 `启动网页.bat`（需已安装 Python 3.9）。

## 系统架构（2026-07-29）

```text
User（浏览器 / cli_api）
  → Query Optimizer → Router
  → L1 精确缓存 → L2 语义缓存 → L3 FAQ（policy 直出）
  → Business API（备货/核价 Mock）
  → Skill 选择器（男装分析模板，可选）
  → MergedRetriever（policy_kb + sop_chunks）
  → LLM + Evidence-bound Prompt
```

本地 **Qwen + LoRA** 入口仍保留在 `cli.py`（可选）；**主展示链路为 API 版** `app_api.py` / `web_server.py`。

## 目录结构

```text
├── app_api.py / web_server.py   # API 主链路 + 网页
├── faq/ cache/ flywheel/        # FAQ、缓存、日志飞轮
├── memory/ secrets/             # 商家记忆（Key 仅本地 example）
├── router.py query_optimizer.py
├── rag/                         # SOP 入库 · 混合检索 · 评测
├── business/                    # 备货 / 核价
├── skills/                      # 分析 Skill 注册表 + 男装 Prompt 包
├── kb/ data/                    # 政策、FAQ、Router/注入评测集
├── scripts/                     # build_policy_faq 等
└── docs/                        # 架构、变更摘要、上传清单
```

## 快速开始（无 Key）

```bash
pip install pydantic jieba rank-bm25
python test_business_api.py
python router_eval.py
python scripts/build_policy_faq.py
python -m rag.run_eval_recall --bm25-only --no-rerank
```

## 核心能力

| 模块 | 说明 |
| --- | --- |
| **FAQ + 缓存** | policy_kb 衍生问法层；高频政策零 LLM 直出 |
| **Router** | 8 类场景，50 条评测 V2 100% |
| **RAG** | 双库 merge、Evidence 约束、Recall/RAGAS |
| **Business** | Pydantic 槽位 + Mock API |
| **Skill 注册表** | `skills/menswear_full_category`；聊天内选择器 + `/tools/menswear` |
| **安全评测** | `run_prompt_injection_eval.py`（50 条） |

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [CHANGELOG_2026-07.md](docs/CHANGELOG_2026-07.md)
- [WHAT_IS_EXCLUDED.md](docs/WHAT_IS_EXCLUDED.md)

## License

MIT — 见 [LICENSE](LICENSE)
