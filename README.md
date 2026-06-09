# Buyer Agent — 电商买手智能助手

基于 **Qwen2.5-7B（QLoRA 微调）+ RAG + 规则 Router + 结构化业务 API** 的垂直领域 Agent，面向 Temu 全托管买手场景：政策咨询、核价、备货、审版、限流申诉等。

> 本仓库为**面试/开源展示版**：仅含核心代码与脱敏政策知识库，不含真实群聊记录与内部研发日志。

## 系统架构

```text
User Input
  → Query Optimizer（上下文补全 / 去噪 / 术语对齐）
  → Router（scene + subtask）
  → Business API（备货 / 核价，Pydantic + Mock API）
  → Retriever（BM25 + 向量混合检索，可选）
  → LLM（本地微调 Qwen 或 DeepSeek 等 OpenAI 兼容 API）
  → 买手口吻回复
```

## 目录结构

```text
mybuyer_agent-github/
├── cli.py              # 本地 GPU + 微调模型
├── cli_api.py          # DeepSeek / OpenAI 兼容 API（无需本地权重）
├── app.py / app_api.py
├── router.py           # 场景路由（8 类 + general）
├── query_optimizer.py  # Query 预处理
├── retriever.py        # 混合检索
├── generator.py        # Prompt 组装 + 本地生成
├── business/           # Pydantic 约束 + 备货/核价 API
├── kb/                 # 政策知识库（脱敏摘要，非原始聊天记录）
├── data/               # Router 评测集（50 条，合成/标注）
├── autodl_eval/        # 云端 SFT 对比推理脚本
└── docs/               # 面试讲解要点
```

## 快速开始

### 1. 仅体验 Agent 逻辑（无 GPU、无 Key）

```bash
pip install pydantic
python test_business_api.py
python router_eval.py
python query_optimizer_test.py
```

### 2. 使用 DeepSeek API 聊天

```bash
pip install pydantic openai
# 在 cli_api.py 顶部填写 API_KEY，或启动时交互输入
python cli_api.py
```

### 3. 使用本地微调 Qwen

```bash
pip install -r requirements.txt
cp .env.example .env   # 填写 MODEL_ID、ADAPTER_PATH
python cli.py
```

## 核心能力

| 模块 | 说明 |
| --- | --- |
| **Router** | 关键词打分 + 优先级消歧，8 类业务场景 |
| **Query Optimizer** | 分散 SPU/SKC 补全、错别字、跨平台术语、多意图标记 |
| **RAG** | `policy_kb.md` / jsonl，BM25 + 语义向量（依赖可选） |
| **Business API** | `StockOrderRequest` / `PriceReviewRequest`，Mock 可换 HTTP |
| **双推理后端** | 自研 SFT 模型 或 通用 API 模型 |

## 评测数据

- `data/router_eval_50.jsonl` — Router 离线评测（50 条）
- `autodl_eval/test_dataset_100.jsonl` — 基座 vs SFT 推理对比用例（合成问句）

## 未包含内容（隐私与体积）

- 原始商家群聊 / 清洗前 JSONL
- 内部研发日志、终端实录、API 调用日志
- `kb/` 下原始问答 txt（仅保留结构化政策库）
- 模型权重与 LoRA checkpoint

## 技术栈

Python 3.9+ · PyTorch · Transformers · PEFT · Pydantic · OpenAI SDK · jieba · rank-bm25 · sentence-transformers

## License

MIT（如需商用请自行替换知识库与模型授权）
