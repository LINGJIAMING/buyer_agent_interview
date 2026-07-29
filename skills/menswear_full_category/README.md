# TEMU 男装全品类买手分析 · Skill 包

> **状态：** V0.2 — Prompt 模板库 + **注册表/选择器已接入** `app_api` / Web  
> **定位：** Skill（场景能力包）；咨询链路仍用 Router+RAG，分析类由模型从目录选用

## 目录

```text
menswear_full_category/
├── README.md
├── prompts.yaml          # Prompt 真相源
└── static/
    └── 全品类.html        # 一键复制 UI

# 包外（skills 根）
../registry.py            # 注册表
../selector.py            # 模型选择 skill_id
```

## 运行时怎么启用

对话走 `BuyerAgentApiApp.chat` 时：

1. `select_skill` 看注册表目录  
2. 命中则注入对应 `body`，**跳过政策 RAG**  
3. 日志字段：`skill_id` / `skill_reason`（见 `log/api_chat/api_chat.jsonl`）

详情：`log/SKILL_REGISTRY_DEVLOG.md`

关闭：`ENABLE_SKILL_SELECTOR=false`

## 页面

- 聊天：`http://127.0.0.1:8001/`（页头显示 `skill=...`）  
- 复制模板：`/tools/menswear`

## 重新从 HTML 抽取

```bash
python skills/_extract_menswear_prompts.py
```

抽完后若进程未重启，可 `from skills.registry import clear_catalog_cache; clear_catalog_cache()`。
