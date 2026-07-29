# 公开仓库上传核对清单

> 维护者：凌嘉铭  
> 用途：每次从 `mybuyer_agent-main` 同步到本仓库前，对照此表确认。**默认「不上传」除非打勾。**

## 请你确认（回复时可改）

| 项 | 默认 | 说明 |
|----|------|------|
| `data/faq_published.jsonl`（27 条政策 FAQ + 平台公开链接） | ✅ 已上传 | 无 Key；含 agentseller / kuajing 等**公开** URL |
| `data/prompt_injection_eval_50.jsonl` | ✅ 已上传 | 仅测试用例，无真实商家对话 |
| `kb/policy_kb.jsonl` / `.md` | ✅ 保留 | 政策摘要（展示用） |
| `rag/data/sop_chunks.jsonl` | ✅ 保留 | 已 redact 内网链 |
| `skills/` 男装 Prompt 包 | ✅ 已上传 | `menswear_full_category`：registry + prompts.yaml + 全品类.html |
| `log/api_chat.jsonl` 真实对话 | ❌ 不上传 | 含商家原话 |
| `data/response_cache.jsonl` | ❌ 不上传 | 可能含用户问法 |
| `data/buyer_memory.db` | ❌ 不上传 | 本地会话库 |
| `secrets/operator_keys.json` | ❌ 不上传 | 仅 example |
| `.env` | ❌ 不上传 | 仅 `.env.example` |
| `log/*_DEVLOG.md` 全文 | ❌ 不上传 | 摘要见 `docs/CHANGELOG_2026-07.md` |
| `eval_output/` 含完整模型回复 | ❌ 不上传 | 评测 JSON 可脱敏后单独加 |
| 简历 / 面试记录 / 手机号邮箱 | ❌ 不上传 | 不进本仓库 |

**2026-07-29 同步说明**：FAQ/缓存/Web 已推送；**skills/** 男装包已按维护者确认补传。
