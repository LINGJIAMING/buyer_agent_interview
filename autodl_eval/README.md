# AutoDL 批量推理评测包

上传到实例后，在 **同目录** 下运行即可。无需 Router/RAG，纯对话推理，对比 **基座 Qwen** vs **SFT LoRA checkpoint**。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `test_dataset_100.jsonl` | 100 条测试对话（50 Agent 场景 + 30 Router 单轮 + 20 测试集真实末轮） |
| `run_inference_autodl.py` | 批量推理脚本，输出带 `base_response` / `sft_response` |
| `inference_results_100.jsonl` | 运行后生成（可下载回本地） |
| `inference_results_100_summary.jsonl` | 精简版，方便 Excel / 人工抽检 |

## 测试集构成

| 来源 | 条数 | id 前缀 | 说明 |
| --- | --- | --- | --- |
| agent_eval | 50 | AE001–AE050 | 含 `reference_reply`（GT 要点，供后续 Judge） |
| router_eval | 30 | RT001–RT030 | 单轮业务句，带 scene 标签 |
| test_holdout | 20 | HD001–HD020 | 从 `qwen_messages.test` 抽的真实末轮对话 + 历史参考回复 |

## 在 AutoDL 上操作步骤

### 1. 上传文件夹

将整个 `autodl_eval` 目录上传到例如：

```
/root/autodl-tmp/buyer_eval/
```

（Jupyter 左侧上传，或 `scp` / AutoDL 网盘）

### 2. 确认模型路径

编辑 `run_inference_autodl.py` 顶部两行（与你截图一致）：

```python
BASE_MODEL_PATH = "/root/.cache/modelscope/hub/models/qwen/Qwen2___5-7B-Instruct"
ADAPTER_PATH = "/root/autodl-tmp/LLaMA-Factory/saves/buyer_agent_v2_1/checkpoint-XXXX"
```

若基座在 `saves/Qwen2.5-7B`，改成对应路径。

### 3. 运行

```bash
cd /root/autodl-tmp/buyer_eval
python run_inference_autodl.py
```

冒烟 5 条：

```bash
python run_inference_autodl.py --limit 5
```

只跑 SFT（基座已跑完）：

```bash
python run_inference_autodl.py --only sft --resume
```

中断后续跑：

```bash
python run_inference_autodl.py --resume
```

### 4. 下载结果

下载这两个文件到本地：

- `inference_results_100.jsonl`
- `inference_results_100_summary.jsonl`

## 输出字段

```json
{
  "id": "AE001",
  "source": "agent_eval",
  "scenario": "信息缺失-需澄清",
  "input": "帮我推进一下这个商品的加站",
  "reference_reply": "先澄清再执行：请提供 SPU/SKC...",
  "messages": [{"role": "user", "content": "..."}],
  "base_response": "基座模型输出",
  "sft_response": "微调模型输出"
}
```

## 本地后续评估（你来做）

1. **人工抽检**：对比 `base_response` vs `sft_response`，看是否更像买手、少幻觉。
2. **LLM-as-Judge**：用 `reference_reply` 作 rubric，对两条回复分别打分（口吻 / 可执行 / 幻觉）。

## 预估耗时

- 100 条 × 2 模型 ≈ 200 次生成，7B + LoRA 约 **30–90 分钟**（视 GPU 而定）
- 建议先 `--limit 5` 确认路径无误再全量跑
