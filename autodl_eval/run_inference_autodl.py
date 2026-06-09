#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoDL 上运行：基座 Qwen vs SFT(LoRA) 批量推理，结果写入 JSONL 便于下载。

用法（在 Jupyter 终端或 Notebook 里）：
  cd /root/autodl-tmp/buyer_eval   # 上传本目录后
  python run_inference_autodl.py
  python run_inference_autodl.py --only base
  python run_inference_autodl.py --only sft --resume
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 按你的 AutoDL 路径修改 ============
BASE_MODEL_PATH = "/root/.cache/modelscope/hub/models/qwen/Qwen2___5-7B-Instruct"
# 若基座在 saves 下，可改为：
# BASE_MODEL_PATH = "/root/autodl-tmp/LLaMA-Factory/saves/Qwen2.5-7B"

ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH",
    "/root/autodl-tmp/LLaMA-Factory/saves/buyer_agent_v2_1/checkpoint-XXXX",
)

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATASET = SCRIPT_DIR / "test_dataset_100.jsonl"
OUTPUT_FILE = SCRIPT_DIR / "inference_results_100.jsonl"

SYSTEM_PROMPT = (
    "你是一个电商平台买手，协助商家解答平台规则、政策、操作流程等问题。"
    "回复要像真实买手在群里说话：简短、直接、可执行，不要AI腔，不要分点标题。"
)

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.6
TOP_P = 0.85
REPETITION_PENALTY = 1.08


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_chat_messages(case: dict) -> list[dict]:
    """组装送入模型的 messages（含 system）。"""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    history = case.get("messages") or [{"role": "user", "content": case["input"]}]
    for m in history:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if content and role in ("user", "assistant"):
            msgs.append({"role": role, "content": content})
    # 若 history 未包含最后一条 input，补上
    if not msgs or msgs[-1].get("content") != case["input"]:
        if msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"] == case["input"]:
            pass
        else:
            last_is_user = msgs and msgs[-1]["role"] == "user"
            if not last_is_user:
                msgs.append({"role": "user", "content": case["input"]})
    return msgs


@torch.no_grad()
def generate_one(tokenizer, model, messages: list[dict]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def load_base_model(device_dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=device_dtype,
    )
    model.eval()
    return tokenizer, model


def load_sft_model(device_dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=device_dtype,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return tokenizer, model


def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    return done


def save_all_results(path: Path, case_order: list[dict], results: dict[str, dict]):
    """按测试集顺序整文件写入，避免 --resume 时重复行。"""
    with path.open("w", encoding="utf-8") as f:
        for case in case_order:
            row = results.get(case["id"])
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_phase(
    phase: str,
    cases: list[dict],
    results: dict[str, dict],
    loader,
    field_name: str,
    output_path: Path,
):
    print(f"\n===== 阶段: {phase} =====")
    tokenizer, model = loader()
    t0 = time.time()

    for i, case in enumerate(cases):
        cid = case["id"]
        row = results.get(cid)
        if row is None:
            row = {
                "id": cid,
                "source": case.get("source", ""),
                "scenario": case.get("scenario", ""),
                "subtask": case.get("subtask", ""),
                "input": case["input"],
                "reference_reply": case.get("reference_reply", ""),
                "messages": case.get("messages", []),
            }
            results[cid] = row

        if row.get(field_name):
            print(f"  [{i+1}/{len(cases)}] {cid} skip (已有结果)")
            continue

        msgs = build_chat_messages(case)
        try:
            reply = generate_one(tokenizer, model, msgs)
            row[field_name] = reply
            row[f"{field_name}_at"] = datetime.now().isoformat(timespec="seconds")
            save_all_results(output_path, cases, results)
            print(f"  [{i+1}/{len(cases)}] {cid} ok | len={len(reply)}")
        except Exception as e:
            row[field_name] = ""
            row[f"{field_name}_error"] = str(e)
            save_all_results(output_path, cases, results)
            print(f"  [{i+1}/{len(cases)}] {cid} ERROR: {e}")

    free_model(model)
    print(f"{phase} 完成，耗时 {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=TEST_DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--only", choices=["base", "sft", "both"], default="both")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="续跑：读取已有 output，跳过已生成字段",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    results: dict[str, dict] = {}
    if args.resume and args.output.exists():
        results = load_existing_results(args.output)
        print(f"续跑：已加载 {len(results)} 条历史结果")
    elif args.output.exists() and not args.resume:
        # 新跑：备份旧文件
        bak = args.output.with_suffix(
            f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        args.output.rename(bak)
        print(f"已备份旧结果 -> {bak}")

    meta = {
        "base_model_path": BASE_MODEL_PATH,
        "adapter_path": ADAPTER_PATH,
        "n_cases": len(cases),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    print("配置:", json.dumps(meta, ensure_ascii=False, indent=2))

    if args.only in ("base", "both"):
        run_phase("base", cases, results, load_base_model, "base_response", args.output)

    if args.only in ("sft", "both"):
        run_phase("sft", cases, results, load_sft_model, "sft_response", args.output)

    # 汇总 CSV 友好版（可选）
    summary_path = args.output.with_name("inference_results_100_summary.jsonl")
    with summary_path.open("w", encoding="utf-8") as f:
        for case in cases:
            row = results.get(case["id"], {})
            f.write(
                json.dumps(
                    {
                        "id": row.get("id", case["id"]),
                        "scenario": row.get("scenario", ""),
                        "input": row.get("input", case["input"]),
                        "reference_reply": row.get("reference_reply", ""),
                        "base_response": row.get("base_response", ""),
                        "sft_response": row.get("sft_response", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"\n全部完成。主结果: {args.output}")
    print(f"精简版（便于下载抽检）: {summary_path}")


if __name__ == "__main__":
    main()
