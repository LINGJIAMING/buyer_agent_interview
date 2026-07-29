# generator.py 修复版
import re

from prompts import (
    BASE_ROLE_PROMPT,
    EVIDENCE_BOUND_PROMPT,
    OUTPUT_FORMAT,
    SCENE_PROMPTS,
    STRONG_HIT_PROMPT,
)
from response_utils import inject_links, postprocess_response

def _safe_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def sanitize_messages(messages):
    """强制清洗消息格式，防止 history 污染"""
    clean_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            clean_messages.append({"role": "user", "content": _safe_str(msg)})
            continue
            
        role = _safe_str(msg.get("role", "user"))
        content = _safe_str(msg.get("content", ""))
        
        # 过滤空消息
        if not content.strip():
            continue
            
        clean_messages.append({"role": role, "content": content})
    return clean_messages


def build_messages(
    user_input,
    scene,
    history=None,
    max_history_rounds=3,
    retrieved_context="",
    strong_hit=False,
    subtask="",
    low_confidence=False,
    follow_up_question="",
    skill_body: str = "",
    use_rag_evidence: bool = True,
    merchant_notes: str = "",
):
    """构建消息。skill_body 非空时注入分析 Skill，并默认弱化聊天格式约束。"""

    # 强制转 int，防止传入字符串 "3"
    max_history_rounds = int(max_history_rounds) if max_history_rounds else 3
    history = history or []

    # 1. 系统提示组装
    system_content = BASE_ROLE_PROMPT

    scene_detail = SCENE_PROMPTS.get(scene, "")
    has_subtask = subtask and subtask != "未细分"

    notes = (merchant_notes or "").strip()
    if notes:
        system_content += (
            "\n\n" + notes +
            "\n（以上为该商家长期便签，请结合本轮问题使用；"
            "不要编造便签未提及的事实。）"
        )

    if skill_body:
        system_content += (
            "\n\n【分析 Skill 已启用】请严格按下列模板完成任务；"
            "若缺图片/标题等必要输入，先用一句话说明缺什么并给出可执行的下一步。"
            f"\n\n{skill_body}"
        )
    else:
        if has_subtask:
            system_content += f"\n\n【当前执行任务】{subtask}"
        elif scene_detail:
            system_content += f"\n\n【场景背景】{scene_detail}"

    if (
        use_rag_evidence
        and retrieved_context
        and retrieved_context != "未检索到明确相关的政策资料。"
    ):
        system_content += (
            f"\n\n{EVIDENCE_BOUND_PROMPT}\n\n【检索证据】\n{retrieved_context}"
        )

    if use_rag_evidence and strong_hit is True:
        system_content += f"\n\n{STRONG_HIT_PROMPT}"

    if use_rag_evidence and low_confidence:
        system_content += (
            "\n\n【低置信度处理要求】当前检索置信度较低，"
            "请不要直接给确定性规则结论，先用一句自然的话补问1-2个关键事实。"
        )
        if follow_up_question:
            system_content += f"\n建议追问：{follow_up_question}"

    # 关键修复：policy 场景强制约束，防止幻觉
    if (
        not skill_body
        and scene == "policy"
        and not retrieved_context
    ):
        system_content += (
            "\n\n【重要】未检索到相关政策资料，请明确告知用户当前无法回答，禁止编造。"
        )

    if skill_body:
        system_content += (
            "\n\n【输出】按 Skill 模板要求的结构输出即可；"
            "不要编造罚款金额、未给出的链接或后台操作承诺。"
        )
    else:
        system_content += f"\n\n{OUTPUT_FORMAT}"

    # 2. 组装 messages
    messages = [{"role": "system", "content": system_content}]
    
    # 历史记录处理（强制清洗）
    if history:
        # 只取最近 N 轮（user+assistant 算一轮）
        keep_history = history[-2 * max_history_rounds:]
        messages.extend(sanitize_messages(keep_history))
    
    messages.append({"role": "user", "content": _safe_str(user_input)})
    return messages


def generate_reply(
    tokenizer,
    model,
    messages,
    retrieved_context="",
    max_new_tokens=2048,
    temperature=0.6,
    top_p=0.8,
    repetition_penalty=1.15,
):
    """生成回复（本地 HuggingFace 模型，需 torch）。云端 cli.py 走此函数。"""
    import torch

    messages = sanitize_messages(messages)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = model_inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    response = postprocess_response(response)
    response = inject_links(response, retrieved_context)

    print("[DEBUG] retrieved_context in generator:", retrieved_context[:200] if retrieved_context else "EMPTY")
    print("[DEBUG] response after postprocess:", response)

    return response


def log_trace(user_input, scene, subtask, retrieved_context, response):
    print("\n" + "="*50)
    print(f"Scene: {scene} | Subtask: {subtask}")
    print(f"User: {user_input[:100]}")
    ctx_preview = retrieved_context[:200] if retrieved_context else "None"
    print(f"Context: {ctx_preview}...")
    print(f"Response: {response[:150]}...")
    print("="*50 + "\n")
