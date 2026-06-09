# generator.py 修复版
import re

from prompts import BASE_ROLE_PROMPT, OUTPUT_FORMAT, SCENE_PROMPTS, STRONG_HIT_PROMPT
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
    follow_up_question=""
):
    """构建消息，修复：history 默认参数、int 类型转换"""
    
    # 强制转 int，防止传入字符串 "3"
    max_history_rounds = int(max_history_rounds) if max_history_rounds else 3
    history = history or []
    
    # 1. 系统提示组装
    
    system_content = BASE_ROLE_PROMPT
    
    scene_detail = SCENE_PROMPTS.get(scene, "")
    has_subtask = subtask and subtask != "未细分"
    
    if has_subtask:
        # 此时模型注意力集中在具体任务上，减少干扰
        system_content += f"\n\n【当前执行任务】{subtask}"
    elif scene_detail:
        # 仅在没有具体子任务时，才提供大场景的背景说明
        system_content += f"\n\n【场景背景】{scene_detail}"
    
    if retrieved_context and retrieved_context != "未检索到明确相关的政策资料。":
        system_content += f"\n\n【参考政策资料】\n{retrieved_context}"
    
    if strong_hit == True :
        system_content += f"\n\n{STRONG_HIT_PROMPT}"

    if low_confidence:
        system_content += (
            "\n\n【低置信度处理要求】当前检索置信度较低，"
            "请不要直接给确定性规则结论，先用一句自然的话补问1-2个关键事实。"
        )
        if follow_up_question:
            system_content += f"\n建议追问：{follow_up_question}"


    VALID_ROLES = {"system", "user", "assistant"}
    


    # 关键修复：policy 场景强制约束，防止幻觉
    if scene == "policy" and not retrieved_context:
        system_content += "\n\n【重要】未检索到相关政策资料，请明确告知用户当前无法回答，禁止编造。"
    
   
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
