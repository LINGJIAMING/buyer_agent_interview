import argparse
import json

from router import route_query, detect_subtask
from retriever import Retriever
from config import POLICY_KB_PATH

retriever = Retriever(kb_path=POLICY_KB_PATH)

test_queries = [
    "极速起量是什么",
    "spu没在售，什么情况？",
    "spu没在售，帮我加站一下",
    "spu图片更新失败，帮我推进",
    "水洗唛有什么要求",
    "能不能免审",
    "帮我下个单，skc123的l码10件",
    "这个价格太低了，能不能涨",
    "审版不通过怎么办",
    "怎么加站",
    "已审spu48789，待审spu465456",
    "活动价格太低了，想退出活动",
    "SPU ID：814473175 这个比价看下是不是比错了 申诉一下",
]


def _safe_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def sanitize_messages(messages):
    clean_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            clean_messages.append({"role": "user", "content": _safe_str(msg)})
            continue
        role = _safe_str(msg.get("role", "user"))
        content = _safe_str(msg.get("content", ""))
        if not content.strip():
            continue
        clean_messages.append({"role": role, "content": content})
    return clean_messages


def debug_router_only():
    """仅 Router + Retriever，不需要 torch。"""
    for q in test_queries:
        scene = route_query(q)
        subtask = detect_subtask(scene, q)
        retrieved_result = retriever.retrieve_context(q, scene=scene)
        ctx_preview = (retrieved_result.get("context") or "")[:80]
        print(f"\n{'#' * 50} {q[:40]}")
        print(f"scene={scene} | subtask={subtask}")
        print(f"strong_hit={retrieved_result.get('strong_hit')} | ctx={ctx_preview}...")


def debug_messages():
    """完整链路含 build_messages，需要安装 torch + transformers。"""
    from generator import build_messages

    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是买手助手。"},
    ]

    for q in test_queries:
        scene = route_query(q)
        subtask = detect_subtask(scene, q)
        retrieved_result = retriever.retrieve_context(q, scene=scene)
        messages = build_messages(
            user_input=q,
            scene=scene,
            history=history,
            max_history_rounds=3,
            retrieved_context=retrieved_result["context"],
            strong_hit=retrieved_result["strong_hit"],
            subtask=subtask,
            low_confidence=retrieved_result.get("low_confidence", False),
            follow_up_question=retrieved_result.get("follow_up_question", ""),
        )
        messages = sanitize_messages(messages)
        print(f"\n{'#' * 50} 测试问题: {q} {'#' * 50}")
        print(f"识别场景: {scene} | 子任务: {subtask}")
        print(str(messages)[:200])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="含 generator（需 torch）；默认仅 router+retriever",
    )
    args = parser.parse_args()
    if args.full:
        debug_messages()
    else:
        debug_router_only()
