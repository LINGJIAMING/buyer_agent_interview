#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""备货 / 核价 业务 API 链路测试（无需 torch）。"""
from __future__ import annotations

import json

from router import route_query, detect_subtask
from business.executor import BusinessActionExecutor
from business.schemas import AgentActionType

CASES = [
    {
        "id": "B01",
        "input": "帮我下个单，skc12345678的L码10件",
        "expect_action": AgentActionType.STOCK_ORDER,
        "expect_api": True,
    },
    {
        "id": "B02",
        "input": "SKC:9088776655 备货 20件",
        "expect_action": AgentActionType.STOCK_ORDER,
        "expect_api": True,
    },
    {
        "id": "B03",
        "input": "帮我备个货",
        "expect_action": AgentActionType.CLARIFY,
        "expect_api": False,
    },
    {
        "id": "B04",
        "input": (
            "*需核价货品SKC：2802453709 *币种 (人民币/美元)：人民币 "
            "*申诉内容：热转印工艺，涤纶材质，报价38人民币。"
        ),
        "expect_action": AgentActionType.PRICE_REVIEW,
        "expect_api": True,
    },
    {
        "id": "B05",
        "input": "这个价能核一下吗",
        "expect_action": AgentActionType.CLARIFY,
        "expect_api": False,
    },
]


def main():
    ex = BusinessActionExecutor(api_mode="mock", enable_file_log=True)
    passed = 0
    for c in CASES:
        scene = route_query(c["input"])
        subtask = detect_subtask(scene, c["input"])
        out = ex.try_execute(c["input"], scene, subtask)
        action = out.action if out else AgentActionType.NONE
        api_ok = bool(out and out.api_called) == c["expect_api"]
        action_ok = c["expect_action"] is None or action == c["expect_action"]
        ok = api_ok and action_ok
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"\n{status} {c['id']} scene={scene} subtask={subtask}")
        print(f"  in : {c['input'][:60]}...")
        if out:
            print(f"  action={action.value} api={out.api_called}")
            print(f"  msg: {out.user_message[:100]}")
            if out.payload:
                print(f"  payload: {json.dumps(out.payload, ensure_ascii=False)[:120]}")
        else:
            print("  -> 未命中业务 API，将走 LLM")

    print(f"\n{passed}/{len(CASES)} passed")
    print("日志: log/business_api/business_api.jsonl")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
