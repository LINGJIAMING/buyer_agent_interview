# -*- coding: utf-8 -*-
"""Query Optimizer 回归用例（对齐 router_eval / agent_eval 风格）。"""
import json
from pathlib import Path

from query_optimizer import QueryOptimizer

CASES = [
    {
        "id": "QO001",
        "query": "这个价格能核多少",
        "history": [{"role": "user", "content": "SPU:814473175 在核价"}],
        "expect_contains": ["SPU:814473175", "核"],
    },
    {
        "id": "QO002",
        "query": "嗯那个 直通车怎么投",
        "history": [],
        "expect_contains": ["开广告"],
    },
    {
        "id": "QO003",
        "query": "这款不是同款 帮我申诉",
        "history": [{"role": "user", "content": "1422624924 加绒裤"}],
        "expect_contains": ["申诉"],
    },
    {
        "id": "QO004",
        "query": "审办不通过",
        "history": [],
        "expect_contains": ["审版"],
    },
    {
        "id": "QO005",
        "query": "另外加站也推进 还有限流申诉",
        "history": [{"role": "user", "content": "SPU:5890623888"}],
        "expect_flags": ["multi_action"],
    },
]


def main():
    opt = QueryOptimizer(enable_file_log=False)
    passed = 0
    for c in CASES:
        opt.reset_session()
        r = opt.optimize(c["query"], history=c.get("history"))
        ok = True
        for kw in c.get("expect_contains", []):
            if kw not in r.optimized_query:
                ok = False
        for fl in c.get("expect_flags", []):
            if fl not in r.flags:
                ok = False
        status = "PASS" if ok else "FAIL"
        passed += int(ok)
        print(f"{status} {c['id']} | {r.raw_query} -> {r.optimized_query} | {r.flags}")

    print(f"\n{passed}/{len(CASES)} passed")
    out = Path(__file__).parent / "log" / "query_opt" / "test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"passed": passed, "total": len(CASES)}, f)
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
