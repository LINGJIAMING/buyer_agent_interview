#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Router 离线评测：scene/subtask 准确率"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from router import route_query, detect_subtask

EVAL_PATH = Path(__file__).parent / "data" / "router_eval_50.jsonl"


def load_cases(path: Path):
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def evaluate(cases):
    scene_ok = 0
    subtask_ok = 0
    joint_ok = 0
    errors = []

    for c in cases:
        q = c["input"]
        gt = c["ground_truth"]
        pred_scene = route_query(q)
        pred_sub = detect_subtask(pred_scene, q)

        s_ok = pred_scene == gt["scene"]
        t_ok = pred_sub == gt["subtask"]
        if s_ok:
            scene_ok += 1
        if t_ok:
            subtask_ok += 1
        if s_ok and t_ok:
            joint_ok += 1
        else:
            errors.append(
                {
                    "id": c["id"],
                    "input": q,
                    "gt_scene": gt["scene"],
                    "pred_scene": pred_scene,
                    "gt_subtask": gt["subtask"],
                    "pred_subtask": pred_sub,
                }
            )

    n = len(cases)
    return {
        "total": n,
        "scene_acc": round(scene_ok / n, 4),
        "subtask_acc": round(subtask_ok / n, 4),
        "joint_acc": round(joint_ok / n, 4),
        "errors": errors,
    }


def scene_metrics(cases):
    """Per-scene precision / recall / F1 (macro 在 main 中汇总)."""
    labels = sorted({c["ground_truth"]["scene"] for c in cases})
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for c in cases:
        gt = c["ground_truth"]["scene"]
        pred = route_query(c["input"])
        if pred == gt:
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1

    rows = []
    for lb in labels:
        p = tp[lb] / (tp[lb] + fp[lb]) if (tp[lb] + fp[lb]) else 0.0
        r = tp[lb] / (tp[lb] + fn[lb]) if (tp[lb] + fn[lb]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append((lb, p, r, f1, tp[lb], fn[lb]))
    return rows


def print_confusion(cases):
    matrix = defaultdict(Counter)
    for c in cases:
        gt_scene = c["ground_truth"]["scene"]
        pred_scene = route_query(c["input"])
        matrix[gt_scene][pred_scene] += 1

    print("\n[Scene Confusion] GT -> Pred")
    for gt_scene in sorted(matrix.keys()):
        preds = ", ".join(f"{k}:{v}" for k, v in matrix[gt_scene].items())
        print(f"  {gt_scene}: {preds}")


def main():
    cases = load_cases(EVAL_PATH)
    result = evaluate(cases)

    print("=== Router Eval Report ===")
    print(f"cases: {result['total']}")
    print(f"scene accuracy: {result['scene_acc']:.2%}")
    print(f"subtask accuracy: {result['subtask_acc']:.2%}")
    print(f"joint accuracy: {result['joint_acc']:.2%}")

    rows = scene_metrics(cases)
    macro_p = sum(r[1] for r in rows) / len(rows)
    macro_r = sum(r[2] for r in rows) / len(rows)
    macro_f1 = sum(r[3] for r in rows) / len(rows)
    print(f"\nscene macro precision: {macro_p:.2%}")
    print(f"scene macro recall: {macro_r:.2%}")
    print(f"scene macro F1: {macro_f1:.2%}")
    print("\n[Per-scene] precision | recall | F1 | support(tp) | missed(fn)")
    for lb, p, r, f1, t, fn in rows:
        print(f"  {lb:20s} {p:6.1%} | {r:6.1%} | {f1:6.1%} | {t:2d} | {fn:2d}")

    if result["errors"]:
        print(f"\nerrors: {len(result['errors'])}")
        for e in result["errors"][:20]:
            print(
                f"- {e['id']} | GT={e['gt_scene']}/{e['gt_subtask'][:12]} "
                f"| PRED={e['pred_scene']}/{e['pred_subtask'][:12]} | {e['input'][:40]}"
            )
        if len(result["errors"]) > 20:
            print(f"... and {len(result['errors']) - 20} more")

    print_confusion(cases)


if __name__ == "__main__":
    main()
