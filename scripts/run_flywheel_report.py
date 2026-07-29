#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成数据飞轮周报：log/flywheel_weekly.md"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flywheel.session_analytics import export_flywheel_report

if __name__ == "__main__":
    out = ROOT / "log" / "flywheel_weekly.md"
    r = export_flywheel_report(output_md=out)
    print(r)
    print(f"Report: {out}")
