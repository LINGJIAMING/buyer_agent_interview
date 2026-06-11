#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 一键流水线（Phase 0～2）

  python -m rag.run_pipeline manifest
  python -m rag.run_pipeline ingest
  python -m rag.run_pipeline index
  python -m rag.run_pipeline eval
  python -m rag.run_pipeline all
"""
from __future__ import annotations

import json
import sys


def main():
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()

    if cmd == "manifest":
        from rag.manifest import build_manifest
        print(json.dumps(build_manifest(), ensure_ascii=False, indent=2))
    elif cmd == "ingest":
        from rag.ingest import ingest_from_manifest
        print(json.dumps(ingest_from_manifest(), ensure_ascii=False, indent=2))
    elif cmd == "index":
        from rag.build_index import build_all
        print(json.dumps(build_all(), ensure_ascii=False, indent=2))
    elif cmd == "eval":
        from rag.run_eval_recall import evaluate
        print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
    elif cmd == "all":
        from rag.manifest import build_manifest
        from rag.ingest import ingest_from_manifest
        from rag.build_index import build_all
        from rag.run_eval_recall import evaluate
        m = build_manifest()
        print("[manifest]", json.dumps(m, ensure_ascii=False))
        i = ingest_from_manifest()
        print("[ingest]", json.dumps(i, ensure_ascii=False))
        x = build_all()
        print("[index]", json.dumps(x, ensure_ascii=False))
        e = evaluate()
        print("[eval]", json.dumps(e, ensure_ascii=False))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
