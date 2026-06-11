# -*- coding: utf-8 -*-
"""Phase 1 入库：manifest → sop_chunks.jsonl + sop_parents.jsonl"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

from rag.chunking import build_parent_child_chunks, file_hash
from rag.constants import (
    CHUNKS_JSONL,
    DATA_DIR,
    DEFAULT_SOURCE_DIR,
    INGEST_REPORT_JSON,
    MANIFEST_CSV,
    PARENTS_JSONL,
)
from rag.loaders import load_document
from rag.manifest import build_manifest


def _safe_file_id(rel_path: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", rel_path)[:60]
    return f"f_{base}"


def ingest_from_manifest(
    source_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    chunks_out: Optional[Path] = None,
    parents_out: Optional[Path] = None,
    limit: Optional[int] = None,
) -> dict:
    source_dir = source_dir or DEFAULT_SOURCE_DIR
    manifest_path = manifest_path or MANIFEST_CSV
    chunks_out = chunks_out or CHUNKS_JSONL
    parents_out = parents_out or PARENTS_JSONL

    if not manifest_path.exists():
        build_manifest(source_dir, manifest_path)

    all_parents: list[dict] = []
    all_children: list[dict] = []
    errors: list[dict] = []
    processed = 0

    with manifest_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("include", "").lower() in ("true", "1", "yes")]

    if limit:
        rows = rows[:limit]

    for row in rows:
        rel = row["relative_path"]
        path = source_dir / rel
        if not path.exists():
            errors.append({"file": rel, "error": "not_found"})
            continue
        try:
            blocks_raw = load_document(path)
            blocks = [
                {
                    "text": b.text,
                    "page_or_slide": b.page_or_slide,
                    "section_title": b.section_title,
                }
                for b in blocks_raw
            ]
            if not blocks:
                errors.append({"file": rel, "error": "no_text_extracted"})
                continue

            file_id = _safe_file_id(rel)
            meta = {
                "kb_lane": row["kb_lane"],
                "scene": row["scene"],
                "doc_type": row["doc_type"],
                "source_hash": row["file_hash"],
                "title_hint": Path(row["filename"]).stem,
            }
            parents, children = build_parent_child_chunks(
                file_id, row["filename"], blocks, meta,
            )
            all_parents.extend(parents)
            all_children.extend(children)
            processed += 1
        except Exception as e:
            errors.append({"file": rel, "error": str(e)})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with parents_out.open("w", encoding="utf-8") as f:
        for p in all_parents:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with chunks_out.open("w", encoding="utf-8") as f:
        for c in all_children:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    report = {
        "files_processed": processed,
        "parent_chunks": len(all_parents),
        "child_chunks": len(all_children),
        "errors_count": len(errors),
        "errors_sample": errors[:20],
        "chunks_path": str(chunks_out),
        "parents_path": str(parents_out),
    }
    with INGEST_REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rebuild-manifest", action="store_true")
    args = ap.parse_args()
    if args.rebuild_manifest:
        build_manifest()
    r = ingest_from_manifest(limit=args.limit)
    print(json.dumps(r, ensure_ascii=False, indent=2))
