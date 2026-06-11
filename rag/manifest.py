# -*- coding: utf-8 -*-
"""扫描 rag_buyer_file，生成 manifest.csv。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from rag.chunking import file_hash
from rag.classify import classify_filename
from rag.constants import (
    DATA_DIR,
    DEFAULT_SOURCE_DIR,
    EXCLUDE_EXTENSIONS,
    INCLUDE_EXTENSIONS,
    MANIFEST_CSV,
)


def scan_source_dir(source_dir: Path) -> list[dict]:
    rows: list[dict] = []
    hash_seen: dict[str, str] = {}

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in EXCLUDE_EXTENSIONS:
            continue
        if ext not in INCLUDE_EXTENSIONS:
            continue
        if path.stat().st_size == 0:
            continue

        fh = file_hash(path)
        rel = path.relative_to(source_dir).as_posix()
        dup_of = ""
        if fh in hash_seen:
            dup_of = hash_seen[fh]
        else:
            hash_seen[fh] = rel

        clf = classify_filename(path.name)
        rows.append({
            "relative_path": rel,
            "filename": path.name,
            "extension": ext,
            "size_bytes": path.stat().st_size,
            "file_hash": fh,
            "include": dup_of == "",
            "duplicate_of": dup_of,
            "kb_lane": clf["kb_lane"],
            "scene": clf["scene"],
            "doc_type": clf["doc_type"],
        })
    return rows


def write_manifest(rows: list[dict], out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "relative_path", "filename", "extension", "size_bytes", "file_hash",
        "include", "duplicate_of", "kb_lane", "scene", "doc_type",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    included = [r for r in rows if r["include"]]
    by_ext: dict[str, int] = defaultdict(int)
    by_lane: dict[str, int] = defaultdict(int)
    for r in included:
        by_ext[r["extension"]] += 1
        by_lane[r["kb_lane"]] += 1
    stats = {
        "total_scanned": len(rows),
        "included": len(included),
        "duplicates_skipped": len(rows) - len(included),
        "by_extension": dict(by_ext),
        "by_lane": dict(by_lane),
    }
    return stats


def build_manifest(
    source_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> dict:
    source_dir = source_dir or DEFAULT_SOURCE_DIR
    out_path = out_path or MANIFEST_CSV
    rows = scan_source_dir(source_dir)
    stats = write_manifest(rows, out_path)
    stats["manifest_path"] = str(out_path)
    stats["source_dir"] = str(source_dir)
    return stats


if __name__ == "__main__":
    s = build_manifest()
    print(json.dumps(s, ensure_ascii=False, indent=2))
