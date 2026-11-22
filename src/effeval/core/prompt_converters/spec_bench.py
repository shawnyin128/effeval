"""Spec-Bench `question.jsonl` (github.com/hemingkx/Spec-Bench) into sample records.

The source has 480 rows {question_id, category, turns, ...}: 80 each of MT-bench (eight categories x 10, two
turns), translation, summarization, qa, math_reasoning and rag. The eight MT-bench categories form the
category `mt_bench`; the sample id keeps the source category: `<source category>:<question_id>`.
"""

from __future__ import annotations

import json

MT_BENCH_CATEGORIES = (
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
)
CATEGORIES = ("mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag")


def convert(raw: bytes) -> list[dict]:
    """Sample records of the Spec-Bench questions in file order. Raises ValueError on an unknown category."""
    records = []
    for number, line in enumerate(raw.decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        source = row["category"]
        category = "mt_bench" if source in MT_BENCH_CATEGORIES else source
        if category not in CATEGORIES:
            raise ValueError(f"Spec-Bench line {number}: unknown category {source!r}")
        records.append({"id": f"{source}:{row['question_id']}", "category": category, "turns": row["turns"]})
    return records
