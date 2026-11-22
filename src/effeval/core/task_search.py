"""Ranked task search shared by server-side callers; the same ranking as the web task picker (web/src/compose/taskIndex.ts)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

ALIASES: dict[str, list[str]] = {
    "ppl": ["perplexity", "wikitext"],
    "perplexity": ["ppl", "wikitext"],
    "wiki": ["wikitext"],
    "wiki2": ["wikitext2", "wikitext"],
    "code": ["humaneval", "mbpp", "livecodebench", "live_code_bench"],
    "coding": ["humaneval", "mbpp", "livecodebench", "live_code_bench"],
    "math": ["gsm8k", "math_500", "aime", "amc"],
    "mc": ["mmlu", "arc", "hellaswag"],
}
"""Words researchers type for what a task measures or is called, mapped to the terms they stand for. Keep identical to ALIASES in taskIndex.ts."""


def metric_label(name: str) -> str:
    """A library metric name as shown: filter suffix and module path dropped ("exact_match,strict-match" -> "exact_match")."""
    return name.split(",")[0].split("/")[-1]


def starts_word(text: str, term: str) -> bool:
    """Whether `term` (lower-case) begins a word of `text`: words start after any character that is not a letter or digit, at a lower-to-upper case change, and where letters meet digits."""
    lowered = text.lower()
    start = lowered.find(term)
    while start >= 0:
        if start == 0 or _boundary(text[start - 1], text[start]):
            return True
        start = lowered.find(term, start + 1)
    return False


def _boundary(before: str, here: str) -> bool:
    if not before.isalnum() or not here.isalnum():
        return not before.isalnum()
    return (before.islower() and here.isupper()) or before.isdigit() != here.isdigit()


def _score(task: Mapping[str, Any], term: str, title: str) -> float:
    """Relevance of `task` to one lower-case `term` (lower is better), or -1 when it does not match."""
    name = task["name"]
    lname = name.lower()
    target = task["target"].lower()
    aliases = ALIASES.get(term, [])
    tags = task["tags"]
    description = task["description"]
    if term in (target, lname):
        return 0
    if lname.startswith(term) or target.startswith(term):
        return 1
    if task["category"] == term:
        return 1.5
    if starts_word(name, term):
        return 2
    if any(target == a or target.startswith(a) for a in aliases):
        return 2.2
    if any(starts_word(name, a) for a in aliases):
        return 2.5
    if any(starts_word(t, term) for t in tags):
        return 3
    if term == title or any(metric_label(m["name"]).lower() == term for m in task["metrics"]):
        return 3.5
    if starts_word(description, term):
        return 4
    if term in lname:
        return 5
    if any(term in t.lower() for t in tags):
        return 5.5
    return 6 if term in description.lower() else -1


def search_tasks(
    tasks: Mapping[str, Mapping[str, Any]], query: str, titles: Mapping[str, str] | None = None
) -> list[str]:
    """Names of the tasks (entries of describe's `tasks`) matching every whitespace-separated term of `query`, most relevant first.

    A term matches a task's name, tags, metric names, category or description; whole words and word
    prefixes ("ppl" in "wikitext2_ppl") rank before bare substrings ("ppl" in "applied"), and a term's
    aliases (ppl, perplexity, wiki, code, math, mc, ...) match too. Exact name or target first, then prefix, category
    name, word in name, alias, tag, metric or category, description, substrings last; at equal rank groups
    precede leaves, then shorter names. Subtasks of a matching group are left out. `titles` maps a category
    to its display title. An empty `query` returns every task by name, subtasks included."""
    terms = query.lower().split()
    titles = {c: t.lower() for c, t in (titles or {}).items()}
    if not terms:
        return sorted(tasks)
    scores: dict[str, float] = {}
    for name, task in tasks.items():
        per_term = [_score(task, term, titles.get(task["category"], "")) for term in terms]
        if min(per_term) >= 0:
            scores[name] = max(per_term)
    parents: dict[str, list[str]] = {}
    for name, task in tasks.items():
        for member in task["subtasks"]:
            parents.setdefault(member, []).append(name)

    def has_matching_ancestor(name: str, seen: frozenset[str] = frozenset()) -> bool:
        return any(
            p in scores or (p not in seen and has_matching_ancestor(p, seen | {name}))
            for p in parents.get(name, [])
        )

    leaves = [n for n in scores if not has_matching_ancestor(n)]
    return sorted(leaves, key=lambda n: (scores[n], not tasks[n]["subtasks"], len(n), n))
