"""Canonical semantic contract shared by Memory extraction entrypoints."""

from __future__ import annotations

__all__ = [
    "DURABLE_MEMORY_QUALITY_RULES",
    "PROJECTION_EXTRACTION_CONTRACT_VERSION",
]


PROJECTION_EXTRACTION_CONTRACT_VERSION = "projection-extraction-v5"

DURABLE_MEMORY_QUALITY_RULES = """Top rules (apply these first; reject candidates that fail any of them):

0. PREFER EMPTY. Returning {{"memories": []}} is the default. The bar for emitting a Memory is high: it must teach a future developer something they would otherwise miss six months from now, after the implementation has been refactored. Routine work, mechanical detail, transient output, and meta-discussion produce zero Memories.

1. CODE-RECOVERABLE FACTS ARE NOT MEMORIES. Reject any candidate a developer could verify by reading the current code, schema, types, configuration, or by running `grep` / `git log -p` in under a minute. Specifically, do not emit Memories that merely restate function or method names, class names, type signatures, parameter lists, ID or constant values, file paths, schema columns, migration numbers, generated query text, or "X passes Y to Z" wiring. Keep a candidate only when it states a reusable constraint, reason, rule, invariant, conclusion, or procedure that survives a future refactor.

2. ONE CLAIM, ONE MEMORY. Pick the single most general accurate phrasing for each underlying claim; do not emit reworded duplicates.

3. FOLD REJECTED ALTERNATIVES INTO THE CHOSEN DECISION. Emit one "picked A over B because <reason>" decision rather than separate Memories for rejected alternatives.

4. FUTURE USEFULNESS CHECK. Skip claims that will be obvious after the next refactor, self-resolve within days, or preserve only one generated or observed instance rather than reusable knowledge.

5. NO META-MEMORIES. Do not emit Memories about the act of working: commit structure, diff splitting, tools used, validation output, work-in-progress state, or whether a project rule was followed. Memory is about the project's durable domain knowledge, not the editing process.

6. OWNED EVIDENCE SETS THE LANGUAGE. For each candidate, preserve the language of its owned source evidence. When that evidence is primarily Chinese, write memory.content in Chinese. Do not translate it to English unless the evidence itself is English or mixed-language phrasing is necessary to preserve exact technical identifiers. Read-only context may resolve meaning but must not change the candidate's language.

"""
