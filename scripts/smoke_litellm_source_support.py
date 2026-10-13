"""Smoke test structured source-support verification through LiteLLM."""

from __future__ import annotations

import asyncio

from memforge.config import load_config
from memforge.runtime import DefaultRuntimeProvider, resolve_models
from memforge.storage.database import Database


PROMPT = """You are verifying whether a source document directly supports existing team memories.

Only mark a memory supported when the document explicitly states the same durable fact.

<candidate_memories>
[
  {
    "memory_id": "mem-smoke",
    "content": "Cutoff validation runs before payroll group creation.",
    "memory_type": "fact",
    "tags": ["payroll"],
    "confidence": 0.9,
    "corroboration_count": 1
  }
]
</candidate_memories>

<document>
Cutoff validation runs before payroll group creation.
</document>

Return a structured object with a decisions array."""


async def main() -> None:
    config = load_config()
    db = Database(config.storage.db_path)
    await db.connect()
    try:
        models = await resolve_models(db, config)
    finally:
        await db.close()

    client = DefaultRuntimeProvider().build_structured_llm_client(models, max_concurrent=1)
    if client is None:
        raise RuntimeError("No generation model is configured; set one in Settings or MEMFORGE_ENRICHMENT_*")
    response = await client.verify_source_support(PROMPT)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
