"""Smoke test structured source-support verification through LiteLLM."""

from __future__ import annotations

import asyncio

from meminception.config import load_config
from meminception.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
from meminception.runtime import get_effective_llm_config
from meminception.storage.database import Database


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
        llm = await get_effective_llm_config(db, config)
    finally:
        await db.close()

    if not llm.enrichment_api_key:
        raise RuntimeError("MEMINCEPTION_ENRICHMENT_API_KEY or DB enrichment_api_key is required")

    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model=llm.enrichment_model,
            base_url=llm.enrichment_base_url or None,
            api_key=llm.enrichment_api_key,
            timeout_s=llm.request_timeout_s,
        )
    )
    response = await client.verify_source_support(PROMPT)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
