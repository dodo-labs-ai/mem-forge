"""Edition-neutral authority contract for explicit Memory corrections."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class CorrectionAuthority(Protocol):
    """Capabilities already resolved for one authenticated correction actor."""

    @property
    def actor_user_id(self) -> str: ...

    def can_manage_source(self, source: Mapping[str, Any]) -> bool: ...

    def can_manage_workspace_memory(self) -> bool: ...


__all__ = ["CorrectionAuthority"]
