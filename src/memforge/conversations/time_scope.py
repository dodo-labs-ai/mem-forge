"""The time scope every conversation Source shares.

Initial History is how far back a conversation Source collects. It is at most
:data:`MAX_INITIAL_HISTORY_DAYS` for every conversation Source, whatever its
Rolling Retention, and never more than a finite retention. A provider that
polls the whole of Initial History on every sync, such as Teams, needs that
bound to keep each sync finite.

This module imports nothing from the reader or storage, so a configuration
check or a migration can use the bound without loading the conversation layer.
"""

from __future__ import annotations

MAX_INITIAL_HISTORY_DAYS = 365
