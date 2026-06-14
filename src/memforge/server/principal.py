"""Resolve the authenticated principal for an HTTP request.

The principal is the only authoritative source of `AccessScope.user_id`. A
request body may carry a `user_id` HINT, but it is ignored by the resolver:
authority comes from the resolved principal, never from a client-declared
field. In v1 (local-only), the resolver returns LOCAL_DEV_USER_ID.
A future authentication integration is the only thing that changes here.
"""

from __future__ import annotations

from fastapi import Request

from memforge.storage.adapters.context import LOCAL_DEV_USER_ID

LOCAL_DEV_WORKSPACE_ROLE = "workspace_admin"


def resolve_principal(request: Request) -> str:
    """Return the authenticated user id for this request.

    v1: local-only, always the dev user. Future auth integration extracts
    the principal from a verified JWT or session here.
    """
    return LOCAL_DEV_USER_ID


def resolve_workspace_role(request: Request) -> str:
    """Return the caller's workspace role for source-management decisions.

    Standalone OSS is a single-user admin surface, so the local principal
    retains full source-management capabilities. Cloud deployments inject a
    resolver through ``create_admin_app`` instead of trusting client input.
    """
    return LOCAL_DEV_WORKSPACE_ROLE
