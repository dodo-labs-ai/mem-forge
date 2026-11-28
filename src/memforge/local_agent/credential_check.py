"""The ``credential_check`` job: a Source's credential checked, or its browser session captured, on the device.

The admin UI's Sign in starts one for any document type (ADR 0058). The
server builds the job from the Source and hands it to the requester's daemon:

- A configuration that signs in with a browser session has the session
  captured from the user's browser, through the type's browser-session
  provider, which also checks it against the provider. A **device Source**
  keeps the session in the device's keyring for the session's origin, and the
  device then holds the Source with that session, so its device report names
  it. A **server
  Source** has the session stored on the server, which serves server Sources
  only.
- A device Source that signs in with a token has the token the device holds
  (``memforge source credentials set``) checked by the type's
  ``authenticate``.

The server holds no credential for a device Source and never sends one in
the job; it builds the job from the configuration without secrets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from memforge.auth.browser_session import (
    CapturedBrowserSession,
    ensure_builtin_providers,
    provider_for_source_type,
)
from memforge.local_agent.device_credentials import DeviceCredentialStore
from memforge.local_agent.document_acquisition import acquisition_job_config
from memforge.local_agent.setup_results import CredentialCheckResult
from memforge.local_agent.source_contract import source_executor
from memforge.source_types import SOURCE_EXECUTOR_DEVICE

__all__ = [
    "CREDENTIAL_CHECK_OPERATION",
    "SessionUploader",
    "credential_check_job_payload",
    "run_credential_check",
    "signs_in_with_browser_session",
]

CREDENTIAL_CHECK_OPERATION = "credential_check"
PRINCIPAL_CHANGED_ERROR = "principal_changed"

SessionUploader = Callable[[str, CapturedBrowserSession], Mapping[str, Any]]
"""Stores a server Source's captured session on the server: ``(provider, session) -> answer``.

The server answers a session of another account than the one it holds with
the provider's ``principal_changed_code``.
"""


def signs_in_with_browser_session(source_type: str, config: Mapping[str, Any]) -> bool:
    """Whether the Source's configuration signs in with a browser session of its type's provider."""

    ensure_builtin_providers()
    provider = provider_for_source_type(source_type)
    return provider is not None and provider.uses_browser_session(dict(config))


def credential_check_job_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    """The job's payload: the Source, where it runs, and its configuration without secrets."""

    source_type = str(source["type"])
    return {
        "source_id": str(source["id"]),
        "source_type": source_type,
        "executor": source_executor(source),
        "config": acquisition_job_config(source_type, dict(source.get("config") or {})),
    }


async def run_credential_check(
    payload: Mapping[str, Any],
    *,
    credentials: DeviceCredentialStore,
    open_gene: Callable[[str, Mapping[str, Any]], Any],
    upload_session: SessionUploader,
    browser: str | None,
) -> CredentialCheckResult:
    """Run one ``credential_check`` job on this device.

    ``open_gene(source_id, config)`` creates the type's Gene for a
    configuration. ``upload_session`` stores a server Source's session on
    the server.
    """

    source_id = str(payload.get("source_id") or "")
    source_type = str(payload.get("source_type") or "")
    config = dict(payload.get("config") or {})
    on_device = payload.get("executor") == SOURCE_EXECUTOR_DEVICE

    def failed(error: str, error_type: str, **fields: Any) -> CredentialCheckResult:
        return CredentialCheckResult(
            operation=CREDENTIAL_CHECK_OPERATION,
            source_id=source_id or None,
            error=error,
            error_type=error_type,
            retryable=False,
            **fields,
        )

    if not source_id or not source_type:
        return failed("source_id and source_type are required", "ValueError")
    try:
        if signs_in_with_browser_session(source_type, config):
            provider = provider_for_source_type(source_type)
            assert provider is not None
            captured = await provider.capture_session(
                str(config.get("base_url") or ""), browser=browser, tls_config=config
            )
            if on_device:
                credentials.hold_session(source_id, provider.provider, captured.origin, captured.cookie_header)
            else:
                uploaded = upload_session(provider.provider, captured)
                if uploaded.get("code") == provider.principal_changed_code:
                    principal = uploaded.get("detail") if isinstance(uploaded.get("detail"), Mapping) else {}
                    return failed(
                        PRINCIPAL_CHANGED_ERROR,
                        "BrowserSessionPrincipalChangedError",
                        origin=principal.get("origin"),
                        old_principal_id=principal.get("old_principal_id"),
                        new_principal_id=principal.get("new_principal_id"),
                    )
                if uploaded.get("error"):
                    return failed(
                        f"Unable to store the {provider.label} session on the server", "BrowserSessionUploadError"
                    )
            return CredentialCheckResult(
                operation=CREDENTIAL_CHECK_OPERATION, source_id=source_id, authenticated=True, origin=captured.origin
            )
        if not on_device:
            return failed("A server Source's credential is checked on the server", "ValueError")
        gene = open_gene(source_id, credentials.gene_config(source_id, source_type, config))
        try:
            await gene.authenticate()
        finally:
            await gene.close()
    except Exception as exc:
        return failed(str(exc), type(exc).__name__)
    return CredentialCheckResult(operation=CREDENTIAL_CHECK_OPERATION, source_id=source_id, authenticated=True)

