"""The result each local setup job reports, one model per setup operation.

A setup job is the admin UI asking the user's local daemon to do one
interactive step while a Source is being set up: open the folder picker, list
a repository tree, check a Source's credential or sign in to its site, sign
in to Teams, or list Teams conversations. The
daemon handler for the operation builds its result through the operation's
model and completes the job with ``model.model_dump()``, so the model is the
one owner of the shape and every field it declares is present, as null when
the handler did not set it.

Example: the admin UI creates a ``local_markdown_pick_root`` job, the user
picks a folder, and the daemon completes the job with::

    LocalMarkdownPickRootResult(operation="local_markdown_pick_root", root="/Users/me/notes")

The admin UI reads the job back and finds ``{"operation": ..., "root": ...,
"cancelled": null, "error": null, ...}`` in its ``result``.
``SETUP_RESULT_MODELS`` maps every setup operation to its model, and the
admin API publishes each model in its OpenAPI document so the admin UI's
generated types describe ``result`` per operation. Sync jobs keep their
free-form result.

A job whose result carries ``error`` failed. A handler that raises instead
completes the job as failed with only ``retryable`` in its result and the
message in the job's ``last_error``. A reader therefore reads ``result`` as
the operation's model once the job has succeeded, and on failure reads
``result.error`` when it is present, otherwise the job's ``last_error``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class LocalSetupJobResult(BaseModel):
    """What every setup job reports.

    ``operation`` names the setup operation. ``error`` is set when the job
    failed, as a sentence the admin UI can show, and ``error_type`` names the
    failure's kind when the daemon knows it. The daemon adds
    ``heartbeat_errors`` after the handler returns when renewing the job's
    lease failed while it ran.
    """

    model_config = ConfigDict(extra="forbid")

    operation: str
    error: str | None = None
    error_type: str | None = None
    heartbeat_errors: list[str] | None = None


# -- local_markdown_pick_root --


class LocalMarkdownPickRootResult(LocalSetupJobResult):
    """The folder the user chose in the daemon's native folder picker.

    ``root`` is the chosen absolute path. ``cancelled`` is true when the user
    closed the picker without choosing, which is not a failure. A daemon with
    no native picker fails the job and asks for the path to be typed instead.
    """

    root: str | None = None
    cancelled: bool | None = None


# -- credential_check --


class CredentialCheckResult(LocalSetupJobResult):
    """A Source's credential checked, and a browser session captured, on the user's computer.

    For a Source whose configuration signs in with a browser session, the
    session is captured from the user's browser and checked against the
    provider. A device Source keeps it in the device's keyring for the
    session's origin; a server Source's session is stored on the server. For
    a device Source that signs in with a token, the device's stored token is
    checked by the type's ``authenticate``.

    On success ``authenticated`` is true and ``origin`` is the site a captured
    session belongs to. ``error`` is ``principal_changed`` when the user
    signed in as a different account than the one the server's stored session
    names; ``old_principal_id`` and ``new_principal_id`` then name both
    accounts. ``retryable`` is false for every failure, because renewing a
    credential needs the user.
    """

    source_id: str | None = None
    authenticated: bool | None = None
    origin: str | None = None
    old_principal_id: str | None = None
    new_principal_id: str | None = None
    retryable: bool | None = None


# -- teams_auth_check --


class TeamsAuthCheckResult(LocalSetupJobResult):
    """Whether the daemon holds a usable Teams session, without opening a browser.

    ``expires_in_minutes`` is the time left on the session's soonest-expiring
    token, or ``null`` when there is no usable session. A missing or expired
    session is reported through ``authenticated: false`` and ``error``.
    """

    region: str
    authenticated: bool
    expires_in_minutes: int | None


# -- teams_auth --


class TeamsAuthResult(LocalSetupJobResult):
    """A Teams sign-in through a browser window on the user's computer.

    On success ``authenticated`` is true and ``token_count`` counts the
    tokens the session holds.
    """

    region: str
    authenticated: bool
    token_count: int | None = None


# -- teams_browse --


class TeamsChannelOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    displayName: str


class TeamsTeamOption(BaseModel):
    """A team the user belongs to, with the channels a Source can read."""

    model_config = ConfigDict(extra="forbid")

    id: str
    displayName: str
    channels: list[TeamsChannelOption]


class TeamsConversationOption(BaseModel):
    """A chat or a favourite channel, named by its topic.

    ``lastActivity`` is the time of the latest message as ISO 8601, when
    Teams reports one; a favourite carries no time.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    topic: str
    lastActivity: str | None = None


class TeamsBrowseResult(LocalSetupJobResult):
    """The Teams conversations a Source can read, for the conversation picker.

    Names follow Teams' own fields (``displayName``, ``lastActivity``).
    Group and individual chats are listed by latest activity first. An
    expired session is renewed once through a browser sign-in before the job
    fails; the four lists are absent from a failed job.
    """

    region: str
    favorites: list[TeamsConversationOption] | None = None
    teams: list[TeamsTeamOption] | None = None
    group_chats: list[TeamsConversationOption] | None = None
    individual_chats: list[TeamsConversationOption] | None = None


SETUP_RESULT_MODELS: dict[str, type[LocalSetupJobResult]] = {
    "credential_check": CredentialCheckResult,
    "local_markdown_pick_root": LocalMarkdownPickRootResult,
    "teams_auth": TeamsAuthResult,
    "teams_auth_check": TeamsAuthCheckResult,
    "teams_browse": TeamsBrowseResult,
}
"""Each local setup operation's result model, keyed by operation name."""
