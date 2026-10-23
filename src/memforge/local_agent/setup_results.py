"""The result each local setup job reports, one model per setup operation.

A setup job is the admin UI asking the user's local daemon to do one
interactive step while a Source is being set up: open the folder picker, list
a repository tree, sign in to Jira or Teams, or list Teams conversations. The
daemon handler for the operation builds its result through the operation's
model and completes the job with ``model.model_dump(exclude_unset=True)``, so
the model is the one owner of the shape and a field the handler did not set
stays absent.

Example: the admin UI creates a ``local_markdown_pick_root`` job, the user
picks a folder, and the daemon completes the job with::

    LocalMarkdownPickRootResult(operation="local_markdown_pick_root", root="/Users/me/notes")

The admin UI reads the job back and finds ``{"operation": ..., "root": ...}``
in its ``result``. ``SETUP_RESULT_MODELS`` maps every setup operation to its
model, and the admin API publishes each model in its OpenAPI document so the
admin UI's generated types describe ``result`` per operation. Sync jobs keep
their free-form result.

A job whose result carries ``error`` failed. A handler that raises instead
completes the job as failed with only ``retryable`` in its result, so a
reader reads ``result`` as the operation's model once the job has succeeded,
and only ``error`` once it has failed.
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


# -- github_repo_preview_tree --


class GitHubSymlinkHop(BaseModel):
    """One symbolic link followed on the way to a file's content."""

    model_config = ConfigDict(extra="forbid")

    path: str
    blob_sha: str
    target_path: str


class GitHubRepoPreviewItem(BaseModel):
    """One repository file the Source would read.

    ``relative_path`` is the path the Source names the file by. When it is a
    symbolic link, ``symlink_chain`` lists the links followed and
    ``resolved_relative_path`` is the file whose content is read.
    """

    model_config = ConfigDict(extra="forbid")

    relative_path: str
    blob_sha: str
    bytes: int
    file_mode: str
    logical_file_mode: str
    symlink_chain: list[GitHubSymlinkHop]
    resolved_relative_path: str | None
    content_type: str


class GitHubRepoPreviewCounts(BaseModel):
    """Files the configured paths and extensions include, and files they leave out."""

    model_config = ConfigDict(extra="forbid")

    included: int
    ignored: int


class GitHubRepoPreviewTreeResult(LocalSetupJobResult):
    """A remote repository's tree as the daemon's Git access reads it.

    The daemon lists the whole tree at ``commit_sha`` of ``ref`` with the
    configured include and exclude paths cleared, so the folder picker can
    offer every folder. ``items`` holds at most the requested number of files,
    and ``counts`` counts all of them. ``extension_counts`` counts the
    included files by extension.
    """

    source_id: str
    profile: str
    repo_url: str
    ref: str
    commit_sha: str
    root_tree_sha: str
    include_paths: list[str]
    exclude_paths: list[str]
    include_extensions: list[str]
    counts: GitHubRepoPreviewCounts
    extension_counts: dict[str, int]
    items: list[GitHubRepoPreviewItem]


# -- local_markdown_pick_root --


class LocalMarkdownPickRootResult(LocalSetupJobResult):
    """The folder the user chose in the daemon's native folder picker.

    ``root`` is the chosen absolute path. ``cancelled`` is true when the user
    closed the picker without choosing, which is not a failure. A daemon with
    no native picker fails the job and asks for the path to be typed instead.
    """

    root: str | None = None
    cancelled: bool | None = None


# -- local_markdown_preview_tree --


class LocalMarkdownPreviewItem(BaseModel):
    """One file under the folder that the Source would read."""

    model_config = ConfigDict(extra="forbid")

    relative_path: str
    title: str
    content_type: str
    raw_hash: str
    bytes: int


class LocalMarkdownPreviewCounts(BaseModel):
    """Files the scan read, and files it left out by reason."""

    model_config = ConfigDict(extra="forbid")

    included: int
    ignored: int
    too_large: int
    invalid_utf8: int
    unreadable: int


class LocalMarkdownPreviewTreeResult(LocalSetupJobResult):
    """The files under a local folder that the configured patterns select.

    ``root`` is the resolved folder and ``vault_id`` the identity its
    Documents are stored under. ``items`` holds at most the requested number
    of files, and ``counts`` counts all of them.
    """

    source_id: str
    profile: str
    root: str
    vault_id: str
    counts: LocalMarkdownPreviewCounts
    items: list[LocalMarkdownPreviewItem]


# -- jira_auth --


class JiraAuthResult(LocalSetupJobResult):
    """A Jira browser session captured on the user's computer and stored for the Source.

    On success ``authenticated`` is true and ``origin`` is the Jira site the
    session belongs to. ``error`` is ``principal_changed`` when the user signed
    in as a different Jira account than the one the stored session names;
    ``old_principal_id`` and ``new_principal_id`` then name both accounts.
    ``retryable`` is false for every failure, because renewing again needs the
    user at the browser.
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
    "github_repo_preview_tree": GitHubRepoPreviewTreeResult,
    "jira_auth": JiraAuthResult,
    "local_markdown_pick_root": LocalMarkdownPickRootResult,
    "local_markdown_preview_tree": LocalMarkdownPreviewTreeResult,
    "teams_auth": TeamsAuthResult,
    "teams_auth_check": TeamsAuthCheckResult,
    "teams_browse": TeamsBrowseResult,
}
"""Each local setup operation's result model, keyed by operation name."""
