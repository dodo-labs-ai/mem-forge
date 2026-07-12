from __future__ import annotations

from pathlib import Path


class FakePage:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.waits: list[int] = []

    def goto(self, url: str, **_kwargs) -> None:
        self.urls.append(url)

    def wait_for_timeout(self, milliseconds: int) -> None:
        self.waits.append(milliseconds)


class FakeContext:
    def __init__(self, cookies: list[dict], *, session_active: bool = True) -> None:
        self.page = FakePage()
        self.cookie_rows = cookies
        self.session_active = session_active
        self.cookie_urls: list[list[str]] = []
        self.validation_urls: list[str] = []
        self.stored_cookies: list[dict[str, str]] = []
        self.closed = False

    def new_page(self) -> FakePage:
        return self.page

    def cookies(self, urls: list[str]) -> list[dict]:
        self.cookie_urls.append(urls)
        return self.cookie_rows

    def jira_session_is_active(self, request_url: str) -> bool:
        self.validation_urls.append(request_url)
        return self.session_active

    def add_cookies(self, cookies: list[dict[str, str]]) -> None:
        self.stored_cookies.extend(cookies)

    def close(self) -> None:
        self.closed = True


class FakeLauncher:
    def __init__(self, context: FakeContext) -> None:
        self.context = context
        self.calls: list[tuple[Path, bool]] = []

    def launch_persistent_context(self, profile_dir: Path, *, headless: bool) -> FakeContext:
        self.calls.append((profile_dir, headless))
        return self.context


class FakeCookieStore:
    def __init__(self, loaded: str | None = None) -> None:
        self.loaded = loaded
        self.saved: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    def load(self, origin: str) -> str | None:
        return self.loaded

    def save(self, origin: str, cookie_header: str) -> None:
        self.saved.append((origin, cookie_header))

    def delete(self, origin: str) -> None:
        self.deleted.append(origin)


def test_jira_browser_session_uses_origin_scoped_headless_profile(tmp_path: Path):
    from memforge.auth.jira_browser_session import JiraBrowserCaptureStatus, JiraBrowserSession

    context = FakeContext([
        {"name": "JSESSIONID", "value": "silent", "expires": -1},
        {"name": "atlassian.xsrf.token", "value": "xsrf", "expires": -1},
    ])
    launcher = FakeLauncher(context)
    cookie_store = FakeCookieStore()
    session = JiraBrowserSession(profile_root=tmp_path, browser_launcher=launcher, cookie_store=cookie_store)

    result = session.capture(
        origin="https://jira.example.test",
        interactive=False,
        timeout_seconds=0,
        poll_interval_seconds=0.1,
        rejected_cookie_hashes=set(),
    )

    assert result.status is JiraBrowserCaptureStatus.CAPTURED
    assert result.cookie_header == "JSESSIONID=silent; atlassian.xsrf.token=xsrf"
    assert launcher.calls[0][1] is True
    assert launcher.calls[0][0].parent == tmp_path
    assert "jira.example.test" in launcher.calls[0][0].name
    assert context.page.urls == ["https://jira.example.test/rest/api/2/myself"]
    assert context.cookie_urls == [["https://jira.example.test/rest/api/2/myself"]]
    assert context.validation_urls == ["https://jira.example.test/rest/api/2/myself"]
    assert context.closed is True
    assert cookie_store.saved == [(
        "https://jira.example.test",
        "JSESSIONID=silent; atlassian.xsrf.token=xsrf",
    )]


def test_jira_browser_session_requests_interaction_when_silent_profile_has_no_session(tmp_path: Path):
    from memforge.auth.jira_browser_session import JiraBrowserCaptureStatus, JiraBrowserSession

    session = JiraBrowserSession(
        profile_root=tmp_path,
        browser_launcher=FakeLauncher(FakeContext([])),
        cookie_store=FakeCookieStore(),
    )

    result = session.capture(
        origin="https://jira.example.test",
        interactive=False,
        timeout_seconds=0,
        poll_interval_seconds=0.1,
        rejected_cookie_hashes=set(),
    )

    assert result.status is JiraBrowserCaptureStatus.INTERACTION_REQUIRED


def test_jira_browser_session_rejects_sso_page_cookies_without_active_principal(tmp_path: Path):
    from memforge.auth.jira_browser_session import JiraBrowserCaptureStatus, JiraBrowserSession

    context = FakeContext(
        [{"name": "SSO", "value": "redirect-cookie", "expires": -1}],
        session_active=False,
    )
    session = JiraBrowserSession(
        profile_root=tmp_path,
        browser_launcher=FakeLauncher(context),
        cookie_store=FakeCookieStore(),
    )

    result = session.capture(
        origin="https://jira.example.test",
        interactive=False,
        timeout_seconds=0,
        poll_interval_seconds=0.1,
        rejected_cookie_hashes=set(),
    )

    assert result.status is JiraBrowserCaptureStatus.INTERACTION_REQUIRED
    assert context.validation_urls == ["https://jira.example.test/rest/api/2/myself"]


def test_jira_browser_session_stores_validated_system_cookies_in_keychain_boundary(tmp_path: Path):
    from memforge.auth.jira_browser_session import JiraBrowserSession

    context = FakeContext([])
    launcher = FakeLauncher(context)
    cookie_store = FakeCookieStore()
    session = JiraBrowserSession(profile_root=tmp_path, browser_launcher=launcher, cookie_store=cookie_store)

    session.store(
        origin="https://jira.example.test",
        cookie_header="JSESSIONID=valid; atlassian.xsrf.token=xsrf",
    )

    assert launcher.calls == []
    assert cookie_store.saved == [(
        "https://jira.example.test",
        "JSESSIONID=valid; atlassian.xsrf.token=xsrf",
    )]


def test_jira_browser_session_injects_keychain_cookie_into_headless_context(tmp_path: Path):
    from memforge.auth.jira_browser_session import JiraBrowserSession

    context = FakeContext([
        {"name": "JSESSIONID", "value": "stored", "expires": -1},
    ])
    session = JiraBrowserSession(
        profile_root=tmp_path,
        browser_launcher=FakeLauncher(context),
        cookie_store=FakeCookieStore("JSESSIONID=stored"),
    )

    result = session.capture(
        origin="https://jira.example.test",
        interactive=False,
        timeout_seconds=0,
        poll_interval_seconds=0.1,
        rejected_cookie_hashes=set(),
    )

    assert result.cookie_header == "JSESSIONID=stored"
    assert context.stored_cookies == [
        {"name": "JSESSIONID", "value": "stored", "url": "https://jira.example.test"},
    ]


def test_jira_browser_session_forgets_keychain_cookie(tmp_path: Path):
    from memforge.auth.jira_browser_session import JiraBrowserSession

    cookie_store = FakeCookieStore("JSESSIONID=stored")
    session = JiraBrowserSession(
        profile_root=tmp_path,
        browser_launcher=FakeLauncher(FakeContext([])),
        cookie_store=cookie_store,
    )

    session.forget(origin="https://jira.example.test")

    assert cookie_store.deleted == ["https://jira.example.test"]
