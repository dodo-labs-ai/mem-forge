"""Turn the signed file URLs in a Notion answer into stable references, so no signature is ever stored.

Notion serves the files people upload (images, PDFs, attachments) from its
file host through signed URLs that expire within an hour and change on every
read, while the path stays the same::

    https://prod-files-secure.s3.us-west-2.amazonaws.com/<space id>/<file id>/diagram.png?X-Amz-Signature=…

So the stored body names such a file by its path, the file key, and the
signed URL is kept only in memory for the item being processed::

    files = NotionFiles()
    text = files.stabilize("![Diagram](https://prod-files-secure…/diagram.png?X-Amz-…)", "page_body")
    # "![Diagram](notion-file:<space id>/<file id>/diagram.png)"
    files.signed_urls  # {"<space id>/<file id>/diagram.png": "https://prod-files-secure…?X-Amz-…"}

A Notion-hosted file is recognized by its host and the shape of its path,
never by one host alone. Any other URL that carries ``X-Amz-*`` query
parameters loses its query, so a signature never reaches storage even when
Notion changes its file host. Other external links are left as they are.
"""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qsl, unquote, urlsplit

from memforge.genes.notion_api import notion_id

__all__ = ["NOTION_FILE_SCHEME", "NotionFiles", "notion_file_reference", "split_signed_url"]

NOTION_FILE_SCHEME = "notion-file:"
_URL = re.compile(r"https?://[^\s\"'<>]+")


def notion_file_reference(url: str) -> str | None:
    """The file key of a Notion-hosted file URL, or ``None`` when ``url`` is not one.

    The key is the URL's path without its query, which stays the same across
    reads while the signature changes::

        notion_file_reference("https://prod-files-secure.s3.us-west-2.amazonaws.com/"
                              "4c1e…/9a7b…/diagram.png?X-Amz-Signature=…")
        # "4c1e…/9a7b…/diagram.png"
        notion_file_reference("https://example.com/diagram.png")  # None

    Two shapes are recognized: ``/<space id>/<file id>/<name>`` on
    ``prod-files-secure.s3.us-west-2.amazonaws.com``, and the older
    ``/secure.notion-static.com/<file id>/<name>`` on
    ``s3.us-west-2.amazonaws.com``.
    """
    parts = urlsplit(url)
    if parts.scheme != "https":
        return None
    segments = parts.path.lstrip("/").split("/")
    host = parts.hostname or ""
    if host == "prod-files-secure.s3.us-west-2.amazonaws.com":
        shaped = len(segments) == 3 and notion_id(segments[0]) and notion_id(segments[1]) and segments[2]
    elif host == "s3.us-west-2.amazonaws.com":
        shaped = len(segments) == 3 and segments[0] == "secure.notion-static.com" and notion_id(segments[1]) and segments[2]
    else:
        shaped = False
    return "/".join(segments) if shaped else None


def split_signed_url(url: str) -> tuple[str, list[tuple[str, str]]]:
    """A signed URL as its address without the query, and the query's parameters in order.

    A request built from the two sends the same URL, while MemForge's retry
    logs and error messages, which name only the address, never show the
    signature. httpx's own request line, which shows the full URL, is kept
    out of the logs where the application configures logging.
    """
    parts = urlsplit(url)
    return parts._replace(query="", fragment="").geturl(), parse_qsl(parts.query, keep_blank_values=True)


class NotionFiles:
    """The Notion-hosted files one fetch met, in the order it met them; see the module docstring.

    ``owners`` maps each file key to the Observation where it first occurs
    (``"page_body"`` before ``"properties"``), and ``signed_urls`` to the
    signed URL that downloads it.
    """

    def __init__(self) -> None:
        self.owners: dict[str, str] = {}
        self.signed_urls: dict[str, str] = {}

    def stabilize(self, text: str, owner: str) -> str:
        """``text`` with each Notion file URL replaced by ``notion-file:<file key>`` and every other signature stripped."""
        return _URL.sub(lambda match: self._replace(match.group(0), owner), text)

    def _replace(self, matched: str, owner: str) -> str:
        url, trailing = _balanced(matched)
        key = notion_file_reference(url)
        if key is not None:
            self.owners.setdefault(key, owner)
            # A URL inside an HTML attribute may write its query's "&" as "&amp;".
            self.signed_urls[key] = html.unescape(url)
            return NOTION_FILE_SCHEME + key + trailing
        parts = urlsplit(url)
        if any(name.lower().startswith("x-amz-") for name, _value in parse_qsl(html.unescape(parts.query))):
            return parts._replace(query="", fragment="").geturl() + trailing
        return matched

    @staticmethod
    def filename(key: str) -> str:
        """The file's name, from the last segment of its key: ``"…/Q3%20plan.pdf"`` is ``"Q3 plan.pdf"``."""
        return unquote(key.rsplit("/", 1)[-1]).strip() or "file"


def _balanced(matched: str) -> tuple[str, str]:
    """Split closing parentheses that end a Markdown link, ``[x](url)``, off the URL they follow."""
    url = matched
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1]
    return url, matched[len(url):]
