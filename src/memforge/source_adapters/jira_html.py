"""Jira's explicitly returned HTML field view and immutable selections.

Offsets address the original provider-rendered string. Syntax highlighting is
presentation decoration; literal code, links and meaningful formatting remain
comparison values. This grammar does not interpret Jira's native wiki strings.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from memforge.source_adapters.contracts import (
    DeclaredTextFormat,
    DeclaredTextFragment,
    DeclaredTextGroup,
    DeclaredTextOrigin,
    ParsedDeclaredText,
)


_VOID = {"br", "hr", "img", "col"}
_TAGS = {
    "p", "div", "span", "b", "strong", "i", "em", "u", "s", "del", "strike",
    "sup", "sub", "a", "br", "hr", "img", "pre", "code", "tt", "blockquote",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "thead",
    "tbody", "tfoot", "tr", "th", "td", "colgroup", "col", "caption", "font",
}
_CODE_SPAN_CLASSES = {"code-tag", "code-quote", "code-keyword", "code-comment"}
_ALIASES = {"b": "strong", "i": "em", "s": "del", "strike": "del", "tt": "code"}
_BLOCKS = {"root", "p", "div", "blockquote", "li", "td", "th", "caption", "h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass(frozen=True, slots=True)
class _Decoration:
    """Provider rendering controls whose authored children retain their meaning."""

    transparent: bool = False
    style_properties: frozenset[str] = frozenset()
    retain_class: bool = False
    erase_literal_class: str | None = None


_PANEL_STYLE = frozenset({"border-width", "border-style", "border-color"})
_DECORATIONS = {
    ("div", frozenset({"code", "panel"})): _Decoration(True, _PANEL_STYLE),
    ("div", frozenset({"codeContent", "panelContent"})): _Decoration(True, _PANEL_STYLE),
    ("div", frozenset({"preformatted", "panel"})): _Decoration(True, _PANEL_STYLE),
    ("div", frozenset({"preformattedContent", "panelContent"})): _Decoration(True, _PANEL_STYLE),
    ("div", frozenset({"panel"})): _Decoration(True, _PANEL_STYLE),
    ("div", frozenset({"panelContent"})): _Decoration(True, _PANEL_STYLE),
    ("div", frozenset({"table-wrap"})): _Decoration(True),
    ("table", frozenset({"confluenceTable"})): _Decoration(),
    ("th", frozenset({"confluenceTh"})): _Decoration(),
    ("td", frozenset({"confluenceTd"})): _Decoration(),
    ("a", frozenset({"external-link"})): _Decoration(retain_class=True, erase_literal_class="external-link"),
    ("a", frozenset({"user-hover"})): _Decoration(),
    ("span", frozenset({"image-wrap"})): _Decoration(True),
    ("img", frozenset({"emoticon"})): _Decoration(retain_class=True),
}
_ATTRIBUTES = {
    "a": {"href", "title", "class", "target", "rel", "id", "file-preview-id", "file-preview-title", "file-preview-type", "data-username"},
    "img": {"src", "alt", "title", "role", "style", "class"},
    "font": {"color", "face", "size"},
    "pre": {"class"}, "span": {"class"}, "div": {"class", "style"},
    "table": {"class"}, "ol": {"start"}, "li": {"value"},
    "th": {"rowspan", "colspan", "scope", "class"}, "td": {"rowspan", "colspan", "class"},
}
_NAVIGATION_REL = {"nofollow", "noopener", "noreferrer"}


@dataclass(slots=True)
class _Text:
    value: str
    start: int
    end: int


@dataclass(slots=True)
class _Node:
    tag: str
    attrs: dict[str, str]
    start: int
    end: int
    children: list[_Node | _Text] = field(default_factory=list)


class _Parser(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.lines = [0, *(m.end() for m in re.finditer("\n", source))]
        self.root = _Node("root", {}, 0, len(source))
        self.stack = [self.root]

    def _position(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs, False)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs, True)

    def _start(self, tag, attrs, closed):
        if tag not in _TAGS:
            raise ValueError(f"unsupported Jira rendered HTML element: {tag}")
        if len(dict(attrs)) != len(attrs) or any(value is None for _, value in attrs):
            raise ValueError("Jira rendered HTML has duplicate or unvalued attributes")
        start = self._position()
        node = _Node(tag, dict(attrs), start, start + len(self.get_starttag_text()))
        self.stack[-1].children.append(node)
        if not closed and tag not in _VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        if tag in _VOID or len(self.stack) == 1 or self.stack[-1].tag != tag:
            raise ValueError("unbalanced Jira rendered HTML elements")
        self.stack.pop().end = self.source.index(">", self._position()) + 1

    def _text(self, value, raw_length):
        start = self._position()
        previous = self.stack[-1].children[-1] if self.stack[-1].children else None
        if isinstance(previous, _Text) and previous.end == start:
            previous.value += value
            previous.end = start + raw_length
        else:
            self.stack[-1].children.append(_Text(value, start, start + raw_length))

    def handle_data(self, data):
        self._text(data, len(data))

    def handle_entityref(self, name):
        raw = f"&{name};"
        value = html.unescape(raw)
        if value == raw:
            raise ValueError("unknown Jira rendered HTML entity")
        self._text(value, len(raw))

    def handle_charref(self, name):
        raw = f"&#{name};"
        self._text(html.unescape(raw), len(raw))

    def handle_comment(self, data):
        raise ValueError("unsupported Jira rendered HTML comment")

    def handle_decl(self, decl):
        raise ValueError("Jira rendered fields must not contain declarations")

    def handle_pi(self, data):
        raise ValueError("Jira rendered fields must not contain processing instructions")

    def unknown_decl(self, data):
        raise ValueError("unsupported Jira rendered HTML declaration")

    def parse(self):
        self.feed(self.source)
        self.close()
        if len(self.stack) != 1:
            raise ValueError("unclosed Jira rendered HTML element")
        return self.root


def _classes(node):
    return frozenset(node.attrs.get("class", "").split())


def _validate_attributes(node, *, literal=False):
    """Only attested decoration or explicitly rendered semantic controls are supported."""
    literal = literal or node.tag in {"pre", "code", "tt"}
    attrs = {key: value for key, value in node.attrs.items() if key not in {"class", "style"} or value.strip()}
    allowed = _ATTRIBUTES.get(node.tag, set())
    if set(attrs) - allowed:
        raise ValueError(f"unsupported Jira rendered HTML attributes on {node.tag}")
    classes = _classes(node)
    if "class" in attrs:
        supported_class = (
            (node.tag, classes) in _DECORATIONS
            or node.tag == "span" and literal and classes <= _CODE_SPAN_CLASSES
            or node.tag == "pre" and re.fullmatch(r"code-[\w+-]+", attrs["class"])
        )
        if not supported_class:
            raise ValueError(f"unsupported Jira rendered HTML class on {node.tag}")
    if "style" in attrs:
        style = attrs["style"].replace(" ", "")
        decoration = _DECORATIONS.get((node.tag, classes))
        declarations = [part.strip() for part in style.split(";") if part.strip()]
        supported_style = bool(decoration and declarations and all(
            ":" in part and part.split(":", 1)[0] in decoration.style_properties
            and part.split(":", 1)[1] for part in declarations
        )) or node.tag == "img" and attrs.get("role") == "presentation" and style == "border:0pxsolidblack"
        if not supported_style:
            raise ValueError(f"unsupported Jira rendered HTML style on {node.tag}")
    if node.tag == "a":
        if "id" in attrs and attrs["id"] != attrs.get("file-preview-id", "") + "_thumb":
            raise ValueError("unsupported Jira rendered HTML anchor identity")
        if attrs.get("target", "_self") not in {"_self", "_blank", "_parent", "_top"}:
            raise ValueError("unsupported Jira rendered HTML link target")
        if classes != {"user-hover"} and set(attrs.get("rel", "").split()) - _NAVIGATION_REL:
            raise ValueError("unsupported Jira rendered HTML link relationship")
        if "data-username" in attrs and classes != {"user-hover"}:
            raise ValueError("Jira author identity requires a rendered mention")
    if node.tag == "img" and attrs.get("role", "presentation") != "presentation":
        raise ValueError("unsupported Jira rendered HTML image role")
    if node.tag == "th" and attrs.get("scope", "col") != "col":
        raise ValueError("unsupported Jira rendered HTML table header scope")
    for child in node.children:
        if isinstance(child, _Node):
            _validate_attributes(child, literal=literal)


def _transparent(node, *, literal=False):
    attrs = {key: value for key, value in node.attrs.items() if key not in {"class", "style"} or value.strip()}
    if node.tag == "span":
        if not attrs:
            return True
        classes = _classes(node)
        if literal and classes and classes <= _CODE_SPAN_CLASSES:
            return set(node.attrs) <= {"class"}
    if node.tag == "div":
        if not attrs:
            return True
    decoration = _DECORATIONS.get((node.tag, _classes(node)))
    return bool(decoration and decoration.transparent and set(attrs) <= {"class", "style"})


def _attributes(node):
    attrs = {key: value for key, value in node.attrs.items() if key not in {"class", "style"} or value.strip()}
    decoration = _DECORATIONS.get((node.tag, _classes(node)))
    if decoration and (not decoration.retain_class or attrs.get("class") == decoration.erase_literal_class):
        attrs.pop("class", None)
    if node.tag == "a":
        # Navigation/browser safety controls do not alter the linked identity.
        attrs.pop("target", None)
        if _classes(node) != {"user-hover"}:
            attrs.pop("rel", None)
        if attrs.get("file-preview-id") and attrs.get("id") == attrs["file-preview-id"] + "_thumb":
            attrs.pop("id")
    if node.tag == "pre" and re.fullmatch(r"code-[\w+-]+", attrs.get("class", "")):
        attrs["language"] = attrs.pop("class")[5:]
    if node.tag == "img" and attrs.get("role") == "presentation":
        attrs.pop("role")
        if attrs.get("style", "").replace(" ", "") == "border:0pxsolidblack":
            attrs.pop("style")
    if "class" in attrs:
        attrs["class"] = " ".join(sorted(attrs["class"].split()))
    return sorted(attrs.items())


def _canonical(node, *, literal=False):
    if isinstance(node, _Text):
        return [("text", node.value if literal else re.sub(r"\s+", " ", node.value))]
    literal = literal or node.tag in {"pre", "code", "tt"}
    children = []
    for child in node.children:
        for value in _canonical(child, literal=literal):
            if value[0] == "text" and children and children[-1][0] == "text":
                children[-1] = ("text", children[-1][1] + value[1])
            else:
                children.append(value)
    if _transparent(node, literal=literal):
        return children
    if not literal and node.tag in _BLOCKS:
        if children and children[0][0] == "text":
            children[0] = ("text", children[0][1].lstrip())
        if children and children[-1][0] == "text":
            children[-1] = ("text", children[-1][1].rstrip())
    return [(_ALIASES.get(node.tag, node.tag), _attributes(node), children)]


def _literal_text(node):
    if isinstance(node, _Text):
        return node.value
    if node.tag == "br":
        return "\n"
    return "".join(_literal_text(child) for child in node.children)


def _render(node):
    if isinstance(node, _Text):
        return re.sub(r"\s+", " ", node.value)
    if node.tag in {"pre", "code", "tt"}:
        if node.tag == "pre" and node.attrs.get("class"):
            return f"Code (language: {node.attrs['class'][5:]}):\n" + _literal_text(node)
        return _literal_text(node)
    text = "".join(_render(child) for child in node.children)
    if node.tag == "br":
        return "\n"
    if node.tag == "hr":
        return "\n—\n"
    if node.tag == "img":
        name = node.attrs.get("alt") or node.attrs.get("title") or "Image reference"
        title = f"; title: {node.attrs['title']}" if node.attrs.get("title") and node.attrs["title"] != name else ""
        return f"{name} (image: {node.attrs.get('src', '')}{title})"
    if node.tag == "a":
        authored_label = text.strip() if _literal_text(node).strip() else ""
        label = authored_label or node.attrs.get("file-preview-title") or text.strip() or node.attrs.get("title") or node.attrs.get("href", "")
        if not authored_label and node.attrs.get("file-preview-title"):
            descriptions = [child.attrs["alt"] for child in _walk(node) if isinstance(child, _Node) and child.tag == "img" and child.attrs.get("alt")]
            if descriptions:
                label += " — " + " / ".join(descriptions)
        details = []
        if _classes(node) == {"user-hover"}:
            identity = node.attrs.get("data-username") or node.attrs.get("rel")
            if identity:
                details.append(f"user: {identity}")
        if node.attrs.get("file-preview-id"):
            details.append(f"attachment: {node.attrs['file-preview-id']}")
        if node.attrs.get("file-preview-type"):
            details.append(f"type: {node.attrs['file-preview-type']}")
        for attribute, name in (("file-preview-title", "filename"), ("title", "title")):
            value = node.attrs.get(attribute)
            if value and value != label and (attribute != "title" or value != node.attrs.get("file-preview-title")):
                details.append(f"{name}: {value}")
        suffix = "; " + "; ".join(details) if details else ""
        return f"{label} ({node.attrs.get('href', '')}{suffix})"
    if node.tag in {"s", "del", "strike"}:
        return f"[deleted: {text}]"
    if node.tag in {"sup", "sub"}:
        return f"[{node.tag}: {text}]"
    if node.tag == "font":
        return text + " [" + "; ".join(f"{key}: {value}" for key, value in sorted(node.attrs.items())) + "]" if node.attrs else text
    if node.tag in {"ol", "ul"}:
        try:
            number = int(node.attrs.get("start", "1"))
        except ValueError as exc:
            raise ValueError("invalid Jira rendered list start") from exc
        lines = []
        for child in node.children:
            if isinstance(child, _Text) and not child.value.strip():
                continue
            if not isinstance(child, _Node) or child.tag != "li":
                raise ValueError("Jira rendered list contains content outside list items")
            if "value" in child.attrs:
                try:
                    number = int(child.attrs["value"])
                except ValueError as exc:
                    raise ValueError("invalid Jira rendered list item value") from exc
            lines.append(f"{number}. {_render(child).strip()}" if node.tag == "ol" else f"• {_render(child).strip()}")
            number += 1
        return "\n".join(lines) + "\n"
    if node.tag in {"p", "div", "blockquote", "li", "caption", "h1", "h2", "h3", "h4", "h5", "h6"}:
        return text + "\n"
    return text


def _table_rows(table):
    rows = []
    captions = []

    def collect(node):
        for child in node.children:
            if isinstance(child, _Text):
                if child.value.strip():
                    raise ValueError("Jira rendered table contains unstructured text")
            elif child.tag in {"thead", "tbody", "tfoot"}:
                collect(child)
            elif child.tag == "tr":
                rows.append(child)
            elif child.tag == "caption":
                captions.append(child)
            elif child.tag not in {"colgroup", "col"}:
                raise ValueError("unsupported Jira rendered table structure")

    collect(table)
    headers = []
    result = []
    header_rows = []
    for row in rows:
        cells = []
        for child in row.children:
            if isinstance(child, _Text) and not child.value.strip():
                continue
            if not isinstance(child, _Node) or child.tag not in {"td", "th"}:
                raise ValueError("Jira rendered table row contains non-cell content")
            if any(child.attrs.get(name, "1") != "1" for name in ("rowspan", "colspan")):
                raise ValueError("Jira rendered table spans require a supported grid representation")
            if any(isinstance(n, _Node) and n.tag == "table" for n in _walk(child)):
                raise ValueError("nested Jira rendered tables are unsupported")
            cells.append(child)
        is_header = bool(cells) and all(cell.tag == "th" for cell in cells)
        if is_header and len(header_rows) == len(result):
            header_rows.append(row)
            if headers and len(headers) != len(cells):
                raise ValueError("inconsistent Jira rendered table header width")
            headers = [" / ".join(filter(None, (headers[i] if headers else "", _render(cell).strip()))) for i, cell in enumerate(cells)]
        if headers and len(headers) != len(cells):
            raise ValueError("inconsistent Jira rendered table row width")
        text = "\n".join(f"{headers[i] if headers else f'Column {i + 1}'}: {_render(cell).strip()}" for i, cell in enumerate(cells))
        result.append((row, text, is_header))
    return result, header_rows, captions


def _walk(node):
    for child in node.children:
        yield child
        if isinstance(child, _Node):
            yield from _walk(child)


def parse_rendered_html(source: str) -> ParsedDeclaredText:
    root = _Parser(source).parse()
    _validate_attributes(root)
    # A parent selection must not conceal unsupported nested structures.
    for node in _walk(root):
        if isinstance(node, _Node) and node.tag == "table":
            _table_rows(node)
    _render(root)
    fragments = []
    groups = []

    def add(node, presentation=None, kind=None, origins=()):
        text = _render(node) if presentation is None else presentation
        if not isinstance(node, _Node) or node.tag != "pre":
            text = text.strip()
        if text:
            fragments.append(DeclaredTextFragment(
                node.start, node.end, kind or (node.tag if isinstance(node, _Node) else "text"),
                text, json.dumps(_canonical(node), ensure_ascii=False, separators=(",", ":")), origins,
            ))

    def select(node):
        if isinstance(node, _Text):
            add(node)
        elif node.tag == "table":
            rows, headers, captions = _table_rows(node)
            for row, text, is_header in rows:
                origins = tuple(DeclaredTextOrigin(h.start, h.end,
                    json.dumps(_canonical(h), ensure_ascii=False, separators=(",", ":")))
                    for h in headers if h is not row)
                add(row, text, "table-header" if is_header else "table-row", origins)
                if not is_header:
                    groups.append(DeclaredTextGroup(row.start, row.end, tuple((n.start, n.end) for n in (*headers, *captions))))
            for caption in captions:
                add(caption)
        elif _transparent(node):
            for child in node.children:
                select(child)
        else:
            add(node)

    for node in root.children:
        select(node)
    for parent in (root, *(node for node in _walk(root) if isinstance(node, _Node))):
        siblings = [child for child in parent.children
                    if isinstance(child, _Node) or child.value.strip()]
        for lead, listing in zip(siblings, siblings[1:]):
            if (isinstance(lead, _Node) and lead.tag == "p"
                    and isinstance(listing, _Node) and listing.tag in {"ul", "ol"}
                    and _render(lead).rstrip().endswith(":")):
                groups.append(DeclaredTextGroup(listing.start, listing.end,
                    ((lead.start, lead.end),), together=True))
    headings = [f for f in fragments if re.fullmatch(r"h[1-6]", f.kind)]
    for i, heading in enumerate(headings):
        end = next((f.start for f in headings[i + 1:] if f.kind <= heading.kind), len(source))
        groups.append(DeclaredTextGroup(heading.start, end, ((heading.start, heading.end),)))
    return ParsedDeclaredText(tuple(fragments), tuple(groups))


RENDERED_HTML_FORMAT = DeclaredTextFormat("jira-rendered-html", 1, parse_rendered_html)
