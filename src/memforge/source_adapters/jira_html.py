"""Jira's explicitly returned HTML field view and immutable selections.

Offsets address the original provider-rendered string. Syntax highlighting is
presentation decoration; literal code, links and meaningful formatting remain
comparison values. This grammar does not interpret Jira's native wiki strings.

The renderer's element, class, attribute and style vocabulary is open. Declared
decoration is erased and declared semantics are presented; anything undeclared
stays in comparison material and is shown beside the text it controls. Only
malformed markup is rejected.
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
# HTML never displays these elements' content.
_NOT_DISPLAYED = {"script", "style"}
_CODE_SPAN_CLASSES = {"code-tag", "code-quote", "code-keyword", "code-comment", "code-object", "code-quote-red"}
_JIRA_FORMATTING = {"ins": "u", "del": "strikethrough", "s": "strikethrough", "strike": "strikethrough"}
_ALIASES = {"b": "strong", "i": "em", "s": "del", "strike": "del", "tt": "code"}
_BLOCKS = {"root", "p", "div", "blockquote", "li", "td", "th", "caption", "h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass(frozen=True, slots=True)
class _Decoration:
    """Provider rendering controls whose authored children retain their meaning."""

    transparent: bool = False
    style_properties: frozenset[str] = frozenset()
    retain_class: bool = False
    erase_literal_class: str | None = None
    omit: bool = False
    ignored_attributes: frozenset[str] = frozenset()


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
    ("span", frozenset({"nobr"})): _Decoration(transparent=True),
    ("span", frozenset({"error"})): _Decoration(retain_class=True),
    ("a", frozenset({"issue-link"})): _Decoration(),
    ("img", frozenset({"rendericon"})): _Decoration(omit=True),
    ("ul", frozenset({"alternate"})): _Decoration(ignored_attributes=frozenset({"type"})),
    ("br", frozenset({"atl-forced-newline"})): _Decoration(),
}
_IMAGE_LAYOUT = frozenset({"width", "height", "align", "border"})
_ATTRIBUTES = {
    "a": {"href", "title", "class", "target", "rel", "id", "name", "file-preview-id", "file-preview-title", "file-preview-type", "data-username", "data-issue-key"},
    "img": {"src", "alt", "title", "role", "style", "class", *_IMAGE_LAYOUT},
    "font": {"color", "face", "size"},
    "pre": {"class"}, "span": {"class"}, "div": {"class", "style"},
    "table": {"class"}, "ol": {"start"}, "ul": {"class", "type"}, "li": {"value"}, "br": {"class"},
    "th": {"rowspan", "colspan", "scope", "class"}, "td": {"rowspan", "colspan", "class"},
}
_NAVIGATION_REL = {"nofollow", "noopener", "noreferrer"}
_EMPTY_BOOKMARK = _Decoration(omit=True)
_CODE_LANGUAGE = re.compile(r"code-([\w+-]+)")


class _IrregularTable(ValueError):
    """Rendered rows, spans or nesting that resolve to no simple grid."""


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
    # Attributes, classes and styles the grammar assigns no meaning.
    controls: tuple[tuple[str, str], ...] = ()


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
        if len(dict(attrs)) != len(attrs):
            raise ValueError("Jira rendered HTML has duplicate attributes")
        start = self._position()
        node = _Node(tag, {key: value or "" for key, value in attrs}, start, start + len(self.get_starttag_text()))
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


def _decoration(node):
    # A control the grammar cannot read is never erased with its element.
    if node.controls:
        return None
    # Jira inserts empty named targets into rendered code/list wrappers.
    # A named link with authored content is not a presentation-only target.
    if node.tag == "a" and set(node.attrs) == {"name"} and not node.children:
        return _EMPTY_BOOKMARK
    decoration = _DECORATIONS.get((node.tag, _classes(node)))
    # A renderer icon carrying authored text is an image, not decoration.
    if decoration and decoration.omit and (node.attrs.get("alt") or node.attrs.get("title")):
        return None
    return decoration


def _classify(node, *, literal=False):
    """Record every attribute, class and style outside the declared grammar.

    Declared decoration and semantic controls keep their own treatment. The
    rest are kept as written: compared across revisions and shown to readers.
    """
    literal = literal or node.tag in {"pre", "code", "tt"}
    attrs = {key: value for key, value in node.attrs.items() if key not in {"class", "style"} or value.strip()}
    undeclared = set(attrs) - _ATTRIBUTES.get(node.tag, set())
    classes = _classes(node)
    decoration = _DECORATIONS.get((node.tag, classes))
    if "class" in attrs and not (
        decoration is not None
        or node.tag == "span" and literal and classes <= _CODE_SPAN_CLASSES
        or node.tag == "pre" and _CODE_LANGUAGE.fullmatch(attrs["class"])
    ):
        undeclared.add("class")
    if "style" in attrs:
        style = attrs["style"].replace(" ", "")
        declarations = [part.strip() for part in style.split(";") if part.strip()]
        declared_style = bool(decoration and declarations and all(
            ":" in part and part.split(":", 1)[0] in decoration.style_properties
            and part.split(":", 1)[1] for part in declarations
        )) or node.tag == "img" and style == "border:0pxsolidblack"
        if not declared_style:
            undeclared.add("style")
    if node.tag == "a":
        if "id" in attrs and attrs["id"] != attrs.get("file-preview-id", "") + "_thumb":
            undeclared.add("id")
        if attrs.get("target", "_self") not in {"_self", "_blank", "_parent", "_top"}:
            undeclared.add("target")
        if classes != {"user-hover"} and set(attrs.get("rel", "").split()) - _NAVIGATION_REL:
            undeclared.add("rel")
        if "data-username" in attrs and classes != {"user-hover"}:
            undeclared.add("data-username")
        if "data-issue-key" in attrs and classes != {"issue-link"}:
            undeclared.add("data-issue-key")
    if node.tag == "img" and attrs.get("role", "presentation") != "presentation":
        undeclared.add("role")
    if node.tag == "ul" and "type" in attrs and attrs["type"] not in {"disc", "circle", "square"}:
        undeclared.add("type")
    if node.tag == "th" and attrs.get("scope", "col") != "col":
        undeclared.add("scope")
    node.controls = tuple(sorted((key, attrs[key]) for key in undeclared))
    for child in node.children:
        if isinstance(child, _Node):
            _classify(child, literal=literal)


def _transparent(node, *, literal=False):
    if node.controls:
        return False
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
    decoration = _decoration(node)
    return bool(decoration and decoration.transparent and set(attrs) <= {"class", "style"})


def _attributes(node):
    attrs = {key: value for key, value in node.attrs.items() if key not in {"class", "style"} or value.strip()}
    decoration = _decoration(node)
    if decoration and (not decoration.retain_class or attrs.get("class") == decoration.erase_literal_class):
        attrs.pop("class", None)
    if decoration:
        for key in decoration.ignored_attributes:
            attrs.pop(key, None)
    if node.tag == "a":
        # Navigation/browser safety controls do not alter the linked identity.
        attrs.pop("target", None)
        if _classes(node) != {"user-hover"}:
            attrs.pop("rel", None)
        if attrs.get("file-preview-id") and attrs.get("id") == attrs["file-preview-id"] + "_thumb":
            attrs.pop("id")
    language = _CODE_LANGUAGE.fullmatch(attrs.get("class", "")) if node.tag == "pre" else None
    if language:
        del attrs["class"]
        attrs["language"] = language.group(1)
    if node.tag == "img":
        attrs.pop("role", None)
        for key in _IMAGE_LAYOUT:
            attrs.pop(key, None)
        if attrs.get("style", "").replace(" ", "") == "border:0pxsolidblack":
            attrs.pop("style")
    attrs.update(node.controls)
    if "class" in attrs:
        attrs["class"] = " ".join(sorted(attrs["class"].split()))
    return sorted(attrs.items())


def _canonical(node, *, literal=False):
    if isinstance(node, _Text):
        return [("text", node.value if literal else re.sub(r"\s+", " ", node.value))]
    decoration = _decoration(node)
    if decoration and decoration.omit:
        return []
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
    text = _element(node)
    if not node.controls:
        return text
    body = text.rstrip("\n")
    controls = "; ".join(f"{key}: {value}" if value else key for key, value in node.controls)
    return f"{body} [{controls}]" + text[len(body):]


def _element(node):
    decoration = _decoration(node)
    if decoration and decoration.omit or node.tag in _NOT_DISPLAYED:
        return ""
    if node.tag in {"pre", "code", "tt"}:
        language = _CODE_LANGUAGE.fullmatch(node.attrs.get("class", "")) if node.tag == "pre" else None
        if language:
            return f"Code (language: {language.group(1)}):\n" + _literal_text(node)
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
    if node.tag == "strikethrough":
        return f"[struck through: {text}]"
    if node.tag in {"s", "del", "strike"}:
        return f"[deleted: {text}]"
    if node.tag == "ins":
        return f"[inserted: {text}]"
    if node.tag in {"sup", "sub"}:
        return f"[{node.tag}: {text}]"
    if node.tag == "font":
        shown = sorted(set(node.attrs.items()) - set(node.controls))
        return text + " [" + "; ".join(f"{key}: {value}" for key, value in shown) + "]" if shown else text
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
                # Content the renderer placed between items stays in reading order.
                lines.append(_render(child).strip())
                continue
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
                    raise _IrregularTable("Jira rendered table contains unstructured text")
            elif child.tag in {"thead", "tbody", "tfoot"}:
                collect(child)
            elif child.tag == "tr":
                rows.append(child)
            elif child.tag == "caption":
                captions.append(child)
            elif child.tag not in {"colgroup", "col"}:
                raise _IrregularTable("Jira rendered table holds content outside its rows")

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
                raise _IrregularTable("Jira rendered table row contains non-cell content")
            if any(child.attrs.get(name, "1") != "1" for name in ("rowspan", "colspan")):
                raise _IrregularTable("Jira rendered table cells span rows or columns")
            if any(isinstance(n, _Node) and n.tag == "table" for n in _walk(child)):
                raise _IrregularTable("Jira rendered table cell holds another table")
            cells.append(child)
        is_header = bool(cells) and all(cell.tag == "th" for cell in cells)
        if is_header and len(header_rows) == len(result):
            header_rows.append(row)
            if headers and len(headers) != len(cells):
                raise _IrregularTable("inconsistent Jira rendered table header width")
            headers = [" / ".join(filter(None, (headers[i] if headers else "", _render(cell).strip()))) for i, cell in enumerate(cells)]
        if headers and len(headers) != len(cells):
            raise _IrregularTable("inconsistent Jira rendered table row width")
        text = "\n".join(f"{headers[i] if headers else f'Column {i + 1}'}: {_render(cell).strip()}" for i, cell in enumerate(cells))
        result.append((row, text, is_header))
    return result, header_rows, captions


def _authored_rows(node):
    """A table without a simple grid, read row by row as rendered."""
    if isinstance(node, _Text):
        return _render(node)
    if node.tag == "tr":
        return " | ".join(filter(None, (_authored_rows(cell).strip() for cell in node.children))) + "\n"
    if node.tag in {"table", "thead", "tbody", "tfoot", "td", "th"} or _transparent(node):
        return "".join(_authored_rows(child) for child in node.children)
    return _render(node)


def _walk(node):
    for child in node.children:
        yield child
        if isinstance(child, _Node):
            yield from _walk(child)


def _parse_rendered_html(source: str, *, jira_formatting: bool) -> ParsedDeclaredText:
    root = _Parser(source).parse()
    _classify(root)
    if jira_formatting:
        # Jira wiki emphasis markers denote formatting, not insertion/deletion events.
        # Normalize only the current tree, preserving provider offsets and children.
        for node in _walk(root):
            if isinstance(node, _Node):
                node.tag = _JIRA_FORMATTING.get(node.tag, node.tag)
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
            try:
                rows, headers, captions = _table_rows(node)
            except _IrregularTable:
                add(node, _authored_rows(node))
                return
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


def _parse_rendered_html_v1(source: str) -> ParsedDeclaredText:
    return _parse_rendered_html(source, jira_formatting=False)


def parse_rendered_html(source: str) -> ParsedDeclaredText:
    return _parse_rendered_html(source, jira_formatting=True)


LEGACY_RENDERED_HTML_FORMAT = DeclaredTextFormat("jira-rendered-html", 1, _parse_rendered_html_v1)
RENDERED_HTML_FORMAT = DeclaredTextFormat("jira-rendered-html", 2, parse_rendered_html)
