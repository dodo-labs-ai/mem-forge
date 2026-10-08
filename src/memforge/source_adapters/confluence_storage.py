"""Confluence storage grammar, comparison and readable selections.

Storage is the authored, version-pinned page body. Its element, attribute and
macro vocabulary is open: installed apps add names freely, while every macro
keeps one envelope of parameters and an optional rich or plain-text body. The
grammar therefore reads structure, never a list of names. Declared rules refine
how known constructs read; anything else is presented by its own shape, and
only malformed markup is rejected. A macro without a body states what the page
embeds; content Confluence generates for it when the page is viewed belongs to
no page revision and is never Evidence.

All provider tags and macro rules live here. Offsets address the *original*
decoded storage value; rendered text is never used as a source selector.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from memforge.source_adapters.contracts import (
    DeclaredTextFormat, DeclaredTextFragment, DeclaredTextGroup, DeclaredTextOrigin, ParsedDeclaredText,
)


_VOID = {"br", "hr", "img", "col", "ri:user", "ri:page", "ri:attachment", "ri:url", "ac:emoticon"}


# Layout containers: their children are selected as if the container were absent.
# A closed child set names the only elements the container directly holds.
_CONTAINERS: dict[str, frozenset[str] | None] = {
    "ac:layout": frozenset({"ac:layout-section"}),
    "ac:layout-section": frozenset({"ac:layout-cell"}),
    "ac:layout-cell": None,
    "ac:inline-comment-marker": None,
    "ac:task-list": frozenset({"ac:task"}),
}
_LAYOUT_CELLS = {"single": 1, "two_equal": 2, "two_left_sidebar": 2, "two_right_sidebar": 2,
                 "three_equal": 3, "three_with_sidebars": 3}
# Template instructions shown only in the editor; a viewed page never contains them.
_EDITOR_ONLY = {"ac:placeholder"}
# Opaque task identities; a task's status and body carry its authored meaning.
_TASK_IDENTITIES = {"ac:task-id", "ac:task-uuid"}
_TABLE_CHILDREN = {
    "table": {"thead", "tbody", "tfoot", "tr", "caption", "colgroup", "col"},
    "thead": {"tr"}, "tbody": {"tr"}, "tfoot": {"tr"}, "colgroup": {"col"},
}


@dataclass(frozen=True, slots=True)
class _Macro:
    parameters: frozenset[str]
    body: str
    identity: str | None = None


_MACROS = {
    "toc": _Macro(frozenset({"type", "outline", "style", "indent", "separator", "minLevel", "maxLevel",
                            "include", "exclude", "printable", "class", "absoluteUrl"}), "none"),
    "status": _Macro(frozenset({"title", "colour"}), "none", "title"),
    "jira": _Macro(frozenset({"key", "server", "serverId", "columnIds", "columns", "showSummary"}), "none", "key"),
    "code": _Macro(frozenset({"language", "title", "collapse", "linenumbers", "firstline", "theme"}), "literal"),
    **{name: _Macro(frozenset({"title", "icon"}), "rich") for name in ("info", "note", "warning", "tip")},
    "panel": _Macro(frozenset({"title", "borderStyle", "borderColor", "borderWidth", "bgColor", "titleBGColor", "titleColor"}), "rich"),
    "expand": _Macro(frozenset({"title"}), "rich"),
    "noformat": _Macro(frozenset(), "literal"),
    "excerpt": _Macro(frozenset({"hidden", "name", "output-type"}), "local"),
}
_BODY_ELEMENTS = {"none": [], "literal": ["ac:plain-text-body"],
                  "rich": ["ac:rich-text-body"], "local": ["ac:rich-text-body"]}
# These identify a rendered macro instance, not its authored content. No other
# attribute is erased from comparison material.
_INSTANCE_ATTRIBUTES = {"ac:macro-id"}


class _IrregularTable(ValueError):
    """Authored rows and spans that resolve to no rectangular grid."""


@dataclass(slots=True)
class _Text:
    value: str
    start: int
    end: int
    literal: bool = False


@dataclass(slots=True)
class _Node:
    tag: str
    attrs: dict[str, str]
    start: int
    end: int = 0
    children: list[_Node | _Text] = field(default_factory=list)


class _StorageParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.lines = [0, *(match.end() for match in re.finditer("\n", source))]
        self.root = _Node("root", {}, 0, len(source))
        self.stack = [self.root]

    def _source_position(self) -> int:
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs, False)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs, True)

    def _start(self, tag, attrs, closed):
        if len(dict(attrs)) != len(attrs) or any(value is None for _, value in attrs):
            raise ValueError("Confluence storage has duplicate or unvalued attributes")
        start = self._source_position()
        node = _Node(tag, dict(attrs), start, start + len(self.get_starttag_text()))
        self.stack[-1].children.append(node)
        if not closed and tag not in _VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        # Native resource elements can be explicitly closed or self-closing.
        if tag in _VOID:
            last = self.stack[-1].children[-1] if self.stack[-1].children else None
            if not isinstance(last, _Node) or last.tag != tag:
                raise ValueError("unmatched Confluence resource close")
            last.end = self.source.index(">", self._source_position()) + 1
            return
        if len(self.stack) == 1 or self.stack[-1].tag != tag:
            raise ValueError("unbalanced Confluence storage elements")
        self.stack.pop().end = self.source.index(">", self._source_position()) + 1

    def _text(self, value, raw_length):
        start = self._source_position()
        siblings = self.stack[-1].children
        previous = siblings[-1] if siblings else None
        # Entity references arrive as separate callbacks of one authored run.
        if isinstance(previous, _Text) and not previous.literal and previous.end == start:
            previous.value += value
            previous.end = start + raw_length
        else:
            siblings.append(_Text(value, start, start + raw_length))

    def handle_data(self, data):
        self._text(data, len(data))

    def handle_entityref(self, name):
        raw = f"&{name};"
        value = html.unescape(raw)
        if value == raw:
            raise ValueError("unknown Confluence storage entity")
        self._text(value, len(raw))

    def handle_charref(self, name):
        raw = f"&#{name};"
        self._text(html.unescape(raw), len(raw))

    def unknown_decl(self, data):
        if not data.startswith("CDATA["):
            raise ValueError("unsupported Confluence storage declaration")
        start = self._source_position() + len("<![CDATA[")
        end = self.source.find("]]>", start)
        if end < 0:
            raise ValueError("unclosed Confluence CDATA")
        self.stack[-1].children.append(_Text(self.source[start:end], start, end, literal=True))

    def handle_decl(self, decl):
        raise ValueError("Confluence storage must not contain a document declaration")

    def handle_pi(self, data):
        raise ValueError("Confluence storage must not contain processing instructions")

    def parse(self) -> _Node:
        self.feed(self.source)
        self.close()
        if len(self.stack) != 1:
            raise ValueError("unclosed Confluence storage element")
        for node in _nodes(self.root):
            held = _CONTAINERS.get(node.tag)
            if held is not None and any(
                isinstance(child, _Text) and child.value.strip()
                or isinstance(child, _Node) and child.tag not in held for child in node.children
            ):
                raise ValueError(f"unclassified Confluence content in {node.tag}")
            if node.tag == "ac:layout-section":
                cells = _LAYOUT_CELLS.get(node.attrs.get("ac:type", ""))
                if cells is not None and sum(isinstance(child, _Node) for child in node.children) != cells:
                    raise ValueError("Confluence layout has inconsistent cell coverage")
        return self.root


def _nodes(node: _Node):
    for child in node.children:
        if isinstance(child, _Node):
            yield child
            yield from _nodes(child)


def _canonical_children(node):
    children = []
    def authored_children(parent):
        for child in parent.children:
            if isinstance(child, _Node) and child.tag in _CONTAINERS:
                yield from authored_children(child)
            else:
                yield child
    for child in authored_children(node):
        value = _canonical(child)
        # Entity spelling and CDATA boundaries can split one text value into
        # parser callbacks; callback boundaries are not authored structures.
        if isinstance(child, _Text) and children and children[-1][0] == value[0]:
            children[-1] = (value[0], children[-1][1] + value[1])
        else:
            children.append(value)
    return children


def _canonical(node: _Node | _Text):
    if isinstance(node, _Text):
        return ("literal" if node.literal else "text", node.value)
    return (
        node.tag,
        sorted((key, value) for key, value in node.attrs.items() if key not in _INSTANCE_ATTRIBUTES),
        _canonical_children(node),
    )


def _content(node: _Node) -> str:
    return json.dumps(_canonical(node), ensure_ascii=False, separators=(",", ":"))


def _envelope(node: _Node) -> tuple[list[_Node], list[_Node | _Text]]:
    """A macro's parameters, and every other authored child as its body."""
    parameters = [child for child in node.children if isinstance(child, _Node) and child.tag == "ac:parameter"]
    body = [child for child in node.children
            if (child.tag != "ac:parameter" if isinstance(child, _Node) else child.value.strip())]
    return parameters, body


def _declared(node: _Node, parameters: list[_Node], body: list[_Node | _Text]) -> _Macro | None:
    """The declared rule, when this instance has exactly the shape it describes."""
    rule = _MACROS.get(node.attrs.get("ac:name", ""))
    if rule is None:
        return None
    names = [child.attrs.get("ac:name", "") for child in parameters]
    conforms = (
        len(set(names)) == len(names) and set(names) <= rule.parameters
        and (rule.identity is None or rule.identity in names)
        and all(set(child.attrs) == {"ac:name"} and not any(isinstance(c, _Node) for c in child.children)
                for child in parameters)
        and [child.tag if isinstance(child, _Node) else "" for child in body] == _BODY_ELEMENTS[rule.body]
        and (rule.body != "literal" or all(isinstance(c, _Text) and c.literal for c in body[0].children))
    )
    return rule if conforms else None


def _setting(parameter: _Node) -> tuple[str, str]:
    return parameter.attrs.get("ac:name", ""), _render(parameter).strip()


def _macro(node: _Node) -> str:
    name = node.attrs.get("ac:name", "")
    parameters, bodies = _envelope(node)
    rule = _declared(node, parameters, bodies)
    if rule is None:
        # The envelope itself: its name, every parameter value and its body.
        settings = [(key, value) for key, value in map(_setting, parameters) if value]
        label = f"Macro {name}".rstrip()
        if not bodies:
            listed = "; ".join(f"{key}={value}" if key else value for key, value in settings)
            return f"[{label}: {listed}]" if listed else f"[{label}]"
        return "\n".join((label, *(f"{key or 'value'}: {value}" for key, value in settings),
                          *(_render(body) for body in bodies)))
    names = [child.attrs.get("ac:name", "") for child in parameters]
    if name == "toc":
        # Native TOC has no authored body: its output repeats page headings.
        # The immutable input retains its configuration; headings remain selectable.
        return ""
    values = {key: _render(value) for key, value in zip(names, parameters, strict=True)}
    if rule.body == "local":
        return _render(bodies[0])
    if name == "jira":
        # Keep a supplied instance name; omit opaque server IDs and column controls.
        # A native issue key proves neither an unseen summary nor issue status.
        server = f" (server: {values['server']})" if "server" in values else ""
        return f"Issue: {values['key']}{server}"
    if name == "status":
        colour = f" (colour: {values['colour']})" if "colour" in values else ""
        return f"Status: {values['title']}{colour}"
    labels = {"status": "Status", "jira": "Issue", "code": "Code", "noformat": "Literal text"}
    lines = [labels.get(name, name)]
    lines.extend(f"{key}: {_render(value)}" for key, value in zip(names, parameters, strict=True))
    lines.extend(_render(body) for body in bodies)
    return "\n".join(lines)


def _render(node: _Node | _Text) -> str:
    if isinstance(node, _Text):
        return node.value
    if node.tag == "ac:structured-macro":
        text = _macro(node)
        return text + "\n"
    if node.tag in _EDITOR_ONLY:
        return ""
    if node.tag == "ac:emoticon":
        name = node.attrs.get("ac:name") or "; ".join(f"{key.removeprefix('ac:')}={value}" for key, value in node.attrs.items())
        return f"[Emoticon: {name}]"
    if node.tag == "ac:task":
        def is_status(child):
            return isinstance(child, _Node) and child.tag == "ac:task-status"
        authored = [child for child in node.children
                    if not (isinstance(child, _Node) and child.tag in _TASK_IDENTITIES)]
        state = "".join(_render(child) for child in authored if is_status(child)).strip()
        body = "".join(_render(child) for child in authored if not is_status(child)).strip()
        return (f"Task ({state}): {body}" if state else f"Task: {body}") + "\n"
    if node.tag == "br":
        return "\n"
    if node.tag == "hr":
        return "\n---\n"
    if node.tag == "table":
        try:
            rows = _table_presentations(node)
        except _IrregularTable:
            return _authored_rows(node) + "\n"
        captions = [_render(child).strip() for child in node.children if isinstance(child, _Node) and child.tag == "caption"]
        return "\n".join((*captions, *(f"Row {number}:\n{text}" for number, (_, text, _) in enumerate(rows, 1)))) + "\n"
    if node.tag == "pre":
        return "Preformatted text:\n" + "".join(_render(child) for child in node.children)
    if node.tag.startswith("ri:"):
        # Opaque identities stay opaque; never invent a display name.
        return "[" + node.tag.removeprefix("ri:") + ": " + "; ".join(
            f"{key.removeprefix('ri:')}={value}" for key, value in node.attrs.items()
        ) + "]"
    content = "".join(
        ("\n" if node.tag == "li" and isinstance(child, _Node) and child.tag in {"ul", "ol"} else "")
        + _render(child) for child in node.children
    )
    if node.tag in {"del", "s"}:
        return f"[Deleted text: {content}]"
    if node.tag in {"ul", "ol"}:
        items = [child for child in node.children if isinstance(child, _Node) and child.tag == "li"]
        try:
            number = int(node.attrs.get("start", str(len(items) if "reversed" in node.attrs else 1)))
        except ValueError as exc:
            raise ValueError("invalid Confluence ordered-list start") from exc
        lines = []
        for child in node.children:
            text = _render(child).strip().splitlines()
            if not (isinstance(child, _Node) and child.tag == "li"):
                # A nested list or text placed between items continues the item above it.
                lines.extend("  " + line for line in text)
                continue
            if "value" in child.attrs:
                try:
                    number = int(child.attrs["value"])
                except ValueError as exc:
                    raise ValueError("invalid Confluence list-item number") from exc
            marker = f"{number}. " if node.tag == "ol" else "- "
            lines.append(marker + (text[0] if text else ""))
            lines.extend("  " + line for line in text[1:])
            number += -1 if "reversed" in node.attrs else 1
        return "\n".join(lines) + "\n"
    if node.tag in {"sup", "sub"}:
        return f"[{node.tag}: {content}]"
    if node.tag == "a" and "href" in node.attrs:
        content += f" ({node.attrs['href']})"
    if node.tag == "img":
        return "[Image: " + "; ".join(f"{key}={value}" for key, value in node.attrs.items()) + "]"
    if node.tag == "time" and "datetime" in node.attrs:
        content += f" ({node.attrs['datetime']})"
    if node.tag == "ac:image":
        content += "".join(f" [{key.removeprefix('ac:')}: {node.attrs[key]}]" for key in ("ac:alt", "ac:title") if key in node.attrs)
    if "style" in node.attrs:
        content += f" [style: {node.attrs['style']}]"
    if node.tag in {"p", "div", "pre", "blockquote", "li"} or node.tag.startswith("h") and node.tag[1:].isdigit():
        return content + "\n"
    return content


def _table_rows(table: _Node) -> tuple[tuple[_Node, tuple[tuple[str, int, int, int], ...], bool], ...]:
    """Resolve authored row/column spans, including cells carried from prior rows.

    Only the table's own rows form its grid; a table inside a cell is that
    cell's content.
    """
    sections = [child for child in table.children if isinstance(child, _Node)]
    for container in (table, *sections):
        allowed = _TABLE_CHILDREN.get(container.tag)
        if allowed is not None and any(
            isinstance(child, _Text) and child.value.strip() or isinstance(child, _Node)
            and child.tag not in allowed for child in container.children
        ):
            raise _IrregularTable("Confluence table has content outside a cell")
    rows = [row for section in sections
            for row in ((section,) if section.tag == "tr"
                        else section.children if section.tag in {"thead", "tbody", "tfoot"} else ())
            if isinstance(row, _Node) and row.tag == "tr"]
    carried: dict[int, tuple[int, tuple[str, int, int, int]]] = {}
    output = []
    width = None
    for row_number, row in enumerate(rows, 1):
        cells = [child for child in row.children if isinstance(child, _Node)]
        if any(child.tag not in {"td", "th"} for child in cells) or any(
            isinstance(child, _Text) and child.value.strip() for child in row.children
        ):
            raise _IrregularTable("Confluence table row has content outside a cell")
        values = {column: value for column, (_, value) in carried.items()}
        next_carried = {column: (left - 1, value) for column, (left, value) in carried.items() if left > 1}
        column = 0
        for cell in cells:
            while column in values:
                column += 1
            try:
                colspan = int(cell.attrs.get("colspan", "1"))
                rowspan = int(cell.attrs.get("rowspan", "1"))
            except ValueError as exc:
                raise _IrregularTable("invalid Confluence table span") from exc
            if colspan < 1 or rowspan < 1 or colspan > 1000 or rowspan > len(rows):
                raise _IrregularTable("Confluence table span is out of bounds")
            text = (_render(cell).strip(), cell.start, cell.end, row_number)
            for offset in range(colspan):
                index = column + offset
                if index in values:
                    raise _IrregularTable("overlapping Confluence table cells")
                values[index] = text
                if rowspan > 1:
                    next_carried[index] = (rowspan - 1, text)
            column += colspan
        row_width = max(values, default=-1) + 1
        width = row_width if width is None else width
        if row_width != width or set(values) != set(range(width)):
            raise _IrregularTable("inconsistent Confluence table column coverage")
        output.append((row, tuple(values[index] for index in range(width)), bool(cells) and all(cell.tag == "th" for cell in cells)))
        carried = next_carried
    if carried:
        raise _IrregularTable("Confluence table span extends past the final row")
    return tuple(output)


def _authored_rows(node: _Node) -> str:
    """A table whose cells form no grid, read row by row as authored."""
    lines = []
    for child in node.children:
        if isinstance(child, _Node) and child.tag == "tr":
            lines.append(" | ".join(filter(None, (_render(cell).strip() for cell in child.children))))
        elif isinstance(child, _Node) and child.tag in {"thead", "tbody", "tfoot"}:
            lines.append(_authored_rows(child))
        else:
            lines.append(_render(child).strip())
    return "\n".join(filter(None, lines))


def _table_presentations(table):
    rows = _table_rows(table)
    header_rows = []
    for row, values, header in rows:
        if not header:
            break
        header_rows.append((row, values))
    headers = tuple(" / ".join(dict.fromkeys(values[column][0] for _, values in header_rows)) for column in range(len(rows[0][1]))) if rows else ()
    output = []
    for row, values, is_header in rows:
        cells: dict[tuple[str, int, int, int], list[int]] = {}
        for column, value in enumerate(values):
            cells.setdefault(value, []).append(column)
        lines = []
        for (value, start, _, source_row), columns in cells.items():
            label = " / ".join(dict.fromkeys(headers[column] or f"Column {column + 1}" for column in columns))
            coverage = f" (columns {columns[0] + 1}–{columns[-1] + 1})" if len(columns) > 1 else ""
            inherited = f" [cell from row {source_row}]" if not (row.start <= start < row.end) else ""
            lines.append(f"{label}{coverage}: {value}{inherited}")
        output.append((row, "\n".join(lines), is_header))
    return tuple(output)


def parse_storage(source: str) -> ParsedDeclaredText:
    root = _StorageParser(source).parse()
    fragments: list[DeclaredTextFragment] = []
    groups: list[DeclaredTextGroup] = []
    # Regions whose headings govern nothing beyond their own end.
    scopes = [node for node in _nodes(root) if node.tag == "ac:layout-cell"]

    def add(node, text=None, kind=None, origins=()):
        literal = node.tag == "pre" or any(child.tag in {"pre", "ac:plain-text-body"} for child in _nodes(node))
        rendered = (_render(node) if literal else _render(node).strip()) if text is None else text
        if rendered:
            fragments.append(DeclaredTextFragment(node.start, node.end, kind or node.tag, rendered, _content(node), origins))

    def select_macro(node):
        parameters, body = _envelope(node)
        rule = _declared(node, parameters, body)
        rich = body and all(isinstance(child, _Node) and child.tag == "ac:rich-text-body" for child in body)
        if rule is not None and rule.body == "local":
            for authored in body[0].children:
                select(authored)
        elif rule is None and rich and not any(
            isinstance(authored, _Text) and authored.value.strip() for child in body for authored in child.children
        ):
            # An undeclared container: its blocks stay individually selectable,
            # read under the macro's own parameter values.
            name = node.attrs.get("ac:name", "")
            settings = []
            for parameter in parameters:
                key, value = _setting(parameter)
                if value:
                    add(parameter, text=f"Macro {name} {key or 'value'}: {value}", kind="macro-parameter")
                    settings.append((parameter.start, parameter.end))
            if settings:
                groups.append(DeclaredTextGroup(node.start, node.end, tuple(settings)))
            for child in body:
                scopes.append(child)
                for authored in child.children:
                    select(authored)
        else:
            add(node)

    def select(node):
        if isinstance(node, _Text):
            if node.value.strip():
                fragments.append(DeclaredTextFragment(
                    node.start, node.end, "text", node.value if node.literal else node.value.strip(), _content(node),
                ))
            return
        if node.tag == "table":
            try:
                rows = _table_presentations(node)
            except _IrregularTable:
                add(node)
                return
            header_rows = []
            for row, _, header in rows:
                if not header:
                    break
                header_rows.append(row)
            native_cells = {(cell.start, cell.end): cell for cell in _nodes(node) if cell.tag in {"td", "th"}}
            grid = {row.start: values for row, values, _ in _table_rows(node)}
            for row, text, is_header in rows:
                external = {(header.start, header.end): header for header in header_rows if header is not row}
                for _, start, end, _ in grid[row.start]:
                    if not row.start <= start < row.end:
                        external[(start, end)] = native_cells[(start, end)]
                origins = tuple(DeclaredTextOrigin(start, end, _content(origin))
                                for (start, end), origin in sorted(external.items()))
                add(row, text=text, kind="table-header" if is_header else "table-row", origins=origins)
                if not is_header:
                    captions = [child for child in node.children if isinstance(child, _Node) and child.tag == "caption"]
                    groups.append(DeclaredTextGroup(row.start, row.end, tuple((header.start, header.end) for header in (*header_rows, *captions))))
            for child in node.children:
                if isinstance(child, _Node) and child.tag == "caption":
                    add(child)
            return
        if node.tag in _CONTAINERS or node.tag == "div" and not node.attrs:
            for child in node.children:
                select(child)
        elif node.tag == "ac:structured-macro":
            select_macro(node)
        else:
            add(node)

    for child in root.children:
        select(child)
    for parent in (root, *_nodes(root)):
        siblings = [child for child in parent.children
                    if isinstance(child, _Node) or child.value.strip()]
        for lead, listing in zip(siblings, siblings[1:]):
            if (isinstance(lead, _Node) and lead.tag == "p"
                    and isinstance(listing, _Node) and listing.tag in {"ul", "ol"}
                    and _render(lead).rstrip().endswith(":")):
                groups.append(DeclaredTextGroup(listing.start, listing.end,
                    ((lead.start, lead.end),), together=True))
    fragments.sort(key=lambda item: (item.start, item.end))
    headings = [item for item in fragments if re.fullmatch(r"h[1-6]", item.kind)]
    for index, heading in enumerate(headings):
        containing_end = min((scope.end for scope in scopes if scope.start <= heading.start < scope.end), default=len(source))
        end = next((item.start for item in headings[index + 1:] if item.kind <= heading.kind and item.start < containing_end), containing_end)
        groups.append(DeclaredTextGroup(heading.start, end, ((heading.start, heading.end),)))
    return ParsedDeclaredText(tuple(fragments), tuple(groups))


STORAGE_FORMAT = DeclaredTextFormat("confluence-storage", 1, parse_storage)
