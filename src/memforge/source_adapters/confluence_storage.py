"""Confluence storage grammar, comparison and readable selections.

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


_VOID = {"br", "hr", "img", "col", "ri:user", "ri:page", "ri:attachment", "ri:url"}
_TAGS = {
    "p", "div", "span", "strong", "b", "em", "i", "u", "s", "del", "sup", "sub",
    "a", "br", "hr", "pre", "code", "blockquote", "ul", "ol", "li", "table",
    "thead", "tbody", "tfoot", "tr", "td", "th", "colgroup", "col", "caption",
    "h1", "h2", "h3", "h4", "h5", "h6", "time", "img",
    "ac:link", "ac:link-body", "ac:plain-text-link-body", "ac:image",
    "ac:structured-macro", "ac:parameter", "ac:plain-text-body", "ac:rich-text-body",
    "ri:user", "ri:page", "ri:attachment", "ri:url",
}
_MACRO_PARAMETERS = {
    "toc": {"type", "outline", "style", "indent", "separator", "minLevel", "maxLevel",
            "include", "exclude", "printable", "class", "absoluteUrl"},
    "status": {"title", "colour"},
    "jira": {"key", "server", "serverId", "columnIds", "columns", "showSummary"},
    "code": {"language", "title", "collapse", "linenumbers", "firstline", "theme"},
    "info": {"title", "icon"}, "note": {"title", "icon"}, "warning": {"title", "icon"},
    "tip": {"title", "icon"}, "panel": {"title", "borderStyle", "borderColor", "borderWidth",
                                            "bgColor", "titleBGColor", "titleColor"},
    "expand": {"title"}, "noformat": set(),
}
# These identify a rendered macro instance, not its authored content. No other
# attribute is erased from comparison material.
_INSTANCE_ATTRIBUTES = {"ac:macro-id"}


@dataclass(slots=True)
class _Text:
    value: str
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
        if tag not in _TAGS:
            raise ValueError(f"unsupported Confluence storage element: {tag}")
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

    def handle_data(self, data):
        self.stack[-1].children.append(_Text(data))

    def handle_entityref(self, name):
        value = html.unescape(f"&{name};")
        if value == f"&{name};":
            raise ValueError("unknown Confluence storage entity")
        self.handle_data(value)

    def handle_charref(self, name):
        value = html.unescape(f"&#{name};")
        self.handle_data(value)

    def unknown_decl(self, data):
        node = self.stack[-1]
        parent = self.stack[-2] if len(self.stack) > 1 else None
        if not data.startswith("CDATA[") or parent is None:
            raise ValueError("unsupported Confluence storage declaration")
        literal = (
            node.tag == "ac:plain-text-body" and parent.tag == "ac:structured-macro"
            and parent.attrs.get("ac:name") in {"code", "noformat"}
        )
        if not literal and node.tag != "ac:plain-text-link-body":
            raise ValueError("Confluence CDATA has no declared body representation")
        start = self._source_position() + len("<![CDATA[")
        end = self.source.find("]]>", start)
        if end < 0:
            raise ValueError("unclosed Confluence CDATA")
        node.children.append(_Text(self.source[start:end], literal=True))

    def handle_decl(self, decl):
        raise ValueError("Confluence storage must not contain a document declaration")

    def handle_pi(self, data):
        raise ValueError("Confluence storage must not contain processing instructions")

    def parse(self) -> _Node:
        self.feed(self.source)
        self.close()
        if len(self.stack) != 1:
            raise ValueError("unclosed Confluence storage element")
        return self.root


def _nodes(node: _Node):
    for child in node.children:
        if isinstance(child, _Node):
            yield child
            yield from _nodes(child)


def _canonical(node: _Node | _Text):
    if isinstance(node, _Text):
        return ("literal" if node.literal else "text", node.value)
    children = []
    for child in node.children:
        value = _canonical(child)
        # Entity spelling and CDATA boundaries can split one text value into
        # parser callbacks; callback boundaries are not authored structures.
        if isinstance(child, _Text) and children and children[-1][0] == value[0]:
            children[-1] = (value[0], children[-1][1] + value[1])
        else:
            children.append(value)
    return (
        node.tag,
        sorted((key, value) for key, value in node.attrs.items() if key not in _INSTANCE_ATTRIBUTES),
        children,
    )


def _content(node: _Node) -> str:
    return json.dumps(_canonical(node), ensure_ascii=False, separators=(",", ":"))


def _macro(node: _Node) -> str:
    name = node.attrs.get("ac:name", "")
    if name not in _MACRO_PARAMETERS:
        raise ValueError(f"unsupported Confluence macro: {name}")
    parameters = [child for child in node.children if isinstance(child, _Node) and child.tag == "ac:parameter"]
    names = [child.attrs.get("ac:name", "") for child in parameters]
    if len(set(names)) != len(names) or set(names) - _MACRO_PARAMETERS[name]:
        raise ValueError("unsupported or duplicate Confluence macro parameter")
    if any(set(child.attrs) != {"ac:name"} or any(isinstance(c, _Node) for c in child.children) for child in parameters):
        raise ValueError("Confluence macro parameter is not a declared scalar")
    if name in {"status", "jira"} and ("title" if name == "status" else "key") not in names:
        raise ValueError("Confluence status/issue macro lacks its identity parameter")
    bodies = [child for child in node.children if isinstance(child, _Node) and child.tag in {"ac:plain-text-body", "ac:rich-text-body"}]
    if any(isinstance(child, _Text) and child.value.strip() for child in node.children):
        raise ValueError("unclassified Confluence macro text")
    if any(isinstance(child, _Node) and child.tag not in {"ac:parameter", "ac:plain-text-body", "ac:rich-text-body"} for child in node.children):
        raise ValueError("unclassified Confluence macro child")
    if name in {"code", "noformat"}:
        if len(bodies) != 1 or bodies[0].tag != "ac:plain-text-body" or any(
            not isinstance(child, _Text) or not child.literal for child in bodies[0].children
        ):
            raise ValueError("Confluence literal macro must have one CDATA body")
    elif name in {"status", "jira", "toc"} and bodies:
        raise ValueError("unexpected Confluence bodyless macro body")
    elif name not in {"status", "jira", "toc"} and (len(bodies) != 1 or bodies[0].tag != "ac:rich-text-body"):
        raise ValueError("Confluence container macro must have one rich-text body")
    if name == "toc":
        # Native TOC has no authored body: its output repeats page headings.
        # The immutable input retains its configuration; headings remain selectable.
        return ""
    values = {key: _render(value) for key, value in zip(names, parameters, strict=True)}
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
    if node.tag == "br":
        return "\n"
    if node.tag == "hr":
        return "\n---\n"
    if node.tag == "table":
        rows = _table_presentations(node)
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
        children = [child for child in node.children if isinstance(child, _Node)]
        if any(child.tag != "li" for child in children) or any(
            isinstance(child, _Text) and child.value.strip() for child in node.children
        ):
            raise ValueError("Confluence list has content outside an item")
        try:
            number = int(node.attrs.get("start", str(len(children) if "reversed" in node.attrs else 1)))
        except ValueError as exc:
            raise ValueError("invalid Confluence ordered-list start") from exc
        lines = []
        for child in children:
            if "value" in child.attrs:
                try:
                    number = int(child.attrs["value"])
                except ValueError as exc:
                    raise ValueError("invalid Confluence list-item number") from exc
            marker = f"{number}. " if node.tag == "ol" else "- "
            text = _render(child).strip().splitlines()
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
    """Resolve authored row/column spans, including cells carried from prior rows."""
    rows = [node for node in _nodes(table) if node.tag == "tr"]
    if any(node.tag == "table" for node in _nodes(table)):
        raise ValueError("nested Confluence tables need a separately declared structure")
    allowed_children = {
        "table": {"thead", "tbody", "tfoot", "tr", "caption", "colgroup", "col"},
        "thead": {"tr"}, "tbody": {"tr"}, "tfoot": {"tr"}, "colgroup": {"col"},
    }
    for container in (table, *_nodes(table)):
        if container.tag not in allowed_children:
            continue
        if any(isinstance(child, _Text) and child.value.strip() or isinstance(child, _Node)
               and child.tag not in allowed_children[container.tag] for child in container.children):
            raise ValueError("Confluence table has unclassified content outside a cell")
    carried: dict[int, tuple[int, tuple[str, int, int, int]]] = {}
    output = []
    width = None
    for row_number, row in enumerate(rows, 1):
        cells = [child for child in row.children if isinstance(child, _Node)]
        if any(child.tag not in {"td", "th"} for child in cells) or any(
            isinstance(child, _Text) and child.value.strip() for child in row.children
        ):
            raise ValueError("Confluence table row has content outside a cell")
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
                raise ValueError("invalid Confluence table span") from exc
            if colspan < 1 or rowspan < 1 or colspan > 1000 or rowspan > len(rows):
                raise ValueError("Confluence table span is out of bounds")
            text = (_render(cell).strip(), cell.start, cell.end, row_number)
            for offset in range(colspan):
                index = column + offset
                if index in values:
                    raise ValueError("overlapping Confluence table cells")
                values[index] = text
                if rowspan > 1:
                    next_carried[index] = (rowspan - 1, text)
            column += colspan
        row_width = max(values, default=-1) + 1
        width = row_width if width is None else width
        if row_width != width or set(values) != set(range(width)):
            raise ValueError("inconsistent Confluence table column coverage")
        output.append((row, tuple(values[index] for index in range(width)), bool(cells) and all(cell.tag == "th" for cell in cells)))
        carried = next_carried
    if carried:
        raise ValueError("Confluence table span extends past the final row")
    return tuple(output)


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
    # Validate controls even when a table/list is the selected parent.
    for node in _nodes(root):
        if node.tag == "ac:structured-macro":
            _macro(node)
    fragments: list[DeclaredTextFragment] = []
    groups: list[DeclaredTextGroup] = []

    def add(node, text=None, kind=None, origins=()):
        literal = node.tag == "pre" or any(child.tag in {"pre", "ac:plain-text-body"} for child in _nodes(node))
        rendered = (_render(node) if literal else _render(node).strip()) if text is None else text
        if rendered:
            fragments.append(DeclaredTextFragment(node.start, node.end, kind or node.tag, rendered, _content(node), origins))

    def select(node):
        if isinstance(node, _Text):
            if node.value.strip():
                raise ValueError("Confluence authored text is outside a selectable native element")
            return
        if node.tag == "table":
            rows = _table_presentations(node)
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
        if node.tag == "div" and not node.attrs:
            for child in node.children:
                select(child)
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
        end = next((item.start for item in headings[index + 1:] if item.kind <= heading.kind), len(source))
        groups.append(DeclaredTextGroup(heading.start, end, ((heading.start, heading.end),)))
    return ParsedDeclaredText(tuple(fragments), tuple(groups))


STORAGE_FORMAT = DeclaredTextFormat("confluence-storage", 1, parse_storage)
