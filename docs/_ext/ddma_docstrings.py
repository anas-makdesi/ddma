"""Render ddma's docstrings, which are written to read well as plain text (``help()``).

ddma's docstrings list parameters as ``name : what it is`` (no separate type line), and use
plain-text notation such as ``|Q|`` and ``(*shape, n)``. Napoleon would read the description
as a type and swallow the paragraphs after the list, and docutils reads ``|Q|`` as a
substitution. This hook runs before napoleon and turns the ``Parameters`` / ``Attributes`` /
``Returns`` sections into field lists, ending each section at the first paragraph that is not
a list entry, and escapes the plain-text notation.
"""

from __future__ import annotations

import re

_ENTRY = re.compile(r"^(?P<names>[A-Za-z_*][\w*]*(?:\s*,\s*[A-Za-z_*][\w*]*)*)\s+:\s?(?P<desc>.*)$")
_FIELD = {"Parameters": "param", "Attributes": "ivar", "Returns": None}


def _escape(line: str) -> str:
    if "``" in line or line.lstrip().startswith(">>>"):
        # leave inline literals alone, escape only outside them
        parts = line.split("``")
        for i in range(0, len(parts), 2):
            parts[i] = _escape(parts[i]) if "``" not in parts[i] else parts[i]
        return "``".join(parts)
    line = re.sub(r"(?<![\\|`])\|([^|\s][^|]*?)\|(?!`)", r"\\|\1\\|", line)
    line = re.sub(r"\(\*(\w)", r"(\\*\1", line)
    return line


def _convert(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        head = ln.strip()
        if head in _FIELD and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            kind = _FIELD[head]
            i += 2
            if kind is None and i < len(lines) and not _ENTRY.match(lines[i]):
                # free-text Returns section: one paragraph
                para = []
                while i < len(lines) and lines[i].strip():
                    para.append(lines[i].strip())
                    i += 1
                out += [":returns: " + " ".join(para), ""]
                continue
            entries: list[tuple[str, list[str]]] = []
            while i < len(lines):
                cur = lines[i]
                m = _ENTRY.match(cur)
                if m and not cur.startswith(" "):
                    entries.append((m["names"], [m["desc"]]))
                    i += 1
                elif cur.startswith(" ") and entries:
                    entries[-1][1].append(cur.strip())
                    i += 1
                elif not cur.strip():
                    # blank line: the section goes on only if the next line is an entry
                    nxt = lines[i + 1] if i + 1 < len(lines) else ""
                    if _ENTRY.match(nxt) and not nxt.startswith(" "):
                        i += 1
                        continue
                    break
                else:
                    break
            if kind is None:
                out += [":returns:"]
                for names, desc in entries:
                    out += [f"   * **{names}** -- {' '.join(desc)}"]
            else:
                for names, desc in entries:
                    text = " ".join(d for d in desc if d)
                    for name in (n.strip() for n in names.split(",")):
                        out.append(f":{kind} {name}: {text}")
            out.append("")
            continue
        out.append(ln)
        i += 1
    return out


def _bare_entries(lines: list[str]) -> list[str]:
    """A paragraph of ``name : description`` entries without a section header (class
    docstrings listing their fields) is read as an Attributes section."""
    out: list[str] = []
    prev_blank = True
    for ln in lines:
        m = _ENTRY.match(ln)
        if prev_blank and m and not ln.startswith(" ") and out and out[-1] != "Attributes":
            out += ["Attributes", "----------"]
        out.append(ln)
        prev_blank = not ln.strip()
    return out


def process(app, what, name, obj, options, lines):
    if not name.startswith("ddma"):
        return
    new = _convert(_bare_entries([_escape(ln) for ln in lines]))
    lines[:] = new


def setup(app):
    app.connect("autodoc-process-docstring", process, priority=400)
    return {"parallel_read_safe": True}
