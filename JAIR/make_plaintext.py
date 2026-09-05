#!/usr/bin/env python3
"""Render the JAIR paper as one plain Markdown file, for a similarity check.

Why this exists: plagiarism checkers extract text from the typeset PDF, and
this paper defeats that. It is set in Libertine with T1 encoding and microtype
protrusion, so extraction returns broken ligatures; the tables are shaded
booktabs floats that come out as columns of loose numbers; and the maths is a
stream of glyphs with no word boundaries. What a checker should be comparing is
the prose. This script emits the prose.

    python JAIR/make_plaintext.py

Output goes to JAIR/submission/plagcheck/. It is derived from paper/sections/
on every run, so it never drifts from the manuscript.

What is preserved: every sentence of running text, every section and
subsection heading in order, every float caption, every list, the pseudo-code,
and the reference list. Citations are expanded to author-year, and cross
references to their numbers, so the text reads as text.

What is dropped, and why: the bodies of tabular environments and the figure
images. Neither is prose, both extract as noise, and a similarity score
computed over columns of floating-point numbers is meaningless. Each float
keeps its caption and leaves a marker saying the data is in the PDF, so nothing
is silently missing.

Maths is kept, converted to Unicode where a character exists. Display equations
are indented as code blocks so Markdown cannot mangle them.

Structure is read from paper/main.tex and paper/jair.tex rather than hardcoded,
so adding a section to the paper adds it here too.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SECTIONS = PAPER / "sections"
OUT = ROOT / "JAIR" / "submission" / "plagcheck"

TITLE = "Initialisation Sets the Conditioning of LoRA Merging"
AUTHORS = "Sankalp Pathak, Sanjay Garg and Piyush Kumar Singh"
AFFIL = ("Department of Computer Science and Engineering, "
         "Jaypee University of Engineering and Technology, Guna, "
         "Madhya Pradesh, India")


# --------------------------------------------------------------------------
# 1. LaTeX comment stripping
# --------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        i, n, cut = 0, len(line), None
        while i < n:
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == "%":
                cut = i
                break
            i += 1
        if cut is None:
            out.append(line)
        elif cut == 0:
            continue  # whole-line comment: drop the line, keep no blank
        else:
            out.append(line[:cut])
    return "\n".join(out)


# --------------------------------------------------------------------------
# 2. Bibliography
# --------------------------------------------------------------------------

COMBINING = {"'": "́", "`": "̀", "^": "̂", '"': "̈",
             "~": "̃", "=": "̄", ".": "̇", "u": "̆",
             "v": "̌", "H": "̋", "c": "̧", "k": "̨",
             "r": "̊", "d": "̣", "b": "̱"}

LIGATURES = {"ss": "ß", "o": "ø", "O": "Ø", "l": "ł",
             "L": "Ł", "aa": "å", "AA": "Å", "ae": "æ",
             "AE": "Æ", "oe": "œ", "OE": "Œ", "i": "ı"}


def debrace(s: str) -> str:
    """Bibliography field to plain text: accents, ligatures, braces.

    Author surnames are the one place in this file where dropping a LaTeX
    command silently would be a factual error rather than a cosmetic one, so
    accents are decoded rather than stripped.
    """
    import unicodedata

    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in "{}":
            i += 1
            continue
        if c != "\\":
            out.append(c)
            i += 1
            continue
        m = re.match(r"\\([a-zA-Z]+)", s[i:])
        if m:
            name = m.group(1)
            i += m.end()
            if name in ("texttt", "emph", "textbf", "textit", "mbox",
                        "textsc", "text", "url", "href"):
                continue
            if name in COMBINING or name in LIGATURES:
                j = i
                while j < n and s[j] in " {":
                    j += 1
                if name in COMBINING and j < n and s[j] not in "}\\":
                    out.append(s[j] + COMBINING[name])
                    i = j + 1
                    continue
                out.append(LIGATURES.get(name, ""))
                continue
            out.append(name)
            continue
        acc = s[i + 1] if i + 1 < n else ""
        if acc in COMBINING:
            j = i + 2
            while j < n and s[j] in " {":
                j += 1
            if j < n:
                out.append(s[j] + COMBINING[acc])
                i = j + 1
                continue
        out.append(acc)
        i += 2
    return unicodedata.normalize("NFC", "".join(out))


def parse_bib(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        start = text.index("{", m.start())
        depth, i = 0, start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[start + 1:i]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            j = fm.end()
            if j >= len(body):
                continue
            if body[j] == "{":
                d, k = 0, j
                while k < len(body):
                    if body[k] == "{":
                        d += 1
                    elif body[k] == "}":
                        d -= 1
                        if d == 0:
                            break
                    k += 1
                val = body[j + 1:k]
            elif body[j] == '"':
                k = body.index('"', j + 1)
                val = body[j + 1:k]
            else:
                k = j
                while k < len(body) and body[k] not in ",\n":
                    k += 1
                val = body[j:k]
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", val).strip()
        entries[m.group(2).strip()] = fields
    return entries


def surnames(author_field: str) -> list:
    """(surname, given) pairs. BibTeX's bare `others` means et al., not a
    person called Others."""
    names = []
    for a in re.split(r"\s+and\s+", debrace(author_field)):
        a = a.strip()
        if not a:
            continue
        if a.lower() == "others":
            names.append(("\x05others", ""))
            continue
        if "," in a:
            last, first = a.split(",", 1)
            names.append((last.strip(), first.strip()))
        else:
            bits = a.split()
            names.append((bits[-1], " ".join(bits[:-1])) if bits else (a, ""))
    return names


def initials(given: str) -> str:
    bits = [b for b in re.split(r"[\s.]+", given) if b]
    return " ".join(b[0] + "." for b in bits)


def cite_short(entry: dict) -> str:
    names = surnames(entry.get("author", ""))
    year = debrace(entry.get("year", "n.d."))
    if not names:
        return debrace(entry.get("title", "Anon"))[:20] + ", " + year
    if any(n[0] == "\x05others" for n in names) or len(names) > 2:
        who = names[0][0] + " et al."
    elif len(names) == 1:
        who = names[0][0]
    else:
        who = names[0][0] + " & " + names[1][0]
    return who + ", " + year


def format_reference(entry: dict) -> str:
    names = surnames(entry.get("author", ""))
    if names:
        etal = any(n[0] == "\x05others" for n in names)
        names = [n for n in names if n[0] != "\x05others"]
        parts = [n[0] + (", " + initials(n[1]) if n[1] else "") for n in names]
        if etal:
            who = ", ".join(parts) + ", et al."
        elif len(parts) == 1:
            who = parts[0]
        else:
            who = ", ".join(parts[:-1]) + ", & " + parts[-1]
    else:
        who = "Anon"
    year = debrace(entry.get("year", "n.d."))
    title = debrace(entry.get("title", "")).rstrip(".")
    venue = ""
    for key in ("booktitle", "journal", "howpublished", "school",
                "institution", "publisher"):
        if entry.get(key):
            venue = debrace(entry[key])
            break
    bits = [who + " (" + year + "). " + title + "."]
    if venue:
        bits.append(venue + ".")
    if entry.get("volume"):
        bits.append("Volume " + debrace(entry["volume"]) + ".")
    eprint = debrace(entry.get("eprint", ""))
    if eprint and eprint not in venue:
        bits.append("arXiv:" + eprint + ".")
    elif eprint:
        pass  # the venue field already spells out "arXiv preprint arXiv:..."
    elif entry.get("doi"):
        bits.append("doi:" + debrace(entry["doi"]) + ".")
    elif entry.get("url"):
        bits.append(debrace(entry["url"]))
    if entry.get("note"):
        bits.append(debrace(entry["note"]) + ".")
    return " ".join(bits)


# --------------------------------------------------------------------------
# 3. Document structure, read out of main.tex and jair.tex
# --------------------------------------------------------------------------

def resolve_topmatter(text: str) -> str:
    """Keep the JAIR branch of every \\ifdefined\\TopmatterAlreadySet block."""
    out, i = [], 0
    tag = r"\ifdefined\TopmatterAlreadySet"
    while True:
        j = text.find(tag, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = text.find(r"\fi", j)
        block = text[j + len(tag):k]
        els = block.find(r"\else")
        out.append(block[:els] if els >= 0 else block)
        i = k + 3
    return "".join(out)


def build_plan() -> list:
    """[(kind, payload)] in document order. kind is section/appendix/file/refs."""
    main = resolve_topmatter(strip_comments((PAPER / "main.tex").read_text(
        encoding="utf-8", errors="ignore")))
    plan = [("file", "abstract_jair.tex")]  # jair.tex sets it in the topmatter
    in_appendix = False
    token = re.compile(
        r"\\(section|subsubsection\*?|appendix|input|printbibliography)"
        r"(?:\{([^}]*)\})?((?:\\label\{[^}]*\})*)")
    for m in token.finditer(main):
        cmd, arg, labels = m.group(1), m.group(2) or "", m.group(3) or ""
        keys = re.findall(r"\\label\{([^}]*)\}", labels)
        if cmd == "appendix":
            in_appendix = True
        elif cmd == "section":
            plan.append(("appendix" if in_appendix else "section", (arg, keys)))
        elif cmd.startswith("subsubsection"):
            plan.append(("unnumbered", (arg, keys)))
        elif cmd == "input":
            name = arg.split("/")[-1]
            if not name.endswith(".tex"):
                name += ".tex"
            if name in ("preamble.tex",):
                continue
            plan.append(("file", name))
        elif cmd == "printbibliography":
            plan.append(("refs", None))
    plan.append(("file", "app_checklist.tex"))  # jair.tex, after main.tex
    return plan


# --------------------------------------------------------------------------
# 4. Numbering pass
# --------------------------------------------------------------------------

FLOATS = {"table": "table", "table*": "table", "figure": "figure",
          "figure*": "figure", "algorithm": "algorithm"}
NUMBERED_MATH = {"equation", "align", "gather", "multline"}
THEOREMS = {"theorem", "lemma", "corollary", "proposition", "definition",
            "claim", "remark", "assumption"}


class Numbering:
    def __init__(self):
        self.labels = {}
        self.sec = 0
        self.app = 0
        self.sub = 0
        self.subsub = 0
        self.tab = 0
        self.fig = 0
        self.alg = 0
        self.eq = 0
        self.thm = 0
        self.in_appendix = False
        self.cur = ""
        self.heading_queue = []

    def section(self, keys):
        if self.in_appendix:
            self.app += 1
            self.cur = chr(ord("A") + self.app - 1)
        else:
            self.sec += 1
            self.cur = str(self.sec)
        self.sub = self.subsub = 0
        for k in keys:
            self.labels[k] = self.cur
        return self.cur

    def subsection(self, keys):
        self.sub += 1
        self.subsub = 0
        num = self.cur.split(".")[0] + "." + str(self.sub)
        for k in keys:
            self.labels[k] = num
        return num

    def subsubsection(self, keys):
        self.subsub += 1
        head = self.cur.split(".")[0]
        num = head + "." + str(self.sub) + "." + str(self.subsub)
        for k in keys:
            self.labels[k] = num
        return num


ENV_OPEN = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
ENV_CLOSE = re.compile(r"\\end\{([a-zA-Z*]+)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
SUBSEC = re.compile(r"\\(subsection|subsubsection)\*?\{")


def scan_numbers(text: str, N: Numbering) -> None:
    """Walk one section file and record every label's number."""
    stack = []  # innermost numbered context
    pos = 0
    pat = re.compile(
        r"\\begin\{([a-zA-Z*]+)\}|\\end\{([a-zA-Z*]+)\}|"
        r"\\label\{([^}]*)\}|\\(subsection|subsubsection)(\*?)\s*\{")
    # subsection headings can carry the label on the same line, so track the
    # heading level as we go and let \label attach to the innermost context.
    ctx = None
    for m in pat.finditer(text):
        if m.group(1):
            env = m.group(1)
            base = env.rstrip("*")
            if env in FLOATS:
                if base == "table":
                    N.tab += 1
                    stack.append(str(N.tab))
                elif base == "figure":
                    N.fig += 1
                    stack.append(str(N.fig))
                else:
                    N.alg += 1
                    stack.append(str(N.alg))
            elif env in NUMBERED_MATH:
                N.eq += 1
                stack.append(str(N.eq))
            elif base in THEOREMS:
                N.thm += 1
                stack.append(str(N.thm))
            else:
                stack.append(None)
        elif m.group(2):
            if stack:
                stack.pop()
        elif m.group(3):
            live = [s for s in stack if s]
            N.labels[m.group(3)] = live[-1] if live else (ctx or N.cur)
        else:
            if m.group(5):     # \subsection*: prints no number, consumes none
                N.heading_queue.append("")
                continue
            ctx = (N.subsection([]) if m.group(4) == "subsection"
                   else N.subsubsection([]))
            # Not every subsection carries a \label, and the ones that do not
            # still have to print their number. Both passes walk the document
            # in the same order, so pass 2 reads this queue off the front.
            N.heading_queue.append(ctx)
    _ = pos


# --------------------------------------------------------------------------
# 5. Maths to Unicode
# --------------------------------------------------------------------------

GREEK = {
    "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "delta": "\u03b4",
    "Delta": "\u0394", "epsilon": "\u03b5", "varepsilon": "\u03b5",
    "zeta": "\u03b6", "eta": "\u03b7", "theta": "\u03b8", "Theta": "\u0398",
    "iota": "\u03b9", "kappa": "\u03ba", "lambda": "\u03bb",
    "Lambda": "\u039b", "mu": "\u03bc", "nu": "\u03bd", "xi": "\u03be",
    "pi": "\u03c0", "Pi": "\u03a0", "rho": "\u03c1", "sigma": "\u03c3",
    "Sigma": "\u03a3", "tau": "\u03c4", "phi": "\u03c6", "varphi": "\u03c6",
    "Phi": "\u03a6", "chi": "\u03c7", "psi": "\u03c8", "Psi": "\u03a8",
    "omega": "\u03c9", "Omega": "\u03a9", "ell": "\u2113",
}

SYMBOLS = {
    "times": "\u00d7", "cdot": "\u00b7", "cdots": "\u00b7\u00b7\u00b7",
    "ldots": "...", "dots": "...", "pm": "\u00b1", "mp": "\u2213",
    "leq": "\u2264", "le": "\u2264", "geq": "\u2265", "ge": "\u2265",
    "neq": "\u2260", "ne": "\u2260", "approx": "\u2248",
    "equiv": "\u2261", "sim": "~", "simeq": "\u2243", "propto": "\u221d",
    "in": "\u2208", "notin": "\u2209", "subseteq": "\u2286",
    "subset": "\u2282", "supseteq": "\u2287", "cup": "\u222a",
    "cap": "\u2229", "emptyset": "\u2205", "to": "\u2192",
    "rightarrow": "\u2192", "mapsto": "\u21a6", "gets": "\u2190",
    "leftarrow": "\u2190", "succ": "\u227b", "prec": "\u227a",
    "succeq": "\u2ab0", "preceq": "\u2aaf", "geqslant": "\u2265",
    "leqslant": "\u2264", "asymp": "\u224d", "triangleq": "\u225c",
    "gg": "\u226b", "ll": "\u226a", "infty": "\u221e", "partial": "\u2202",
    "nabla": "\u2207", "forall": "\u2200", "exists": "\u2203",
    "sum": "\u2211", "prod": "\u220f", "int": "\u222b", "square": "\u220e",
    "star": "*", "ast": "*", "circ": "\u2218", "oplus": "\u2295",
    "otimes": "\u2297", "top": "\u1d40", "perp": "\u22a5", "mid": "|",
    "langle": "\u27e8", "rangle": "\u27e9", "lVert": "||", "rVert": "||",
    "lvert": "|", "rvert": "|", "|": "||", "setminus": "\\",
    "colon": ":", "quad": " ", "qquad": "  ", ",": " ", ";": " ",
    "!": "", ":": " ", " ": " ",
}

WORDS = ["max", "min", "log", "ln", "exp", "inf", "sup", "dim", "cos", "sin",
         "det", "arg", "lim", "Pr", "rank", "range", "tr", "deg", "gcd"]

DROP = ["bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl", "Biggr",
        "big", "Big", "bigg", "Bigg", "bigm", "Bigm", "left", "right",
        "displaystyle", "textstyle", "scriptstyle", "limits", "nolimits",
        "notag", "nonumber", "centering", "small", "footnotesize",
        "scriptsize", "smallskip", "medskip", "bigskip", "hfill", "noindent",
        "vspace", "hspace", "linebreak", "newline"]

ACCENTS = {"bar": "\u0304", "hat": "\u0302", "widehat": "\u0302",
           "tilde": "\u0303", "widetilde": "\u0303", "dot": "\u0307",
           "vec": "\u20d7", "check": "\u030c"}


def take_group(s: str, i: int):
    """Take the argument at s[i]. Returns (contents, index after it).

    Skips the space in `\\bar c`, which TeX treats as part of the command name
    rather than as the argument. Getting this wrong put the macron on the
    space instead of the letter.
    """
    while i < len(s) and s[i] in " \t\n":
        i += 1
    if i >= len(s) or s[i] != "{":
        if i < len(s) and s[i] == "\\":
            m = re.match(r"\\[a-zA-Z]+", s[i:])
            if m:
                return m.group(0), i + m.end()
        return (s[i], i + 1) if i < len(s) else ("", i)
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def math_to_text(m: str) -> str:
    m = m.strip()
    # primes before anything else: the prose pass turns a bare '' into a
    # closing quotation mark, which would silently corrupt g''(x)
    m = m.replace("'''", "‴").replace("''", "″").replace("'", "′")
    # project macros first
    m = m.replace(r"\deff", "d_eff")
    m = m.replace(r"\tauH", "\u03c4\u0304_H")
    m = re.sub(r"\\R\b", "\u211d", m)
    m = re.sub(r"\\E\b", "E", m)

    out, i, n = [], 0, len(m)
    while i < n:
        c = m[i]
        if c != "\\":
            if c == "&":
                out.append(" ")
            elif c == "~":
                out.append(" ")
            else:
                out.append(c)
            i += 1
            continue
        mm = re.match(r"\\([a-zA-Z]+)", m[i:])
        if not mm:
            nxt = m[i + 1] if i + 1 < n else ""
            if nxt in "{}%$&#_":
                out.append(nxt)
                i += 2
                continue
            out.append(SYMBOLS.get(nxt, nxt))
            i += 2
            continue
        cmd = mm.group(1)
        i += mm.end()
        if cmd in ACCENTS:
            body, i = take_group(m, i)
            out.append(math_to_text(body) + ACCENTS[cmd])
        elif cmd in ("frac", "tfrac", "dfrac"):
            a, i = take_group(m, i)
            b, i = take_group(m, i)
            a, b = math_to_text(a), math_to_text(b)
            if len(a) > 1 and not (a[0] == "(" and a[-1] == ")"):
                a = "(" + a + ")"
            if len(b) > 1 and not (b[0] == "(" and b[-1] == ")"):
                b = "(" + b + ")"
            out.append(a + "/" + b)
        elif cmd == "sqrt":
            body, i = take_group(m, i)
            out.append("sqrt(" + math_to_text(body) + ")")
        elif cmd == "textsc":
            body, i = take_group(m, i)
            out.append(math_to_text(body).upper())
        elif cmd in ("mathrm", "mathbf", "mathit", "mathsf", "mathtt",
                     "boldsymbol", "bm", "text", "textrm", "textbf", "textit",
                     "mathbb", "mathcal", "mathscr", "operatorname",
                     "textnormal", "texttt", "emph"):
            body, i = take_group(m, i)
            out.append(math_to_text(body))
        elif cmd == "phantom":
            _, i = take_group(m, i)
        elif cmd in GREEK:
            out.append(GREEK[cmd])
        elif cmd in WORDS:
            out.append(cmd)
        elif cmd in DROP:
            pass
        elif cmd in SYMBOLS:
            out.append(SYMBOLS[cmd])
        else:
            out.append(cmd)
    s = "".join(out)
    # tidy braces left by sub/superscripts
    s = re.sub(r"\^\{([^{}]*)\}", lambda g: "^" + g.group(1), s)
    s = re.sub(r"_\{([^{}]*)\}", lambda g: "_" + g.group(1), s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,;.)])", lambda g: g.group(1), s)
    return s.strip()


# --------------------------------------------------------------------------
# 6. Pseudo-code
# --------------------------------------------------------------------------

def plain(s: str) -> str:
    """Drop Markdown emphasis. Pseudo-code goes inside a code block, where
    asterisks and backticks would print literally."""
    s = re.sub(r"\*\*([^*]*)\*\*", lambda m: m.group(1), s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: m.group(1), s)
    return s.replace("`", "").strip()


def render_algorithmic(body: str) -> str:
    lines, indent = [], 0
    body = re.sub(r"\n\s*", " ", body)
    body = re.sub(r"\\(State|Require|Ensure|For|EndFor|If|ElsIf|Else|EndIf|"
                  r"While|EndWhile|Return|Repeat|Until|Function|EndFunction)",
                  lambda g: "\n\\" + g.group(1), body)
    for raw in body.split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r"\\([A-Za-z]+)\s*", raw)
        if not m:
            continue
        cmd, rest = m.group(1), raw[m.end():]
        comment = ""
        cm = re.search(r"\\Comment", rest)
        if cm:
            arg, _ = take_group(rest, cm.end())
            comment = "    // " + plain(inline_text(arg))
            rest = rest[:cm.start()]
        if rest.startswith("{"):
            arg, after = take_group(rest, 0)
            rest = arg + rest[after:]
        text = plain(inline_text(rest))
        if cmd == "State" and not text:
            continue  # \State \Return, already split onto the Return line
        if cmd in ("EndFor", "EndIf", "EndWhile", "EndFunction", "Until",
                   "ElsIf", "Else"):
            indent = max(0, indent - 1)
        pre = "  " * indent
        if cmd == "Require":
            lines.append(pre + "Require: " + text)
        elif cmd == "Ensure":
            lines.append(pre + "Ensure:  " + text)
        elif cmd == "For":
            lines.append(pre + "for " + text + " do")
            indent += 1
        elif cmd == "While":
            lines.append(pre + "while " + text + " do")
            indent += 1
        elif cmd == "If":
            lines.append(pre + "if " + text + " then")
            indent += 1
        elif cmd == "ElsIf":
            lines.append(pre + "else if " + text + " then")
            indent += 1
        elif cmd == "Else":
            lines.append(pre + "else")
            indent += 1
        elif cmd in ("EndFor", "EndIf", "EndWhile", "EndFunction"):
            lines.append(pre + "end")
        elif cmd == "Return":
            lines.append(pre + "return " + text)
        else:
            lines.append(pre + text)
        if comment and lines:
            lines[-1] = lines[-1] + comment
    return "\n".join("    " + ln for ln in lines)


# --------------------------------------------------------------------------
# 7. Inline conversion
# --------------------------------------------------------------------------

BIB = {}
LABELS = {}


def cite(keys: str, textual: bool) -> str:
    parts = []
    for k in [x.strip() for x in keys.split(",") if x.strip()]:
        e = BIB.get(k)
        if not e:
            parts.append(k)
            continue
        short = cite_short(e)
        if textual:
            who, _, yr = short.rpartition(", ")
            parts.append(who + " (" + yr + ")")
        else:
            parts.append(short)
    return "; ".join(parts) if textual else "(" + "; ".join(parts) + ")"


def inline_text(s: str) -> str:
    """LaTeX inline markup to Markdown. Assumes floats already removed."""
    # escaped braces out of the way; stray ones are stripped at the end
    s = s.replace(r"\{", "\x03").replace(r"\}", "\x04")
    # \paragraph is a run-in heading, so it stays inline but opens a block.
    # Brace-aware, because the titles carry \emph and maths of their own; the
    # contents go on through the rules below like any other text.
    while True:
        pm = re.search(r"\\paragraph\*?\s*\{", s)
        if not pm:
            break
        inner, after = take_group(s, pm.end() - 1)
        title = inner.strip()
        if not title.endswith((".", "?", "!", ":")):
            title += "."
        s = s[:pm.start()] + "\n\n**" + title + "** " + s[after:]
    # citations
    s = re.sub(r"\\citep\*?(?:\[[^\]]*\])*\{([^}]*)\}",
               lambda m: cite(m.group(1), False), s)
    s = re.sub(r"\\cite\{([^}]*)\}", lambda m: cite(m.group(1), False), s)
    s = re.sub(r"\\citet\*?(?:\[[^\]]*\])*\{([^}]*)\}",
               lambda m: cite(m.group(1), True), s)
    # cross references
    s = re.sub(r"\\eqref\{([^}]*)\}",
               lambda m: "(" + LABELS.get(m.group(1), "?") + ")", s)
    s = re.sub(r"\\ref\{([^}]*)\}",
               lambda m: LABELS.get(m.group(1), "?"), s)
    # \S is the section symbol, but the negative lookahead is load-bearing:
    # an unconditional replace turns \Sigma into "Section igma".
    s = re.sub(r"\\S(?=\s*[0-9A-Z])(?![a-zA-Z])", "Section ", s)
    s = re.sub(r"\\S(?![a-zA-Z])", "Section ", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:url|href)\{([^}]*)\}(?:\{([^}]*)\})?",
               lambda m: m.group(2) or m.group(1), s)
    s = s.replace(r"\repourl",
                  "https://github.com/K144U/"
                  "Initialisation-Sets-the-Conditioning-of-LoRA-Merging")
    s = s.replace(r"\repohost",
                  "github.com/K144U/"
                  "Initialisation-Sets-the-Conditioning-of-LoRA-Merging")
    # \texorpdfstring{TeX}{bookmark}: keep the TeX arm and drop the other,
    # or the heading reads "Rate-RR merging codes".
    while True:
        tm = re.search(r"\\texorpdfstring\s*\{", s)
        if not tm:
            break
        tex, after = take_group(s, tm.end() - 1)
        _, after = take_group(s, after)
        s = s[:tm.start()] + tex + s[after:]

    # maths, longest delimiters first. \[ ... \] is display, so it gets the
    # same indented block as an equation environment.
    s = re.sub(r"\\\[(.+?)\\\]",
               lambda m: "\n\n    " + math_to_text(m.group(1)) + "\n\n", s,
               flags=re.S)
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: math_to_text(m.group(1)), s,
               flags=re.S)
    s = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$",
               lambda m: math_to_text(m.group(1)), s, flags=re.S)
    s = re.sub(r"\\\((.+?)\\\)", lambda m: math_to_text(m.group(1)), s,
               flags=re.S)

    # verbatim and code
    s = re.sub(r"\\verb\|([^|]*)\|", lambda m: "`" + m.group(1) + "`", s)
    s = re.sub(r"\\verb\+([^+]*)\+", lambda m: "`" + m.group(1) + "`", s)

    # nested brace commands, innermost first
    def group_cmd(pattern, wrap):
        nonlocal s
        prev = None
        while prev != s:
            prev = s
            s = re.sub(pattern, wrap, s)

    group_cmd(r"\\texttt\{([^{}]*)\}", lambda m: "`" + m.group(1) + "`")
    group_cmd(r"\\textsc\{([^{}]*)\}", lambda m: m.group(1).upper())
    group_cmd(r"\\(?:textbf|bf)\{([^{}]*)\}",
              lambda m: "**" + m.group(1) + "**" if m.group(1).strip() else "")
    group_cmd(r"\\(?:emph|textit|it)\{([^{}]*)\}",
              lambda m: "*" + m.group(1) + "*" if m.group(1).strip() else "")
    group_cmd(r"\\(?:textrm|textnormal|text|mbox)\{([^{}]*)\}",
              lambda m: m.group(1))
    group_cmd(r"\\footnote\{([^{}]*)\}",
              lambda m: " [footnote: " + m.group(1) + "]")
    s = re.sub(r"\{\\bf\s+([^{}]*)\}", lambda m: "**" + m.group(1) + "**", s)
    s = re.sub(r"\{\\(?:it|em)\s+([^{}]*)\}",
               lambda m: "*" + m.group(1) + "*", s)

    # leftovers
    for cmd in DROP + ["hline", "toprule", "midrule", "bottomrule",
                       "addlinespace", "protect", "relax"]:
        s = re.sub(r"\\" + cmd + r"\b", "", s)
    s = re.sub(r"\\(?:setlength|tabcolsep|itemsep|topsep|leftmargin)"
               r"(?:\{[^{}]*\})*", "", s)

    # dashes and quotes, faithful to how the PDF sets them
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("~", " ")
    for a, b in ((r"\%", "%"), (r"\&", "&"), (r"\#", "#"), (r"\$", "$"),
                 (r"\_", "_")):
        s = s.replace(a, b)
    s = re.sub(r"\\[a-zA-Z]+\b", "", s)   # any command still standing
    s = s.replace("\\", "")
    s = re.sub(r"[{}]", "", s)            # braces those commands left behind
    s = s.replace("\x03", "{").replace("\x04", "}")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" ([,.;:)])", lambda m: m.group(1), s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    return s.strip()


# --------------------------------------------------------------------------
# 8. Block conversion
# --------------------------------------------------------------------------

def find_env(text: str, start: int, name: str):
    """Return (body, end_index) for the environment opening at `start`."""
    open_pat = "\\begin{" + name + "}"
    close_pat = "\\end{" + name + "}"
    depth, i = 0, start
    while i < len(text):
        if text.startswith(open_pat, i):
            depth += 1
            i += len(open_pat)
            continue
        if text.startswith(close_pat, i):
            depth -= 1
            if depth == 0:
                return text[start + len(open_pat):i], i + len(close_pat)
            i += len(close_pat)
            continue
        i += 1
    return text[start + len(open_pat):], len(text)


def caption_of(body: str) -> str:
    m = re.search(r"\\caption", body)
    if not m:
        return ""
    arg, _ = take_group(body, m.end())
    return inline_text(arg)


def render_list(body: str, ordered: bool) -> str:
    body = re.sub(r"^\s*\[[^\]]*\]", "", body.strip())
    items = re.split(r"\\item\b", body)[1:]
    out, k = [], 0
    for it in items:
        it = re.sub(r"^\s*\[[^\]]*\]\s*", "", it)
        txt = convert_body(it).strip()
        if not txt:
            continue
        k += 1
        bullet = (str(k) + ". ") if ordered else "- "
        first, *rest = txt.split("\n")
        out.append(bullet + first)
        for r in rest:
            out.append("  " + r if r.strip() else "")
    return "\n".join(out)


THM_NAMES = {"theorem": "Theorem", "lemma": "Lemma", "corollary": "Corollary",
             "proposition": "Proposition", "definition": "Definition",
             "claim": "Claim", "remark": "Remark",
             "assumption": "Assumption"}

# module-level counters, shared with the numbering pass
COUNT = {"table": 0, "figure": 0, "algorithm": 0, "equation": 0, "thm": 0}


def convert_body(text: str) -> str:
    """Convert a run of LaTeX body text to Markdown."""
    out = []
    i = 0
    while i < len(text):
        m = ENV_OPEN.search(text, i)
        if not m:
            out.append(inline_text_block(text[i:]))
            break
        out.append(inline_text_block(text[i:m.start()]))
        env = m.group(1)
        base = env.rstrip("*")
        body, end = find_env(text, m.start(), env)
        i = end

        if env in FLOATS:
            kind = FLOATS[env]
            COUNT[kind] += 1
            num = COUNT[kind]
            cap = caption_of(body)
            label = {"table": "Table", "figure": "Figure",
                     "algorithm": "Algorithm"}[kind]
            head = "**" + label + " " + str(num) + ".**"
            out.append("\n\n" + (head + " " + cap).strip() + "\n\n")
            if kind == "algorithm":
                am = re.search(r"\\begin\{algorithmic\}", body)
                if am:
                    ab, _ = find_env(body, am.start(), "algorithmic")
                    ab = re.sub(r"^\s*\[[^\]]*\]", "", ab)
                    out.append(render_algorithmic(ab) + "\n\n")
            elif kind == "table":
                out.append("*[Tabular data omitted from this text extract; "
                           "see Table " + str(num) + " in the typeset PDF.]*"
                           "\n\n")
            else:
                out.append("*[Figure omitted from this text extract; see "
                           "Figure " + str(num) + " in the typeset PDF.]*\n\n")
        elif env in NUMBERED_MATH:
            COUNT["equation"] += 1
            body = re.sub(r"\\label\{[^}]*\}", "", body)
            out.append("\n\n" + render_display(body, COUNT["equation"])
                       + "\n\n")
        elif base in ("equation", "align", "gather", "multline", "displaymath",
                      "eqnarray") or env.endswith("*"):
            body = re.sub(r"\\label\{[^}]*\}", "", body)
            out.append("\n\n" + render_display(body, None) + "\n\n")
        elif base in THEOREMS:
            COUNT["thm"] += 1
            name = THM_NAMES.get(base, base.capitalize())
            opt = ""
            b = body.lstrip()
            if b.startswith("["):
                j = b.index("]")
                opt = " (" + inline_text(b[1:j]) + ")"
                b = b[j + 1:]
            b = re.sub(r"^\s*\\label\{[^}]*\}", "", b)
            out.append("\n\n**" + name + " " + str(COUNT["thm"]) + opt + ".** "
                       + convert_body(b).strip() + "\n\n")
        elif base == "proof":
            out.append("\n\n*Proof.* " + convert_body(body).strip()
                       + " \u220e\n\n")
        elif base == "itemize":
            out.append("\n\n" + render_list(body, False) + "\n\n")
        elif base == "enumerate":
            out.append("\n\n" + render_list(body, True) + "\n\n")
        elif base in ("center", "minipage", "acks", "small", "quote",
                      "figurehere"):
            body = re.sub(r"^\s*\{[^{}]*\}", "", body)  # minipage width arg
            out.append("\n\n" + convert_body(body).strip() + "\n\n")
        elif base == "tabular":
            pass  # only reached outside a float; the data is not prose
        else:
            out.append(convert_body(body))
    return "".join(out)


def render_display(body: str, num) -> str:
    lines = []
    for chunk in re.split(r"\\\\", body):
        t = math_to_text(chunk)
        if t:
            lines.append(t)
    if not lines:
        return ""
    if num is not None:
        lines[-1] = lines[-1] + "        (" + str(num) + ")"
    return "\n".join("    " + ln for ln in lines)


HEAD_RE = re.compile(r"\\(subsection|subsubsection)(\*?)\s*\{")


def inline_text_block(text: str) -> str:
    """Split a run of body text into headings and paragraphs, then convert.

    Headings are lifted out by a brace-aware scan rather than a regex, because
    the titles carry \\texorpdfstring and inline maths with braces of their own.
    Everything else goes through inline_text, which is the only path that
    strips LaTeX; anything that skips it shows up as a raw command in the
    output, and the run reports those.
    """
    if not text.strip():
        return "\n\n" if "\n\n" in text else " "

    blocks, i = [], 0
    while True:
        m = HEAD_RE.search(text, i)
        if not m:
            blocks.append(("text", text[i:]))
            break
        blocks.append(("text", text[i:m.start()]))
        title, j = take_group(text, m.end() - 1)
        keys = []
        while True:
            lm = re.match(r"\s*\\label\{([^}]*)\}", text[j:])
            if not lm:
                break
            keys.append(lm.group(1))
            j += lm.end()
        blocks.append(("head", m.group(1), m.group(2), title, keys))
        i = j

    done = []
    for b in blocks:
        if b[0] == "head":
            _, lvl, star, title, keys = b
            mark = "###" if lvl == "subsection" else "####"
            num = HEADING_NUMBERS.pop(0) if HEADING_NUMBERS else ""
            for k in keys:
                if k in LABELS:
                    num = LABELS[k]
                    break
            prefix = (num + ". ") if num and not star else ""
            # several titles wrap across source lines; a heading is one line
            flat = re.sub(r"\s+", " ", inline_text(title)).strip()
            done.append(mark + " " + prefix + flat)
            continue
        chunk = re.sub(r"\n{3,}", "\n\n", b[1])
        for p in re.split(r"\n[ \t]*\n", chunk):
            t = inline_text(p)
            if t:
                done.append(t)
    return "\n\n".join(done)


# --------------------------------------------------------------------------
# 9. Drive it
# --------------------------------------------------------------------------

def load(name: str) -> str:
    p = SECTIONS / name
    if not p.exists():
        raise SystemExit("missing section file: " + str(p))
    return strip_comments(p.read_text(encoding="utf-8", errors="ignore"))


def main() -> int:
    global BIB, LABELS, HEADING_NUMBERS
    BIB = parse_bib(PAPER / "references.bib")
    plan = build_plan()

    # ---- pass 1: numbers ----
    N = Numbering()
    for kind, payload in plan:
        if kind == "section":
            N.section(payload[1])
        elif kind == "appendix":
            N.in_appendix = True
            N.section(payload[1])
        elif kind == "file":
            text = load(payload)
            if payload == "app_checklist.tex":
                sm = re.search(r"\\section\{([^}]*)\}", text)
                if sm:
                    N.in_appendix = True
                    N.section(re.findall(
                        r"\\label\{([^}]*)\}",
                        text[sm.end():sm.end() + 120]))
            scan_numbers(text, N)
    LABELS = N.labels
    HEADING_NUMBERS = list(N.heading_queue)

    # ---- pass 2: text ----
    md = []
    md.append("# " + TITLE)
    md.append("")
    md.append(AUTHORS)
    md.append("")
    md.append(AFFIL)
    md.append("")
    md.append("Corresponding author: Sankalp Pathak "
              "(pathaksankalp04@gmail.com), ORCID 0009-0006-5666-8271. "
              "Sanjay Garg, ORCID 0000-0002-2279-9373. "
              "Piyush Kumar Singh, ORCID 0009-0000-8033-3777.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("*Note on this file. This is a plain-text rendering of the "
              "manuscript, produced from the LaTeX sources for a text "
              "similarity check. It carries the complete prose of the paper: "
              "every section, every float caption, every list and the "
              "pseudo-code. The bodies of the data tables and the figure "
              "images are not reproduced here, because neither is prose and "
              "both extract as noise; each is marked in place and appears in "
              "the typeset PDF. Mathematics is rendered in plain characters. "
              "Generated by `JAIR/make_plaintext.py`.*")
    md.append("")
    md.append("---")

    sec_no, app_no = 0, 0
    in_appendix = False
    for kind, payload in plan:
        if kind == "section":
            sec_no += 1
            md.append("")
            md.append("## " + str(sec_no) + ". " + inline_text(payload[0]))
        elif kind == "appendix":
            in_appendix = True
            app_no += 1
            md.append("")
            md.append("## Appendix " + chr(ord("A") + app_no - 1) + ". "
                      + inline_text(payload[0]))
        elif kind == "unnumbered":
            md.append("")
            md.append("## " + inline_text(payload[0]))
        elif kind == "refs":
            md.append("")
            md.append("## References")
            md.append("")
            for key in sorted(BIB, key=lambda k: (
                    surnames(BIB[k].get("author", "zz"))[0][0].lower()
                    if BIB[k].get("author") else "zz",
                    BIB[k].get("year", ""))):
                md.append(format_reference(BIB[key]))
                md.append("")
        elif kind == "file":
            name = payload
            text = load(name)
            if name == "abstract_jair.tex":
                md.append("")
                md.append("## Abstract")
            if name == "acks.tex":
                # main.tex wraps this in acmart's acks environment, which
                # prints the heading; spelled the way acmart spells it
                md.append("")
                md.append("## Acknowledgments")
            if name == "app_checklist.tex":
                sm = re.search(
                    r"\\section\{((?:[^{}]|\{[^{}]*\})*)\}"
                    r"((?:\s*\\label\{[^}]*\})*)", text)
                if sm:
                    app_no += 1
                    md.append("")
                    md.append("## Appendix "
                              + chr(ord("A") + app_no - 1) + ". "
                              + inline_text(sm.group(1)))
                    text = text[sm.end():]
            md.append("")
            md.append(convert_body(text).strip())
    _ = in_appendix

    body = "\n".join(md)
    body = re.sub(r"\n{4,}", "\n\n\n", body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n\n+(###)", lambda m: "\n\n" + m.group(1), body)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "Pathak_Garg_Singh_Initialisation_LoRA_Merging_TEXT.md"
    dest.write_text(body + "\n", encoding="utf-8")

    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", body))
    print("wrote " + str(dest))
    print("  %.1f KB, %d lines, ~%d words"
          % (dest.stat().st_size / 1024, body.count("\n") + 1, words))
    leftovers = re.findall(r"\\[a-zA-Z]+", body)
    if leftovers:
        from collections import Counter
        print("  WARNING, unconverted commands: "
              + str(Counter(leftovers).most_common(12)))
    unresolved = body.count("Section ?") + body.count("Table ?")
    if unresolved:
        print("  WARNING, unresolved cross references: " + str(unresolved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
