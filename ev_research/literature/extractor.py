"""Rule-based extraction of literature-matrix fields from PDF text.

Design principles
-----------------
* Nothing is invented. If a rule finds nothing, the value is
  ``"Not reported in the paper."`` and the status says the extractor did not
  find it, because the information may still be in the paper in a form the
  rules miss.
* Every extracted value carries the page number(s) and a verbatim evidence
  snippet, so the researcher can check it against the PDF.
* The application's own interpretation (for example "this construct name
  appears, but its role as a variable is not established") is stored in a
  separate ``interpretation`` attribute, never mixed into the value.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ev_research import config
from ev_research.config import NOT_REPORTED
from ev_research.literature.pdf_reader import PDFDocument

MAX_VALUE_CHARS = 1500
MAX_EVIDENCE_CHARS = 350


@dataclass
class FieldResult:
    field: str
    value: str = NOT_REPORTED
    pages: str = ""              # e.g. "3" or "3, 7"
    evidence: str = ""           # verbatim snippet(s) from the PDF
    method: str = ""             # how the value was obtained
    status: str = config.STATUS_NOT_FOUND
    interpretation: str = ""     # the application's commentary, kept separate

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
@dataclass
class Sentence:
    text: str
    page: int


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _strip_references(pages: list[str]) -> list[str]:
    """Remove the reference list (everything after the last 'References'
    heading) so that cited works do not pollute the extraction."""
    pattern = re.compile(r"^\s*(references|bibliography|reference list)\s*$", re.I | re.M)
    for idx in range(len(pages) - 1, -1, -1):
        matches = list(pattern.finditer(pages[idx]))
        if matches:
            cut = matches[-1].start()
            # Only treat it as the reference list if it is in the second half
            if idx >= len(pages) // 2:
                return pages[:idx] + [pages[idx][:cut]]
            break
    return pages


_HEADING_RE = None


def _is_heading(line: str) -> bool:
    global _HEADING_RE
    if _HEADING_RE is None:
        pats = [p for v in config.SECTION_HEADINGS.values() for p in v]
        _HEADING_RE = re.compile(
            r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:" + "|".join(pats) + r")\s*[:.\-\u2013\u2014]?\s*$", re.I)
    return len(line) <= 90 and bool(_HEADING_RE.match(line.strip()))


def _sentences(pages: list[str]) -> list[Sentence]:
    """Split pages into sentences, remembering the page of each sentence.
    Heading lines are treated as sentence boundaries and dropped."""
    out: list[Sentence] = []
    for i, text in enumerate(pages, start=1):
        blocks, current = [], []
        for line in text.split("\n"):
            if _is_heading(line):
                blocks.append(" ".join(current))
                current = []
            else:
                current.append(line.strip())
        blocks.append(" ".join(current))
        for block in blocks:
            block = re.sub(r"(?<=[a-z])- (?=[a-z])", "", block)  # de-hyphenate line breaks
            for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])", block):
                sent = sent.strip()
                if len(sent) > 15:
                    out.append(Sentence(sent, i))
    return out


FUTURE_RE = re.compile(r"future (research|studies|work|scholars|researchers)|further (research|studies)|should (be )?(use|examine|explore|include|consider)|could (be )?(compare|explore|examine)", re.I)


def _current_study(sentences: list[Sentence]) -> list[Sentence]:
    """Drop sentences about future research so they are not mistaken for a
    description of the present study's design, sample or variables."""
    return [s for s in sentences if not FUTURE_RE.search(s.text)]


@dataclass
class Section:
    name: str
    page: int
    text: str


def _find_sections(pages: list[str]) -> dict[str, Section]:
    """Locate sections by heading lines.  A heading is a short line that
    matches one of the configured heading patterns."""
    lines: list[tuple[int, str]] = []
    for i, text in enumerate(pages, start=1):
        for line in text.split("\n"):
            lines.append((i, line.strip()))

    compiled = {
        name: [re.compile(rf"^(?:\d+(?:\.\d+)*\.?\s*)?{p}\s*[:.\-–—]?\s*$", re.I) for p in pats]
        for name, pats in config.SECTION_HEADINGS.items()
    }
    # Some headings are inline ("Abstract: text...", "Keywords: a; b")
    inline = {
        name: re.compile(rf"^(?:\d+(?:\.\d+)*\.?\s*)?(?:{'|'.join(pats)})\s*[:.–—-]\s*(.+)$", re.I)
        for name, pats in config.SECTION_HEADINGS.items()
        if name in {"abstract", "keywords"}
    }

    heads: list[tuple[int, str, int, str]] = []  # (line_idx, name, page, inline_text)
    for idx, (page, line) in enumerate(lines):
        if not line or len(line) > 90:
            m_found = False
        else:
            m_found = False
            for name, pats in compiled.items():
                if any(p.match(line) for p in pats):
                    heads.append((idx, name, page, ""))
                    m_found = True
                    break
        if not m_found and line:
            for name, pat in inline.items():
                m = pat.match(line)
                if m:
                    heads.append((idx, name, page, m.group(1)))
                    break

    sections: dict[str, Section] = {}
    for n, (idx, name, page, inline_text) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        body_lines = [inline_text] if inline_text else []
        body_lines += [l for _, l in lines[idx + 1:end]]
        if name == "keywords":
            body_lines = body_lines[:2]   # keywords never run on for long
        body = " ".join(l for l in body_lines if l)
        body = re.sub(r"-\s(?=[a-z])", "", body)
        if name not in sections and body.strip():
            sections[name] = Section(name, page, body.strip())
    return sections


def _dictionary_matches(sentences: list[Sentence], dictionary: dict[str, list[str]]):
    """Return {label: (count, first_sentence)} for dictionary entries found."""
    found: dict[str, list] = {}
    compiled = {k: [re.compile(p, re.I) for p in v] for k, v in dictionary.items()}
    for s in sentences:
        for label, pats in compiled.items():
            if any(p.search(s.text) for p in pats):
                if label not in found:
                    found[label] = [0, s]
                found[label][0] += 1
    return found


def _sentences_matching(sentences: list[Sentence], patterns: list[str], limit: int = 4) -> list[Sentence]:
    comp = [re.compile(p, re.I) for p in patterns]
    hits = [s for s in sentences if any(c.search(s.text) for c in comp)]
    # de-duplicate
    seen, out = set(), []
    for s in hits:
        key = s.text[:80]
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:limit]


def _pages_str(pages) -> str:
    uniq = sorted({int(p) for p in pages if p})
    return ", ".join(str(p) for p in uniq)


def _from_sentences(field: str, hits: list[Sentence], method: str, interpretation: str = "") -> FieldResult:
    if not hits:
        return FieldResult(field, method=method, interpretation=interpretation)
    value = " ".join(h.text for h in hits)
    evidence = " | ".join(f"p.{h.page}: \"{_truncate(h.text, MAX_EVIDENCE_CHARS)}\"" for h in hits)
    return FieldResult(
        field,
        value=_truncate(value, MAX_VALUE_CHARS),
        pages=_pages_str(h.page for h in hits),
        evidence=evidence,
        method=method,
        status=config.STATUS_EXTRACTED,
        interpretation=interpretation,
    )


def _from_section(field: str, section: Section | None, method: str,
                  max_chars: int = MAX_VALUE_CHARS, interpretation: str = "") -> FieldResult | None:
    if section is None:
        return None
    text = _truncate(section.text, max_chars)
    note = "Text taken from the section that begins on the cited page; it may continue onto later pages."
    if len(section.text) > max_chars:
        note += " The section was shortened; read the full section in the PDF."
    return FieldResult(
        field,
        value=text,
        pages=str(section.page),
        evidence=f"p.{section.page}: \"{_truncate(section.text, MAX_EVIDENCE_CHARS)}\"",
        method=method,
        status=config.STATUS_EXTRACTED,
        interpretation=(interpretation + " " + note).strip(),
    )


def _from_dictionary(field: str, matches: dict, method: str, interpretation: str) -> FieldResult:
    if not matches:
        return FieldResult(field, method=method, interpretation=interpretation)
    ordered = sorted(matches.items(), key=lambda kv: -kv[1][0])
    value = "; ".join(label for label, _ in ordered)
    evidence = " | ".join(
        f"{label} - p.{s.page}: \"{_truncate(s.text, 200)}\" ({n} mention{'s' if n != 1 else ''})"
        for label, (n, s) in ordered
    )
    return FieldResult(
        field,
        value=value,
        pages=_pages_str(s.page for _, (_, s) in ordered),
        evidence=evidence,
        method=method,
        status=config.STATUS_EXTRACTED,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Individual field extractors
# ---------------------------------------------------------------------------
def _title(doc: PDFDocument) -> FieldResult:
    meta_title = (doc.metadata.get("Title") or "").strip()
    junk = re.compile(r"^(untitled|microsoft word|document\d*|.*\.(docx?|pdf|tex))", re.I)
    if doc.title_candidate and len(doc.title_candidate) > 10:
        return FieldResult(
            "Paper title", value=doc.title_candidate, pages="1",
            evidence=f"p.1 (largest font on first page): \"{_truncate(doc.title_candidate, 300)}\"",
            method="Largest-font text on page 1", status=config.STATUS_EXTRACTED,
            interpretation="Heuristic: journal names or running headers are sometimes set in a large font. Check this is the article title.",
        )
    if meta_title and not junk.match(meta_title) and len(meta_title) > 10:
        return FieldResult(
            "Paper title", value=meta_title, pages="",
            evidence=f"PDF metadata 'Title' field: \"{meta_title}\"",
            method="PDF metadata", status=config.STATUS_EXTRACTED,
            interpretation="Taken from file metadata, not from the page text. Metadata is sometimes wrong.",
        )
    return FieldResult("Paper title", method="Largest font on page 1 / PDF metadata")


def _authors(doc: PDFDocument) -> FieldResult:
    meta = (doc.metadata.get("Author") or "").strip()
    if meta and len(meta) > 2 and not re.search(r"(admin|user|owner|microsoft)", meta, re.I):
        return FieldResult(
            "Authors", value=meta, evidence=f"PDF metadata 'Author' field: \"{meta}\"",
            method="PDF metadata", status=config.STATUS_EXTRACTED,
            interpretation="Taken from file metadata. Check it against the author line on page 1.",
        )
    name_like = re.compile(r"^[A-Z][A-Za-z.\-' ]+(,|\band\b|&)\s*[A-Z][A-Za-z.\-' ]+")
    for line in doc.lines_after_title:
        cleaned = re.sub(r"[\d*†‡,]+$", "", line).strip()
        if name_like.match(cleaned) and not re.search(r"(journal|university|department|abstract|vol\.)", cleaned, re.I):
            return FieldResult(
                "Authors", value=cleaned, pages="1",
                evidence=f"p.1 (line below title): \"{_truncate(line, 250)}\"",
                method="Line following the title on page 1", status=config.STATUS_EXTRACTED,
                interpretation="Heuristic: the line under the title usually lists authors. Check it.",
            )
    return FieldResult("Authors", method="PDF metadata / line below title")


def _year(doc: PDFDocument) -> FieldResult:
    first = doc.pages[0] if doc.pages else ""
    patterns = [
        (r"(?:©|\(c\)|copyright)\s*((?:19|20)\d{2})", "Copyright notice"),
        (r"published(?: online)?[:\s]+[^\n]{0,30}?((?:19|20)\d{2})", "'Published' date"),
        (r"accepted[:\s]+[^\n]{0,30}?((?:19|20)\d{2})", "'Accepted' date (publication may be later)"),
        (r"(?:vol(?:ume)?\.?\s*\d+[^\n]{0,30}?)\(((?:19|20)\d{2})\)", "Volume/issue line"),
        (r"\b((?:19|20)\d{2})\s*,\s*vol", "Year before volume number"),
    ]
    for pat, label in patterns:
        m = re.search(pat, first, re.I)
        if m:
            start = max(0, m.start() - 60)
            return FieldResult(
                "Publication year", value=m.group(1), pages="1",
                evidence=f"p.1: \"{_truncate(first[start:m.end() + 40], 200)}\"",
                method=f"Pattern on page 1: {label}", status=config.STATUS_EXTRACTED,
                interpretation="" if "Accepted" not in label else "The acceptance year may differ from the publication year.",
            )
    return FieldResult(
        "Publication year", method="Copyright / published / volume patterns on page 1",
        interpretation="No clearly labelled year was found on page 1. Years elsewhere in the text are usually citations, so none was guessed.",
    )


def _country(sentences: list[Sentence]) -> FieldResult:
    names = sorted(config.COUNTRIES, key=len, reverse=True)
    ctx = re.compile(
        r"(?:respondents?|consumers?|participants?|sample|survey|data|study|studies|conducted|collected|residents?|households?|buyers?|users?)"
        r"[^.]{0,80}?\b(?:in|from|across|of)\s+(?:the\s+)?(?:[\w-]+,?\s+){0,3}?(" + "|".join(re.escape(n) for n in names) + r")\b",
        re.I,
    )
    loc = re.compile(r"\b(" + "|".join(re.escape(n) for n in config.INDIA_LOCATIONS) + r")\b")
    hits: dict[str, list[Sentence]] = {}
    for s in sentences:
        for m in ctx.finditer(s.text):
            name = m.group(1)
            canon = {"USA": "United States", "UK": "United Kingdom", "Korea": "South Korea"}.get(name, name)
            hits.setdefault(canon, []).append(s)
    india_loc = [s for s in sentences if loc.search(s.text)]
    if not hits:
        interp = ""
        if india_loc:
            interp = (f"No explicit country statement found, but Indian place names appear (e.g. p.{india_loc[0].page}). "
                      "This suggests, but does not establish, an India-based study.")
        return FieldResult("Country of study", method="Phrases like 'respondents in <country>'", interpretation=interp)
    ordered = sorted(hits.items(), key=lambda kv: -len(kv[1]))
    evidence = " | ".join(f"{c} - p.{v[0].page}: \"{_truncate(v[0].text, 200)}\"" for c, v in ordered)
    interp = "Country names found next to words such as 'respondents', 'data' or 'conducted in'."
    if len(ordered) > 1:
        interp += " More than one country was found; some may be comparisons or citations of other studies."
    return FieldResult(
        "Country of study", value="; ".join(c for c, _ in ordered),
        pages=_pages_str(v[0].page for _, v in ordered), evidence=evidence,
        method="Context pattern: '<respondents/data/...> in <country>'",
        status=config.STATUS_EXTRACTED, interpretation=interp,
    )


SAMPLE_PATTERNS = [
    r"\b[nN]\s*=\s*(\d{2,6}(?:,\d{3})?)\b",
    r"sample (?:size )?(?:of|was|comprised|consisted of|included)\s+(\d{2,6}(?:,\d{3})?)\b",
    r"\b(\d{2,6}(?:,\d{3})?)\s+(?:usable |valid |complete |completed )?(?:respondents|participants|responses|questionnaires|consumers|individuals|car buyers|households)\b",
]


def _sample_size(sentences: list[Sentence]) -> FieldResult:
    cands: dict[str, list[Sentence]] = {}
    for s in sentences:
        for p in SAMPLE_PATTERNS:
            for m in re.finditer(p, s.text, re.I):
                num = m.group(1).replace(",", "")
                if num.isdigit() and 10 <= int(num) <= 1_000_000 and not (1900 <= int(num) <= 2100):
                    cands.setdefault(num, []).append(s)
    if not cands:
        return FieldResult("Sample size", method="Patterns: 'n = 350', 'sample of 350', '350 respondents'")
    ordered = sorted(cands.items(), key=lambda kv: (-len(kv[1]), kv[1][0].page))
    value = "; ".join(n for n, _ in ordered)
    evidence = " | ".join(f"{n} - p.{v[0].page}: \"{_truncate(v[0].text, 200)}\"" for n, v in ordered[:5])
    interp = ""
    if len(ordered) > 1:
        interp = ("Several candidate numbers were found (e.g. questionnaires distributed versus usable responses, "
                  "or subgroup sizes). Listed most frequent first. Confirm the final analysed sample.")
    return FieldResult(
        "Sample size", value=value, pages=_pages_str(v[0].page for _, v in ordered[:5]),
        evidence=evidence, method="Patterns: 'n = 350', 'sample of 350', '350 respondents'",
        status=config.STATUS_EXTRACTED, interpretation=interp,
    )


def _construct_spans(text: str) -> list[tuple[int, int, str]]:
    """Positions of EV construct names in a sentence (longest match wins)."""
    spans = []
    for label, pats in config.EV_CONSTRUCTS.items():
        for p in pats:
            for m in re.finditer(p, text, re.I):
                spans.append((m.start(), m.end(), label))
    spans.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    out, last_end = [], -1
    for st, en, lab in spans:
        if st >= last_end:
            out.append((st, en, lab))
            last_end = en
    return out


def _nearest_before(spans, pos, window):
    cands = [sp for sp in spans if sp[1] <= pos and pos - sp[1] <= window]
    return cands[-1] if cands else None


def _nearest_after(spans, pos, window):
    cands = [sp for sp in spans if sp[0] >= pos and sp[0] - pos <= window]
    return cands[0] if cands else None


EXPLICIT_ROLES = [
    ("Independent variables", re.compile(r"\bindependent variables?|\bexogenous (variables?|constructs?)", re.I)),
    ("Dependent variables", re.compile(r"(?<!in)dependent variables?|\bendogenous (variables?|constructs?)|\boutcome variables?", re.I)),
]
# 'X mediates' -> X before; 'mediating role of X' / 'mediated by X' -> X after
MEDIATOR_BEFORE = re.compile(r"\b(partially |fully |significantly )?mediates?\b|\bacts? as an? mediat", re.I)
MEDIATOR_AFTER = re.compile(r"mediat(ing|ion)? (role|effect) of|mediated (by|through)|\bmediators? such as", re.I)
MODERATOR_BEFORE = re.compile(r"\b(significantly )?moderates?\b|\bacts? as an? moderat", re.I)
MODERATOR_AFTER = re.compile(r"moderat(ing|ion)? (role|effect) of|moderated by|\bmoderators? such as", re.I)
EFFECT_VERB = re.compile(
    r"(effects?|influences?|impacts?) on|\b(positively|negatively|significantly) (influences?|affects?|predicts?|impacts?)\b"
    r"|\b(influences?|affects?|predicts?|impacts?|determines?)\b(?! of)", re.I)

GAP_RE = re.compile(r"little is known|few studies|remains? (unexplored|under[- ]?explored|unclear)|research gap|dearth of|scant", re.I)

VARIABLE_FIELDS = ["Independent variables", "Dependent variables", "Mediating variables", "Moderating variables"]


def _variables(sentences: list[Sentence]) -> dict[str, FieldResult]:
    """Assign construct names to variable roles using position rules:

    1. Explicit labels: a construct is assigned to the nearest role phrase
       ('independent variable', 'dependent variable') in the same sentence.
    2. Mediation/moderation verbs: 'X mediates ...', 'the mediating role of X'.
    3. Effect statements: 'X has a significant effect on Y' / 'X influences Y'
       -> X is an IV candidate, Y a DV candidate.
    """
    found: dict[str, dict[str, tuple[Sentence, str]]] = {f: {} for f in VARIABLE_FIELDS}

    def add(field, label, sent, rule):
        found[field].setdefault(label, (sent, rule))

    for s in _current_study(sentences):
        t = s.text
        if GAP_RE.search(t):
            continue   # gap statements describe what is NOT studied
        spans = _construct_spans(t)
        if not spans:
            continue
        roles = [(f, m.start(), m.end()) for f, rx in EXPLICIT_ROLES for m in rx.finditer(t)]
        if roles:
            for st, en, lab in spans:
                if any(rs <= st < re_ for _, rs, re_ in roles):
                    continue
                copula = [r for r in roles if r[1] >= en and re.fullmatch(
                    r"\s*(is|are|was|were|served as|acts? as)\s+(the|an?|our)?\s*(main |key |sole )?", t[en:r[1]], re.I)]
                f = copula[0][0] if copula else min(roles, key=lambda r: min(abs(st - r[2]), abs(r[1] - en)))[0]
                add(f, lab, s, "explicit role label")
        for rx_b, rx_a, f in ((MEDIATOR_BEFORE, MEDIATOR_AFTER, "Mediating variables"),
                              (MODERATOR_BEFORE, MODERATOR_AFTER, "Moderating variables")):
            for m in rx_b.finditer(t):
                sp = _nearest_before(spans, m.start(), 25)
                if sp:
                    add(f, sp[2], s, "'X mediates/moderates' pattern")
            for m in rx_a.finditer(t):
                sp = _nearest_after(spans, m.end(), 25)
                if sp:
                    add(f, sp[2], s, "'mediating/moderating role of X' pattern")
        if not re.search(r"mediat|moderat", t, re.I):
            for m in EFFECT_VERB.finditer(t):
                iv = _nearest_before(spans, m.start(), 60)
                dv = _nearest_after(spans, m.end(), 40)
                if iv and dv and iv[2] != dv[2]:
                    add("Independent variables", iv[2], s, "effect statement 'X ... effect on Y'")
                    add("Dependent variables", dv[2], s, "effect statement 'X ... effect on Y'")

    all_mentions = sorted(_dictionary_matches(sentences, config.EV_CONSTRUCTS))
    results: dict[str, FieldResult] = {}
    method = ("Position rules: explicit role labels, 'X mediates/moderates', "
              "'X has an effect on Y' (EV construct dictionary)")
    for field in VARIABLE_FIELDS:
        hits = found[field]
        if hits:
            results[field] = FieldResult(
                field, value="; ".join(hits),
                pages=_pages_str(sent.page for sent, _ in hits.values()),
                evidence=" | ".join(f"{k} [{rule}] - p.{sent.page}: \"{_truncate(sent.text, 200)}\""
                                    for k, (sent, rule) in hits.items()),
                method=method, status=config.STATUS_EXTRACTED,
                interpretation=("Candidates from sentence patterns. A sentence may describe another study or a "
                                "hypothesis that was not supported. Check against the paper's model or hypotheses. "
                                "Constructs not in the dictionary are not detected."),
            )
        else:
            interp = "No candidate found by the position rules."
            if all_mentions and field == "Independent variables":
                interp += (" Constructs mentioned anywhere in the paper (role not determined): "
                           + "; ".join(all_mentions) + ".")
            results[field] = FieldResult(field, method=method, interpretation=interp)
    return results


def extract_fields(doc: PDFDocument) -> dict[str, FieldResult]:
    """Run all extractors and return a FieldResult for every matrix field."""
    pages = _strip_references(doc.pages)
    sentences = _sentences(pages)
    sections = _find_sections(pages)
    present = _current_study(sentences)
    res: dict[str, FieldResult] = {}

    res["Paper title"] = _title(doc)
    res["Authors"] = _authors(doc)
    res["Publication year"] = _year(doc)
    res["Country of study"] = _country(present)

    res["Research objectives"] = _from_section("Research objectives", sections.get("objectives"), "Section headed 'Objectives'") or \
        _from_sentences("Research objectives", _sentences_matching(sentences, [
            r"(objective|aim|purpose)s? of (this|the present|our) (study|paper|research)",
            r"this (study|paper|research) (aims|seeks|attempts|intends|investigates|examines)",
            r"we (aim|seek) to",
        ], 3), "Sentences stating the aim/purpose/objective of 'this study'")

    res["Research questions"] = _from_section("Research questions", sections.get("questions"), "Section headed 'Research questions'") or \
        _from_sentences("Research questions", _sentences_matching(sentences, [
            r"\bRQ\s?\d", r"research questions? (is|are|was|were|addressed)", r"following (research )?questions",
        ], 4), "Sentences containing 'RQ1' or 'research question(s)'")

    res["Research gaps"] = _from_section("Research gaps", sections.get("gap"), "Section headed 'Research gap'") or \
        _from_sentences("Research gaps", _sentences_matching(sentences, [
            r"research gap", r"gap in (the )?(existing )?literature", r"little is known", r"(limited|scant|scarce|few) (research|studies|attention)",
            r"dearth of", r"has (not|rarely) been (studied|examined|explored|investigated)", r"remains? (unexplored|under[- ]?explored|unclear)",
        ], 4), "Sentences with gap phrases ('research gap', 'little is known', 'few studies')",
            interpretation="These sentences state gaps in the authors' own words.")

    res["Theories used"] = _from_dictionary(
        "Theories used", _dictionary_matches(sentences, config.THEORIES),
        "Dictionary of named theories (see ev_research/config.py)",
        "A theory being NAMED does not prove the paper USES it as its framework; it may only be cited. Check the theoretical framework section.",
    )
    res["Sample size"] = _sample_size(present)
    res["Sampling method"] = _from_dictionary(
        "Sampling method", _dictionary_matches(present, config.SAMPLING_METHODS),
        "Dictionary of sampling-method names", "Term found in text; confirm it describes this study's sampling.",
    )
    res["Research design"] = _from_dictionary(
        "Research design", _dictionary_matches(present, config.RESEARCH_DESIGNS),
        "Dictionary of research-design terms", "Term found in text; confirm it describes this study's design.",
    )
    res.update(_variables(sentences))
    res["Statistical techniques"] = _from_dictionary(
        "Statistical techniques", _dictionary_matches(present, config.STAT_TECHNIQUES),
        "Dictionary of statistical technique names",
        "Some mentions may refer to other studies discussed in the literature review.",
    )

    findings_sec = sections.get("results") or sections.get("discussion")
    finding_pats = [r"\b(results?|findings?) (show|indicate|reveal|suggest|confirm|demonstrate)", r"\bwas found\b", r"\bwere found\b",
                    r"(significant(ly)?|positive(ly)?|negative(ly)?) (influence|effect|impact|relationship|associat)"]
    hits = _sentences_matching(sentences, finding_pats, 5)
    res["Key findings"] = _from_sentences(
        "Key findings", hits, "Sentences reporting results ('results show', 'significant effect', ...)",
        interpretation="Selected sentences only; they are not a complete summary of the findings.",
    ) if hits else (_from_section("Key findings", findings_sec, "Start of the Results/Discussion section", 800)
                    or FieldResult("Key findings", method="Result-reporting sentences / Results section"))

    res["Conclusions"] = _from_section("Conclusions", sections.get("conclusion"), "Section headed 'Conclusion'", 1200) or \
        _from_sentences("Conclusions", _sentences_matching(sentences, [r"(we|this study) conclude", r"in conclusion", r"to conclude"], 3),
                        "Sentences with 'in conclusion' / 'we conclude'")

    fut = sections.get("future") or sections.get("limitations")
    fut_hits = _sentences_matching(sentences, [r"future (research|studies|work|scholars|researchers)", r"further (research|studies|investigation)"], 5)
    res["Future research directions"] = _from_sentences(
        "Future research directions", fut_hits, "Sentences mentioning 'future research' / 'further studies'",
        interpretation="These directions are stated by the authors.",
    ) if fut_hits else (_from_section("Future research directions", fut, "Limitations / Future research section", 1000)
                        or FieldResult("Future research directions", method="'Future research' sentences / section"))

    kw = sections.get("keywords")
    if kw:
        value = _truncate(re.split(r"\b(?:1\.?\s*)?introduction\b|\babstract\b", kw.text, flags=re.I)[0], 400).strip(" .;")
        res["Keywords"] = FieldResult(
            "Keywords", value=value, pages=str(kw.page), evidence=f"p.{kw.page}: \"{_truncate(kw.text, 250)}\"",
            method="Line starting with 'Keywords'", status=config.STATUS_EXTRACTED,
        )
    else:
        res["Keywords"] = FieldResult("Keywords", method="Line starting with 'Keywords'")

    # Guarantee every field exists, in order.
    return {f: res.get(f, FieldResult(f)) for f in config.MATRIX_FIELDS}


def empty_fields() -> dict[str, FieldResult]:
    """All fields set to 'Not reported' (for manual entry)."""
    return {f: FieldResult(f, method="Manual entry") for f in config.MATRIX_FIELDS}
