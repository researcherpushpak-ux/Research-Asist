"""Research-gap support built on the literature matrix.

Three clearly separated outputs:

1. **Explicit gaps** - gaps and future-research directions stated by the
   authors themselves (copied from the matrix, with page numbers).
2. **Cross-paper patterns** - counts of theories, designs, countries,
   variables, etc. across the reviewed papers, with the papers behind each
   count.
3. **Potential gaps (inferred)** - statements produced by transparent rules
   from the patterns.  These are the application's inferences, limited to the
   uploaded papers. Each lists the supporting papers, the rule used and the
   counter-evidence, and must be checked against the wider literature before
   being presented as a research gap.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd

from ev_research import config
from ev_research.literature.matrix import split_list

PATTERN_FIELDS = [
    "Country of study", "Theories used", "Research design", "Sampling method",
    "Statistical techniques", "Independent variables", "Dependent variables",
    "Mediating variables", "Moderating variables",
]

PROBABILITY_SAMPLING = {"Stratified random sampling", "Simple random sampling", "Systematic sampling",
                        "Cluster sampling", "Multistage sampling", "Random sampling (unspecified)"}

# Themes searched for in the authors' own gap / future-research statements.
THEMES = {
    "Longitudinal / over-time designs": r"longitudinal|over time|panel",
    "Rural / non-metro / tier-2 & tier-3 populations": r"rural|tier[- ]?(2|3|ii|iii|two|three)|semi[- ]urban|small(er)? cities",
    "Two-wheelers / three-wheelers / vehicle segments": r"two[- ]wheeler|three[- ]wheeler|2W|3W|four[- ]wheeler|segment",
    "Moderating variables / heterogeneity": r"moderat|multi[- ]group|heterogene",
    "Mediating mechanisms": r"mediat",
    "Government policy and incentives": r"polic(y|ies)|incentive|subsid",
    "Charging infrastructure": r"charging|infrastructure",
    "Actual purchase behaviour (not only intention)": r"actual (purchase|behaviou?r|adoption)|intention[- ]behaviou?r gap",
    "Probability / representative sampling": r"probability sampl|representative|random sampl",
    "Cross-country / cross-cultural comparison": r"cross[- ](country|cultural|national)|other countries|compar\w+ (across|between) countries",
    "Additional theories or constructs": r"other (theories|constructs|variables|factors)|additional (constructs|variables|factors)|extend",
    "Qualitative / mixed-methods insight": r"qualitative|interview|mixed[- ]method",
}


@dataclass
class GapReport:
    explicit: pd.DataFrame
    patterns: pd.DataFrame
    themes: pd.DataFrame
    inferred: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _value(rec: dict, f: str) -> str:
    return rec["fields"].get(f, {}).get("value", config.NOT_REPORTED)


def _verified(rec: dict, f: str) -> bool:
    return rec["fields"].get(f, {}).get("status") in (config.STATUS_VERIFIED, config.STATUS_EDITED)


def explicit_gaps(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        for f in ("Research gaps", "Future research directions"):
            d = r["fields"].get(f, {})
            v = d.get("value", config.NOT_REPORTED)
            if v and v != config.NOT_REPORTED:
                rows.append({"Paper": r["paper_id"], "Title": _value(r, "Paper title"), "Type": f,
                             "Statement (authors' words, as extracted)": v, "Page(s)": d.get("pages", ""),
                             "Review status": d.get("status", "")})
    return pd.DataFrame(rows, columns=["Paper", "Title", "Type", "Statement (authors' words, as extracted)", "Page(s)", "Review status"])


def patterns(records: list[dict]) -> pd.DataFrame:
    rows = []
    n = len(records)
    for f in PATTERN_FIELDS:
        reported = [r for r in records if split_list(_value(r, f))]
        support: dict[str, list[str]] = {}
        for r in reported:
            for item in set(split_list(_value(r, f))):
                support.setdefault(item, []).append(r["paper_id"])
        for item, papers in sorted(support.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            rows.append({"Field": f, "Item": item, "Papers (count)": len(papers),
                         "Share of papers reporting this field": f"{len(papers)}/{len(reported)}",
                         "Supporting papers": ", ".join(sorted(papers))})
        missing = n - len(reported)
        if missing:
            rows.append({"Field": f, "Item": config.NOT_REPORTED, "Papers (count)": missing,
                         "Share of papers reporting this field": f"{missing}/{n}",
                         "Supporting papers": ", ".join(r["paper_id"] for r in records if not split_list(_value(r, f)))})
    return pd.DataFrame(rows, columns=["Field", "Item", "Papers (count)", "Share of papers reporting this field", "Supporting papers"])


def theme_counts(records: list[dict]) -> pd.DataFrame:
    rows = []
    for theme, pat in THEMES.items():
        rx = re.compile(pat, re.I)
        hits = []
        for r in records:
            text = " ".join(_value(r, f) for f in ("Research gaps", "Future research directions") if _value(r, f) != config.NOT_REPORTED)
            if rx.search(text):
                hits.append(r["paper_id"])
        if hits:
            rows.append({"Theme in authors' gap / future-research statements": theme, "Papers (count)": len(hits),
                         "Supporting papers": ", ".join(hits)})
    df = pd.DataFrame(rows, columns=["Theme in authors' gap / future-research statements", "Papers (count)", "Supporting papers"])
    return df.sort_values("Papers (count)", ascending=False).reset_index(drop=True)


def _support_status(records, ids, f):
    ver = sum(1 for r in records if r["paper_id"] in ids and _verified(r, f))
    return f"{ver} of {len(ids)} supporting values verified by researcher"


def inferred_gaps(records: list[dict], min_papers: int = 3) -> pd.DataFrame:
    """Rule-based potential gaps. Returns an empty table if fewer than
    `min_papers` papers are available (too few for any pattern)."""
    cols = ["Potential gap (inference - not an established fact)", "Rule / basis", "Supporting papers",
            "Counter-evidence (papers)", "Papers without data for this field", "Evidence status"]
    n = len(records)
    if n < min_papers:
        return pd.DataFrame(columns=cols)
    rows = []

    def ids(pred, f):
        return [r["paper_id"] for r in records if split_list(_value(r, f)) and pred(set(split_list(_value(r, f))))]

    def no_data(f):
        return ", ".join(r["paper_id"] for r in records if not split_list(_value(r, f))) or "none"

    # 1. Research design
    f = "Research design"
    cross = ids(lambda s: "Cross-sectional survey" in s, f)
    longi = ids(lambda s: "Longitudinal" in s or "Experimental" in s, f)
    reported = ids(lambda s: True, f)
    if reported and len(cross) / len(reported) >= 0.6 and len(longi) <= max(1, len(reported) // 5):
        rows.append({cols[0]: f"Evidence may rely mainly on cross-sectional surveys; longitudinal or experimental designs appear rare among the reviewed papers ({len(longi)} of {len(reported)}).",
                     cols[1]: ">= 60% of papers reporting a design are cross-sectional surveys", cols[2]: ", ".join(cross),
                     cols[3]: ", ".join(longi) or "none", cols[4]: no_data(f), cols[5]: _support_status(records, cross, f)})

    # 2. Sampling
    f = "Sampling method"
    prob = ids(lambda s: bool(s & PROBABILITY_SAMPLING), f)
    nonprob = ids(lambda s: not (s & PROBABILITY_SAMPLING), f)
    if nonprob and len(prob) <= max(0, len(nonprob) // 4):
        rows.append({cols[0]: f"Probability-based samples appear uncommon ({len(prob)} of {len(prob) + len(nonprob)} papers reporting a sampling method), which may limit generalisability of findings.",
                     cols[1]: "Non-probability sampling reported in at least 80% of papers that state a method", cols[2]: ", ".join(nonprob),
                     cols[3]: ", ".join(prob) or "none", cols[4]: no_data(f), cols[5]: _support_status(records, nonprob, f)})

    # 3. Moderators / mediators
    for f, label in (("Moderating variables", "moderating (boundary) conditions"), ("Mediating variables", "mediating mechanisms")):
        with_v = ids(lambda s: True, f)
        without = [r["paper_id"] for r in records if not split_list(_value(r, f))]
        if len(with_v) <= n / 3:
            rows.append({cols[0]: f"Few reviewed papers ({len(with_v)} of {n}) report {label}.",
                         cols[1]: f"'{f}' reported in at most one third of papers", cols[2]: ", ".join(without),
                         cols[3]: ", ".join(with_v) or "none", cols[4]: "(the supporting papers are those without data)",
                         cols[5]: f"'Not reported' may mean the extractor missed it; {sum(_verified(r, f) for r in records if r['paper_id'] in without)} of {len(without)} verified"})

    # 4. Theory concentration
    f = "Theories used"
    t_support: dict[str, list[str]] = {}
    for r in records:
        for t in set(split_list(_value(r, f))):
            t_support.setdefault(t, []).append(r["paper_id"])
    t_reported = ids(lambda s: True, f)
    if t_reported:
        top, top_ids = max(t_support.items(), key=lambda kv: len(kv[1]))
        if len(top_ids) / len(t_reported) >= 0.5 and len(t_reported) >= 3:
            others = sorted(set(t_reported) - set(top_ids))
            rows.append({cols[0]: f"The reviewed papers draw heavily on {top} ({len(top_ids)} of {len(t_reported)} papers naming a theory); alternative or integrated theoretical lenses may be under-used.",
                         cols[1]: "One theory named in >= 50% of papers that name any theory", cols[2]: ", ".join(top_ids),
                         cols[3]: ", ".join(others) or "none", cols[4]: no_data(f), cols[5]: _support_status(records, top_ids, f)})

    # 5. Country concentration (relevant for an India-focused study)
    f = "Country of study"
    india = ids(lambda s: "India" in s, f)
    c_reported = ids(lambda s: True, f)
    if c_reported and len(india) <= len(c_reported) / 3:
        rows.append({cols[0]: f"Only {len(india)} of {len(c_reported)} reviewed papers reporting a country studied India; India-specific evidence may be limited in this set.",
                     cols[1]: "India reported in at most one third of papers that state a country",
                     cols[2]: ", ".join(sorted(set(c_reported) - set(india))), cols[3]: ", ".join(india) or "none",
                     cols[4]: no_data(f), cols[5]: _support_status(records, c_reported, f)})

    # 6. Constructs never examined as variables
    var_fields = ["Independent variables", "Dependent variables", "Mediating variables", "Moderating variables"]
    var_reported = [r["paper_id"] for r in records if any(split_list(_value(r, vf)) for vf in var_fields)]
    if len(var_reported) >= min_papers:
        used = {item.lower() for r in records for vf in var_fields for item in split_list(_value(r, vf))}
        candidates = [c for c in config.EV_CONSTRUCTS if "(demographic)" not in c and c != "Purchase intention"]
        absent = [c for c in candidates if c.lower() not in used]
        if absent:
            rows.append({cols[0]: "Constructs from the application's EV construct list that no reviewed paper lists as a variable: " + "; ".join(absent) + ". Some may be worth examining, if theoretically justified.",
                         cols[1]: "Construct absent from all variable fields (comparison with a fixed dictionary; variable names typed differently are not matched)",
                         cols[2]: ", ".join(var_reported), cols[3]: "n/a",
                         cols[4]: ", ".join(r["paper_id"] for r in records if r["paper_id"] not in var_reported) or "none",
                         cols[5]: "Absence in this set of papers is not evidence of absence in the literature"})

    # 7. Recurring calls by authors
    themes = theme_counts(records)
    for _, t in themes.iterrows():
        if t["Papers (count)"] >= 2:
            rows.append({cols[0]: f"Several authors call for research on: {t.iloc[0]} ({t['Papers (count)']} papers).",
                         cols[1]: "Theme keywords found in >= 2 papers' gap / future-research statements",
                         cols[2]: t["Supporting papers"], cols[3]: "n/a", cols[4]: no_data("Future research directions"),
                         cols[5]: "Based on the authors' own statements; check whether later studies have addressed them"})
    return pd.DataFrame(rows, columns=cols)


def build_report(records: list[dict]) -> GapReport:
    n = len(records)
    notes = [
        f"Based on {n} uploaded paper(s) only. Patterns and inferred gaps describe this set of papers, not the literature as a whole.",
        "Explicit gaps are the authors' statements. Inferred gaps are the application's rule-based inferences and must be verified.",
        "Values that have not been verified by the researcher may contain extraction errors.",
    ]
    if n < 3:
        notes.append("Fewer than 3 papers: no inferred gaps are produced because patterns would be meaningless.")
    if n < 15:
        notes.append("A small number of papers was reviewed; patterns may change substantially as more papers are added.")
    return GapReport(explicit_gaps(records), patterns(records), theme_counts(records), inferred_gaps(records), notes)


def export_excel(report: GapReport) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Notes": report.notes}).to_excel(xw, sheet_name="Read me", index=False)
        report.explicit.to_excel(xw, sheet_name="1 Explicit gaps (authors)", index=False)
        report.patterns.to_excel(xw, sheet_name="2 Cross-paper patterns", index=False)
        report.themes.to_excel(xw, sheet_name="2b Recurring themes", index=False)
        report.inferred.to_excel(xw, sheet_name="3 Inferred gaps (app)", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 45
    return buf.getvalue()
