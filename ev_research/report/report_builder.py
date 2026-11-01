"""Preliminary research report (Word .docx and Markdown).

The report contains ONLY:
* results of analyses that the researcher ran and explicitly added,
* charts the researcher added,
* literature-matrix content (as extracted / reviewed),
* text the researcher typed.

Sections the application cannot write (e.g. the methodology narrative) are
left as clearly marked placeholders instead of being filled with generic
text.  No references are generated other than a list of the uploaded papers
as recorded in the matrix.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ev_research import config
from ev_research.analysis.result import AnalysisResult

PLACEHOLDER = "[To be written by the researcher.]"
MAX_TABLE_ROWS = 60

CATEGORY_DESCRIPTIVE = "Descriptive findings"
CATEGORY_STATISTICAL = "Statistical results"
CATEGORY_PREPARATION = "Preparation for specialist software (not estimated by the application)"


@dataclass
class ReportInputs:
    title: str = "Factors Affecting Purchase Intention for Electric Vehicles in India - Preliminary Analysis"
    author: str = ""
    dataset_name: str = ""
    dataset_shape: tuple[int, int] | None = None
    dataset_inspection: AnalysisResult | None = None
    data_handling_notes: list[str] = field(default_factory=list)
    methodology_text: str = ""            # typed by researcher
    results: list[tuple[str, AnalysisResult]] = field(default_factory=list)   # (category, result)
    figures: list[tuple[str, bytes]] = field(default_factory=list)            # (caption, png)
    literature_records: list[dict] = field(default_factory=list)
    gap_report: object | None = None       # GapReport
    limitations_text: str = ""             # typed by researcher
    further_text: str = ""                 # typed by researcher
    sample_is_synthetic: bool = False


def auto_limitations(inp: ReportInputs) -> list[str]:
    """Limitations derived from what is actually in the inputs."""
    out = []
    if inp.sample_is_synthetic:
        out.append("The dataset used is SYNTHETIC test data. None of the results describe real consumers and they must not be reported as findings.")
    if inp.dataset_shape:
        out.append("The survey data are analysed as a single cross-sectional dataset; associations cannot establish causal effects or their direction.")
        out.append("Responses are self-reported; common method bias and social desirability bias have not been assessed by the application.")
    if inp.data_handling_notes:
        out.append("Missing-value handling affects the results: " + " ".join(inp.data_handling_notes))
    for cat, r in inp.results:
        for w in r.warnings:
            out.append(f"{r.title}: {w}")
    if any(not r.performed_by_app for _, r in inp.results):
        out.append("PLS-SEM was not estimated by this application. Only preparation outputs (specification, screening, HTMT, approximate indices) are included.")
    if inp.literature_records:
        unrev = sum(1 for r in inp.literature_records for d in r["fields"].values()
                    if d.get("status") not in (config.STATUS_VERIFIED, config.STATUS_EDITED))
        total = sum(len(r["fields"]) for r in inp.literature_records)
        out.append(f"The literature matrix was produced by rule-based extraction; {unrev} of {total} cells have not been verified by the researcher.")
        out.append(f"The literature summary covers only the {len(inp.literature_records)} uploaded paper(s), not a systematic search.")
    return out


def auto_suggestions(inp: ReportInputs) -> list[str]:
    """Suggestions that follow directly from the included outputs."""
    out = []
    titles = " ".join(r.title for _, r in inp.results)
    if inp.results and "PLS-SEM" not in titles and "Exploratory factor" not in titles:
        out.append("Assess construct validity (e.g. EFA, then CFA or the PLS-SEM measurement model) before testing the structural relationships.")
    if any(not r.performed_by_app for _, r in inp.results):
        out.append("Estimate the specified PLS-SEM model in SmartPLS or R (seminr/cSEM), and report outer loadings, reliability, AVE, HTMT with bootstrap CIs, path coefficients with bootstrap CIs, R-squared, f-squared and PLSpredict.")
    if any("Mediation" in r.title for _, r in inp.results):
        out.append("Mediation results are based on cross-sectional data; consider longitudinal or experimental designs to support causal claims.")
    if any("heteroscedasticity" in w.lower() for _, r in inp.results for w in r.warnings):
        out.append("Re-estimate regressions with heteroscedasticity-robust standard errors.")
    if inp.gap_report is not None and len(getattr(inp.gap_report, "inferred", [])):
        out.append("Verify each inferred literature gap against a systematic search (e.g. Scopus / Web of Science) before using it to motivate the study.")
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame) -> str:
    df = df.head(MAX_TABLE_ROWS).copy()
    df = df.fillna("")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [(f"{v:.3f}" if isinstance(v, float) else str(v)).replace("|", "/").replace("\n", " ") for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _md_result(r: AnalysisResult) -> list[str]:
    out = [f"### {r.title}", ""]
    if not r.performed_by_app:
        out += ["> **Preparation only - this model was not estimated by the application.**", ""]
    if r.method:
        out += [f"*Method:* {r.method}", ""]
    for name, t in r.tables.items():
        out += [f"**{name}**", "", _md_table(t), ""]
        if len(t) > MAX_TABLE_ROWS:
            out += [f"*(Table shortened to {MAX_TABLE_ROWS} rows.)*", ""]
    out += [f"- {t}" for t in r.text]
    out += [f"- **Caution:** {w}" for w in r.warnings]
    return out + [""]


def build_markdown(inp: ReportInputs) -> str:
    L = [f"# {inp.title}", ""]
    if inp.author:
        L.append(f"Prepared by: {inp.author}  ")
    L += [f"Date: {date.today().isoformat()}  ",
          "Status: **Preliminary, generated with the EV Purchase Intention Research Assistant. Review all content before use.**", ""]
    if inp.sample_is_synthetic:
        L += ["> **WARNING: The dataset is SYNTHETIC test data. Results are for software demonstration only.**", ""]

    L += ["## 1. Dataset description", ""]
    if inp.dataset_shape:
        L.append(f"- File: {inp.dataset_name}")
        L.append(f"- {inp.dataset_shape[0]} rows (responses) and {inp.dataset_shape[1]} columns (variables).")
        if inp.dataset_inspection:
            L += [f"- {t}" for t in inp.dataset_inspection.text]
            L += [f"- **Caution:** {w}" for w in inp.dataset_inspection.warnings]
        L += [f"- {n}" for n in inp.data_handling_notes]
    else:
        L.append("No survey dataset was included.")
    L.append("")

    L += ["## 2. Methodology", "", inp.methodology_text.strip() or PLACEHOLDER, "",
          "**Analyses performed by the application** (methods as executed):", ""]
    performed = [r for _, r in inp.results if r.performed_by_app]
    L += [f"- {r.title}: {r.method}" for r in performed] or ["- None added."]
    prep = [r for _, r in inp.results if not r.performed_by_app]
    if prep:
        L += ["", "**Not performed by the application (require specialist software):**", ""]
        L += [f"- {r.title}: PLS-SEM estimation must be carried out in SmartPLS, R (seminr/cSEM) or similar." for r in prep]
    L.append("")

    for num, cat in ((3, CATEGORY_DESCRIPTIVE), (4, CATEGORY_STATISTICAL), (5, CATEGORY_PREPARATION)):
        items = [r for c, r in inp.results if c == cat]
        if not items and cat == CATEGORY_PREPARATION:
            continue
        L += [f"## {num}. {cat}", ""]
        if not items:
            L += ["No results of this type were added to the report.", ""]
        for r in items:
            L += _md_result(r)

    L += ["## 6. Visualizations", ""]
    if inp.figures:
        for i, (cap, _) in enumerate(inp.figures, start=1):
            L += [f"![Figure {i}](figure_{i}.png)", f"*Figure {i}. {cap}*", ""]
        L += ["*(Figures are embedded in the Word version of this report.)*", ""]
    else:
        L += ["No charts were added.", ""]

    if inp.literature_records:
        L += _md_literature(inp)

    L += ["## 8. Limitations", "", "**Researcher's notes:**", "", inp.limitations_text.strip() or PLACEHOLDER, "",
          "**Limitations identified automatically from the analyses:**", ""]
    L += [f"- {x}" for x in auto_limitations(inp)] or ["- None."]
    L += ["", "## 9. Suggestions for further investigation", "", inp.further_text.strip() or PLACEHOLDER, "",
          "**Suggestions following from the included outputs (application-generated):**", ""]
    L += [f"- {x}" for x in auto_suggestions(inp)] or ["- None."]
    L.append("")
    if inp.literature_records:
        L += ["## Appendix: Papers reviewed (as recorded in the matrix)", "",
              "*These entries are as extracted/edited in the matrix and are not formatted citations. Verify every detail against the original paper.*", ""]
        for r in inp.literature_records:
            f = r["fields"]
            L.append(f"- [{r['paper_id']}] {f.get('Authors', {}).get('value', config.NOT_REPORTED)} "
                     f"({f.get('Publication year', {}).get('value', config.NOT_REPORTED)}). "
                     f"{f.get('Paper title', {}).get('value', config.NOT_REPORTED)}. File: {r['filename']}")
    return "\n".join(L) + "\n"


def _lit_overview(records) -> pd.DataFrame:
    rows = []
    for r in records:
        f = r["fields"]
        rows.append({"ID": r["paper_id"], **{k: f.get(k, {}).get("value", config.NOT_REPORTED)
                                             for k in ("Paper title", "Publication year", "Country of study", "Theories used", "Sample size")}})
    return pd.DataFrame(rows)


def _md_literature(inp: ReportInputs) -> list[str]:
    L = ["## 7. Literature review summary", "",
         f"{len(inp.literature_records)} paper(s) were uploaded and recorded in the literature matrix.", "",
         _md_table(_lit_overview(inp.literature_records)), ""]
    g = inp.gap_report
    if g is not None:
        L += ["### 7.1 Gaps stated by the authors", ""]
        L += [_md_table(g.explicit), ""] if len(g.explicit) else ["None recorded.", ""]
        L += ["### 7.2 Patterns across the reviewed papers", ""]
        L += [_md_table(g.patterns), ""] if len(g.patterns) else ["None.", ""]
        L += ["### 7.3 Potential gaps inferred by the application", "",
              "> These are rule-based inferences from the uploaded papers only. They are not established facts about the literature.", ""]
        L += [_md_table(g.inferred), ""] if len(g.inferred) else ["None produced.", ""]
        L += [f"- {n}" for n in g.notes] + [""]
    return L


# ---------------------------------------------------------------------------
# Word (.docx)
# ---------------------------------------------------------------------------
def _docx_table(doc, df: pd.DataFrame):
    df = df.head(MAX_TABLE_ROWS).fillna("")
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row.values):
            cells[i].text = f"{v:.3f}" if isinstance(v, float) else str(v)
    from docx.shared import Pt
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8)
    return t


def build_docx(inp: ReportInputs) -> bytes:
    from docx import Document
    from docx.shared import Inches, RGBColor

    doc = Document()
    doc.add_heading(inp.title, 0)
    if inp.author:
        doc.add_paragraph(f"Prepared by: {inp.author}")
    doc.add_paragraph(f"Date: {date.today().isoformat()}")
    p = doc.add_paragraph()
    run = p.add_run("Preliminary report generated with the EV Purchase Intention Research Assistant. Review all content before use.")
    run.bold = True
    if inp.sample_is_synthetic:
        r = doc.add_paragraph().add_run("WARNING: The dataset is SYNTHETIC test data. Results are for software demonstration only.")
        r.bold = True
        r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    def bullet(text):
        doc.add_paragraph(text, style="List Bullet")

    def add_result(r: AnalysisResult):
        doc.add_heading(r.title, 2)
        if not r.performed_by_app:
            doc.add_paragraph().add_run("Preparation only - this model was not estimated by the application.").bold = True
        if r.method:
            para = doc.add_paragraph()
            para.add_run("Method: ").italic = True
            para.add_run(r.method)
        for name, t in r.tables.items():
            doc.add_paragraph().add_run(name).bold = True
            _docx_table(doc, t)
            if len(t) > MAX_TABLE_ROWS:
                doc.add_paragraph(f"(Table shortened to {MAX_TABLE_ROWS} rows.)")
        for t in r.text:
            bullet(t)
        for w in r.warnings:
            bullet("Caution: " + w)

    doc.add_heading("1. Dataset description", 1)
    if inp.dataset_shape:
        bullet(f"File: {inp.dataset_name}")
        bullet(f"{inp.dataset_shape[0]} rows (responses) and {inp.dataset_shape[1]} columns (variables).")
        if inp.dataset_inspection:
            for t in inp.dataset_inspection.text:
                bullet(t)
            for w in inp.dataset_inspection.warnings:
                bullet("Caution: " + w)
        for n in inp.data_handling_notes:
            bullet(n)
    else:
        doc.add_paragraph("No survey dataset was included.")

    doc.add_heading("2. Methodology", 1)
    doc.add_paragraph(inp.methodology_text.strip() or PLACEHOLDER)
    doc.add_paragraph().add_run("Analyses performed by the application:").bold = True
    performed = [r for _, r in inp.results if r.performed_by_app]
    for r in performed:
        bullet(f"{r.title}: {r.method}")
    if not performed:
        bullet("None added.")
    prep = [r for _, r in inp.results if not r.performed_by_app]
    if prep:
        doc.add_paragraph().add_run("Not performed by the application (require specialist software):").bold = True
        for r in prep:
            bullet(f"{r.title}: PLS-SEM estimation must be carried out in SmartPLS, R (seminr/cSEM) or similar.")

    for num, cat in ((3, CATEGORY_DESCRIPTIVE), (4, CATEGORY_STATISTICAL), (5, CATEGORY_PREPARATION)):
        items = [r for c, r in inp.results if c == cat]
        if not items and cat == CATEGORY_PREPARATION:
            continue
        doc.add_heading(f"{num}. {cat}", 1)
        if not items:
            doc.add_paragraph("No results of this type were added to the report.")
        for r in items:
            add_result(r)

    doc.add_heading("6. Visualizations", 1)
    if not inp.figures:
        doc.add_paragraph("No charts were added.")
    for i, (cap, png) in enumerate(inp.figures, start=1):
        doc.add_picture(io.BytesIO(png), width=Inches(6))
        doc.add_paragraph().add_run(f"Figure {i}. {cap}").italic = True

    if inp.literature_records:
        doc.add_heading("7. Literature review summary", 1)
        doc.add_paragraph(f"{len(inp.literature_records)} paper(s) were uploaded and recorded in the literature matrix.")
        _docx_table(doc, _lit_overview(inp.literature_records))
        g = inp.gap_report
        if g is not None:
            doc.add_heading("7.1 Gaps stated by the authors", 2)
            _docx_table(doc, g.explicit) if len(g.explicit) else doc.add_paragraph("None recorded.")
            doc.add_heading("7.2 Patterns across the reviewed papers", 2)
            _docx_table(doc, g.patterns) if len(g.patterns) else doc.add_paragraph("None.")
            doc.add_heading("7.3 Potential gaps inferred by the application", 2)
            doc.add_paragraph().add_run("These are rule-based inferences from the uploaded papers only. They are not established facts about the literature.").bold = True
            _docx_table(doc, g.inferred) if len(g.inferred) else doc.add_paragraph("None produced.")
            for n in g.notes:
                bullet(n)

    doc.add_heading("8. Limitations", 1)
    doc.add_paragraph().add_run("Researcher's notes:").bold = True
    doc.add_paragraph(inp.limitations_text.strip() or PLACEHOLDER)
    doc.add_paragraph().add_run("Limitations identified automatically from the analyses:").bold = True
    for x in auto_limitations(inp) or ["None."]:
        bullet(x)

    doc.add_heading("9. Suggestions for further investigation", 1)
    doc.add_paragraph(inp.further_text.strip() or PLACEHOLDER)
    doc.add_paragraph().add_run("Suggestions following from the included outputs (application-generated):").bold = True
    for x in auto_suggestions(inp) or ["None."]:
        bullet(x)

    if inp.literature_records:
        doc.add_heading("Appendix: Papers reviewed (as recorded in the matrix)", 1)
        doc.add_paragraph("These entries are as extracted/edited in the matrix and are not formatted citations. Verify every detail against the original paper.")
        for r in inp.literature_records:
            f = r["fields"]
            bullet(f"[{r['paper_id']}] {f.get('Authors', {}).get('value', config.NOT_REPORTED)} "
                   f"({f.get('Publication year', {}).get('value', config.NOT_REPORTED)}). "
                   f"{f.get('Paper title', {}).get('value', config.NOT_REPORTED)}. File: {r['filename']}")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
