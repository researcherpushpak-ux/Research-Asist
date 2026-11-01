"""Literature matrix: storage, review/correction and Excel export/import.

Internal format: a list of "paper records"::

    {"paper_id": "P01", "filename": "smith2021.pdf",
     "fields": {"Paper title": {FieldResult as dict}, ...}}

Two tabular views are derived from it:

* long table: one row per (paper, field) with value, pages, evidence,
  method, status and interpretation.  This is what the researcher edits.
* wide table: one row per paper, one column per field (values only).  This
  is the classic literature matrix.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from ev_research import config
from ev_research.config import NOT_REPORTED
from ev_research.literature.extractor import FieldResult

LONG_COLUMNS = ["paper_id", "filename", "field", "value", "pages", "evidence",
                "method", "status", "interpretation"]


def new_record(paper_id: str, filename: str, fields: dict[str, FieldResult]) -> dict:
    return {
        "paper_id": paper_id,
        "filename": filename,
        "fields": {k: (v.to_dict() if isinstance(v, FieldResult) else dict(v)) for k, v in fields.items()},
    }


def next_paper_id(records: list[dict]) -> str:
    used = {r["paper_id"] for r in records}
    i = 1
    while f"P{i:02d}" in used:
        i += 1
    return f"P{i:02d}"


def to_long(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        for f in config.MATRIX_FIELDS:
            d = r["fields"].get(f, FieldResult(f).to_dict())
            rows.append({
                "paper_id": r["paper_id"], "filename": r["filename"], "field": f,
                "value": d.get("value", NOT_REPORTED), "pages": d.get("pages", ""),
                "evidence": d.get("evidence", ""), "method": d.get("method", ""),
                "status": d.get("status", config.STATUS_NOT_FOUND),
                "interpretation": d.get("interpretation", ""),
            })
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


def to_wide(records: list[dict], include_pages: bool = False) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"Paper ID": r["paper_id"], "File": r["filename"]}
        for f in config.MATRIX_FIELDS:
            d = r["fields"].get(f, {})
            row[f] = d.get("value", NOT_REPORTED)
            if include_pages:
                row[f"{f} (page)"] = d.get("pages", "")
        rows.append(row)
    return pd.DataFrame(rows)


def apply_edits(records: list[dict], edited_long: pd.DataFrame) -> list[dict]:
    """Write researcher edits from the long table back into the records.

    If the researcher changes a value but leaves the status as an automatic
    one, the status becomes 'Corrected by researcher'.  An empty value is
    stored as 'Not reported in the paper.'
    """
    by_id = {r["paper_id"]: r for r in records}
    for _, row in edited_long.iterrows():
        rec = by_id.get(row["paper_id"])
        if rec is None:
            continue
        old = rec["fields"].setdefault(row["field"], FieldResult(row["field"]).to_dict())
        value = str(row["value"]).strip() if pd.notna(row["value"]) else ""
        value = value or NOT_REPORTED
        status = row["status"] if row["status"] in config.STATUSES else old.get("status")
        if value != old.get("value") and status in (config.STATUS_EXTRACTED, config.STATUS_NOT_FOUND):
            status = config.STATUS_EDITED
        old["value"] = value
        old["status"] = status
        old["pages"] = "" if pd.isna(row["pages"]) else str(row["pages"])
        if "interpretation" in row and pd.notna(row["interpretation"]):
            old["interpretation"] = str(row["interpretation"])
    return records


def review_progress(records: list[dict]) -> tuple[int, int]:
    """(number of fields reviewed by the researcher, total fields)."""
    total = done = 0
    for r in records:
        for d in r["fields"].values():
            total += 1
            if d.get("status") in (config.STATUS_VERIFIED, config.STATUS_EDITED):
                done += 1
    return done, total


README_ROWS = [
    ("What is this?", "Literature review matrix produced by the EV Purchase Intention Research Assistant."),
    ("Sheet 'Matrix'", "One row per paper, one column per field (values only)."),
    ("Sheet 'Matrix with pages'", "Same, with the supporting page number(s) next to each value."),
    ("Sheet 'Evidence and status'", "One row per paper x field: value, page, verbatim evidence snippet, extraction method and review status. Re-import this sheet to continue reviewing."),
    ("Sheet 'App interpretations'", "The application's commentary on each extracted value. These are NOT statements from the papers."),
    ("'Not reported in the paper.'", "The extractor did not find this information. Check the status column: unless it says 'Verified by researcher', the information may exist in the paper in a form the rules missed."),
    ("Extraction method", "Rule-based (patterns and dictionaries), not AI. All values need manual verification before use."),
]


def export_excel(records: list[dict]) -> bytes:
    """Return an .xlsx file (bytes) with the matrix, evidence and notes."""
    if not records:
        raise ValueError("There are no papers in the matrix to export.")
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    long_df = to_long(records)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        to_wide(records).to_excel(xw, sheet_name="Matrix", index=False)
        to_wide(records, include_pages=True).to_excel(xw, sheet_name="Matrix with pages", index=False)
        long_df.drop(columns=["interpretation"]).to_excel(xw, sheet_name="Evidence and status", index=False)
        long_df[["paper_id", "filename", "field", "value", "interpretation"]].to_excel(
            xw, sheet_name="App interpretations", index=False)
        readme = pd.DataFrame(README_ROWS + [("Exported", datetime.now().strftime("%Y-%m-%d %H:%M"))],
                              columns=["Item", "Explanation"])
        readme.to_excel(xw, sheet_name="Read me", index=False)

        header_fill = PatternFill("solid", fgColor="DDEBF7")
        for ws in xw.book.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.fill = header_fill
            for col_idx in range(1, ws.max_column + 1):
                ws.column_dimensions[get_column_letter(col_idx)].width = 40 if ws.max_column > 3 else 70
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "A2"
    return buf.getvalue()


def import_excel(data: bytes) -> list[dict]:
    """Rebuild records from the 'Evidence and status' sheet of a previous export."""
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
    except Exception as exc:
        raise ValueError(f"Could not read the Excel file: {exc}") from exc
    if "Evidence and status" not in sheets:
        raise ValueError("This file has no 'Evidence and status' sheet. Import only works with files exported by this app.")
    ev = sheets["Evidence and status"].fillna("")
    missing = {"paper_id", "filename", "field", "value"} - set(ev.columns)
    if missing:
        raise ValueError(f"The 'Evidence and status' sheet is missing columns: {sorted(missing)}")
    interp = {}
    if "App interpretations" in sheets:
        for _, r in sheets["App interpretations"].fillna("").iterrows():
            interp[(r["paper_id"], r["field"])] = r.get("interpretation", "")
    records: dict[str, dict] = {}
    for _, r in ev.iterrows():
        rec = records.setdefault(r["paper_id"], {"paper_id": r["paper_id"], "filename": r["filename"], "fields": {}})
        rec["fields"][r["field"]] = FieldResult(
            field=r["field"], value=r["value"] or NOT_REPORTED, pages=r.get("pages", ""),
            evidence=r.get("evidence", ""), method=r.get("method", ""),
            status=r.get("status", "") or config.STATUS_NOT_FOUND,
            interpretation=interp.get((r["paper_id"], r["field"]), ""),
        ).to_dict()
    return list(records.values())


def split_list(value: str) -> list[str]:
    """Split a multi-value cell ('TPB; TAM') into items. Returns [] for
    'Not reported'."""
    if not value or str(value).strip() == NOT_REPORTED:
        return []
    parts = [p.strip() for p in str(value).replace("\n", ";").split(";")]
    return [p for p in parts if p]
