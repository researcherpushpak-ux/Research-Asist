"""Shared Streamlit helpers (session state, result display, report buttons)."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from ev_research.analysis.result import AnalysisResult
from ev_research.report.report_builder import (CATEGORY_DESCRIPTIVE, CATEGORY_PREPARATION,
                                               CATEGORY_STATISTICAL)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_data"

DEFAULTS = {
    "papers": [],               # literature matrix records
    "df_raw": None,             # dataset as uploaded
    "df": None,                 # working dataset (after missing handling, recoding, composites)
    "df_name": "",
    "df_synthetic": False,
    "data_notes": [],
    "inspection": None,
    "groups": {},               # construct -> items
    "report_items": [],         # list of (category, AnalysisResult)
    "report_figures": [],       # list of (caption, png bytes)
    "gap_report": None,
}

CATEGORIES = [CATEGORY_DESCRIPTIVE, CATEGORY_STATISTICAL, CATEGORY_PREPARATION]


def init_state() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, (list, dict)) else v


def page_header(title: str, intro: str) -> None:
    init_state()
    st.title(title)
    st.caption(intro)


def safe_key(text: str) -> str:
    return re.sub(r"\W+", "_", text)[:60]


def add_result_to_report(category: str, result: AnalysisResult) -> None:
    items = st.session_state["report_items"]
    items[:] = [(c, r) for c, r in items if r.title != result.title]   # replace same-titled result
    items.append((category, result))


def show_result(result: AnalysisResult, category: str, key: str) -> None:
    """Render an AnalysisResult with tables, statements, warnings, a CSV/Excel
    download and an 'Add to report' button."""
    if not result.performed_by_app:
        st.info("**Preparation only.** This model is not estimated by the application.")
    if result.method:
        st.markdown(f"**Method:** {result.method}")
    for name, table in result.tables.items():
        st.markdown(f"**{name}**")
        st.dataframe(table, hide_index=True)
    if result.text:
        st.markdown("**Results**")
        for t in result.text:
            st.markdown(f"- {t}")
    for w in result.warnings:
        st.warning(w)
    for cap, png in result.figures.items():
        st.image(png, caption=cap)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download tables (Excel)", data=tables_to_excel(result),
                           file_name=f"{safe_key(result.title)}.xlsx", key=f"dl_{key}",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        if st.button("Add to report", key=f"add_{key}"):
            add_result_to_report(category, result)
            st.success(f"Added '{result.title}' to the report.")


def tables_to_excel(result: AnalysisResult) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        used = set()
        for name, t in result.tables.items():
            sheet = re.sub(r"[\[\]\*\?/\\:]", "", name)[:31] or "Sheet"
            while sheet in used:
                sheet = sheet[:28] + str(len(used))
            used.add(sheet)
            t.to_excel(xw, sheet_name=sheet, index=False)
        notes = [("Method", result.method)] + [("Result", t) for t in result.text] + [("Caution", w) for w in result.warnings]
        pd.DataFrame(notes, columns=["Type", "Text"]).to_excel(xw, sheet_name="Notes", index=False)
    return buf.getvalue()


def figure_actions(png: bytes, caption: str, key: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download PNG", data=png, file_name=f"{safe_key(caption)}.png", mime="image/png", key=f"png_{key}")
    with c2:
        if st.button("Add chart to report", key=f"fig_{key}"):
            figs = st.session_state["report_figures"]
            figs[:] = [(c, p) for c, p in figs if c != caption]
            figs.append((caption, png))
            st.success("Chart added to the report.")


def require_data() -> pd.DataFrame | None:
    df = st.session_state.get("df")
    if df is None:
        st.info("No dataset loaded yet. Go to **Data Analysis** and upload a CSV/Excel file (or load the synthetic sample).")
        return None
    if st.session_state.get("df_synthetic"):
        st.warning("You are using the SYNTHETIC sample dataset. Results are for learning the software only.")
    return df


def parse_groups(text: str) -> dict[str, list[str]]:
    """Parse 'ATT: ATT1, ATT2' lines into a dict."""
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, items = line.split(":", 1)
        its = [i.strip() for i in items.split(",") if i.strip()]
        if name.strip() and its:
            out[name.strip()] = its
    return out


def groups_to_text(groups: dict[str, list[str]]) -> str:
    return "\n".join(f"{k}: {', '.join(v)}" for k, v in groups.items())
