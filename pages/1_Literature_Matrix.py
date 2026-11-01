import pandas as pd
import streamlit as st

from ev_research import config
from ev_research.literature import matrix
from ev_research.literature.extractor import empty_fields, extract_fields
from ev_research.literature.pdf_reader import PDFReadError, read_pdf
from ev_research.ui import SAMPLE_DIR, page_header

st.set_page_config(page_title="Literature Matrix", layout="wide")
page_header("Module 1 - Literature Review Matrix",
            "Upload research papers (PDF). The application extracts candidate details with page references; you review and correct them.")

papers: list[dict] = st.session_state["papers"]

# ---------------------------------------------------------------- upload
with st.expander("1. Add papers", expanded=not papers):
    files = st.file_uploader("Upload one or more PDF papers", type=["pdf"], accept_multiple_files=True)
    if st.button("Extract details from uploaded PDFs", disabled=not files, type="primary"):
        existing = {p["filename"] for p in papers}
        for f in files:
            if f.name in existing:
                st.info(f"'{f.name}' is already in the matrix and was skipped.")
                continue
            try:
                doc = read_pdf(f.getvalue(), f.name)
                fields = extract_fields(doc)
                pid = matrix.next_paper_id(papers)
                papers.append(matrix.new_record(pid, f.name, fields))
                found = sum(1 for v in fields.values() if v.value != config.NOT_REPORTED)
                st.success(f"{pid}: '{f.name}' ({doc.n_pages} pages) - candidates found for {found} of {len(fields)} fields.")
            except PDFReadError as exc:
                st.error(str(exc))
            except Exception as exc:  # unexpected - show but continue with other files
                st.error(f"Unexpected error while processing '{f.name}': {exc}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Add a paper manually** (e.g. a scanned PDF)")
        manual_name = st.text_input("Short name / file name", key="manual_name")
        if st.button("Add empty record", disabled=not manual_name.strip()):
            pid = matrix.next_paper_id(papers)
            papers.append(matrix.new_record(pid, manual_name.strip(), empty_fields()))
            st.success(f"Added {pid}. Fill it in below.")
    with c2:
        st.markdown("**Continue earlier work**: import a matrix exported by this app")
        prev = st.file_uploader("Exported matrix (.xlsx)", type=["xlsx"], key="import_xlsx")
        if st.button("Import matrix", disabled=prev is None):
            try:
                imported = matrix.import_excel(prev.getvalue())
                ids = {p["paper_id"] for p in papers}
                clash = [r["paper_id"] for r in imported if r["paper_id"] in ids]
                if clash:
                    st.error(f"Paper IDs already present: {clash}. Remove them first or start a new session.")
                else:
                    papers.extend(imported)
                    st.success(f"Imported {len(imported)} papers.")
            except ValueError as exc:
                st.error(str(exc))

    fictional = sorted((SAMPLE_DIR / "fictional_papers").glob("*.pdf"))
    if fictional and st.button("Load the 3 FICTIONAL test papers (for learning only)"):
        for path in fictional:
            if path.name in {p["filename"] for p in papers}:
                continue
            doc = read_pdf(path.read_bytes(), path.name)
            papers.append(matrix.new_record(matrix.next_paper_id(papers), path.name, extract_fields(doc)))
        st.warning("Fictional test papers loaded. They are not real research and must not be cited.")

if not papers:
    st.stop()

# ---------------------------------------------------------------- review
done, total = matrix.review_progress(papers)
st.progress(done / total if total else 0, text=f"Reviewed by you: {done} of {total} cells")

st.subheader("2. Review and correct")
st.markdown(
    "Select a paper, then check each value against the PDF using the **page** and **evidence** columns. "
    "Edit the *value* (and *pages*) where needed and set the *status*. Leave a value empty if the paper truly does not report it; "
    f"it will be saved as \"{config.NOT_REPORTED}\". The *app interpretation* column is the application's commentary, not the paper's content."
)
labels = {p["paper_id"]: f"{p['paper_id']} - {p['fields'].get('Paper title', {}).get('value', '')[:80]} ({p['filename']})" for p in papers}
pid = st.selectbox("Paper", list(labels), format_func=lambda k: labels[k])
rec = next(p for p in papers if p["paper_id"] == pid)
long_df = matrix.to_long([rec])

with st.form(f"edit_{pid}"):
    edited = st.data_editor(
        long_df,
        hide_index=True,
        disabled=["paper_id", "filename", "field", "evidence", "method", "interpretation"],
        column_order=["field", "value", "pages", "status", "evidence", "interpretation", "method"],
        column_config={
            "field": st.column_config.TextColumn("Field", width="small"),
            "value": st.column_config.TextColumn("Value (editable)", width="large"),
            "pages": st.column_config.TextColumn("Page(s)", width="small"),
            "status": st.column_config.SelectboxColumn("Status", options=config.STATUSES, width="medium"),
            "evidence": st.column_config.TextColumn("Evidence from PDF (verbatim)", width="large"),
            "interpretation": st.column_config.TextColumn("App interpretation (not from paper)", width="large"),
            "method": st.column_config.TextColumn("Extraction method", width="medium"),
        },
        key=f"editor_{pid}",
    )
    if st.form_submit_button("Save changes for this paper", type="primary"):
        matrix.apply_edits(papers, edited)
        st.success("Saved. Values you changed are marked 'Corrected by researcher'.")
        st.rerun()

with st.expander("Remove this paper"):
    if st.button(f"Remove {pid} from the matrix", type="secondary"):
        papers[:] = [p for p in papers if p["paper_id"] != pid]
        st.rerun()

# ---------------------------------------------------------------- matrix + export
st.subheader("3. Matrix and export")
st.dataframe(matrix.to_wide(papers), hide_index=True)
try:
    xlsx = matrix.export_excel(papers)
    st.download_button("Download literature matrix (Excel)", data=xlsx, file_name="literature_matrix.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
except Exception as exc:
    st.error(f"Export failed: {exc}")
st.caption("The Excel file contains: the matrix, the matrix with page numbers, evidence and review status, the app's interpretations (separate sheet) and an explanation sheet.")
