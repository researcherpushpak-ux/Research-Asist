import streamlit as st

from ev_research.report.report_builder import ReportInputs, build_docx, build_markdown
from ev_research.ui import CATEGORIES, page_header

st.set_page_config(page_title="Report", layout="wide")
page_header("Module 5 - Preliminary Research Report",
            "The report contains only the results and charts you added, plus your own text. Nothing is invented to fill gaps.")
S = st.session_state

st.subheader("1. Your text")
title = st.text_input("Report title", "Factors Affecting Purchase Intention for Electric Vehicles in India - Preliminary Analysis")
author = st.text_input("Researcher name (optional)")
methodology = st.text_area("Methodology (research design, population, sampling, questionnaire, data collection)", height=150,
                           help="The application cannot know how you collected the data, so this section is left for you. If empty, a placeholder is shown.")
limitations = st.text_area("Limitations (your own)", height=100)
further = st.text_area("Suggestions for further investigation (your own)", height=100)

st.subheader("2. Content added from other pages")
items = S["report_items"]
if not items:
    st.info("No analysis results added yet. Use the 'Add to report' buttons on the Data Analysis page.")
for i, (cat, r) in enumerate(list(items)):
    c1, c2, c3 = st.columns([5, 3, 1])
    c1.markdown(f"**{r.title}**" + ("  _(preparation only)_" if not r.performed_by_app else ""))
    new_cat = c2.selectbox("Section", CATEGORIES, index=CATEGORIES.index(cat), key=f"cat_{i}", label_visibility="collapsed")
    if new_cat != cat:
        items[i] = (new_cat, r)
    if c3.button("Remove", key=f"rm_{i}"):
        items.pop(i)
        st.rerun()

figs = S["report_figures"]
for i, (cap, png) in enumerate(list(figs)):
    c1, c2 = st.columns([8, 1])
    c1.markdown(f"Chart: {cap}")
    if c2.button("Remove", key=f"rmf_{i}"):
        figs.pop(i)
        st.rerun()
if S.get("gap_report") is not None:
    st.markdown("Gap analysis: included (section 7).")
st.markdown(f"Literature matrix: {len(S['papers'])} paper(s) will be summarised." if S["papers"] else "Literature matrix: no papers.")

df = S.get("df")
inputs = ReportInputs(
    title=title, author=author,
    dataset_name=S.get("df_name", ""), dataset_shape=None if df is None else df.shape,
    dataset_inspection=S.get("inspection"), data_handling_notes=list(S.get("data_notes", [])),
    methodology_text=methodology, results=list(items), figures=list(figs),
    literature_records=S["papers"], gap_report=S.get("gap_report"),
    limitations_text=limitations, further_text=further, sample_is_synthetic=bool(S.get("df_synthetic")),
)

st.subheader("3. Generate")
if S.get("df_synthetic"):
    st.warning("The report will be clearly marked as based on SYNTHETIC data.")
c1, c2 = st.columns(2)
try:
    c1.download_button("Download report (Word .docx)", build_docx(inputs), "preliminary_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary")
except Exception as exc:
    c1.error(f"Word export failed: {exc}")
md = build_markdown(inputs)
c2.download_button("Download report (Markdown)", md.encode("utf-8"), "preliminary_report.md", "text/markdown")
with st.expander("Preview (Markdown)"):
    st.markdown(md)
