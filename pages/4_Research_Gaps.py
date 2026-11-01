import streamlit as st

from ev_research.gaps import gap_finder
from ev_research.ui import page_header

st.set_page_config(page_title="Research Gaps", layout="wide")
page_header("Module 4 - Research Gap Identification",
            "Built from the literature matrix. Review the matrix first: gaps can only be as good as the extracted values.")
S = st.session_state
papers = S["papers"]
if not papers:
    st.info("Add papers on the Literature Matrix page first.")
    st.stop()

report = gap_finder.build_report(papers)
for n in report.notes:
    st.caption(n)

st.subheader("1. Gaps stated explicitly by the authors")
st.markdown("Copied from the *Research gaps* and *Future research directions* fields, with page numbers. These are the authors' claims.")
if len(report.explicit):
    st.dataframe(report.explicit, hide_index=True)
else:
    st.info("No explicit gaps or future-research statements are recorded in the matrix.")

st.subheader("2. Patterns observed across the reviewed papers")
st.markdown("Counts of what the papers report, with the papers behind each count. These are descriptive facts about **this set** of papers.")
st.dataframe(report.patterns, hide_index=True)
if len(report.themes):
    st.markdown("**Recurring themes in the authors' gap and future-research statements** (keyword-based):")
    st.dataframe(report.themes, hide_index=True)

st.subheader("3. Potential gaps inferred by the application")
st.warning("These are **inferences**, produced by simple rules from the patterns above. They are not established facts. "
           "Each one lists the papers that support it and those that contradict it. Check every inferred gap against a "
           "systematic literature search before using it.")
if len(report.inferred):
    st.dataframe(report.inferred, hide_index=True)
else:
    st.info("No inferred gaps (at least 3 papers are needed, and no rule was triggered).")

c1, c2 = st.columns(2)
c1.download_button("Download gap analysis (Excel)", gap_finder.export_excel(report), "research_gaps.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
if c2.button("Add gap analysis to report"):
    S["gap_report"] = report
    st.success("Gap analysis added to the report (section 7).")
