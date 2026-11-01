"""EV Purchase Intention Research Assistant - home page.

Run with:  streamlit run app.py
"""

import streamlit as st

from ev_research import __version__
from ev_research.ui import init_state

st.set_page_config(page_title="EV Research Assistant", page_icon=":material/electric_car:", layout="wide")
init_state()

st.title("EV Purchase Intention Research Assistant")
st.caption(f"Version {__version__} - support tool for research on factors affecting purchase intention for electric vehicles in India")

st.markdown(
    """
Use the pages in the left sidebar, roughly in this order:

| Step | Page | What you do there |
|---|---|---|
| 1 | **Literature Matrix** | Upload PDFs, review/correct the extracted details, export the matrix to Excel |
| 2 | **Data Analysis** | Upload your questionnaire data (CSV/Excel), inspect it, run reliability, EFA, correlation, regression, ANOVA, mediation and PLS-SEM preparation |
| 3 | **Visualizations** | Bar charts, histograms, correlation heatmaps, publication trends, frequent theories and variables |
| 4 | **Research Gaps** | See gaps stated by authors, patterns across papers, and application-inferred potential gaps (with supporting papers) |
| 5 | **Report** | Build a preliminary Word/Markdown report from the results you chose to add |
"""
)

st.subheader("What this application does and does not do")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
**Performed by the application**
- Rule-based text extraction from PDFs (with page numbers and evidence)
- Data inspection, missing-value analysis, descriptive statistics, frequencies
- Cronbach's alpha, KMO/Bartlett, exploratory factor analysis
- Pearson/Spearman correlation, OLS multiple regression with diagnostics
- One-way ANOVA (+ Welch, Kruskal-Wallis, Tukey HSD)
- Simple mediation with bootstrap confidence intervals
- HTMT from item correlations
"""
    )
with c2:
    st.markdown(
        """
**Not performed (needs specialist software or your judgement)**
- **PLS-SEM estimation**: the app only prepares the data, the model specification, an R script template and a reporting checklist. Estimate the model in SmartPLS or R (seminr / cSEM).
- CB-SEM / CFA, moderation/multi-group analysis, logistic regression
- OCR of scanned PDFs, AI summarisation of papers
- Writing your methodology and interpretation; these are left as placeholders in the report
"""
    )

st.subheader("Important")
st.markdown(
    """
- PDF extraction uses **patterns and dictionaries, not AI**. It can miss information or pick up text that describes *other* studies. Every value starts as *needs review*, and every value shows the page and the sentence it came from. Check each one.
- Information that the extractor cannot find is marked **"Not reported in the paper."** It may still be in the paper; confirm it yourself.
- Your work is kept only while the browser tab is open. **Export the matrix to Excel regularly.** You can re-import it later on the Literature Matrix page.
- The sample files in `sample_data/` are **synthetic or fictional**. Use them only to learn the software.
"""
)

papers = len(st.session_state["papers"])
df = st.session_state["df"]
st.subheader("Current session")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Papers in matrix", papers)
m2.metric("Dataset rows", 0 if df is None else len(df))
m3.metric("Results in report", len(st.session_state["report_items"]))
m4.metric("Charts in report", len(st.session_state["report_figures"]))
