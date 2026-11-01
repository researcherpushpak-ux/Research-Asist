import streamlit as st

from ev_research.analysis import data_io, inferential
from ev_research.analysis.result import AnalysisError
from ev_research.ui import figure_actions, page_header
from ev_research.viz import charts

st.set_page_config(page_title="Visualizations", layout="wide")
page_header("Module 3 - Visualizations", "Charts from your survey data and from the literature matrix. Add any chart to the report.")
S = st.session_state


def draw(make, caption: str, key: str):
    try:
        fig = make()
    except AnalysisError as exc:
        st.error(str(exc))
        return None
    png = charts.to_png(fig)
    st.image(png)
    figure_actions(png, caption, key)
    return png


survey_tab, lit_tab = st.tabs(["Survey data", "Literature"])

with survey_tab:
    df = S.get("df")
    if df is None:
        st.info("Load a dataset on the Data Analysis page first.")
    else:
        if S.get("df_synthetic"):
            st.warning("SYNTHETIC sample data: charts are for learning only.")
        kind = st.radio("Chart", ["Bar chart (categories)", "Histogram (numeric)", "Correlation heatmap"], horizontal=True)
        if kind.startswith("Bar"):
            col = st.selectbox("Variable", data_io.categorical_columns(df) or list(df.columns))
            pct = st.checkbox("Show percentages")
            draw(lambda: charts.bar_chart(df, col, pct), f"Bar chart of {col}", f"bar_{col}")
        elif kind.startswith("Histogram"):
            nums = data_io.analysis_columns(df) or data_io.numeric_columns(df)
            col = st.selectbox("Variable", nums)
            bins = st.slider("Bins", 3, 60, 15)
            draw(lambda: charts.histogram(df, col, bins), f"Histogram of {col}", f"hist_{col}")
        else:
            nums = data_io.numeric_columns(df)
            default = [g for g in S["groups"] if g in nums] or data_io.analysis_columns(df)[:6]
            cols = st.multiselect("Variables", nums, default=default)
            meth = st.radio("Method", ["pearson", "spearman"], horizontal=True)
            if len(cols) >= 2:
                try:
                    r = inferential.correlation(df, cols, meth)
                    draw(lambda: charts.correlation_heatmap(r.raw["r"], r.raw["p"], f"{meth.title()} correlations"),
                         f"{meth.title()} correlation heatmap ({', '.join(cols)})", "heat")
                except AnalysisError as exc:
                    st.error(str(exc))
            else:
                st.info("Select at least two variables.")

with lit_tab:
    papers = S["papers"]
    if not papers:
        st.info("Add papers on the Literature Matrix page first.")
    else:
        st.caption("Charts use the current matrix values, including values you have not yet reviewed. Counts describe the uploaded papers only.")
        st.markdown("#### Publication trend")
        draw(lambda: charts.publication_trend(papers), "Reviewed papers by publication year", "trend")
        st.markdown("#### Frequently studied items")
        field = st.selectbox("Field", ["Theories used", "Independent variables", "Dependent variables", "Mediating variables",
                                       "Moderating variables", "Statistical techniques", "Country of study", "Research design",
                                       "Sampling method"])
        try:
            fig, table = charts.field_frequency(papers, field)
            png = charts.to_png(fig)
            st.image(png)
            figure_actions(png, f"{field} across reviewed papers", f"freq_{field}")
            st.markdown("Supporting papers for each count:")
            st.dataframe(table, hide_index=True)
        except AnalysisError as exc:
            st.info(str(exc))
