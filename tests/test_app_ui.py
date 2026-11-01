"""Headless UI tests using Streamlit's AppTest harness: each page loads, and
the main workflows can be clicked through without exceptions."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

TIMEOUT = 120
ROOT = Path(__file__).resolve().parent.parent


def _app(page):
    return AppTest.from_file(str(ROOT / page), default_timeout=TIMEOUT)


def _click(at, label):
    btn = next((b for b in at.button if b.label == label), None)
    assert btn is not None, f"button '{label}' not found; have: {[b.label for b in at.button]}"
    btn.click().run(timeout=TIMEOUT)
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.mark.parametrize("page", ["Home.py", "pages/1_Literature_Matrix.py", "pages/2_Data_Analysis.py",
                                  "pages/3_Visualizations.py", "pages/4_Research_Gaps.py", "pages/5_Report.py"])
def test_pages_load_empty(page):
    at = _app(page).run()
    assert not at.exception, [e.message for e in at.exception]


def test_literature_workflow():
    at = _app("pages/1_Literature_Matrix.py").run()
    _click(at, "Load the 3 FICTIONAL test papers (for learning only)")
    assert len(at.session_state["papers"]) == 3
    assert any("Download literature matrix" in d.label for d in at.get("download_button"))
    _click(at, "Save changes for this paper")
    assert not at.exception


def test_data_analysis_workflow():
    at = _app("pages/2_Data_Analysis.py").run()
    _click(at, "Load SYNTHETIC sample dataset (for learning only)")
    assert at.session_state["df"] is not None and len(at.session_state["groups"]) == 6
    _click(at, "Compute construct mean scores (adds columns)")
    assert "PI" in at.session_state["df"].columns
    for label in ["Run descriptive statistics", "Run frequency table", "Compute Cronbach's alpha for all constructs",
                  "1. Check suitability (KMO, Bartlett, eigenvalues)", "2. Run EFA", "Run correlation analysis",
                  "Run ANOVA", "Run PLS-SEM preparation checks"]:
        _click(at, label)
    for key in ["res_desc", "res_freq", "res_rel_all", "res_efa_s", "res_efa", "res_corr", "res_anova", "res_pls"]:
        assert at.session_state[key] is not None, key
    # regression needs predictors selected
    at.multiselect(key="reg_iv").set_value(["ATT", "SN", "PBC"]).run(timeout=TIMEOUT)
    _click(at, "Run regression")
    assert at.session_state["res_reg"] is not None
    at.selectbox(key="med_x").set_value("EC")
    at.selectbox(key="med_m").set_value("ATT")
    at.selectbox(key="med_y").set_value("PI").run(timeout=TIMEOUT)
    _click(at, "Run mediation analysis")
    assert at.session_state["res_med"] is not None
    btn = next(b for b in at.button if b.key == "add_reg")
    btn.click().run(timeout=TIMEOUT)
    assert any(r.title.startswith("Multiple regression") for _, r in at.session_state["report_items"])


def test_report_page_with_content():
    at = _app("pages/2_Data_Analysis.py").run()
    _click(at, "Load SYNTHETIC sample dataset (for learning only)")
    _click(at, "Compute Cronbach's alpha for all constructs")
    next(b for b in at.button if b.key == "add_rel_all").click().run(timeout=TIMEOUT)
    state = {k: at.session_state[k] for k in ["df", "df_name", "df_synthetic", "inspection", "groups", "report_items", "data_notes"]}
    rep = _app("pages/5_Report.py")
    for k, v in state.items():
        rep.session_state[k] = v
    rep.run()
    assert not rep.exception, [e.message for e in rep.exception]
    labels = [d.label for d in rep.get("download_button")]
    assert "Download report (Word .docx)" in labels and "Download report (Markdown)" in labels
