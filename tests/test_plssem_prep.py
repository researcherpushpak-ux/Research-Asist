import numpy as np
import pandas as pd
import pytest

from ev_research.analysis import plssem_prep
from ev_research.analysis.result import AnalysisError


def test_inverse_square_root():
    assert plssem_prep.inverse_square_root_n(0.15) == 275      # ceil((2.486/.15)^2) = ceil(274.68)
    assert plssem_prep.inverse_square_root_n(0.197) == 160      # ceil(159.25)
    with pytest.raises(AnalysisError):
        plssem_prep.inverse_square_root_n(1.5)


def test_htmt_hand_computed():
    # Build data with a known correlation matrix
    R = np.array([[1, .6, .3, .2], [.6, 1, .25, .35], [.3, .25, 1, .5], [.2, .35, .5, 1]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(np.zeros(4), R, size=20000)
    df = pd.DataFrame(X, columns=["a1", "a2", "b1", "b2"])
    emp = df.corr().abs().values
    expected = emp[np.ix_([0, 1], [2, 3])].mean() / np.sqrt(emp[0, 1] * emp[2, 3])
    h = plssem_prep.htmt(df, {"A": ["a1", "a2"], "B": ["b1", "b2"]})
    assert h.loc["B", "A"] == pytest.approx(expected)
    assert h.loc["B", "A"] == pytest.approx(0.275 / np.sqrt(.3), abs=0.03)   # population value


def test_validate_spec_finds_problems(survey_df):
    probs = plssem_prep.validate_spec(survey_df, {"A": ["ATT1", "NOPE"], "B": ["ATT1"]}, [("A", "B"), ("B", "A")])
    text = " ".join(probs)
    assert "NOPE" in text and "both" in text and "loop" in text


def test_prepare_is_labelled_not_estimated(survey_df, groups):
    res = plssem_prep.prepare(survey_df, groups, [("ATT", "PI"), ("SN", "PI")])
    assert res.performed_by_app is False
    assert any("does not estimate PLS-SEM" in w for w in res.warnings)
    assert "Approximate reliability and convergent validity (NOT PLS results)" in res.tables


def test_exports(survey_df, groups):
    data, note = plssem_prep.export_smartpls_csv(survey_df, groups, ["Gender"], "listwise")
    df = pd.read_csv(pd.io.common.BytesIO(data))
    assert df.isna().sum().sum() == 0 and "Gender" in df.columns
    data2, _ = plssem_prep.export_smartpls_csv(survey_df, groups, None, "code")
    assert (pd.read_csv(pd.io.common.BytesIO(data2)) == -99).any().any()
    script = plssem_prep.seminr_script(groups, [("ATT", "PI"), ("SN", "PI")])
    assert 'paths(from = c("ATT", "SN"), to = "PI")' in script and "NOT run" in script
