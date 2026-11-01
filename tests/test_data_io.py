import io

import numpy as np
import pandas as pd
import pytest

from ev_research.analysis import data_io
from ev_research.analysis.result import AnalysisError


def test_load_csv_and_excel(survey_df):
    buf = io.BytesIO()
    survey_df.to_excel(buf, index=False)
    xl = data_io.load_dataset(buf.getvalue(), "x.xlsx")
    assert xl.shape == survey_df.shape


def test_semicolon_csv_detected():
    df = data_io.load_dataset(b"a;b\n1;2\n3;4\n", "x.csv")
    assert list(df.columns) == ["a", "b"] and len(df) == 2


@pytest.mark.parametrize("data,name,msg", [
    (b"", "x.csv", "empty"),
    (b"a,b\n1,2", "x.pdf", "Unsupported"),
])
def test_load_errors(data, name, msg):
    with pytest.raises(AnalysisError, match=msg):
        data_io.load_dataset(data, name)


def test_detect_item_groups():
    g = data_io.detect_item_groups(["ATT1", "ATT2", "SN_1", "SN_2", "Age", "PI1"])
    assert g == {"ATT": ["ATT1", "ATT2"], "SN": ["SN_1", "SN_2"]}


def test_likert_detection_excludes_id(survey_df):
    lik = data_io.detect_likert_columns(survey_df)
    assert "RespondentID" not in lik and "ATT1" in lik


def test_inspect_counts_missing(survey_df):
    res = data_io.inspect(survey_df)
    t = res.tables["Variables"].set_index("Variable")
    assert t.loc["ATT1", "Missing"] == survey_df["ATT1"].isna().sum()
    assert int(t["Missing"].sum()) == int(survey_df.isna().sum().sum())


def test_handle_missing_listwise_and_mean():
    df = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [1, 2, np.nan, 4]})
    out, note = data_io.handle_missing(df, ["a", "b"], "listwise")
    assert len(out) == 2 and "removed 2" in note
    out, _ = data_io.handle_missing(df, ["a"], "mean")
    assert out["a"].isna().sum() == 0 and out.loc[1, "a"] == pytest.approx(8 / 3)
    assert out["b"].isna().sum() == 1     # untouched


def test_reverse_code():
    df = pd.DataFrame({"x": [1, 2, 5]})
    assert data_io.reverse_code(df, ["x"], 1, 5)["x"].tolist() == [5, 4, 1]
    with pytest.raises(AnalysisError):
        data_io.reverse_code(pd.DataFrame({"x": [0, 6]}), ["x"], 1, 5)


def test_composite_scores_requires_half_items():
    df = pd.DataFrame({"A1": [1, 2, np.nan], "A2": [3, np.nan, np.nan], "A3": [5, 4, 3]})
    s = data_io.composite_scores(df, {"A": ["A1", "A2", "A3"]})["A"]
    assert s.iloc[0] == 3 and s.iloc[1] == 3 and np.isnan(s.iloc[2])


def test_analysis_columns_exclude_ids(survey_df):
    cols = data_io.analysis_columns(survey_df)
    assert "RespondentID" not in cols and "ATT1" in cols
