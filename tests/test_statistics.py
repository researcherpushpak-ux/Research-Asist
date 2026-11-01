import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ev_research.analysis import descriptive, efa, inferential, mediation, reliability
from ev_research.analysis.result import AnalysisError


def _alpha_alt(df):
    """Independent formula: alpha = k*c / (v + (k-1)*c), c = mean covariance, v = mean variance."""
    C = df.cov().values
    k = C.shape[0]
    v = np.mean(np.diag(C))
    c = (C.sum() - np.trace(C)) / (k * (k - 1))
    return k * c / (v + (k - 1) * c)


def test_cronbach_alpha_matches_alternative_formula(survey_df, groups):
    for items in groups.values():
        sub = survey_df[items].dropna()
        assert reliability.cronbach_alpha(sub) == pytest.approx(_alpha_alt(sub), abs=1e-10)


def test_cronbach_alpha_hand_example():
    # Totals 6, 9, 9, 14, 14 -> var(total) = 49.2/4 = 12.3; item variances 2.5, 1.3, 1.0 (sum 4.8)
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 3, 5, 4], "c": [3, 4, 3, 5, 5]})
    assert reliability.cronbach_alpha(df) == pytest.approx(1.5 * (1 - 4.8 / 12.3))


def test_cronbach_alpha_identical_items_is_one():
    x = np.arange(10, dtype=float)
    assert reliability.cronbach_alpha(pd.DataFrame({"a": x, "b": x, "c": x})) == pytest.approx(1.0)


def test_reliability_flags_reverse_item():
    rng = np.random.default_rng(0)
    t = rng.normal(size=200)
    df = pd.DataFrame({f"i{j}": t + rng.normal(0, .5, 200) for j in range(3)})
    df["neg"] = -t + rng.normal(0, .5, 200)
    res = reliability.reliability_analysis(df, list(df.columns), "S")
    assert any("reverse-coding" in w for w in res.warnings)


def test_descriptives_match_pandas(survey_df):
    res = descriptive.descriptives(survey_df, ["ATT1"])
    row = res.tables["Descriptive statistics"].iloc[0]
    s = survey_df["ATT1"].dropna()
    assert row["Mean"] == pytest.approx(round(s.mean(), 3)) and row["SD"] == pytest.approx(round(s.std(), 3))
    with pytest.raises(AnalysisError):
        descriptive.descriptives(survey_df, ["Gender"])


def test_frequencies_percentages(survey_df):
    t = descriptive.frequencies(survey_df, "Gender").tables["Frequencies - Gender"]
    assert t["Count"].sum() == len(survey_df)
    assert t["Percent"].sum() == pytest.approx(100, abs=0.05)


def test_correlation_matches_scipy(scored_df):
    res = inferential.correlation(scored_df, ["ATT", "PI", "SN"])
    pair = scored_df[["ATT", "PI"]].dropna()
    r, p = stats.pearsonr(pair["ATT"], pair["PI"])
    assert res.raw["r"].loc["ATT", "PI"] == pytest.approx(r)
    assert res.raw["p"].loc["PI", "ATT"] == pytest.approx(p)


def test_regression_matches_numpy(scored_df):
    res = inferential.regression(scored_df, "PI", ["ATT", "SN", "PBC"])
    d = scored_df[["PI", "ATT", "SN", "PBC"]].dropna()
    X = np.column_stack([np.ones(len(d)), d[["ATT", "SN", "PBC"]].values])
    b, *_ = np.linalg.lstsq(X, d["PI"].values, rcond=None)
    coef = res.tables["Coefficients"]
    assert coef["B"].values == pytest.approx(np.round(b, 4), abs=1e-4)
    resid = d["PI"].values - X @ b
    r2 = 1 - resid.var() / d["PI"].var(ddof=0)
    assert res.tables["Model fit"]["R-squared"].iloc[0] == pytest.approx(r2, abs=1e-4)


def test_regression_categorical_and_errors(scored_df):
    res = inferential.regression(scored_df, "PI", ["ATT", "Gender"], categorical=["Gender"])
    assert any(p.startswith("Gender_") for p in res.tables["Coefficients"]["Predictor"])
    with pytest.raises(AnalysisError):
        inferential.regression(scored_df, "PI", ["PI"])
    with pytest.raises(AnalysisError):
        inferential.regression(scored_df, "PI", ["Gender"])   # text predictor not ticked as categorical


def test_anova_matches_scipy(scored_df):
    res = inferential.one_way_anova(scored_df, "PI", "Income")
    d = scored_df[["PI", "Income"]].dropna()
    F, p = stats.f_oneway(*[g["PI"].values for _, g in d.groupby("Income")])
    row = res.tables["Tests"].iloc[0]
    assert row["Statistic"] == pytest.approx(round(F, 3)) and row["p"] == pytest.approx(round(p, 4))
    assert "Tukey HSD pairwise comparisons" in res.tables


def test_mediation_recovers_known_indirect_effect():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(size=n)
    y = 0.4 * m + 0.2 * x + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "m": m, "y": y})
    res = mediation.simple_mediation(df, "x", "m", "y", n_boot=1000, seed=3)
    ind = res.tables["Indirect effect"].iloc[0]
    assert ind["Indirect effect (a x b)"] == pytest.approx(0.2, abs=0.04)
    assert ind.iloc[2] < 0.2 < ind.iloc[3]
    # total = direct + indirect holds exactly for OLS without covariates
    paths = res.tables["Path coefficients"].set_index("Path")["Estimate"]
    assert paths.iloc[2] == pytest.approx(paths.iloc[3] + ind["Indirect effect (a x b)"], abs=1e-3)


def test_mediation_null_effect_ci_includes_zero():
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame({"x": rng.normal(size=n), "m": rng.normal(size=n), "y": rng.normal(size=n)})
    res = mediation.simple_mediation(df, "x", "m", "y", n_boot=1000)
    ind = res.tables["Indirect effect"].iloc[0]
    assert ind.iloc[2] < 0 < ind.iloc[3]
    assert "not supported" in res.text[0]


def test_mediation_reproducible_with_seed(scored_df):
    a = mediation.simple_mediation(scored_df, "EC", "ATT", "PI", n_boot=600, seed=9).tables["Indirect effect"]
    b = mediation.simple_mediation(scored_df, "EC", "ATT", "PI", n_boot=600, seed=9).tables["Indirect effect"]
    pd.testing.assert_frame_equal(a, b)


def test_efa_recovers_simulated_structure(survey_df, groups):
    items = [i for v in groups.values() for i in v]
    res = efa.run_efa(survey_df, items, len(groups), "oblimin")
    load = res.tables["Factor loadings"].set_index("Item")
    for g, its in groups.items():
        assert load.loc[its, "Primary factor"].nunique() == 1, f"{g} items split across factors"
    s = efa.suitability(survey_df, items)
    assert s.tables["Suitability summary"]["Overall KMO"].iloc[0] > 0.6


def test_efa_input_validation(survey_df):
    with pytest.raises(AnalysisError):
        efa.run_efa(survey_df, ["ATT1", "ATT2"], 1)
    with pytest.raises(AnalysisError):
        efa.run_efa(survey_df, ["ATT1", "ATT2", "ATT3", "ATT4"], 4)
