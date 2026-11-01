"""Correlation, multiple regression and one-way ANOVA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ev_research.analysis.data_io import numeric_columns
from ev_research.analysis.result import AnalysisError, AnalysisResult, fmt_p, sig_word


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def correlation(df: pd.DataFrame, columns: list[str], method: str = "pearson") -> AnalysisResult:
    if len(columns) < 2:
        raise AnalysisError("Select at least two numeric variables.")
    bad = [c for c in columns if c not in numeric_columns(df)]
    if bad:
        raise AnalysisError(f"Non-numeric variables: {bad}.")
    if method not in ("pearson", "spearman"):
        raise AnalysisError("Method must be 'pearson' or 'spearman'.")
    sub = df[columns]
    k = len(columns)
    r = pd.DataFrame(np.eye(k), index=columns, columns=columns)
    p = pd.DataFrame(np.zeros((k, k)), index=columns, columns=columns)
    n = pd.DataFrame(np.zeros((k, k), dtype=int), index=columns, columns=columns)
    func = stats.pearsonr if method == "pearson" else stats.spearmanr
    for i, a in enumerate(columns):
        for j, b in enumerate(columns):
            if j <= i:
                continue
            pair = sub[[a, b]].dropna()
            n.loc[a, b] = n.loc[b, a] = len(pair)
            if len(pair) < 3 or pair[a].std() == 0 or pair[b].std() == 0:
                r.loc[a, b] = r.loc[b, a] = np.nan
                p.loc[a, b] = p.loc[b, a] = np.nan
                continue
            rv, pv = func(pair[a], pair[b])
            r.loc[a, b] = r.loc[b, a] = float(rv)
            p.loc[a, b] = p.loc[b, a] = float(pv)
        n.loc[a, a] = int(sub[a].notna().sum())

    def stars(pv):
        return "" if pd.isna(pv) else "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else ""
    display = r.round(3).astype(str)
    for a in columns:
        for b in columns:
            if a != b:
                display.loc[a, b] = f"{r.loc[a, b]:.3f}{stars(p.loc[a, b])}" if pd.notna(r.loc[a, b]) else "n/a"
            else:
                display.loc[a, b] = "1"
    res = AnalysisResult(
        f"{method.title()} correlation analysis",
        method=f"{method.title()} correlations with pairwise deletion; two-tailed p-values, not adjusted for multiple comparisons. * p < .05, ** p < .01, *** p < .001.",
        tables={"Correlation matrix (with significance)": display.reset_index().rename(columns={"index": ""}),
                "Correlation coefficients": r.round(4).reset_index().rename(columns={"index": ""}),
                "p-values": p.round(4).reset_index().rename(columns={"index": ""}),
                "Pairwise N": n.reset_index().rename(columns={"index": ""})},
    )
    res.raw = {"r": r, "p": p}
    n_tests = k * (k - 1) // 2
    n_sig = int(sum((p.values[np.triu_indices(k, 1)] < 0.05)))
    res.text.append(f"{n_sig} of {n_tests} correlations have p < .05 (unadjusted).")
    if n_tests > 10:
        res.add_warning(f"{n_tests} correlations were tested; some 'significant' results are expected by chance. Consider a Bonferroni or Holm correction.")
    high = [(a, b, r.loc[a, b]) for i, a in enumerate(columns) for b in columns[i + 1:] if abs(r.loc[a, b]) > 0.85]
    if high:
        res.add_warning("Very high correlations (|r| > .85) may indicate overlapping constructs or multicollinearity: "
                        + "; ".join(f"{a}-{b} ({v:.2f})" for a, b, v in high))
    res.text.append("Correlation does not establish causation or direction of influence.")
    return res


# ---------------------------------------------------------------------------
# Multiple regression
# ---------------------------------------------------------------------------
def regression(df: pd.DataFrame, dv: str, ivs: list[str], categorical: list[str] | None = None) -> AnalysisResult:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson

    categorical = categorical or []
    if not ivs:
        raise AnalysisError("Select at least one independent variable.")
    if dv in ivs:
        raise AnalysisError("The dependent variable cannot also be an independent variable.")
    if dv not in numeric_columns(df):
        raise AnalysisError(f"The dependent variable '{dv}' must be numeric. For a categorical outcome, logistic regression is needed (not implemented).")
    num_ivs = [c for c in ivs if c not in categorical]
    bad = [c for c in num_ivs if c not in numeric_columns(df)]
    if bad:
        raise AnalysisError(f"Non-numeric predictors {bad}: tick them as categorical so they are dummy-coded.")
    data = df[[dv] + ivs].dropna()
    n_dropped = len(df) - len(data)
    X = data[num_ivs].astype(float)
    dummy_notes = []
    for c in categorical:
        d = pd.get_dummies(data[c].astype(str), prefix=c, drop_first=True, dtype=float)
        ref = sorted(data[c].astype(str).unique())[0]
        dummy_notes.append(f"'{c}' was dummy-coded; reference category = '{ref}'.")
        X = pd.concat([X, d], axis=1)
    if X.shape[1] == 0:
        raise AnalysisError("No usable predictors after coding.")
    y = data[dv].astype(float)
    if len(data) < X.shape[1] + 10:
        raise AnalysisError(f"Only {len(data)} complete cases for {X.shape[1]} predictors. More data are needed.")
    const = [c for c in X.columns if X[c].std() == 0]
    if const:
        raise AnalysisError(f"Predictors with no variation: {const}.")
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()

    # standardised coefficients (z-scored X and y)
    zX = (X - X.mean()) / X.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    beta = sm.OLS(zy, sm.add_constant(zX)).fit().params

    ci = model.conf_int()
    coef = pd.DataFrame({
        "Predictor": model.params.index,
        "B": model.params.values, "SE": model.bse.values,
        "Beta (standardised)": [np.nan] + list(beta.values[1:]),
        "t": model.tvalues.values, "p": model.pvalues.values,
        "95% CI lower": ci[0].values, "95% CI upper": ci[1].values,
    })
    if X.shape[1] > 1:
        vif = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    else:
        vif = [1.0]
    coef["VIF"] = [np.nan] + vif
    coef = coef.round(4)

    resid = model.resid
    bp_p = het_breuschpagan(resid, Xc)[1]
    dw = durbin_watson(resid)
    norm_name, norm_p = ("Shapiro-Wilk", stats.shapiro(resid)[1]) if len(resid) <= 5000 else ("Jarque-Bera", stats.jarque_bera(resid)[1])
    fit = pd.DataFrame([{
        "N": int(model.nobs), "R-squared": model.rsquared, "Adjusted R-squared": model.rsquared_adj,
        "F": model.fvalue, "df model": int(model.df_model), "df residual": int(model.df_resid), "p (F)": model.f_pvalue,
    }]).round(4)
    diag = pd.DataFrame([
        {"Check": "Breusch-Pagan (heteroscedasticity)", "Value": round(bp_p, 4), "Interpretation": "p < .05 suggests non-constant error variance" },
        {"Check": "Durbin-Watson (autocorrelation)", "Value": round(dw, 3), "Interpretation": "values near 2 suggest no autocorrelation"},
        {"Check": f"{norm_name} on residuals (normality)", "Value": round(norm_p, 4), "Interpretation": "p < .05 suggests non-normal residuals (less important for large N)"},
        {"Check": "Maximum VIF", "Value": round(max(vif), 3), "Interpretation": "> 5 (strict: > 3.3) suggests multicollinearity"},
    ])
    res = AnalysisResult(
        f"Multiple regression: {dv}",
        method=f"Ordinary least squares regression of {dv} on {', '.join(ivs)}; listwise deletion (N = {int(model.nobs)}).",
        tables={"Model fit": fit, "Coefficients": coef, "Assumption checks": diag},
    )
    res.text.append(f"The model explains {model.rsquared * 100:.1f}% of the variance in {dv} (R-squared = {model.rsquared:.3f}, adjusted R-squared = {model.rsquared_adj:.3f}); "
                    f"F({int(model.df_model)}, {int(model.df_resid)}) = {model.fvalue:.2f}, p {fmt_p(model.f_pvalue)}.")
    for _, row in coef.iloc[1:].iterrows():
        res.text.append(f"{row['Predictor']}: B = {row['B']:.3f}, beta = {row['Beta (standardised)']:.3f}, t = {row['t']:.2f}, p {fmt_p(row['p'])} ({sig_word(row['p'])} at .05).")
    res.text += dummy_notes
    if n_dropped:
        res.text.append(f"{n_dropped} rows with missing values were excluded.")
    if bp_p < 0.05:
        res.add_warning("Breusch-Pagan p < .05: heteroscedasticity is likely. Standard errors may be biased; consider robust (HC3) standard errors.")
    if max(vif) > 5:
        res.add_warning("At least one VIF exceeds 5: multicollinearity may make individual coefficients unstable.")
    if norm_p < 0.05 and len(resid) < 100:
        res.add_warning("Residuals deviate from normality in a small sample; p-values may be inaccurate.")
    res.text.append("Regression on cross-sectional survey data shows associations, not causal effects.")
    return res


# ---------------------------------------------------------------------------
# One-way ANOVA
# ---------------------------------------------------------------------------
def one_way_anova(df: pd.DataFrame, dv: str, group: str, min_group_n: int = 2) -> AnalysisResult:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.oneway import anova_oneway

    if dv not in numeric_columns(df):
        raise AnalysisError(f"The outcome '{dv}' must be numeric.")
    if dv == group:
        raise AnalysisError("Outcome and grouping variable must be different.")
    data = df[[dv, group]].dropna()
    data[group] = data[group].astype(str)
    sizes = data[group].value_counts()
    small = sizes[sizes < min_group_n].index.tolist()
    if small:
        data = data[~data[group].isin(small)]
    groups = sorted(data[group].unique())
    if len(groups) < 2:
        raise AnalysisError(f"'{group}' needs at least two groups with {min_group_n}+ responses.")
    if len(groups) > 20:
        raise AnalysisError(f"'{group}' has {len(groups)} groups; ANOVA needs a categorical grouping variable.")
    samples = [data.loc[data[group] == g, dv].values for g in groups]
    desc = pd.DataFrame({
        "Group": groups, "N": [len(s) for s in samples], "Mean": [s.mean() for s in samples],
        "SD": [s.std(ddof=1) if len(s) > 1 else np.nan for s in samples],
    }).round(3)
    F, p = stats.f_oneway(*samples)
    grand = data[dv].mean()
    ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    ss_total = ((data[dv] - grand) ** 2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    lev_stat, lev_p = stats.levene(*samples, center="median")
    welch = anova_oneway(data[dv], data[group], use_var="unequal")
    kw_stat, kw_p = stats.kruskal(*samples)
    k, N = len(groups), len(data)
    tests = pd.DataFrame([
        {"Test": "One-way ANOVA (equal variances)", "Statistic": round(F, 3), "df": f"{k - 1}, {N - k}", "p": p},
        {"Test": "Welch ANOVA (unequal variances)", "Statistic": round(float(welch.statistic), 3),
         "df": f"{k - 1}, {float(welch.df[1]):.1f}", "p": float(welch.pvalue)},
        {"Test": "Levene (median) - equality of variances", "Statistic": round(lev_stat, 3), "df": f"{k - 1}, {N - k}", "p": lev_p},
        {"Test": "Kruskal-Wallis (non-parametric)", "Statistic": round(kw_stat, 3), "df": f"{k - 1}", "p": kw_p},
    ])
    tests["p"] = tests["p"].astype(float).round(4)
    res = AnalysisResult(
        f"One-way ANOVA: {dv} by {group}",
        method=f"Comparison of mean {dv} across {k} groups of {group} (N = {N}). Welch ANOVA and Kruskal-Wallis are given as robust alternatives; Tukey HSD for pairwise comparisons.",
        tables={"Group descriptives": desc, "Tests": tests},
    )
    use_welch = lev_p < 0.05
    main_p = float(welch.pvalue) if use_welch else p
    res.text.append(f"{'Welch' if use_welch else 'Standard'} ANOVA: the difference in mean {dv} across {group} groups is {sig_word(main_p)} (p {fmt_p(main_p)}); eta-squared = {eta2:.3f}.")
    if use_welch:
        res.add_warning("Levene's test p < .05: group variances differ, so the Welch ANOVA result is reported as the main result.")
    if min(len(s) for s in samples) < 20:
        res.add_warning("At least one group has fewer than 20 responses; results may be unstable.")
    if small:
        res.add_warning(f"Groups with fewer than {min_group_n} responses were excluded: {small}.")
    if k > 2:
        tk = pairwise_tukeyhsd(data[dv], data[group])
        tk_df = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
        res.tables["Tukey HSD pairwise comparisons"] = tk_df
        if use_welch:
            res.add_warning("Tukey HSD assumes equal variances; with unequal variances, Games-Howell would be more appropriate (not implemented).")
    return res
