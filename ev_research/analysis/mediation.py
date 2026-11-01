"""Simple mediation analysis (X -> M -> Y) with a percentile bootstrap
confidence interval for the indirect effect.

This is the regression-based approach used by, for example, Hayes' PROCESS
Model 4 with one mediator.  It is computed here with ordinary least squares;
results should match PROCESS up to bootstrap sampling variation, but this
has not been formally cross-validated against PROCESS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_research.analysis.data_io import numeric_columns
from ev_research.analysis.result import AnalysisError, AnalysisResult, fmt_p


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def simple_mediation(df: pd.DataFrame, x: str, m: str, y: str, covariates: list[str] | None = None,
                     n_boot: int = 5000, ci: float = 0.95, seed: int = 12345) -> AnalysisResult:
    import statsmodels.api as sm

    covariates = covariates or []
    names = [x, m, y] + covariates
    if len(set([x, m, y])) < 3:
        raise AnalysisError("X, M and Y must be three different variables.")
    bad = [c for c in names if c not in numeric_columns(df)]
    if bad:
        raise AnalysisError(f"Non-numeric variables: {bad}.")
    if not 500 <= n_boot <= 50000:
        raise AnalysisError("Number of bootstrap samples must be between 500 and 50,000 (5,000 is common).")
    data = df[names].dropna().astype(float)
    n = len(data)
    if n < 30:
        raise AnalysisError(f"Only {n} complete cases; mediation analysis needs more data.")

    cov = data[covariates].values if covariates else np.empty((n, 0))
    ones = np.ones((n, 1))
    Xa = np.column_stack([ones, data[x].values, cov])                   # M ~ X
    Xb = np.column_stack([ones, data[x].values, data[m].values, cov])   # Y ~ X + M
    Xc = np.column_stack([ones, data[x].values, cov])                   # Y ~ X

    path_a = sm.OLS(data[m].values, Xa).fit()
    path_b = sm.OLS(data[y].values, Xb).fit()
    path_c = sm.OLS(data[y].values, Xc).fit()
    a, b = path_a.params[1], path_b.params[2]
    c_prime, c_total = path_b.params[1], path_c.params[1]
    indirect = a * b

    rng = np.random.default_rng(seed)
    yv, mv = data[y].values, data[m].values
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a_i = _ols(mv[idx], Xa[idx])[1]
        b_i = _ols(yv[idx], Xb[idx])[2]
        boots[i] = a_i * b_i
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])

    # Sobel test (normal-theory; shown only for comparison)
    se_a, se_b = path_a.bse[1], path_b.bse[2]
    sobel_se = np.sqrt(b ** 2 * se_a ** 2 + a ** 2 * se_b ** 2)
    from scipy import stats
    sobel_z = indirect / sobel_se if sobel_se > 0 else np.nan
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    paths = pd.DataFrame([
        {"Path": f"a: {x} -> {m}", "Estimate": a, "SE": se_a, "p": path_a.pvalues[1]},
        {"Path": f"b: {m} -> {y} (controlling {x})", "Estimate": b, "SE": se_b, "p": path_b.pvalues[2]},
        {"Path": f"c: total effect {x} -> {y}", "Estimate": c_total, "SE": path_c.bse[1], "p": path_c.pvalues[1]},
        {"Path": f"c': direct effect {x} -> {y}", "Estimate": c_prime, "SE": path_b.bse[1], "p": path_b.pvalues[1]},
    ]).round(4)
    ind = pd.DataFrame([{
        "Indirect effect (a x b)": round(indirect, 4), "Bootstrap SE": round(boots.std(ddof=1), 4),
        f"{int(ci * 100)}% CI lower (percentile)": round(lo, 4), f"{int(ci * 100)}% CI upper (percentile)": round(hi, 4),
        "Bootstrap samples": n_boot, "Seed": seed, "Sobel z": round(sobel_z, 3), "Sobel p": round(sobel_p, 4),
    }])
    res = AnalysisResult(
        f"Mediation: {x} -> {m} -> {y}",
        method=(f"Simple mediation with OLS regressions (unstandardised coefficients), N = {n}"
                + (f", covariates: {', '.join(covariates)}" if covariates else "")
                + f". Indirect effect tested with a percentile bootstrap ({n_boot} resamples, seed {seed})."),
        tables={"Path coefficients": paths, "Indirect effect": ind},
    )
    excludes_zero = lo > 0 or hi < 0
    res.text.append(f"The indirect effect of {x} on {y} through {m} is {indirect:.3f}, {int(ci * 100)}% bootstrap CI [{lo:.3f}, {hi:.3f}]. "
                    + ("The interval does not include zero, which is consistent with an indirect effect in this sample."
                       if excludes_zero else "The interval includes zero, so an indirect effect is not supported in this sample."))
    res.text.append(f"Direct effect c' = {c_prime:.3f} (p {fmt_p(path_b.pvalues[1])}); total effect c = {c_total:.3f} (p {fmt_p(path_c.pvalues[1])}).")
    if excludes_zero:
        if path_b.pvalues[1] >= 0.05:
            res.text.append("The direct effect is not significant at .05; some authors call this pattern 'full' (indirect-only) mediation. A non-significant result does not prove the direct effect is zero.")
        else:
            res.text.append("The direct effect is also significant at .05 ('partial' / complementary or competitive mediation, depending on the signs).")
    res.add_warning("Mediation with cross-sectional data cannot establish causal order. The model assumes X causes M and M causes Y, "
                    "and that there are no unmeasured confounders.")
    res.raw = {"boot": boots}
    return res
