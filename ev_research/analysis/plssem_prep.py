"""PLS-SEM *preparation and reporting support*.

This module does NOT estimate a PLS-SEM model.  No validated PLS algorithm is
implemented here.  What it does:

1. Builds a measurement-model specification (construct -> items) and a
   structural-model specification (paths).
2. Screens the indicators: missing values, variance, item correlations.
3. Computes HTMT (heterotrait-monotrait ratio), which is defined directly on
   the item correlation matrix and therefore does not depend on PLS
   estimation.
4. Computes *approximate* composite reliability (CR) and AVE from a
   one-factor common-factor solution per construct.  These are NOT the PLS
   outer-loading-based values that SmartPLS/seminr report and must not be
   reported as PLS results.
5. Estimates minimum sample size with the inverse square-root method.
6. Exports a clean data file for SmartPLS and an R script template for the
   'seminr' package, plus a checklist of what to report.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from ev_research.analysis.data_io import numeric_columns
from ev_research.analysis.efa import one_factor_loadings
from ev_research.analysis.reliability import cronbach_alpha
from ev_research.analysis.result import AnalysisError, AnalysisResult

NOT_ESTIMATED = ("This application does not estimate PLS-SEM. Estimate the model in SmartPLS, "
                 "R (seminr / cSEM) or ADANCO and report those outputs.")


def validate_spec(df: pd.DataFrame, constructs: dict[str, list[str]], paths: list[tuple[str, str]]) -> list[str]:
    """Return a list of problems (empty if the specification is usable)."""
    problems = []
    if len(constructs) < 2:
        problems.append("Define at least two constructs.")
    seen: dict[str, str] = {}
    for name, items in constructs.items():
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
            problems.append(f"Construct name '{name}' should start with a letter and contain only letters, digits and underscores.")
        if not items:
            problems.append(f"Construct '{name}' has no items.")
        for it in items:
            if it not in df.columns:
                problems.append(f"Item '{it}' (construct {name}) is not in the dataset.")
            elif it not in numeric_columns(df):
                problems.append(f"Item '{it}' (construct {name}) is not numeric.")
            if it in seen:
                problems.append(f"Item '{it}' is assigned to both {seen[it]} and {name}.")
            seen[it] = name
    for a, b in paths:
        if a not in constructs or b not in constructs:
            problems.append(f"Path {a} -> {b} uses an undefined construct.")
        if a == b:
            problems.append(f"Path {a} -> {b} points to itself.")
    if paths and _has_cycle(paths):
        problems.append("The structural model contains a loop (cycle). PLS-SEM requires a recursive model.")
    return problems


def _has_cycle(paths: list[tuple[str, str]]) -> bool:
    graph: dict[str, list[str]] = {}
    for a, b in paths:
        graph.setdefault(a, []).append(b)
    state: dict[str, int] = {}

    def visit(n):
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        if any(visit(m) for m in graph.get(n, [])):
            return True
        state[n] = 2
        return False
    return any(visit(n) for n in list(graph))


def htmt(df: pd.DataFrame, constructs: dict[str, list[str]]) -> pd.DataFrame:
    """Heterotrait-monotrait ratio of correlations (Henseler, Ringle &
    Sarstedt, 2015) computed from the Pearson item correlation matrix
    (complete cases on all items).

    HTMT(i, j) = mean(|r| between items of i and items of j)
                 / sqrt(mean(|r| among items of i) * mean(|r| among items of j))
    A construct with a single item has no monotrait correlations; its HTMT is
    not defined.
    """
    items = [it for v in constructs.values() for it in v]
    R = df[items].dropna().corr().abs()
    names = list(constructs)
    out = pd.DataFrame(np.nan, index=names, columns=names)

    def mono(c):
        its = constructs[c]
        if len(its) < 2:
            return np.nan
        sub = R.loc[its, its].values
        return sub[np.triu_indices(len(its), 1)].mean()
    mono_vals = {c: mono(c) for c in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            hetero = R.loc[constructs[a], constructs[b]].values.mean()
            denom = np.sqrt(mono_vals[a] * mono_vals[b])
            out.loc[b, a] = hetero / denom if denom and not np.isnan(denom) else np.nan
    return out


def inverse_square_root_n(min_path: float, alpha: float = 0.05) -> int:
    """Minimum sample size by the inverse square-root method
    (Kock & Hadaya, 2018): n_min > (z / |p_min|)^2, with z = 2.486 for
    alpha = .05 (one-tailed, power .80), 3.168 for alpha = .01."""
    if not 0 < abs(min_path) < 1:
        raise AnalysisError("The expected minimum path coefficient must be between 0 and 1 (e.g. 0.15).")
    z = {0.05: 2.486, 0.01: 3.168, 0.10: 2.124}.get(alpha)
    if z is None:
        raise AnalysisError("alpha must be 0.10, 0.05 or 0.01.")
    return int(np.ceil((z / abs(min_path)) ** 2))


def prepare(df: pd.DataFrame, constructs: dict[str, list[str]], paths: list[tuple[str, str]],
            expected_min_path: float = 0.15) -> AnalysisResult:
    problems = validate_spec(df, constructs, paths)
    if problems:
        raise AnalysisError("Fix the model specification first:\n- " + "\n- ".join(problems))
    items = [it for v in constructs.values() for it in v]
    complete = df[items].dropna()
    res = AnalysisResult("PLS-SEM preparation (model NOT estimated)", performed_by_app=False,
                         method="Specification checks, indicator screening, HTMT from item correlations, and approximate reliability/convergent-validity indices. " + NOT_ESTIMATED)
    res.add_warning(NOT_ESTIMATED)

    # Specification tables
    res.tables["Measurement model specification"] = pd.DataFrame(
        [{"Construct": c, "Items": ", ".join(v), "Number of items": len(v), "Assumed mode": "Reflective (Mode A)"} for c, v in constructs.items()])
    if paths:
        res.tables["Structural model specification"] = pd.DataFrame([{"From": a, "To": b} for a, b in paths])

    # Indicator screening
    rows = []
    for c, its in constructs.items():
        for it in its:
            s = df[it]
            rows.append({"Construct": c, "Item": it, "Missing %": round(s.isna().mean() * 100, 2),
                         "Mean": round(s.mean(), 3), "SD": round(s.std(ddof=1), 3),
                         "Min": s.min(), "Max": s.max()})
    screen = pd.DataFrame(rows)
    res.tables["Indicator screening"] = screen
    if (screen["SD"] == 0).any():
        res.add_warning(f"Items with zero variance: {screen.loc[screen['SD'] == 0, 'Item'].tolist()}. SmartPLS cannot use them.")
    if (screen["Missing %"] > 5).any():
        res.add_warning("Some items have more than 5% missing values. Decide on a missing-value treatment before exporting.")

    # Approximate reliability / validity
    rel = []
    for c, its in constructs.items():
        row = {"Construct": c, "Items": len(its)}
        sub = complete[its]
        if len(its) >= 2:
            row["Cronbach's alpha"] = round(cronbach_alpha(sub), 3)
            try:
                lam = one_factor_loadings(complete, its).clip(upper=0.999)
                cr = lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum())
                ave = (lam ** 2).mean()
                row["Approx. CR (factor-based)"] = round(float(cr), 3)
                row["Approx. AVE (factor-based)"] = round(float(ave), 3)
                row["Min. loading (factor-based)"] = round(float(lam.min()), 3)
            except Exception as exc:  # pragma: no cover - numerical failure
                row["Note"] = f"Loadings could not be computed: {exc}"
        else:
            row["Note"] = "Single-item construct: reliability cannot be assessed."
        rel.append(row)
    res.tables["Approximate reliability and convergent validity (NOT PLS results)"] = pd.DataFrame(rel)
    res.text.append("CR and AVE above come from a one-factor common-factor model per construct. PLS-SEM uses composite-based "
                    "outer loadings, which are usually higher, so these values are only a screening aid. Report the CR, AVE and "
                    "loadings produced by your PLS software.")

    # HTMT
    h = htmt(df, constructs)
    res.tables["HTMT (from item correlations)"] = h.round(3).reset_index().rename(columns={"index": ""})
    res.raw["htmt"] = h
    hi = [(r, c, h.loc[r, c]) for r in h.index for c in h.columns if pd.notna(h.loc[r, c]) and h.loc[r, c] > 0.85]
    res.text.append("HTMT is calculated directly from item correlations, the same quantity PLS software reports; "
                    "the bootstrap confidence intervals for HTMT are not computed here.")
    if hi:
        res.add_warning("HTMT above .85 (conservative threshold; .90 is used for conceptually similar constructs) suggests weak discriminant validity: "
                        + "; ".join(f"{a}-{b} = {v:.2f}" for a, b, v in hi))

    # Sample size
    n_min = inverse_square_root_n(expected_min_path)
    max_arrows = max([sum(1 for a, b in paths if b == c) for c in constructs] + [0])
    max_items = max(len(v) for v in constructs.values())
    res.tables["Sample size guidance"] = pd.DataFrame([
        {"Criterion": f"Inverse square-root method (min. path = {expected_min_path}, alpha = .05, power = .80)", "Minimum N": n_min},
        {"Criterion": "'10-times rule' (10 x max. arrows pointing at a construct or max. items) - widely criticised; shown for reference only",
         "Minimum N": 10 * max(max_arrows, max_items)},
        {"Criterion": "Complete cases available on all items", "Minimum N": len(complete)},
    ])
    if len(complete) < n_min:
        res.add_warning(f"Only {len(complete)} complete cases; the inverse square-root method suggests at least {n_min} to detect path coefficients of {expected_min_path}.")

    res.tables["What to obtain and report from your PLS software"] = reporting_checklist()
    return res


def reporting_checklist() -> pd.DataFrame:
    rows = [
        ("Measurement model", "Indicator (outer) loadings", "Commonly >= .708; consider removing items .40-.70 only if CR/AVE improve and content validity is kept"),
        ("Measurement model", "Cronbach's alpha, rho_A, composite reliability (rho_C)", "Commonly .70-.95"),
        ("Measurement model", "Average variance extracted (AVE)", "Commonly >= .50"),
        ("Measurement model", "HTMT with bootstrap confidence intervals", "< .85 / .90; CI should not include 1"),
        ("Structural model", "Collinearity (inner VIF)", "Commonly < 3 (ideally), < 5"),
        ("Structural model", "Path coefficients with bootstrap t-values, p-values and CIs", "Use e.g. 5,000 or 10,000 bootstrap samples"),
        ("Structural model", "R-squared (and adjusted) of endogenous constructs", "Interpret relative to the research context"),
        ("Structural model", "f-squared effect sizes", ".02 / .15 / .35 conventionally small / medium / large"),
        ("Structural model", "Q-squared / PLSpredict", "Predictive relevance / out-of-sample prediction"),
        ("Mediation (if any)", "Specific indirect effects with bootstrap CIs", "CI excluding zero supports mediation"),
        ("Robustness (optional)", "Common method bias (e.g. full collinearity VIF), unobserved heterogeneity, MGA with MICOM", "As required by your design"),
    ]
    return pd.DataFrame(rows, columns=["Part", "Statistic", "Common guideline (verify for your field)"])


def export_smartpls_csv(df: pd.DataFrame, constructs: dict[str, list[str]], extra: list[str] | None = None,
                        missing: str = "listwise") -> tuple[bytes, str]:
    """CSV with only the item columns (and optional extra columns such as
    grouping variables). SmartPLS reads CSV with a header row; missing
    values are written as -99 when missing='code', or rows are removed when
    missing='listwise'."""
    items = [it for v in constructs.values() for it in v]
    cols = items + [c for c in (extra or []) if c not in items]
    out = df[cols].copy()
    if missing == "listwise":
        before = len(out)
        out = out.dropna(subset=items)
        note = f"Listwise deletion: {before - len(out)} rows removed, {len(out)} rows exported."
    elif missing == "code":
        out = out.fillna(-99)
        note = "Missing values written as -99. Declare -99 as the missing-value marker when importing into SmartPLS."
    else:
        raise AnalysisError("missing must be 'listwise' or 'code'.")
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8"), note


def seminr_script(constructs: dict[str, list[str]], paths: list[tuple[str, str]], data_file: str = "pls_data.csv",
                  nboot: int = 5000) -> str:
    """R script template for the 'seminr' package. NOT executed or tested by
    this application; check it against the seminr documentation."""
    def q(xs):
        return ", ".join(f'"{x}"' for x in xs)
    mm = ",\n  ".join(f'composite("{c}", c({q(its)}))' for c, its in constructs.items())
    targets: dict[str, list[str]] = {}
    for a, b in paths:
        targets.setdefault(b, []).append(a)
    sm = ",\n  ".join(f"paths(from = c({q(src)}), to = \"{dst}\")" for dst, src in targets.items())
    return f"""# PLS-SEM script template generated by the EV Purchase Intention Research Assistant.
# This script was NOT run by the application. Check it against the seminr
# documentation (https://cran.r-project.org/package=seminr) before use.
# install.packages("seminr")
library(seminr)

data <- read.csv("{data_file}")

measurement_model <- constructs(
  {mm}
)

structural_model <- relationships(
  {sm if sm else '# add paths, e.g. paths(from = c("ATT", "SN"), to = "PI")'}
)

pls_model <- estimate_pls(data = data,
                          measurement_model = measurement_model,
                          structural_model = structural_model)
summary_pls <- summary(pls_model)
summary_pls$loadings        # outer loadings
summary_pls$reliability     # alpha, rhoC, AVE, rhoA
summary_pls$validity$htmt   # HTMT
summary_pls$paths           # path coefficients and R-squared

boot_model <- bootstrap_model(seminr_model = pls_model, nboot = {nboot}, seed = 123)
summary(boot_model)          # bootstrapped paths, CIs
"""
