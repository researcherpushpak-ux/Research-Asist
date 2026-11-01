"""Internal-consistency reliability (Cronbach's alpha)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_research.analysis.data_io import numeric_columns
from ev_research.analysis.result import AnalysisError, AnalysisResult


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha for complete cases:
    alpha = k/(k-1) * (1 - sum(item variances) / variance(total score))."""
    items = items.dropna()
    k = items.shape[1]
    if k < 2:
        raise AnalysisError("Cronbach's alpha needs at least 2 items.")
    if len(items) < 3:
        raise AnalysisError("Too few complete responses to compute Cronbach's alpha.")
    item_var = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        raise AnalysisError("The total score has zero variance; alpha cannot be computed.")
    return float(k / (k - 1) * (1 - item_var.sum() / total_var))


def alpha_label(a: float) -> str:
    """Conventional descriptive labels (rule of thumb, not a statistical test)."""
    if a >= 0.9:
        return "excellent (>= .90; very high values can indicate redundant items)"
    if a >= 0.8:
        return "good (.80-.89)"
    if a >= 0.7:
        return "acceptable (.70-.79)"
    if a >= 0.6:
        return "questionable (.60-.69)"
    if a >= 0.5:
        return "poor (.50-.59)"
    return "unacceptable (< .50)"


def reliability_analysis(df: pd.DataFrame, items: list[str], scale_name: str = "Scale") -> AnalysisResult:
    if len(items) < 2:
        raise AnalysisError("Select at least two items for the scale.")
    bad = [c for c in items if c not in numeric_columns(df)]
    if bad:
        raise AnalysisError(f"Non-numeric items: {bad}. Recode text answers (e.g. 'Agree') into numbers first.")
    sub = df[items].dropna()
    n_dropped = len(df) - len(sub)
    alpha = cronbach_alpha(sub)
    total = sub.sum(axis=1)
    rows = []
    for c in items:
        rest = total - sub[c]
        r_it = np.corrcoef(sub[c], rest)[0, 1] if sub[c].std() > 0 and rest.std() > 0 else np.nan
        a_del = cronbach_alpha(sub.drop(columns=c)) if len(items) > 2 else np.nan
        rows.append({"Item": c, "Mean": sub[c].mean(), "SD": sub[c].std(ddof=1),
                     "Corrected item-total r": r_it, "Alpha if item deleted": a_del})
    table = pd.DataFrame(rows).round(3)
    res = AnalysisResult(
        f"Reliability: {scale_name}",
        method="Cronbach's alpha on complete cases (listwise deletion), with corrected item-total correlations and alpha-if-item-deleted.",
        tables={f"Item statistics - {scale_name}": table,
                f"Reliability summary - {scale_name}": pd.DataFrame([{
                    "Scale": scale_name, "Items": len(items), "N (complete cases)": len(sub),
                    "Cronbach's alpha": round(alpha, 3), "Conventional label": alpha_label(alpha)}])},
    )
    res.text.append(f"{scale_name}: Cronbach's alpha = {alpha:.3f} ({len(items)} items, N = {len(sub)}), conventionally described as {alpha_label(alpha)}.")
    if n_dropped:
        res.text.append(f"{n_dropped} rows with missing values on these items were excluded.")
    low = table.loc[table["Corrected item-total r"] < 0.3, "Item"].tolist()
    if low:
        res.add_warning(f"Items with corrected item-total correlation below .30 (rule of thumb): {', '.join(low)}. Check wording or reverse-coding.")
    neg = table.loc[table["Corrected item-total r"] < 0, "Item"].tolist()
    if neg:
        res.add_warning(f"Negative item-total correlation for {', '.join(neg)}: the item may need reverse-coding.")
    if len(items) > 2:
        better = table.loc[table["Alpha if item deleted"] > alpha + 0.01, "Item"].tolist()
        if better:
            res.text.append(f"Alpha would increase if these items were removed: {', '.join(better)}. Do not drop items on this basis alone; consider content validity.")
    res.text.append("Alpha assumes the items measure one dimension with roughly equal loadings (tau-equivalence). It is not evidence of validity.")
    return res


def reliability_table(df: pd.DataFrame, groups: dict[str, list[str]]) -> AnalysisResult:
    """Alpha for several constructs at once."""
    rows, warnings = [], []
    for name, items in groups.items():
        try:
            sub = df[items].dropna()
            a = cronbach_alpha(sub)
            rows.append({"Construct": name, "Items": len(items), "N": len(sub),
                         "Cronbach's alpha": round(a, 3), "Conventional label": alpha_label(a)})
        except AnalysisError as exc:
            warnings.append(f"{name}: {exc}")
    res = AnalysisResult("Reliability of all constructs", method="Cronbach's alpha per construct (listwise deletion within construct).",
                         tables={"Reliability by construct": pd.DataFrame(rows)}, warnings=warnings)
    for r in rows:
        alpha_value = r["Cronbach's alpha"]
        res.text.append(f"{r['Construct']}: alpha = {alpha_value:.3f} ({r['Items']} items, N = {r['N']}).")
    return res
