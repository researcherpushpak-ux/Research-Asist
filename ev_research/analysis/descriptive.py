"""Descriptive statistics and frequency tables."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from ev_research.analysis.data_io import numeric_columns
from ev_research.analysis.result import AnalysisError, AnalysisResult


def descriptives(df: pd.DataFrame, columns: list[str]) -> AnalysisResult:
    if not columns:
        raise AnalysisError("Select at least one numeric variable.")
    bad = [c for c in columns if c not in numeric_columns(df)]
    if bad:
        raise AnalysisError(f"These variables are not numeric: {bad}. Use frequency tables for categorical variables.")
    rows = []
    for c in columns:
        s = df[c].dropna()
        rows.append({
            "Variable": c, "N": int(s.count()), "Missing": int(df[c].isna().sum()),
            "Mean": s.mean(), "SD": s.std(ddof=1), "Median": s.median(),
            "Min": s.min(), "Max": s.max(),
            "Skewness": stats.skew(s, bias=False) if len(s) > 2 else float("nan"),
            "Excess kurtosis": stats.kurtosis(s, bias=False) if len(s) > 3 else float("nan"),
        })
    table = pd.DataFrame(rows).round(3)
    res = AnalysisResult("Descriptive statistics", method="Mean, standard deviation (n-1), median, range, skewness and excess kurtosis (bias-corrected). Missing values excluded per variable.",
                         tables={"Descriptive statistics": table})
    nonnormal = table.loc[(table["Skewness"].abs() > 2) | (table["Excess kurtosis"].abs() > 7), "Variable"].tolist()
    if nonnormal:
        res.add_warning(f"Marked non-normality (|skew| > 2 or |excess kurtosis| > 7, a common rule of thumb): {', '.join(nonnormal)}.")
    return res


def frequencies(df: pd.DataFrame, column: str, max_levels: int = 50) -> AnalysisResult:
    if column not in df.columns:
        raise AnalysisError(f"Column '{column}' not found.")
    s = df[column]
    if s.nunique(dropna=True) > max_levels:
        raise AnalysisError(f"'{column}' has more than {max_levels} distinct values; a frequency table is not useful. Use descriptive statistics or a histogram.")
    counts = s.value_counts(dropna=False).sort_index(key=lambda idx: idx.map(lambda v: (pd.isna(v), str(v))))
    n_total = len(s)
    n_valid = int(s.notna().sum())
    table = pd.DataFrame({
        "Value": ["Missing" if pd.isna(v) else v for v in counts.index],
        "Count": counts.values,
        "Percent": (counts.values / n_total * 100).round(2),
    })
    table["Valid percent"] = [
        round(c / n_valid * 100, 2) if not pd.isna(v) and n_valid else None for v, c in zip(counts.index, counts.values)
    ]
    valid = table[table["Value"] != "Missing"].copy()
    valid["Cumulative valid percent"] = valid["Valid percent"].cumsum().round(2)
    table = table.merge(valid[["Value", "Cumulative valid percent"]], on="Value", how="left")
    return AnalysisResult(f"Frequency distribution: {column}", method="Counts and percentages; 'valid percent' excludes missing values.",
                          tables={f"Frequencies - {column}": table},
                          text=[f"'{column}': {n_valid} valid responses, {n_total - n_valid} missing."])
