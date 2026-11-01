"""A common container for analysis output so that every analysis can be shown
in the interface and added to the report in the same way."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class AnalysisError(Exception):
    """An analysis cannot be run with the given input. The message is written
    for the end user."""


@dataclass
class AnalysisResult:
    title: str
    method: str = ""                      # what was done, in plain language
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    text: list[str] = field(default_factory=list)       # neutral statements of results
    warnings: list[str] = field(default_factory=list)   # assumption problems etc.
    figures: dict[str, bytes] = field(default_factory=dict)   # PNG bytes
    performed_by_app: bool = True          # False => preparation only, not estimated here
    raw: dict = field(default_factory=dict)   # unrounded objects for charts (not exported)

    def add_warning(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)


def fmt_p(p: float) -> str:
    """APA-style p-value formatting."""
    if p is None or pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "< .001"
    return f"= {p:.3f}".replace("0.", ".", 1)


def sig_word(p: float, alpha: float = 0.05) -> str:
    if p is None or pd.isna(p):
        return "could not be tested"
    return "statistically significant" if p < alpha else "not statistically significant"
