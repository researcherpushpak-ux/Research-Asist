"""Exploratory factor analysis (EFA) using the factor_analyzer package."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_research.analysis.data_io import numeric_columns
from ev_research.analysis.result import AnalysisError, AnalysisResult


_SHIM_APPLIED = False


def _require_fa():
    """Import factor_analyzer and, if needed, patch a known incompatibility:
    factor_analyzer 0.5.1 calls sklearn's check_array(force_all_finite=...),
    an argument that newer scikit-learn versions renamed to
    ensure_all_finite.  The shim only renames that argument."""
    global _SHIM_APPLIED
    try:
        import factor_analyzer.factor_analyzer as fa_mod
    except ImportError as exc:  # pragma: no cover
        raise AnalysisError("The 'factor_analyzer' package is not installed. Run: pip install factor_analyzer") from exc
    if _SHIM_APPLIED:
        return
    import inspect

    from sklearn.utils import validation
    params = inspect.signature(validation.check_array).parameters
    if "force_all_finite" not in params and "ensure_all_finite" in params:
        original = validation.check_array

        def check_array_compat(*args, **kwargs):
            if "force_all_finite" in kwargs:
                kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
            return original(*args, **kwargs)
        fa_mod.check_array = check_array_compat
    _SHIM_APPLIED = True


def suitability(df: pd.DataFrame, items: list[str]) -> AnalysisResult:
    """KMO and Bartlett's test of sphericity, plus eigenvalues."""
    _require_fa()
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

    sub = _prepare(df, items)
    chi2, p = calculate_bartlett_sphericity(sub)
    kmo_per, kmo_total = calculate_kmo(sub)
    eig = np.linalg.eigvalsh(sub.corr().values)[::-1]
    res = AnalysisResult("EFA suitability checks", method="Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy; Bartlett's test of sphericity; eigenvalues of the correlation matrix.")
    res.tables["KMO per item"] = pd.DataFrame({"Item": items, "KMO": np.round(kmo_per, 3)})
    res.tables["Eigenvalues"] = pd.DataFrame({"Component": np.arange(1, len(eig) + 1), "Eigenvalue": np.round(eig, 3),
                                              "Cumulative % variance": np.round(np.cumsum(eig) / eig.sum() * 100, 2)})
    res.tables["Suitability summary"] = pd.DataFrame([{"N": len(sub), "Items": len(items), "Overall KMO": round(kmo_total, 3),
                                                       "Bartlett chi-square": round(chi2, 2),
                                                       "df": len(items) * (len(items) - 1) // 2, "Bartlett p": p}])
    res.text.append(f"Overall KMO = {kmo_total:.3f}; Bartlett's chi-square = {chi2:.2f}, p {'< .001' if p < 0.001 else f'= {p:.3f}'}.")
    n_kaiser = int((eig > 1).sum())
    res.text.append(f"{n_kaiser} eigenvalue(s) exceed 1 (Kaiser criterion). The Kaiser rule often over-extracts; also use the scree plot and theory.")
    if kmo_total < 0.6:
        res.add_warning("Overall KMO is below .60: the data may be unsuitable for factor analysis.")
    if p >= 0.05:
        res.add_warning("Bartlett's test is not significant: the correlations may be too weak for factor analysis.")
    low = [i for i, v in zip(items, kmo_per) if v < 0.5]
    if low:
        res.add_warning(f"Items with KMO below .50: {', '.join(low)}. Consider removing them.")
    _sample_warning(res, len(sub), len(items))
    return res


def _prepare(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    if len(items) < 3:
        raise AnalysisError("Select at least 3 items for factor analysis.")
    bad = [c for c in items if c not in numeric_columns(df)]
    if bad:
        raise AnalysisError(f"Non-numeric items: {bad}.")
    sub = df[items].dropna()
    if len(sub) < len(items) + 2:
        raise AnalysisError(f"Only {len(sub)} complete responses for {len(items)} items. More data are needed.")
    zero = [c for c in items if sub[c].std() == 0]
    if zero:
        raise AnalysisError(f"Items with no variation: {zero}. Remove them.")
    if np.linalg.matrix_rank(sub.corr().values) < len(items):
        raise AnalysisError("The correlation matrix is singular (some items are perfectly correlated or are combinations of others). Remove duplicate or total-score columns.")
    return sub


def _sample_warning(res: AnalysisResult, n: int, k: int) -> None:
    if n < 100:
        res.add_warning(f"N = {n} is small for factor analysis (commonly recommended: at least 100-200, or 5-10 respondents per item).")
    elif n / k < 5:
        res.add_warning(f"Only {n / k:.1f} respondents per item (a common rule of thumb asks for at least 5).")


def run_efa(df: pd.DataFrame, items: list[str], n_factors: int, rotation: str = "oblimin",
            method: str = "minres", loading_cutoff: float = 0.4) -> AnalysisResult:
    _require_fa()
    from factor_analyzer import FactorAnalyzer

    sub = _prepare(df, items)
    if not 1 <= n_factors < len(items):
        raise AnalysisError(f"Number of factors must be between 1 and {len(items) - 1}.")
    rot = None if rotation in (None, "none") or n_factors == 1 else rotation
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rot, method=method)
    try:
        fa.fit(sub.values)
    except Exception as exc:
        raise AnalysisError(f"Factor analysis failed: {exc}. Try fewer factors or another extraction method.") from exc
    cols = [f"Factor {i + 1}" for i in range(n_factors)]
    load = pd.DataFrame(fa.loadings_, index=items, columns=cols).round(3)
    load["Communality"] = np.round(fa.get_communalities(), 3)
    load["Primary factor"] = load[cols].abs().idxmax(axis=1)
    load.loc[load[cols].abs().max(axis=1) < loading_cutoff, "Primary factor"] = f"none >= {loading_cutoff}"
    cross = (load[cols].abs() >= loading_cutoff).sum(axis=1) > 1
    load["Cross-loading"] = np.where(cross, "yes", "")
    var, prop, cum = fa.get_factor_variance()
    variance = pd.DataFrame({"Factor": cols, "SS loadings": np.round(var, 3),
                             "% variance": np.round(prop * 100, 2), "Cumulative %": np.round(cum * 100, 2)})
    res = AnalysisResult(
        "Exploratory factor analysis",
        method=f"EFA with {method} extraction, {rot or 'no'} rotation, {n_factors} factor(s), N = {len(sub)} complete cases. Loadings below {loading_cutoff} are treated as weak.",
        tables={"Factor loadings": load.reset_index().rename(columns={"index": "Item"}), "Variance explained": variance},
    )
    if rot in ("oblimin", "promax") and getattr(fa, "phi_", None) is not None:
        res.tables["Factor correlations"] = pd.DataFrame(fa.phi_, index=cols, columns=cols).round(3).reset_index().rename(columns={"index": ""})
    if rot in ("oblimin", "promax"):
        res.text.append("With oblique rotation the loadings shown are pattern coefficients; variance explained by factors overlaps.")
    res.text.append(f"The {n_factors} factor(s) account for {cum[-1] * 100:.1f}% of the item variance (common-factor solution).")
    weak = load.index[load[cols].abs().max(axis=1) < loading_cutoff].tolist()
    if weak:
        res.add_warning(f"Items without any loading >= {loading_cutoff}: {', '.join(weak)}.")
    if cross.any():
        res.add_warning(f"Items loading >= {loading_cutoff} on more than one factor: {', '.join(load.index[cross])}.")
    low_comm = load.index[load["Communality"] < 0.3].tolist()
    if low_comm:
        res.add_warning(f"Items with communality below .30: {', '.join(low_comm)}.")
    if (load[cols].abs() > 1).any().any():
        res.add_warning("A loading above 1 (Heywood case) was found; the solution may be improper. Try fewer factors.")
    _sample_warning(res, len(sub), len(items))
    res.text.append("EFA is exploratory: the factor structure should be confirmed (e.g. CFA or the PLS-SEM measurement model) preferably on new data.")
    return res


def one_factor_loadings(df: pd.DataFrame, items: list[str]) -> pd.Series:
    """Standardised loadings of a single-factor ML solution (used for the
    approximate CR/AVE in PLS-SEM preparation)."""
    _require_fa()
    from factor_analyzer import FactorAnalyzer

    sub = df[items].dropna()
    if len(items) < 2:
        raise AnalysisError("At least 2 items are needed.")
    if len(items) == 2:
        # One-factor model with 2 items is not identified; use sqrt(r) for both.
        r = sub.corr().iloc[0, 1]
        val = np.sqrt(r) if r > 0 else np.nan
        return pd.Series([val, val], index=items)
    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    fa.fit(sub.values)
    load = pd.Series(fa.loadings_[:, 0], index=items)
    if load.sum() < 0:
        load = -load
    return load
