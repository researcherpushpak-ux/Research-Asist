"""Loading, validating and inspecting questionnaire datasets."""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from ev_research.analysis.result import AnalysisError, AnalysisResult

MAX_FILE_MB = 200


def load_dataset(data: bytes, filename: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame with clear error messages."""
    if not data:
        raise AnalysisError(f"'{filename}' is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise AnalysisError(f"'{filename}' is larger than {MAX_FILE_MB} MB.")
    name = filename.lower()
    try:
        if name.endswith(".csv") or name.endswith(".txt"):
            df = None
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise AnalysisError("Could not decode the CSV file. Save it as UTF-8 and try again.")
        elif name.endswith((".xlsx", ".xlsm", ".xls")):
            df = pd.read_excel(io.BytesIO(data), sheet_name=sheet_name)
        else:
            raise AnalysisError("Unsupported file type. Upload a .csv, .xlsx or .xls file.")
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError(f"Could not read '{filename}': {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise AnalysisError(f"'{filename}' contains no data rows.")
    # Tidy column names
    df.columns = [str(c).strip() for c in df.columns]
    dupes = [c for c in df.columns[df.columns.duplicated()]]
    if dupes:
        raise AnalysisError(f"Duplicate column names found: {dupes}. Rename them and upload again.")
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if unnamed and df[unnamed].isna().all().all():
        df = df.drop(columns=unnamed)
    df = df.dropna(how="all")          # completely empty rows
    return df


def excel_sheet_names(data: bytes) -> list[str]:
    try:
        return list(pd.ExcelFile(io.BytesIO(data)).sheet_names)
    except Exception as exc:
        raise AnalysisError(f"Could not open the Excel file: {exc}") from exc


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_levels: int = 20) -> list[str]:
    out = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if not pd.api.types.is_numeric_dtype(df[c]) or nun <= 7:
            if 1 < nun <= max_levels:
                out.append(c)
    return out


def likely_id_column(s: pd.Series) -> bool:
    return bool(re.search(r"(^id$|_id$|^id_|respondent|serial|timestamp)", str(s.name), re.I)) or (
        pd.api.types.is_integer_dtype(s) and s.is_unique and len(s) > 20)


def analysis_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns excluding ID-like columns (used for default selections)."""
    return [c for c in numeric_columns(df) if not likely_id_column(df[c])]


def detect_likert_columns(df: pd.DataFrame, max_points: int = 11) -> list[str]:
    """Numeric columns with few integer values (e.g. 1-5, 1-7) that are not IDs."""
    out = []
    for c in numeric_columns(df):
        s = df[c].dropna()
        if s.empty or likely_id_column(df[c]):
            continue
        if np.allclose(s, s.round()) and s.nunique() <= max_points and s.min() >= 0 and s.max() <= 11:
            out.append(c)
    return out


def detect_item_groups(columns: list[str]) -> dict[str, list[str]]:
    """Group item columns by alphabetic prefix: ATT1, ATT2 -> 'ATT'.
    Also handles ATT_1, ATT.1 and 'ATT 1'. Groups with one item are dropped."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        m = re.match(r"^([A-Za-z][A-Za-z_]*?)[_\.\s-]?(\d+)$", str(c))
        if m:
            groups.setdefault(m.group(1).rstrip("_"), []).append(c)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def inspect(df: pd.DataFrame) -> AnalysisResult:
    res = AnalysisResult("Data inspection", method="Overview of variables, data types and missing values.")
    n, k = df.shape
    overview = pd.DataFrame({
        "Variable": df.columns,
        "Type": ["numeric" if c in numeric_columns(df) else "text/categorical" for c in df.columns],
        "Non-missing": df.notna().sum().values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Distinct values": df.nunique(dropna=True).values,
        "Example values": [", ".join(map(str, df[c].dropna().unique()[:4])) for c in df.columns],
    })
    res.tables["Variables"] = overview
    complete = int(df.dropna().shape[0])
    dup_rows = int(df.duplicated().sum())
    res.text.append(f"The dataset has {n} rows (responses) and {k} columns (variables).")
    res.text.append(f"{complete} rows ({complete / n * 100:.1f}%) have no missing values on any variable.")
    total_missing = int(df.isna().sum().sum())
    res.text.append(f"There are {total_missing} missing cells ({total_missing / (n * k) * 100:.2f}% of all cells).")
    if dup_rows:
        res.add_warning(f"{dup_rows} rows are exact duplicates of another row. Check whether these are double submissions.")
    high = overview.loc[overview["Missing %"] > 10, "Variable"].tolist()
    if high:
        res.add_warning(f"Variables with more than 10% missing values: {', '.join(high)}. Results using them may be biased.")
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        res.add_warning(f"Variables with only one value (no variation): {', '.join(const)}.")
    likert = detect_likert_columns(df)
    if likert:
        # Straight-lining: same answer on every Likert item
        sub = df[likert].dropna()
        if len(likert) >= 5 and len(sub):
            straight = int((sub.nunique(axis=1) == 1).sum())
            res.text.append(f"{straight} complete responses gave the same answer to all {len(likert)} rating-scale items (possible 'straight-lining').")
    # Missing-data pattern per row
    per_row = df.isna().sum(axis=1).value_counts().sort_index()
    res.tables["Missing values per row"] = pd.DataFrame({"Missing values in row": per_row.index, "Number of rows": per_row.values})
    return res


def little_mcar_note() -> str:
    return ("The application does not test whether data are Missing Completely At Random (e.g. Little's MCAR test). "
            "Listwise deletion is the default; it is unbiased only if data are MCAR.")


def handle_missing(df: pd.DataFrame, columns: list[str], method: str) -> tuple[pd.DataFrame, str]:
    """Return a copy with missing values handled for `columns`.

    method: 'listwise' (drop rows with any missing in columns),
            'mean' (replace with column mean; reduces variance - use with care),
            'median', or 'none'.
    """
    out = df.copy()
    before = len(out)
    if method == "listwise":
        out = out.dropna(subset=columns)
        note = f"Listwise deletion removed {before - len(out)} of {before} rows."
    elif method in ("mean", "median"):
        nums = [c for c in columns if c in numeric_columns(out)]
        filled = int(out[nums].isna().sum().sum())
        fill = out[nums].mean() if method == "mean" else out[nums].median()
        out[nums] = out[nums].fillna(fill)
        note = (f"{filled} missing values in {len(nums)} numeric variables were replaced by the variable {method}. "
                "Single imputation underestimates variability; report this choice.")
    elif method == "none":
        note = "Missing values were left in place; each analysis drops incomplete cases for the variables it uses."
    else:
        raise AnalysisError(f"Unknown missing-value method: {method}")
    return out, note


def reverse_code(df: pd.DataFrame, columns: list[str], scale_min: int, scale_max: int) -> pd.DataFrame:
    if scale_max <= scale_min:
        raise AnalysisError("The scale maximum must be larger than the minimum.")
    out = df.copy()
    for c in columns:
        if c not in numeric_columns(out):
            raise AnalysisError(f"'{c}' is not numeric and cannot be reverse-coded.")
        s = out[c]
        if (s.dropna() < scale_min).any() or (s.dropna() > scale_max).any():
            raise AnalysisError(f"'{c}' has values outside {scale_min}-{scale_max}.")
        out[c] = scale_max + scale_min - s
    return out


def composite_scores(df: pd.DataFrame, groups: dict[str, list[str]], min_items_fraction: float = 0.5) -> pd.DataFrame:
    """Mean score per construct. A respondent gets a score only if they
    answered at least `min_items_fraction` of the construct's items."""
    out = pd.DataFrame(index=df.index)
    for name, items in groups.items():
        sub = df[items]
        answered = sub.notna().sum(axis=1)
        score = sub.mean(axis=1)
        score[answered < np.ceil(len(items) * min_items_fraction)] = np.nan
        out[name] = score
    return out
