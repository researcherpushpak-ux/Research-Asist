"""Static charts (matplotlib) for the interface and the report.

Each function returns a matplotlib Figure. Use ``to_png`` to get PNG bytes
for downloads and for embedding in the Word report.
"""

from __future__ import annotations

import io
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ev_research.analysis.result import AnalysisError  # noqa: E402
from ev_research.literature.matrix import split_list  # noqa: E402

# Palette (validated reference palette: categorical slot 1 + blue/red diverging pair)
PRIMARY = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def _style(ax, title: str, xlabel: str = "", ylabel: str = "", grid_axis: str = "y"):
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def to_png(fig, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _hbar(labels: list[str], values: list[float], title: str, xlabel: str, note: str = ""):
    if not labels:
        raise AnalysisError("Nothing to plot.")
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.38 * len(labels) + 1.2)))
    bars = ax.barh(labels, values, color=PRIMARY, height=0.6, edgecolor=SURFACE, linewidth=2)
    _style(ax, title, xlabel=xlabel, grid_axis="x")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(BASELINE)
    for b, v in zip(bars, values):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:g}", va="center", fontsize=9, color=INK_2)
    ax.xaxis.get_major_locator().set_params(integer=True)
    if note:
        fig.text(0.01, -0.02, note, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Survey data
# ---------------------------------------------------------------------------
def bar_chart(df: pd.DataFrame, column: str, percent: bool = False, max_levels: int = 25):
    if column not in df.columns:
        raise AnalysisError(f"Column '{column}' not found.")
    counts = df[column].dropna().astype(str).value_counts()
    if counts.empty:
        raise AnalysisError(f"'{column}' has no non-missing values.")
    if len(counts) > max_levels:
        raise AnalysisError(f"'{column}' has {len(counts)} categories; a bar chart would be unreadable. Use a histogram for numeric data.")
    # Keep natural order for numeric codes (e.g. Likert 1..5)
    try:
        counts = counts.sort_index(key=lambda idx: idx.astype(float))
    except (ValueError, TypeError):
        pass
    values = counts / counts.sum() * 100 if percent else counts
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(values) + 2), 4))
    bars = ax.bar(values.index, values.values, color=PRIMARY, width=0.65, edgecolor=SURFACE, linewidth=2)
    _style(ax, f"{column} (n = {int(counts.sum())})", ylabel="Percent of valid responses" if percent else "Count")
    for b, v in zip(bars, values.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}%" if percent else f"{int(v)}",
                ha="center", va="bottom", fontsize=9, color=INK_2)
    if len(values) > 6:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def histogram(df: pd.DataFrame, column: str, bins: int | str = "auto"):
    if column not in df.columns:
        raise AnalysisError(f"Column '{column}' not found.")
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        raise AnalysisError(f"'{column}' has no numeric values.")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=bins, color=PRIMARY, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(s.mean(), color=INK, linewidth=1.5, linestyle="--")
    ax.text(s.mean(), ax.get_ylim()[1] * 0.97, f" mean = {s.mean():.2f}", color=INK, fontsize=9, va="top")
    _style(ax, f"Distribution of {column} (n = {len(s)})", xlabel=column, ylabel="Count")
    fig.tight_layout()
    return fig


def correlation_heatmap(r: pd.DataFrame, p: pd.DataFrame | None = None, title: str = "Correlation matrix"):
    if r.shape[0] < 2:
        raise AnalysisError("At least two variables are needed for a heatmap.")
    k = r.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, 0.75 * k + 2), max(4, 0.65 * k + 1.5)))
    mask = np.triu(np.ones_like(r, dtype=bool), 1)
    data = np.ma.array(r.values.astype(float), mask=mask)
    im = ax.imshow(data, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(k), r.columns, rotation=45, ha="right")
    ax.set_yticks(range(k), r.index)
    if k <= 15:
        for i in range(k):
            for j in range(i + 1):
                v = r.values[i, j]
                if np.isnan(v):
                    continue
                star = ""
                if p is not None and i != j and not np.isnan(p.values[i, j]):
                    pv = p.values[i, j]
                    star = "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else ""
                ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(v) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    _style(ax, title, grid_axis="")
    ax.spines["bottom"].set_visible(False)
    if p is not None:
        fig.text(0.01, 0.0, "* p < .05, ** p < .01, *** p < .001 (two-tailed, unadjusted)", fontsize=8, color=MUTED)
    fig.tight_layout()
    return fig


def scree_plot(eigenvalues: list[float]):
    ev = np.asarray(eigenvalues, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, len(ev) + 1)
    ax.plot(x, ev, color=PRIMARY, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
    ax.text(x[-1], 1, " eigenvalue = 1", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x)
    _style(ax, "Scree plot", xlabel="Component", ylabel="Eigenvalue")
    fig.tight_layout()
    return fig


def bootstrap_histogram(boots: np.ndarray, lo: float, hi: float, title: str):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(boots, bins=50, color=PRIMARY, edgecolor=SURFACE, linewidth=1)
    for v in (lo, hi):
        ax.axvline(v, color=INK, linestyle="--", linewidth=1.2)
    ax.axvline(0, color="#e34948", linewidth=1.5)
    _style(ax, title, xlabel="Indirect effect (bootstrap samples)", ylabel="Count")
    fig.text(0.01, 0.0, "Dashed lines: confidence interval limits. Red line: zero.", fontsize=8, color=MUTED)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Literature
# ---------------------------------------------------------------------------
def publication_trend(records: list[dict]):
    years = []
    unknown = 0
    for r in records:
        v = r["fields"].get("Publication year", {}).get("value", "")
        try:
            y = int(str(v).strip()[:4])
            if 1950 <= y <= 2100:
                years.append(y)
                continue
        except ValueError:
            pass
        unknown += 1
    if not years:
        raise AnalysisError("No paper has a valid publication year yet. Add or correct years in the matrix.")
    counts = Counter(years)
    xs = list(range(min(years), max(years) + 1))
    ys = [counts.get(x, 0) for x in xs]
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(xs) + 2), 3.8))
    ax.bar([str(x) for x in xs], ys, color=PRIMARY, width=0.65, edgecolor=SURFACE, linewidth=2)
    _style(ax, f"Reviewed papers by publication year (n = {len(years)})", ylabel="Number of papers")
    ax.yaxis.get_major_locator().set_params(integer=True)
    if len(xs) > 8:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    note = "Counts describe the uploaded papers only, not the whole literature."
    if unknown:
        note += f" {unknown} paper(s) without a year are excluded."
    fig.text(0.01, -0.02, note, fontsize=8, color=MUTED, va="top")
    fig.tight_layout()
    return fig


def field_frequency(records: list[dict], field: str, top: int = 15) -> tuple[object, pd.DataFrame]:
    """Bar chart of how many papers mention each item of a multi-value field
    (e.g. 'Theories used'). Returns (figure, table with supporting papers)."""
    support: dict[str, list[str]] = {}
    for r in records:
        for item in set(split_list(r["fields"].get(field, {}).get("value", ""))):
            support.setdefault(item, []).append(r["paper_id"])
    if not support:
        raise AnalysisError(f"No paper has a value for '{field}' yet.")
    table = pd.DataFrame([{"Item": k, "Number of papers": len(v), "Papers": ", ".join(sorted(v))}
                          for k, v in support.items()]).sort_values(["Number of papers", "Item"], ascending=[False, True])
    shown = table.head(top)
    fig = _hbar(shown["Item"].tolist(), shown["Number of papers"].tolist(),
                f"{field}: number of reviewed papers (N = {len(records)})", "Number of papers",
                note="Each paper is counted once per item. Based on the matrix values as currently reviewed.")
    return fig, table.reset_index(drop=True)
