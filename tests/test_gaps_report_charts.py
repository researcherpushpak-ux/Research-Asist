import copy
import io

import pandas as pd
import pytest

from ev_research.analysis import data_io, inferential, plssem_prep, reliability
from ev_research.gaps import gap_finder
from ev_research.report import report_builder as rb
from ev_research.viz import charts


def test_gap_report_three_parts_and_traceable(fictional_records):
    rep = gap_finder.build_report(fictional_records)
    ids = {r["paper_id"] for r in fictional_records}
    assert len(rep.explicit) >= 3                       # each fictional paper states gaps / future research
    assert set(rep.explicit["Paper"]) <= ids
    assert len(rep.inferred) > 0
    for _, row in rep.inferred.iterrows():
        assert "inference" in rep.inferred.columns[0].lower()
        sup = {s.strip() for s in row["Supporting papers"].split(",") if s.strip()}
        assert sup and sup <= ids, row.iloc[0]


def test_gap_rules_on_fictional_set(fictional_records):
    rep = gap_finder.build_report(fictional_records)
    text = " ".join(rep.inferred.iloc[:, 0])
    assert "cross-sectional" in text                       # all 3 are cross-sectional surveys
    assert "Probability-based samples appear uncommon" in text
    assert "Theory of Planned Behaviour" in text            # TPB in 2 of 3 papers


def test_no_inferred_gaps_with_too_few_papers(fictional_records):
    rep = gap_finder.build_report(fictional_records[:2])
    assert rep.inferred.empty
    assert any("Fewer than 3" in n for n in rep.notes)


def test_gap_excel_export(fictional_records):
    data = gap_finder.export_excel(gap_finder.build_report(fictional_records))
    assert len(pd.read_excel(io.BytesIO(data), sheet_name=None)) == 5


def test_charts_render_png(scored_df, fictional_records):
    png_sig = b"\x89PNG"
    assert charts.to_png(charts.bar_chart(scored_df, "Income")).startswith(png_sig)
    assert charts.to_png(charts.histogram(scored_df, "PI")).startswith(png_sig)
    r = inferential.correlation(scored_df, ["ATT", "SN", "PI"])
    assert charts.to_png(charts.correlation_heatmap(r.raw["r"], r.raw["p"])).startswith(png_sig)
    assert charts.to_png(charts.publication_trend(fictional_records)).startswith(png_sig)
    fig, table = charts.field_frequency(fictional_records, "Theories used")
    assert charts.to_png(fig).startswith(png_sig)
    tpb = table.set_index("Item").loc["Theory of Planned Behaviour (TPB)"]
    assert tpb["Number of papers"] == 2 and tpb["Papers"] == "P01, P03"


def test_publication_trend_without_years(fictional_records):
    recs = copy.deepcopy(fictional_records)
    for r in recs:
        r["fields"]["Publication year"]["value"] = "Not reported in the paper."
    with pytest.raises(Exception, match="valid publication year"):
        charts.publication_trend(recs)


def _inputs(survey_df, groups, fictional_records, synthetic=True):
    df = survey_df
    return rb.ReportInputs(
        dataset_name="ev_survey_synthetic.csv", dataset_shape=df.shape, dataset_inspection=data_io.inspect(df),
        results=[(rb.CATEGORY_DESCRIPTIVE, reliability.reliability_table(df, groups)),
                 (rb.CATEGORY_PREPARATION, plssem_prep.prepare(df, groups, [("ATT", "PI")]))],
        figures=[("Income", charts.to_png(charts.bar_chart(df, "Income")))],
        literature_records=fictional_records, gap_report=gap_finder.build_report(fictional_records),
        sample_is_synthetic=synthetic,
    )


def test_report_markdown_contents(survey_df, groups, fictional_records):
    md = rb.build_markdown(_inputs(survey_df, groups, fictional_records))
    assert "SYNTHETIC" in md
    assert rb.PLACEHOLDER in md                               # methodology not invented
    assert "Not performed by the application" in md and "PLS-SEM estimation must be carried out" in md
    assert "not established facts" in md
    assert "No results of this type were added" in md          # statistical section empty -> stated, not filled
    for r in fictional_records:
        assert r["paper_id"] in md


def test_report_docx_builds(survey_df, groups, fictional_records):
    from docx import Document
    data = rb.build_docx(_inputs(survey_df, groups, fictional_records))
    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "1. Dataset description" in text and "8. Limitations" in text and "SYNTHETIC" in text
    assert len(doc.inline_shapes) == 1 and len(doc.tables) > 3


def test_empty_report_has_no_fabricated_content():
    md = rb.build_markdown(rb.ReportInputs())
    assert "No survey dataset was included" in md and "No charts were added" in md
    assert "Appendix" not in md and "p <" not in md
