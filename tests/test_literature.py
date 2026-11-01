import pandas as pd
import pytest

from ev_research import config
from ev_research.config import NOT_REPORTED
from ev_research.literature import matrix
from ev_research.literature.extractor import extract_fields
from ev_research.literature.pdf_reader import PDFReadError, read_pdf


def _f(recs, i, field):
    return recs[i]["fields"][field]


def test_all_fields_present(fictional_records):
    for r in fictional_records:
        assert list(r["fields"]) == config.MATRIX_FIELDS


def test_extracts_core_fields_with_pages(fictional_records):
    a = fictional_records[0]
    assert _f(fictional_records, 0, "Publication year")["value"] == "2019"
    assert _f(fictional_records, 0, "Sample size")["value"].split(";")[0] == "412"
    assert "Theory of Planned Behaviour (TPB)" in _f(fictional_records, 0, "Theories used")["value"]
    assert _f(fictional_records, 0, "Sampling method")["value"] == "Convenience sampling"
    assert _f(fictional_records, 0, "Sampling method")["pages"] == "2"     # methodology was placed on page 2
    assert _f(fictional_records, 0, "Dependent variables")["value"] == "Purchase intention"
    assert _f(fictional_records, 1, "Mediating variables")["value"] == "Perceived risk"
    assert _f(fictional_records, 2, "Moderating variables")["value"] == "Gender (demographic)"
    assert "India" in _f(fictional_records, 2, "Country of study")["value"]
    assert a["fields"]["Keywords"]["value"].startswith("electric vehicles")


def test_missing_information_marked_not_reported(fictional_records):
    d = _f(fictional_records, 0, "Research questions")
    assert d["value"] == NOT_REPORTED and d["status"] == config.STATUS_NOT_FOUND


def test_every_extracted_value_has_evidence(fictional_records):
    for r in fictional_records:
        for d in r["fields"].values():
            if d["value"] != NOT_REPORTED:
                assert d["evidence"], d["field"]
                assert d["status"] == config.STATUS_EXTRACTED


def test_reference_list_is_ignored(fictional_records):
    # The fictional reference list mentions Germany, 999 respondents and TAM
    for r in fictional_records:
        assert "Germany" not in r["fields"]["Country of study"]["value"]
        assert "999" not in r["fields"]["Sample size"]["value"]
        assert "TAM" not in r["fields"]["Theories used"]["value"]


def test_future_research_not_counted_as_design(fictional_records):
    # Paper A only mentions 'longitudinal' in its future-research sentence
    assert "Longitudinal" not in _f(fictional_records, 0, "Research design")["value"]


def test_interpretation_kept_separate(fictional_records):
    d = _f(fictional_records, 0, "Theories used")
    assert "NAMED" in d["interpretation"] and "NAMED" not in d["value"]


def test_pdf_errors():
    with pytest.raises(PDFReadError, match="empty"):
        read_pdf(b"", "a.pdf")
    with pytest.raises(PDFReadError, match="does not look like a PDF"):
        read_pdf(b"hello world", "a.pdf")


def test_scanned_pdf_detected(tmp_path):
    from reportlab.pdfgen import canvas
    p = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(p))
    c.rect(10, 10, 100, 100)
    c.showPage()
    c.save()
    with pytest.raises(PDFReadError, match="scanned"):
        read_pdf(p.read_bytes(), "blank.pdf")


def test_excel_export_import_roundtrip(fictional_records):
    import copy
    recs = copy.deepcopy(fictional_records)
    data = matrix.export_excel(recs)
    xl = pd.read_excel(pd.io.common.BytesIO(data), sheet_name=None)
    assert set(xl) == {"Matrix", "Matrix with pages", "Evidence and status", "App interpretations", "Read me"}
    assert len(xl["Matrix"]) == 3 and "Paper title" in xl["Matrix"].columns
    back = matrix.import_excel(data)
    for a, b in zip(recs, back):
        for f in config.MATRIX_FIELDS:
            assert str(a["fields"][f]["value"]) == str(b["fields"][f]["value"])
            assert a["fields"][f]["status"] == b["fields"][f]["status"]
            assert a["fields"][f]["interpretation"] == b["fields"][f]["interpretation"]


def test_export_empty_raises():
    with pytest.raises(ValueError):
        matrix.export_excel([])


def test_import_rejects_foreign_excel():
    buf = pd.io.common.BytesIO()
    pd.DataFrame({"x": [1]}).to_excel(buf, index=False)
    with pytest.raises(ValueError, match="Evidence and status"):
        matrix.import_excel(buf.getvalue())


def test_apply_edits_sets_status(fictional_records):
    import copy
    recs = copy.deepcopy(fictional_records)
    long_df = matrix.to_long([recs[0]])
    i = long_df.index[long_df["field"] == "Research questions"][0]
    long_df.loc[i, "value"] = "RQ1: What drives EV intention?"
    long_df.loc[i, "pages"] = "3"
    j = long_df.index[long_df["field"] == "Sample size"][0]
    long_df.loc[j, "status"] = config.STATUS_VERIFIED
    k = long_df.index[long_df["field"] == "Keywords"][0]
    long_df.loc[k, "value"] = ""
    matrix.apply_edits(recs, long_df)
    f = recs[0]["fields"]
    assert f["Research questions"]["status"] == config.STATUS_EDITED and f["Research questions"]["pages"] == "3"
    assert f["Sample size"]["status"] == config.STATUS_VERIFIED
    assert f["Keywords"]["value"] == NOT_REPORTED
    assert matrix.review_progress(recs)[0] == 3


def test_split_list():
    assert matrix.split_list("TPB; TAM ;") == ["TPB", "TAM"]
    assert matrix.split_list(NOT_REPORTED) == []
