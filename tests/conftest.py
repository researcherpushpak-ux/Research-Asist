import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "sample_data"


@pytest.fixture(scope="session")
def survey_df():
    from ev_research.analysis.data_io import load_dataset
    return load_dataset((SAMPLE / "ev_survey_synthetic.csv").read_bytes(), "ev_survey_synthetic.csv")


@pytest.fixture(scope="session")
def groups(survey_df):
    from ev_research.analysis.data_io import detect_item_groups, detect_likert_columns
    return detect_item_groups(detect_likert_columns(survey_df))


@pytest.fixture(scope="session")
def scored_df(survey_df, groups):
    from ev_research.analysis.data_io import composite_scores
    return pd.concat([survey_df, composite_scores(survey_df, groups)], axis=1)


@pytest.fixture(scope="session")
def fictional_records():
    from ev_research.literature.extractor import extract_fields
    from ev_research.literature.matrix import new_record, next_paper_id
    from ev_research.literature.pdf_reader import read_pdf
    recs = []
    for p in sorted((SAMPLE / "fictional_papers").glob("*.pdf")):
        recs.append(new_record(next_paper_id(recs), p.name, extract_fields(read_pdf(p.read_bytes(), p.name))))
    return recs
