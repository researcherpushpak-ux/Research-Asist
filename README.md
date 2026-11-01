# EV Purchase Intention Research Assistant

A beginner-friendly web application (Python + Streamlit) that supports academic research on
**"Factors Affecting Purchase Intention for Electric Vehicles in India."** It helps you:

1. build a **literature review matrix** from PDF papers, with page references,
2. **analyse questionnaire data** (CSV/Excel),
3. create **charts**,
4. identify **research gaps** in a traceable way, and
5. generate a **preliminary report** (Word and Markdown).

> **Read the [Limitations](#limitations-and-untested-features) section before relying on any output.**
> The application is a support tool. It does not replace reading the papers, checking the statistics,
> or your own judgement.

---

## 1. Installation

You need **Python 3.10 or newer** (tested on Python 3.13). Check with `python --version`.

```bash
git clone https://github.com/researcherpushpak-ux/Research-Asist.git
```

```bash
cd Research-Asist
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, use `.venv\Scripts\activate` instead.

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 2. Starting the application

```bash
streamlit run Home.py
```

Your browser opens at <http://localhost:8501>. If it does not, open that address yourself.
Stop the application with `Ctrl+C` in the terminal.

**Your work is kept only while the browser tab is open.** Download the literature matrix (Excel)
regularly. You can re-import it later to continue reviewing.

## 3. Using the application

Use the pages in the left sidebar.

### Module 1: Literature Matrix
1. Upload one or more PDFs, then click **Extract details from uploaded PDFs**.
2. Choose a paper and check every row of the review table against the PDF:
   * **Value**: the extracted information (editable).
   * **Page(s)** and **Evidence from PDF (verbatim)**: where it came from.
   * **App interpretation**: the application's comments (for example "a theory being *named* does not prove it is *used*"). These are **not** content from the paper and are exported to a separate sheet.
   * **Status**: set to *Verified by researcher* when you have checked a value. Values you change are marked *Corrected by researcher* automatically.
3. Missing information is shown as **"Not reported in the paper."** The status *Not found by extractor* means the rules did not find the information. It may still be in the paper, so check it.
4. Click **Download literature matrix (Excel)**. The workbook contains: *Matrix*, *Matrix with pages*, *Evidence and status*, *App interpretations* and *Read me*.
5. For scanned PDFs, use **Add empty record** and type the details yourself.

### Module 2: Data Analysis
Data format: one row per respondent and one column per question. Item names such as `ATT1, ATT2, ATT3`
are grouped into constructs automatically; you can edit the groups.

| Tab | What it does |
|---|---|
| Inspect & prepare | Variable overview, missing values per variable and row, duplicates, straight-lining; listwise deletion or mean/median replacement; reverse-coding; construct definitions; construct mean scores |
| Descriptives / Frequencies | Mean, SD, median, range, skewness, kurtosis; frequency tables with valid and cumulative percentages |
| Reliability | Cronbach's alpha, corrected item-total correlations, alpha-if-item-deleted |
| Factor analysis (EFA) | KMO, Bartlett's test, eigenvalues and scree plot; EFA (minres/ML/principal; oblimin/promax/varimax) with loadings, communalities, cross-loading flags |
| Correlation | Pearson or Spearman, p-values, pairwise N, heatmap |
| Regression | OLS multiple regression: B, standardised beta, CIs, VIF, R-squared, F-test, Breusch-Pagan, Durbin-Watson, residual normality; categorical predictors dummy-coded |
| ANOVA | One-way ANOVA, Welch ANOVA, Levene, Kruskal-Wallis, eta-squared, Tukey HSD |
| Mediation | Simple mediation X -> M -> Y (optional covariates) with a percentile bootstrap CI for the indirect effect; Sobel test for comparison |
| PLS-SEM preparation | See below |

Each result has **Download tables (Excel)** and **Add to report** buttons.

#### About PLS-SEM
**This application does not estimate PLS-SEM models.** No validated PLS algorithm is included.
The *PLS-SEM preparation* tab provides:

* specification checks (unknown or duplicated items, loops in the structural model),
* indicator screening,
* **HTMT**, computed from item correlations (the same quantity PLS software reports, without bootstrap CIs),
* **approximate** CR and AVE from a one-factor model per construct. These are labelled *NOT PLS results*: PLS software uses composite-based loadings, which usually differ,
* minimum sample size (inverse square-root method; the "10-times rule" is shown for reference only),
* a clean **CSV for SmartPLS**, an **R script template for `seminr`** (not run or tested by this app), and a checklist of what to report.

Estimate the model in SmartPLS, R (`seminr`/`cSEM`) or similar, and report those results.

### Module 3: Visualizations
Bar charts, histograms and correlation heatmaps from survey data. Publication trends and the most
frequent theories, variables, techniques, countries, designs and sampling methods from the matrix,
with the supporting papers for each count. Charts can be downloaded (PNG) or added to the report.

### Module 4: Research Gaps
Three separate sections:
1. **Gaps stated explicitly by authors**: the *Research gaps* and *Future research directions* values, with pages.
2. **Patterns across reviewed papers**: counts with the paper IDs behind each count.
3. **Potential gaps inferred by the application**: produced by simple, documented rules (for example "at least 60% of papers reporting a design are cross-sectional"). Each row lists the **supporting papers**, the **counter-evidence**, papers without data, and how many supporting values you have verified. These are inferences about the uploaded papers only, not established facts. At least 3 papers are needed.

### Module 5: Report
Type your methodology, limitations and suggestions, then download a **Word (.docx)** or **Markdown** report.
The report includes only what you added. Sections the application cannot know about (such as your methodology)
show *[To be written by the researcher.]* instead of generic text. No references are generated: the appendix lists
only the uploaded papers as recorded in your matrix.

## 4. Trying it with the sample data
* **Data Analysis → Load SYNTHETIC sample dataset**: 320 simulated responses.
* **Literature Matrix → Load the 3 FICTIONAL test papers**.

Both are clearly marked in the interface and in the report. They are **not real evidence**; see
[`sample_data/README.md`](sample_data/README.md).

## 5. Running the tests

```bash
python -m pytest -q
```

The test suite (63 tests at the time of writing) covers:
* statistics checked against independent calculations: Cronbach's alpha (hand-worked example and an alternative formula), correlation/ANOVA against SciPy, regression against NumPy least squares, mediation recovering a known simulated indirect effect, a null mediation with a CI that includes zero, HTMT against a hand computation, and sample-size values,
* EFA recovering the simulated factor structure,
* PDF extraction on the fictional PDFs: fields, page numbers, "Not reported", reference-list exclusion, scanned-PDF detection,
* Excel export → import round trip, researcher edits and status handling,
* gap traceability (every inferred gap cites existing papers),
* Word/Markdown report contents (placeholders, synthetic-data warning, no invented sections),
* headless UI tests (Streamlit `AppTest`) that load every page and click through the main workflows.

## 6. Project structure

```
Home.py                       Streamlit entry page
pages/                        One file per module (1-5)
ev_research/
  config.py                   Matrix fields, statuses, term dictionaries (editable)
  ui.py                       Shared Streamlit helpers
  literature/                 pdf_reader.py, extractor.py, matrix.py (edit/export/import)
  analysis/                   data_io, descriptive, reliability, efa, inferential, mediation, plssem_prep
  viz/charts.py               matplotlib charts
  gaps/gap_finder.py          explicit gaps, patterns, inferred gaps
  report/report_builder.py    Word and Markdown report
sample_data/                  synthetic survey, codebook, fictional PDFs, generator script
tests/                        pytest suite
```

To recognise more theories, constructs or techniques, add patterns to the dictionaries in
`ev_research/config.py`.

## Limitations and untested features

**PDF extraction**
* Extraction is **rule-based** (headings, regular expressions, dictionaries). It is not AI and does not "understand" papers.
* It has been tested **only on the three simple, single-column fictional PDFs** generated for this project. Real journal PDFs (two-column layouts, running headers, footnotes, tables) will extract **less accurately**. Expect missed fields and sentences that describe *other* studies (for example from the literature review).
* Variables are found only if they match the EV-construct dictionary. Roles (IV/DV/mediator/moderator) come from sentence patterns and are candidates only.
* Titles and authors use font-size and metadata heuristics. Publication year is read only from labelled patterns on page 1 (©, "Published", volume line); otherwise it is left as not reported.
* Scanned PDFs are detected but **not OCR'd**. Password-protected PDFs cannot be read.

**Statistics**
* No PLS-SEM, CB-SEM/CFA, moderation/interaction models, multi-group analysis, logistic regression, Games-Howell post-hoc tests, robust (HC3) standard errors or multiple-comparison correction are implemented. Where relevant, the output warns you.
* Mediation: one mediator only (equivalent in design to PROCESS Model 4). It has **not** been cross-checked against the PROCESS macro itself.
* Approximate CR/AVE in the PLS-SEM tab are factor-based screening values, **not** PLS results.
* `factor_analyzer` 0.5.1 is incompatible with recent scikit-learn versions. The app patches this automatically (`ev_research/analysis/efa.py`, `_require_fa`).
* Little's MCAR test and multiple imputation are not available.

**Untested**
* The generated **R `seminr` script** has not been run (R is not part of this project). Check it against the seminr documentation.
* The interface has been tested headlessly (Streamlit AppTest) and with automated browser screenshots on desktop width, but **not with real research PDFs, very large datasets (> 50,000 rows), or on mobile screens**.
* Word reports were checked by re-opening them with `python-docx`, not visually in Microsoft Word.

**General**
* Nothing is saved on a server. Closing the tab loses unsaved work.
* All "rules of thumb" (alpha ≥ .70, KMO ≥ .60, HTMT < .85, VIF < 5, etc.) are conventions. Check what your field and supervisor expect.
