# Sample data: synthetic and fictional, for testing only

**Nothing in this folder is real research evidence. Do not cite it or report its results.**

| File | What it is |
|---|---|
| `ev_survey_synthetic.csv` / `.xlsx` | 320 simulated questionnaire responses generated from random numbers by `generate_samples.py` (seed 2024). Columns: demographics (Gender, AgeGroup, Income, City, OwnsVehicle) and 5-point items for six constructs: ATT (attitude), SN (subjective norms), PBC (perceived behavioural control), EC (environmental concern), CI (charging infrastructure) and PI (purchase intention). About 2% of item values are deliberately missing. |
| `codebook.csv` | Item wording and construct mapping for the synthetic survey. The wording was written for this demo; it is not a validated scale. |
| `fictional_papers/*.pdf` | Three invented "papers" with invented authors and a fictional journal. Each says *FICTIONAL TEST DOCUMENT* on page 1. They exist to test PDF extraction, gap identification and charts. |

The simulated relationships (e.g. EC -> ATT -> PI, an income effect on PI) were chosen by the developer
so that the analyses produce non-trivial output. They say nothing about real Indian consumers.

Regenerate everything with:

```bash
python sample_data/generate_samples.py
```
