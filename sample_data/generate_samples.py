"""Generate SYNTHETIC sample files for testing the application.

    python sample_data/generate_samples.py

Creates:
  * ev_survey_synthetic.csv / .xlsx - a simulated questionnaire dataset
  * codebook.csv                     - item wording / construct mapping
  * fictional_papers/*.pdf           - three FICTIONAL test "papers"

IMPORTANT: Nothing here is real research evidence. The survey responses are
random numbers drawn from a model chosen by the developer, and the "papers"
are invented documents with invented authors. Use them only to learn the
software. Never cite them or report their results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 2024

CONSTRUCTS = {
    "ATT": ("Attitude towards EVs", ["Buying an EV would be a good idea", "I like the idea of driving an EV",
                                     "Using an EV would be beneficial for me", "I have a positive opinion of EVs"]),
    "SN": ("Subjective norms", ["People important to me think I should buy an EV",
                                "My family would approve of me buying an EV",
                                "My friends expect me to consider an EV"]),
    "PBC": ("Perceived behavioural control", ["I could afford an EV if I wanted to",
                                              "Buying an EV is entirely within my control",
                                              "I have the knowledge to choose an EV"]),
    "EC": ("Environmental concern", ["I am worried about air pollution in my city",
                                     "Vehicle emissions harm the environment",
                                     "I try to reduce my environmental impact"]),
    "CI": ("Charging infrastructure", ["Charging stations are available where I live",
                                       "I could charge an EV easily at home or work",
                                       "Public charging in my city is reliable"]),
    "PI": ("Purchase intention", ["I intend to buy an EV in the next two years",
                                  "I will consider an EV for my next vehicle",
                                  "I would recommend EVs to others"]),
}


def _likert(latent: np.ndarray, rng, noise=0.7) -> np.ndarray:
    raw = latent + rng.normal(0, noise, size=latent.shape)
    return np.clip(np.round(3 + raw), 1, 5).astype(int)


def make_survey(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ec = rng.normal(0, 1, n)
    ci = rng.normal(0, 1, n)
    sn = rng.normal(0, 1, n)
    pbc = 0.3 * ci + rng.normal(0, 0.95, n)
    att = 0.5 * ec + rng.normal(0, 0.85, n)          # EC -> ATT (simulated mediation path)
    pi = 0.45 * att + 0.25 * sn + 0.2 * pbc + 0.1 * ec + rng.normal(0, 0.7, n)
    latents = {"ATT": att, "SN": sn, "PBC": pbc, "EC": ec, "CI": ci, "PI": pi}

    df = pd.DataFrame({"RespondentID": np.arange(1, n + 1)})
    df["Gender"] = rng.choice(["Male", "Female", "Prefer not to say"], n, p=[0.55, 0.43, 0.02])
    df["AgeGroup"] = rng.choice(["18-25", "26-35", "36-45", "46-60", "60+"], n, p=[0.25, 0.32, 0.22, 0.16, 0.05])
    df["Income"] = rng.choice(["<5 lakh", "5-10 lakh", "10-20 lakh", ">20 lakh"], n, p=[0.2, 0.35, 0.3, 0.15])
    df["City"] = rng.choice(["Delhi NCR", "Mumbai", "Bengaluru", "Pune", "Chennai"], n)
    df["OwnsVehicle"] = rng.choice(["Yes", "No"], n, p=[0.7, 0.3])
    for code, (_, items) in CONSTRUCTS.items():
        for i in range(len(items)):
            df[f"{code}{i + 1}"] = _likert(latents[code], rng)
    # Income effect on PI (for ANOVA demonstration)
    shift = df["Income"].map({"<5 lakh": -0.4, "5-10 lakh": 0.0, "10-20 lakh": 0.2, ">20 lakh": 0.5}).to_numpy()
    for c in [c for c in df.columns if c.startswith("PI")]:
        df[c] = np.clip(df[c] + np.round(shift + rng.normal(0, 0.3, n)), 1, 5).astype(int)
    # Introduce ~2% missing values in item columns
    item_cols = [c for c in df.columns if any(c.startswith(k) and c[len(k):].isdigit() for k in CONSTRUCTS)]
    mask = rng.random((n, len(item_cols))) < 0.02
    items = df[item_cols].astype("float")
    items = items.mask(mask)
    df[item_cols] = items
    return df


def make_codebook() -> pd.DataFrame:
    rows = []
    for code, (name, items) in CONSTRUCTS.items():
        for i, text in enumerate(items, start=1):
            rows.append({"item": f"{code}{i}", "construct_code": code, "construct": name,
                         "wording": text, "scale": "1 = Strongly disagree ... 5 = Strongly agree"})
    return pd.DataFrame(rows)


FICTIONAL_PAPERS = [
    {
        "file": "fictional_paper_A_2019.pdf",
        "title": "Consumer Attitudes and Electric Vehicle Adoption Intention: A Test Document",
        "authors": "Alpha Tester, Beta Example and Gamma Placeholder",
        "year": "2019",
        "keywords": "Keywords: electric vehicles; purchase intention; theory of planned behaviour; India",
        "body": [
            ("Abstract", "This is a FICTIONAL document generated to test software. It is not a real study. "
             "The purpose of this study is to examine the factors that influence purchase intention for electric vehicles. "
             "Data were collected from 412 respondents in India using a structured questionnaire."),
            ("1. Introduction", "Little is known about how subjective norms shape EV adoption in tier-2 cities. "
             "This study uses the theory of planned behaviour (TPB) as its framework."),
            ("2. Research objectives", "To examine the effect of attitude, subjective norms and perceived behavioural control on purchase intention."),
            ("3. Methodology", "A cross-sectional survey design was adopted. Convenience sampling was used to recruit car owners in Delhi and Pune. "
             "The final sample of 412 usable responses was analysed with PLS-SEM in SmartPLS. Cronbach's alpha was used to assess reliability."),
            ("4. Results", "The results show that attitude has a significant positive effect on purchase intention. "
             "Subjective norms were found to have a weaker effect. Attitude, subjective norms and perceived behavioural control are the independent variables and purchase intention is the dependent variable."),
            ("5. Conclusion", "Attitude is the strongest predictor of purchase intention in this fictional dataset."),
            ("6. Limitations and future research", "Future research should use longitudinal designs and examine charging infrastructure as a moderator."),
        ],
    },
    {
        "file": "fictional_paper_B_2021.pdf",
        "title": "Charging Infrastructure, Price and EV Purchase Intention: Another Test Document",
        "authors": "Delta Sample and Epsilon Mock",
        "year": "2021",
        "keywords": "Keywords: charging infrastructure; perceived price; UTAUT2; mediation",
        "body": [
            ("Abstract", "This is a FICTIONAL document generated to test software. It is not a real study. "
             "This paper aims to investigate how charging infrastructure and perceived price shape intention to purchase electric vehicles."),
            ("1. Introduction", "Few studies have examined the role of range anxiety in emerging markets. "
             "The study draws on the unified theory of acceptance and use of technology (UTAUT2)."),
            ("2. Methodology", "A questionnaire survey was conducted among consumers in India (n = 286) using purposive sampling. "
             "Multiple regression and mediation analysis using the PROCESS macro with bootstrapping were performed."),
            ("3. Results", "Findings indicate that charging infrastructure positively influences purchase intention. "
             "Perceived risk mediates the relationship between perceived price and purchase intention."),
            ("4. Conclusions", "Investment in charging infrastructure may raise purchase intention, according to this fictional analysis."),
            ("5. Future research", "Further research could compare two-wheeler and four-wheeler buyers."),
        ],
    },
    {
        "file": "fictional_paper_C_2023.pdf",
        "title": "Environmental Concern and Green Purchase of Electric Vehicles: A Third Test Document",
        "authors": "Zeta Fixture and Eta Dummy",
        "year": "2023",
        "keywords": "Keywords: environmental concern; attitude; theory of planned behaviour; gender",
        "body": [
            ("Abstract", "This is a FICTIONAL document generated to test software. It is not a real study. "
             "The objective of this study is to analyse the role of environmental concern in EV purchase intention."),
            ("1. Introduction", "Research on the moderating role of gender remains underexplored. "
             "We extend the theory of planned behaviour with environmental concern."),
            ("2. Methodology", "A cross-sectional survey of 350 respondents in Bengaluru, India was conducted using convenience sampling. "
             "Exploratory factor analysis and structural equation modelling (AMOS) were used."),
            ("3. Results", "The results reveal that environmental concern has a significant positive effect on attitude. "
             "Attitude mediates the effect of environmental concern on purchase intention. Gender moderates the attitude-intention link."),
            ("4. Conclusion", "Environmental concern matters indirectly through attitude in this fictional example."),
            ("5. Future research directions", "Future studies should include government incentives and use probability sampling."),
        ],
    },
]


def make_fictional_pdf(spec: dict, out: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=18, leading=22)
    body = styles["BodyText"]
    head = styles["Heading2"]
    doc = SimpleDocTemplate(str(out), pagesize=A4, title=spec["title"], author=spec["authors"])
    story = [
        Paragraph(spec["title"], title),
        Paragraph(spec["authors"], body),
        Paragraph(f"Fictional Journal for Software Testing, Vol. 1 ({spec['year']}) - © {spec['year']} Test Fixtures", body),
        Paragraph("FICTIONAL TEST DOCUMENT - NOT A REAL RESEARCH PAPER", head),
        Spacer(1, 8),
        Paragraph(spec["keywords"], body),
    ]
    for i, (h, text) in enumerate(spec["body"]):
        if i == 3:
            story.append(PageBreak())   # put the methodology on page 2+ to test page numbers
        story += [Paragraph(h, head), Paragraph(text, body)]
    story += [PageBreak(), Paragraph("References", head),
              Paragraph("Placeholder, X. (2010). A non-existent reference used only to test reference stripping. "
                        "Data from 999 respondents in Germany using the technology acceptance model.", body)]
    doc.build(story)


def main() -> None:
    survey = make_survey()
    survey.to_csv(HERE / "ev_survey_synthetic.csv", index=False)
    survey.to_excel(HERE / "ev_survey_synthetic.xlsx", index=False)
    make_codebook().to_csv(HERE / "codebook.csv", index=False)
    pdf_dir = HERE / "fictional_papers"
    pdf_dir.mkdir(exist_ok=True)
    for spec in FICTIONAL_PAPERS:
        make_fictional_pdf(spec, pdf_dir / spec["file"])
    print(f"Wrote synthetic survey ({len(survey)} rows), codebook and {len(FICTIONAL_PAPERS)} fictional PDFs to {HERE}")


if __name__ == "__main__":
    main()
