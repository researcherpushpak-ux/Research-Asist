"""Shared constants: matrix fields, status labels and the term dictionaries
used by the rule-based extractor.

The dictionaries are deliberately simple and editable.  A dictionary match
only shows that a term *appears* in the paper; it does not show how the paper
uses the term.  The extractor records this distinction explicitly.
"""

NOT_REPORTED = "Not reported in the paper."

# Order matters: this is the column order of the literature matrix.
MATRIX_FIELDS = [
    "Paper title",
    "Authors",
    "Publication year",
    "Country of study",
    "Research objectives",
    "Research questions",
    "Research gaps",
    "Theories used",
    "Sample size",
    "Sampling method",
    "Research design",
    "Independent variables",
    "Dependent variables",
    "Mediating variables",
    "Moderating variables",
    "Statistical techniques",
    "Key findings",
    "Conclusions",
    "Future research directions",
    "Keywords",
]

# Status of each cell in the matrix.
STATUS_EXTRACTED = "Auto-extracted (needs review)"
STATUS_NOT_FOUND = "Not found by extractor (needs review)"
STATUS_VERIFIED = "Verified by researcher"
STATUS_EDITED = "Corrected by researcher"
STATUSES = [STATUS_EXTRACTED, STATUS_NOT_FOUND, STATUS_VERIFIED, STATUS_EDITED]

# ---------------------------------------------------------------------------
# Dictionaries (case-insensitive).  Key = canonical label, value = patterns.
# Patterns are regular expressions matched with word boundaries.
# ---------------------------------------------------------------------------
THEORIES = {
    "Theory of Planned Behaviour (TPB)": [r"theory of planned behaviou?r", r"\bTPB\b"],
    "Theory of Reasoned Action (TRA)": [r"theory of reasoned action", r"\bTRA\b"],
    "Technology Acceptance Model (TAM)": [r"technology acceptance model", r"\bTAM\b"],
    "UTAUT / UTAUT2": [r"unified theory of acceptance and use of technology", r"\bUTAUT2?\b"],
    "Diffusion of Innovations (DOI)": [r"diffusion of innovations?", r"\bDOI\b"],
    "Value-Belief-Norm theory (VBN)": [r"value[- ]belief[- ]norm", r"\bVBN\b"],
    "Norm Activation Model (NAM)": [r"norm activation model", r"\bNAM\b"],
    "Stimulus-Organism-Response (S-O-R)": [r"stimulus[- ]organism[- ]response", r"\bS-O-R\b", r"\bSOR\b"],
    "Protection Motivation Theory": [r"protection motivation theory"],
    "Consumption Values Theory": [r"theory of consumption values?", r"consumption values? theory"],
    "Perceived Risk Theory": [r"perceived risk theory"],
    "Innovation Resistance Theory": [r"innovation resistance theory", r"\bIRT\b"],
    "Social Cognitive Theory": [r"social cognitive theory"],
    "Self-Determination Theory": [r"self[- ]determination theory"],
}

STAT_TECHNIQUES = {
    "PLS-SEM": [r"\bPLS[- ]SEM\b", r"partial least squares", r"SmartPLS"],
    "CB-SEM / SEM": [r"\bCB[- ]SEM\b", r"structural equation model", r"\bAMOS\b", r"\blavaan\b"],
    "Confirmatory factor analysis (CFA)": [r"confirmatory factor analysis", r"\bCFA\b"],
    "Exploratory factor analysis (EFA)": [r"exploratory factor analysis", r"\bEFA\b"],
    "Multiple regression": [r"multiple regression", r"\bOLS\b", r"linear regression"],
    "Logistic regression": [r"logistic regression", r"\blogit\b"],
    "ANOVA": [r"\bANOVA\b", r"analysis of variance"],
    "t-test": [r"\bt[- ]test"],
    "Chi-square test": [r"chi[- ]square", r"χ\s*2"],
    "Correlation analysis": [r"correlation analysis", r"pearson.{0,20}correlation", r"spearman"],
    "Cronbach's alpha": [r"cronbach"],
    "Mediation analysis": [r"mediation analysis", r"PROCESS macro", r"indirect effect"],
    "Moderation / multi-group analysis": [r"moderation analysis", r"multi[- ]group analysis", r"\bMGA\b"],
    "fsQCA": [r"\bfsQCA\b", r"fuzzy[- ]set qualitative"],
    "Conjoint / discrete choice": [r"conjoint", r"discrete choice", r"choice experiment"],
    "Thematic / qualitative analysis": [r"thematic analysis", r"content analysis", r"NVivo"],
    "Bootstrapping": [r"bootstrap"],
}

RESEARCH_DESIGNS = {
    "Cross-sectional survey": [r"cross[- ]sectional", r"questionnaire survey", r"survey[- ]based"],
    "Longitudinal": [r"longitudinal", r"panel data"],
    "Experimental": [r"\bexperiment(al)? design", r"randomi[sz]ed"],
    "Qualitative (interviews / focus groups)": [r"semi[- ]structured interviews?", r"focus groups?"],
    "Mixed methods": [r"mixed[- ]methods?"],
    "Systematic / bibliometric review": [r"systematic (literature )?review", r"bibliometric", r"meta[- ]analysis"],
    "Quantitative (unspecified)": [r"quantitative (research|approach|study|method)"],
    "Descriptive": [r"descriptive research design"],
}

SAMPLING_METHODS = {
    "Convenience sampling": [r"convenience sampl"],
    "Purposive / judgmental sampling": [r"purposive sampl", r"judg(e)?mental sampl"],
    "Snowball sampling": [r"snowball sampl"],
    "Quota sampling": [r"quota sampl"],
    "Stratified random sampling": [r"stratified (random )?sampl"],
    "Simple random sampling": [r"simple random sampl"],
    "Systematic sampling": [r"systematic sampl"],
    "Cluster sampling": [r"cluster sampl"],
    "Multistage sampling": [r"multi[- ]?stage sampl"],
    "Random sampling (unspecified)": [r"\brandom sampl"],
    "Non-probability sampling (unspecified)": [r"non[- ]probability sampl"],
}

# Constructs frequently studied in EV purchase-intention research.  A match
# means "this term appears"; the role (IV/DV/mediator/moderator) is NOT
# inferred from the dictionary.
EV_CONSTRUCTS = {
    "Purchase intention": [r"purchase intentions?", r"adoption intentions?", r"intention to (buy|purchase|adopt)"],
    "Attitude": [r"\battitudes?\b"],
    "Subjective norms": [r"subjective norms?"],
    "Perceived behavioural control": [r"perceived behaviou?ral control", r"\bPBC\b"],
    "Environmental concern": [r"environmental concerns?", r"environmental awareness", r"environmental knowledge"],
    "Perceived price / cost": [r"perceived (price|cost)", r"purchase price", r"price sensitivity", r"financial cost"],
    "Charging infrastructure": [r"charging (infrastructure|stations?|facilit)"],
    "Range anxiety": [r"range anxiety", r"driving range"],
    "Government incentives / policy": [r"government (incentives?|subsid|polic)", r"financial incentives?", r"\bFAME\b", r"subsid(y|ies)"],
    "Perceived risk": [r"perceived risks?"],
    "Perceived usefulness": [r"perceived usefulness"],
    "Perceived ease of use": [r"perceived ease of use"],
    "Performance expectancy": [r"performance expectancy"],
    "Effort expectancy": [r"effort expectancy"],
    "Social influence": [r"social influence"],
    "Facilitating conditions": [r"facilitating conditions?"],
    "Hedonic motivation": [r"hedonic motivation"],
    "Price value": [r"price value"],
    "Perceived benefits": [r"perceived benefits?"],
    "Knowledge / awareness of EVs": [r"(EV|product) knowledge", r"awareness of (EVs?|electric vehicles?)"],
    "Trust": [r"\btrust\b"],
    "Personal norms": [r"personal norms?"],
    "Green self-identity": [r"green self[- ]identity", r"self[- ]identity"],
    "Innovativeness": [r"consumer innovativeness", r"personal innovativeness"],
    "Brand image": [r"brand image"],
    "Income (demographic)": [r"\bincome\b"],
    "Age (demographic)": [r"\bage groups?\b", r"respondents'? age"],
    "Gender (demographic)": [r"\bgender\b"],
}

COUNTRIES = [
    "India", "China", "United States", "USA", "United Kingdom", "UK", "Germany",
    "Norway", "Netherlands", "Sweden", "France", "Italy", "Spain", "Japan",
    "South Korea", "Korea", "Malaysia", "Indonesia", "Thailand", "Vietnam",
    "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Australia", "Canada",
    "Brazil", "Mexico", "Turkey", "Iran", "Saudi Arabia", "UAE", "Singapore",
    "Taiwan", "Philippines", "South Africa", "Nigeria", "Egypt", "Poland",
    "Portugal", "Denmark", "Finland", "Ireland", "New Zealand",
]

# Indian states / major cities help confirm India-based studies.
INDIA_LOCATIONS = [
    "Delhi", "NCR", "Mumbai", "Bengaluru", "Bangalore", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kerala", "Tamil Nadu",
    "Karnataka", "Maharashtra", "Gujarat", "Punjab", "Uttar Pradesh", "Rajasthan",
    "West Bengal", "Telangana", "Andhra Pradesh", "Odisha", "Haryana",
]

# Section headings (lower-case, matched at the start of a line).
SECTION_HEADINGS = {
    "abstract": [r"abstract"],
    "keywords": [r"key\s?words", r"index terms"],
    "introduction": [r"\d*\.?\s*introduction"],
    "objectives": [r"\d*\.?\s*(research )?objectives?( of the study)?"],
    "questions": [r"\d*\.?\s*research questions?"],
    "gap": [r"\d*\.?\s*(research gaps?|gaps? in (the )?literature)"],
    "methodology": [r"\d*\.?\s*(research )?(methodology|methods?|research design)"],
    "results": [r"\d*\.?\s*(results|findings|results and discussion|data analysis)"],
    "discussion": [r"\d*\.?\s*discussion"],
    "conclusion": [r"\d*\.?\s*(conclusions?|concluding remarks)"],
    "limitations": [r"\d*\.?\s*(limitations?( and future (research|scope|directions))?)"],
    "future": [r"\d*\.?\s*(future (research|scope|directions|studies)( directions| agenda)?)"],
    "references": [r"references", r"bibliography"],
}
