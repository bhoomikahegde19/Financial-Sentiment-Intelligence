"""Synthetic financial news and public-text corpus for the sentiment case study."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

COMPANIES = [("Apex Motors", "APX", "Manufacturing"), ("Brightline Retail", "BLR", "Retail"), ("Cedar Health", "CDH", "Healthcare"), ("Northstar Cloud", "NSC", "Technology"), ("Oakstone Energy", "OAE", "Energy"), ("Summit Properties", "SMP", "Real Estate")]
SOURCES = ["Earnings Call", "Business News", "Regulatory Filing", "Analyst Note"]
TOPICS = ["earnings", "guidance", "operations", "regulation", "capital markets"]

# The templates intentionally vary the subject, verb, evidence, and sentence shape.
TEXT_TEMPLATES = {
    "Positive": [
        "{company} finished the quarter ahead of internal plans as orders held up across its core business.",
        "A better mix of products helped {company} lift gross margin, although costs remain under review.",
        "Management at {company} described customer retention as encouraging and raised its outlook modestly.",
        "New contracts in {company}'s pipeline provide a firmer base for growth into the next reporting period.",
        "Cash generation at {company} improved from last year, giving the balance sheet more room to invest.",
        "The latest filing shows {company} making steady progress on its operating targets without major setbacks.",
        "Demand was stronger than expected for {company}; analysts now see a constructive path for earnings.",
        "{company} reported a solid start to the year and said execution is tracking better than planned.",
        "Investors responded favorably after {company} clarified its strategy and maintained healthy liquidity.",
        "A successful rollout added momentum for {company}, with early customer feedback described as positive.",
        "The company converted several delayed bids into signed work, which supports {company}'s revenue outlook.",
        "{company} appears to be gaining share in its main market while keeping spending disciplined.",
        "An improved renewal rate and stable pricing left {company} in a stronger position this quarter.",
        "Leadership pointed to dependable demand and a manageable cost base in the {company} update.",
        "{company} exceeded the street's cautious expectations and reiterated its longer-term ambitions.",
        "Operational changes are beginning to pay off at {company}, even though the improvement is gradual.",
        "The quarter brought reassuring progress on cash flow and product delivery for {company}.",
        "A new commercial partnership broadens {company}'s reach and strengthens its growth narrative.",
        "Compared with the prior period, {company} delivered better margins and fewer execution issues.",
        "{company} kept its promises on the main milestones, a welcome sign for investors watching the turnaround.",
    ],
    "Neutral": [
        "{company} left its annual forecast unchanged while the finance team continues to review assumptions.",
        "The quarterly release from {company} was broadly in line with expectations and offered few surprises.",
        "Management discussed demand, staffing, and costs at {company}; no material policy change was announced.",
        "{company} will provide a fuller update next quarter after completing its ordinary planning cycle.",
        "Revenue was steady for {company}, but the effect of pricing and volume changes was mixed.",
        "The filing records routine progress at {company} and repeats the previously disclosed risk factors.",
        "Analysts described {company}'s update as uneventful, with both helpful and difficult details in view.",
        "{company} continues to monitor market conditions before deciding whether to adjust its investment pace.",
        "There was no clear change in the outlook for {company}, despite a few moving pieces in the quarter.",
        "The company answered questions about liquidity and demand, but {company} did not publish new targets.",
        "{company} completed the scheduled leadership transition and said day-to-day operations remain stable.",
        "Results at {company} were acceptable overall; a soft area was offset by an improvement elsewhere.",
        "The note on {company} is descriptive rather than directional and mostly repeats known information.",
        "Management at {company} acknowledged some pressure in one segment while retaining the existing guidance.",
        "{company} plans to revisit its expansion timetable once the regulatory review has concluded.",
        "Investors received a mixed set of indicators from {company}, with no decisive signal either way.",
        "The update from {company} was modest in tone and did not materially change the prior narrative.",
        "{company} is still evaluating supplier costs and customer activity before making a formal forecast.",
        "A routine compliance notice from {company} contained administrative details and no new financial commitment.",
        "The latest numbers from {company} were close to plan, though management said visibility remains limited.",
    ],
    "Negative": [
        "{company} missed its quarterly plan as weaker orders and cautious customers weighed on revenue.",
        "Higher input costs narrowed {company}'s margin, while management trimmed its forecast for the year.",
        "A delayed launch left {company} with excess inventory and a less certain near-term outlook.",
        "Investors focused on falling cash generation and rising execution risk in the {company} update.",
        "{company} reported disappointing demand and said a recovery may take longer than previously expected.",
        "The filing points to project slippage at {company}, with additional spending now likely.",
        "Shares of {company} came under pressure after management acknowledged slower growth in its core market.",
        "Analysts lowered their estimates for {company} as cancellations and pricing pressure became more visible.",
        "{company} is pausing part of its expansion plan because funding and customer commitments are less secure.",
        "A difficult quarter exposed gaps in {company}'s delivery process and put the margin target at risk.",
        "The company warned that regulatory delays could hurt {company}'s timetable and increase operating costs.",
        "Liquidity remains a concern for {company}, despite management's efforts to preserve cash.",
        "{company} lost momentum in a key segment, and the latest guidance implies a weaker second half.",
        "The update from {company} was below expectations, with muted demand and several unresolved issues.",
        "Cost inflation and staff turnover are creating fresh headwinds for {company} this year.",
        "After a series of missed milestones, {company} admitted that the recovery is not yet dependable.",
        "The quarter brought a sharper decline in activity than {company} had signaled in its previous release.",
        "{company} faces a difficult refinancing window as earnings soften and lenders ask for more detail.",
        "Customer complaints and delivery interruptions have damaged confidence in {company}'s operating plan.",
        "Management at {company} sounded cautious, citing weak visibility and a growing list of execution risks.",
    ],
}

SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "steady": ("stable", "consistent", "firm"),
    "stronger": ("better", "healthier", "more resilient"),
    "weaker": ("softer", "slower", "less robust"),
    "improved": ("lifted", "strengthened", "recovered"),
    "difficult": ("challenging", "tough", "strained"),
    "review": ("assessment", "check", "revisit"),
    "outlook": ("forecast", "view", "guidance"),
    "customers": ("clients", "buyers", "accounts"),
    "costs": ("expenses", "spending", "cost pressures"),
}

PREFIXES = ("In the latest release, ", "According to the update, ", "For this period, ", "")
FOLLOW_UPS = (
    " The commentary was brief.",
    " Visibility remains dependent on the next reporting cycle.",
    " That view could change as market conditions develop.",
    "",
)


def _vary_text(text: str, rng: np.random.Generator, label: str) -> str:
    """Add natural variation without making the class signal perfectly clean."""
    for word, alternatives in SYNONYMS.items():
        if rng.random() < 0.22:
            replacement = str(rng.choice(alternatives))
            text = text.replace(word, replacement)
    if rng.random() < 0.28:
        text = str(rng.choice(PREFIXES)) + text
    if rng.random() < 0.40:
        text += str(rng.choice(FOLLOW_UPS))

    # Borderline language makes the adjacent classes harder to distinguish.
    if label == "Neutral" and rng.random() < 0.18:
        text += str(rng.choice((
            " Some investors found the update disappointing, but no deterioration was confirmed.",
            " There are mild concerns about execution, though the overall position is unchanged.",
            " The result was not exciting, yet it did not point to a clear setback.",
        )))
    elif label == "Positive" and rng.random() < 0.12:
        text += str(rng.choice((
            " A few implementation risks remain.",
            " The improvement is encouraging rather than conclusive.",
        )))
    elif label == "Negative" and rng.random() < 0.12:
        text += str(rng.choice((
            " Management said corrective work is already under way.",
            " Some parts of the business are still performing acceptably.",
        )))

    # A small number of informal artifacts occur in public text.
    if rng.random() < 0.07:
        text = text.replace("and", "&", 1)
    if rng.random() < 0.04:
        words = text.split()
        if len(words) > 8:
            position = int(rng.integers(2, len(words) - 2))
            words[position] = words[position][:-1] if len(words[position]) > 4 else words[position]
            text = " ".join(words)
    return text


def _noisy_label(label: str, rng: np.random.Generator) -> str:
    """Model occasional adjacent-class annotation disagreement."""
    if rng.random() >= 0.04:
        return label
    adjacent = {
        "Positive": ("Neutral", "Neutral"),
        "Neutral": ("Positive", "Negative"),
        "Negative": ("Neutral", "Neutral"),
    }
    return str(rng.choice(adjacent[label]))


def generate_sentiment_dataset(n_documents: int = 1800, years: Tuple[int, ...] = (2022, 2023, 2024, 2025, 2026), seed: int = 42) -> pd.DataFrame:
    """Generate reproducible, timestamped financial text with sentiment labels."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(["Positive", "Neutral", "Negative"], n_documents, p=[0.38, 0.30, 0.32])
    records = []
    for index in range(n_documents):
        company, ticker, sector = COMPANIES[index % len(COMPANIES)]
        year = int(rng.choice(years))
        source_label = str(labels[index])
        label = _noisy_label(source_label, rng)
        source = str(rng.choice(SOURCES))
        topic = str(rng.choice(TOPICS))
        text = str(rng.choice(TEXT_TEMPLATES[source_label])).format(company=company)
        text = _vary_text(text, rng, source_label)
        sarcasm = bool(rng.random() < 0.06)
        misinformation = bool(rng.random() < 0.05)
        if sarcasm:
            text += str(rng.choice((
                " The market calls this a 'brilliant' turnaround, with obvious irony.",
                " One commentator called the result 'excellent' in a plainly sarcastic post.",
                " The upbeat headline drew skepticism from readers who noticed the irony.",
            )))
        if misinformation:
            text += str(rng.choice((
                " Unverified online claims should be treated cautiously.",
                " An unconfirmed social post is circulating and has not been corroborated.",
                " Readers should separate this report from unsupported online claims.",
            )))
        score = {"Positive": 0.75, "Neutral": 0.0, "Negative": -0.75}[label]
        records.append({"document_id": f"DOC_{index + 1:05d}", "company": company, "ticker": ticker, "sector": sector, "publication_date": f"{year}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}", "year": year, "source": source, "topic": topic, "text": text, "sentiment_label": label, "sentiment_score": score, "sarcasm_flag": sarcasm, "misinformation_flag": misinformation})
    return pd.DataFrame(records).sort_values("publication_date").reset_index(drop=True)


# Compatibility bridge for older notebooks.
generate_financial_dataset = generate_sentiment_dataset
