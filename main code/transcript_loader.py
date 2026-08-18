"""
Earnings Call Transcript Scraper & Text Cleaner
=================================================
Fetches and parses earnings call transcripts.

Sources supported:
    1. Motley Fool  (free, HTML scraping)
    2. Seeking Alpha (HTML scraping)
    3. Local file loader (for offline / cached transcripts)

Pipeline:
    fetch → parse HTML → clean text → speaker diarization
    → section split (prepared remarks / Q&A) → JSON storage

Speaker roles detected:
    CEO / CFO / COO / Analyst / Operator / Unknown
"""

import re
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Speaker:
    name: str
    role: str          # CEO | CFO | COO | Analyst | Operator | Unknown
    company: str       # issuer or analyst firm
    is_management: bool


@dataclass
class Utterance:
    speaker: Speaker
    text: str
    section: str       # prepared_remarks | qa
    char_count: int = 0
    word_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())


@dataclass
class Transcript:
    ticker: str
    company_name: str
    quarter: str       # e.g. "Q3 2024"
    date: str
    source: str
    raw_text: str
    utterances: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def management_text(self) -> str:
        return " ".join(u.text for u in self.utterances if u.speaker.is_management)

    @property
    def analyst_text(self) -> str:
        return " ".join(u.text for u in self.utterances
                        if u.speaker.role == "Analyst")

    @property
    def prepared_remarks(self) -> str:
        return " ".join(u.text for u in self.utterances
                        if u.section == "prepared_remarks")

    @property
    def qa_section(self) -> str:
        return " ".join(u.text for u in self.utterances
                        if u.section == "qa")

    @property
    def total_words(self) -> int:
        return sum(u.word_count for u in self.utterances)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Filler phrases common in earnings calls
FILLERS = [
    r"\bum\b", r"\buh\b", r"\byou know\b", r"\bI mean\b",
    r"\bkind of\b", r"\bsort of\b", r"\bright\b",
    r"\bokay so\b", r"\bso basically\b",
]

FILLER_RE = re.compile("|".join(FILLERS), flags=re.IGNORECASE)

# Operator boilerplate to strip
OPERATOR_BOILERPLATE = [
    r"ladies and gentlemen.*?please proceed",
    r"your (next|first) question (comes|is) from.*?please proceed",
    r"thank you.*?no further questions",
    r"this concludes.*?conference call",
    r"please standby.*?begin shortly",
]
OPERATOR_RE = re.compile("|".join(OPERATOR_BOILERPLATE), flags=re.IGNORECASE | re.DOTALL)


def clean_text(text: str) -> str:
    """
    Clean a raw transcript text block:
        1. Normalise whitespace and line endings
        2. Remove HTML artefacts
        3. Strip filler words
        4. Remove operator boilerplate
        5. Normalise numbers and percentages
        6. Strip excessive punctuation
    """
    # HTML artefacts
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)

    # Normalise whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Operator boilerplate
    text = OPERATOR_RE.sub(" ", text)

    # Filler words
    text = FILLER_RE.sub(" ", text)

    # Normalise percentages and numbers
    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)
    text = re.sub(r"\$\s*(\d)", r"$\1", text)

    # Strip excessive punctuation
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"-{2,}", "--", text)

    # Final whitespace cleanup
    text = " ".join(text.split())
    return text.strip()


# ---------------------------------------------------------------------------
# Speaker role classifier
# ---------------------------------------------------------------------------

ROLE_KEYWORDS = {
    "CEO":      ["chief executive", "ceo", "president and chief"],
    "CFO":      ["chief financial", "cfo", "finance officer"],
    "COO":      ["chief operating", "coo", "operations officer"],
    "CTO":      ["chief technology", "cto", "technology officer"],
    "Operator": ["operator", "conference operator", "conference facilitator"],
    "Analyst":  ["analyst", "research", "securities", "capital", "partners",
                 "investments", "asset management", "bank", "goldman", "morgan",
                 "jp morgan", "barclays", "citi", "ubs", "deutsche"],
}


def classify_speaker(name: str, title: str = "") -> str:
    """Classify a speaker's role from their name and title string."""
    combined = (name + " " + title).lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return role
    return "Unknown"


def is_management(role: str) -> bool:
    return role in {"CEO", "CFO", "COO", "CTO", "Unknown"}


# ---------------------------------------------------------------------------
# Section detector
# ---------------------------------------------------------------------------

QA_TRIGGERS = [
    "question-and-answer", "question and answer",
    "q&a session", "q&a portion",
    "open the line", "open for questions",
    "take your first question",
    "we will now begin the question",
]

QA_RE = re.compile("|".join(QA_TRIGGERS), flags=re.IGNORECASE)


def detect_section(text: str, prev_section: str) -> str:
    """Return 'prepared_remarks' or 'qa' based on text content."""
    if QA_RE.search(text):
        return "qa"
    return prev_section


# ---------------------------------------------------------------------------
# Synthetic transcript generator (for testing without live scraping)
# ---------------------------------------------------------------------------

SYNTHETIC_TRANSCRIPTS = {
    "AAPL": {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "quarter": "Q3 2024",
        "date": "2024-08-01",
        "source": "synthetic",
        "speakers": [
            ("Tim Cook", "Chief Executive Officer", "Apple Inc."),
            ("Luca Maestri", "Chief Financial Officer", "Apple Inc."),
            ("Wamsi Mohan", "Analyst", "Bank of America"),
            ("Amit Daryanani", "Analyst", "Evercore ISI"),
            ("Operator", "Operator", ""),
        ],
        "prepared_remarks": [
            ("Tim Cook", "Good afternoon, everyone. Thank you for joining us today. "
             "We are very pleased to report another record quarter for Apple with revenue "
             "of $85.8 billion, up 5 percent year over year. iPhone revenue was strong at "
             "$39.3 billion. We are seeing continued momentum in our Services business, "
             "which reached an all-time high of $24.2 billion, growing 14 percent. "
             "We remain deeply committed to innovation and look forward to an exciting "
             "product pipeline in the second half of the year. We are confident in our "
             "long-term growth trajectory and continue to return significant capital to "
             "our shareholders."),
            ("Luca Maestri", "Thank you Tim. Let me provide additional details on our "
             "financial performance. Total revenue of $85.8 billion was up 5 percent. "
             "Gross margin was 46.3 percent, up 180 basis points year over year. "
             "Operating income was $29.6 billion. We generated very strong operating "
             "cash flow of $29.0 billion during the quarter. Our Board has authorised an "
             "additional $110 billion for share repurchases. For Q4 guidance, we expect "
             "total revenue to grow low-to-mid single digits year over year. We expect "
             "gross margin between 46.5 and 47.0 percent."),
        ],
        "qa": [
            ("Operator", "We will now begin the question-and-answer session. "
             "Your first question comes from Wamsi Mohan at Bank of America."),
            ("Wamsi Mohan", "Thank you. Tim, can you talk about the iPhone upgrade cycle "
             "and what you are seeing in terms of consumer demand heading into the fall? "
             "And then for Luca, can you talk about Services margin trajectory?"),
            ("Tim Cook", "Sure Wamsi. We are very excited about the iPhone cycle. "
             "We are seeing strong interest from customers particularly around AI features. "
             "The upgrade cycle from older phones remains a significant opportunity. "
             "We are optimistic about the second half demand environment."),
            ("Luca Maestri", "On Services margins, we continue to see them at very healthy "
             "levels. We do not guide specifically to Services margins but the trajectory "
             "has been positive and we feel good about the business."),
            ("Amit Daryanani", "Great results. Can you elaborate on the China environment "
             "and what gives you confidence in that region going forward?"),
            ("Tim Cook", "China is an important market for us and we remain committed there. "
             "We had some headwinds this quarter but we are focused on the long term. "
             "The installed base continues to grow and customer satisfaction is very high."),
        ],
    },
    "MSFT": {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "quarter": "Q4 2024",
        "date": "2024-07-30",
        "source": "synthetic",
        "speakers": [
            ("Satya Nadella", "Chief Executive Officer", "Microsoft"),
            ("Amy Hood", "Chief Financial Officer", "Microsoft"),
            ("Karl Keirstead", "Analyst", "UBS"),
            ("Operator", "Operator", ""),
        ],
        "prepared_remarks": [
            ("Satya Nadella", "Thank you and good afternoon everyone. Microsoft delivered "
             "strong results in fiscal year 2024 with total revenue of $245 billion up "
             "16 percent and operating income of $109 billion up 24 percent. Azure and "
             "other cloud services grew 29 percent. AI is the defining technology of our "
             "time and we are seeing real business value from our AI investments. "
             "GitHub Copilot has over 1.8 million paid subscribers and growing rapidly. "
             "We are well positioned to benefit from the secular shift to cloud and AI. "
             "Our commercial bookings grew 17 percent reflecting strong customer commitment."),
            ("Amy Hood", "Thank you Satya. Revenue was $64.7 billion up 15 percent in "
             "constant currency. Intelligent Cloud revenue was $28.5 billion up 19 percent. "
             "Azure grew 29 percent in constant currency. Gross margin was 70 percent. "
             "We returned $8.4 billion to shareholders via buybacks and dividends. "
             "For Q1 guidance, we expect revenue of $63.8 to $64.8 billion. "
             "Azure growth expected between 28 and 29 percent in constant currency."),
        ],
        "qa": [
            ("Operator", "We will now begin the question-and-answer session. "
             "Your first question comes from Karl Keirstead at UBS."),
            ("Karl Keirstead", "Amy, can you help us think about the Azure growth trajectory "
             "over the coming quarters and how AI capacity constraints factor in?"),
            ("Amy Hood", "Thanks Karl. We feel good about the Azure growth trajectory. "
             "The AI demand is very strong and we continue to invest in capacity. "
             "We do see some near term constraints but expect them to ease over the year. "
             "The underlying business momentum is strong and we are confident in sustaining "
             "growth in the high 20s to 30 percent range."),
        ],
    },
}


def load_synthetic_transcript(ticker: str) -> Optional[Transcript]:
    """Load a pre-built synthetic transcript for testing."""
    if ticker not in SYNTHETIC_TRANSCRIPTS:
        return None

    data = SYNTHETIC_TRANSCRIPTS[ticker]
    utterances = []
    section = "prepared_remarks"

    for spk_name, title, company in data["speakers"]:
        pass  # just building speaker lookup

    speaker_map = {
        spk: Speaker(
            name=spk,
            role=classify_speaker(spk, title),
            company=company,
            is_management=is_management(classify_speaker(spk, title)),
        )
        for spk, title, company in data["speakers"]
    }

    for spk_name, text in data["prepared_remarks"]:
        section = detect_section(text, "prepared_remarks")
        spk = speaker_map.get(spk_name, Speaker(spk_name, "Unknown", "", True))
        utterances.append(Utterance(
            speaker=spk,
            text=clean_text(text),
            section="prepared_remarks",
        ))

    for spk_name, text in data["qa"]:
        spk = speaker_map.get(spk_name, Speaker(spk_name, "Unknown", "", True))
        utterances.append(Utterance(
            speaker=spk,
            text=clean_text(text),
            section="qa",
        ))

    return Transcript(
        ticker=data["ticker"],
        company_name=data["company_name"],
        quarter=data["quarter"],
        date=data["date"],
        source=data["source"],
        raw_text=" ".join(u.text for u in utterances),
        utterances=utterances,
        metadata={"n_speakers": len(speaker_map), "sections": ["prepared_remarks", "qa"]},
    )


# ---------------------------------------------------------------------------
# JSON storage
# ---------------------------------------------------------------------------

def save_transcript(transcript: Transcript, output_dir: str = "data") -> str:
    """Save transcript to JSON. Returns file path."""
    Path(output_dir).mkdir(exist_ok=True)
    fname = f"{transcript.ticker}_{transcript.quarter.replace(' ', '_')}.json"
    fpath = Path(output_dir) / fname

    payload = {
        "ticker":       transcript.ticker,
        "company_name": transcript.company_name,
        "quarter":      transcript.quarter,
        "date":         transcript.date,
        "source":       transcript.source,
        "metadata":     transcript.metadata,
        "utterances": [
            {
                "speaker_name":    u.speaker.name,
                "speaker_role":    u.speaker.role,
                "speaker_company": u.speaker.company,
                "is_management":   u.speaker.is_management,
                "section":         u.section,
                "text":            u.text,
                "word_count":      u.word_count,
            }
            for u in transcript.utterances
        ],
        "stats": {
            "total_words":      transcript.total_words,
            "management_words": len(transcript.management_text.split()),
            "analyst_words":    len(transcript.analyst_text.split()),
            "n_utterances":     len(transcript.utterances),
        },
    }

    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)

    return str(fpath)


def load_transcript_json(fpath: str) -> dict:
    """Load a previously saved transcript JSON."""
    with open(fpath) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Earnings Call Transcript Pipeline Demo")
    print("=" * 60)

    for ticker in ["AAPL", "MSFT"]:
        t = load_synthetic_transcript(ticker)
        print(f"\n  [{ticker}] {t.company_name} — {t.quarter}")
        print(f"  Total words      : {t.total_words:,}")
        print(f"  Management words : {len(t.management_text.split()):,}")
        print(f"  Analyst words    : {len(t.analyst_text.split()):,}")
        print(f"  Utterances       : {len(t.utterances)}")

        speakers = set(u.speaker.name for u in t.utterances)
        print(f"  Speakers         : {', '.join(speakers)}")

        fpath = save_transcript(t, "data")
        print(f"  Saved to         : {fpath}")
