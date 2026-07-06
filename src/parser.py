"""
parser.py
---------
Extracts raw text from resumes (PDF / DOCX / TXT) and pulls out
structured fields: candidate name, skills, years of experience, education.

Design choice: extraction is fully rule-based (regex + keyword matching)
so the pipeline never depends on an LLM call to function. This keeps the
core agent fast, deterministic, and free to run. An optional LLM pass
(see llm.py) can refine/augment these fields when an API key is present.
"""

import os
import re
from pathlib import Path

import pdfplumber
import docx


# ---------------------------------------------------------------------------
# Text extraction per file type
# ---------------------------------------------------------------------------

def extract_text(file_path: str) -> str:
    """Return plain text content of a resume, regardless of PDF/DOCX/TXT."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        text_chunks = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)
        return "\n".join(text_chunks)

    elif ext == ".docx":
        d = docx.Document(file_path)
        return "\n".join(p.text for p in d.paragraphs)

    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported resume format: {ext}")


# ---------------------------------------------------------------------------
# Canonical skills list used for matching (kept small & explicit on purpose —
# see README tradeoffs for why this beats a giant fuzzy taxonomy for a
# 24-hour build).
# ---------------------------------------------------------------------------

SKILL_SYNONYMS = {
    "python": ["python"],
    "sql": ["sql", "postgresql", "mysql", "t-sql", "pl/sql"],
    "machine learning": ["machine learning", "ml ", "scikit-learn", "sklearn"],
    "deep learning": ["deep learning", "pytorch", "tensorflow", "keras"],
    "nlp": ["nlp", "natural language processing", "spacy", "transformers"],
    "data visualization": ["data visualization", "tableau", "power bi", "matplotlib", "seaborn"],
    "statistics": ["statistics", "statistical", "hypothesis testing", "a/b testing"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "excel": ["excel", "spreadsheets"],
    "cloud": ["aws", "azure", "gcp", "google cloud", "cloud computing"],
    "docker": ["docker", "kubernetes", "containerization"],
    "communication": ["communication", "stakeholder management", "presentation"],
    "leadership": ["leadership", "mentored", "led a team", "managed a team"],
    "etl": ["etl", "data pipeline", "data pipelines", "airflow"],
    "spark": ["spark", "pyspark", "hadoop"],
    "r programming": [" r programming", "r language", "rstudio"],
    "java": ["java"],
    "javascript": ["javascript", "typescript", "react", "node.js"],
    "api development": ["rest api", "api development", "flask", "fastapi", "django"],
    "git": ["git", "github", "version control"],
}


def _contains_variant(lowered_text: str, variant: str) -> bool:
    """Word-boundary aware match so short tokens (e.g. 'excel') don't
    false-positive inside unrelated words (e.g. 'excellent')."""
    stripped = variant.strip()
    if not stripped:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(stripped) + r"(?![a-z0-9])"
    return re.search(pattern, lowered_text) is not None


def extract_skills(text: str) -> list:
    """Keyword/synonym match against a canonical skill taxonomy."""
    lowered = text.lower()
    found = []
    for canonical, variants in SKILL_SYNONYMS.items():
        if any(_contains_variant(lowered, v) for v in variants):
            found.append(canonical)
    return sorted(found)


def extract_years_experience(text: str) -> float:
    """
    Estimate total years of experience.

    Strategy (in priority order):
    1. Explicit phrase like "5+ years of experience" / "3 years experience".
    2. Fallback: sum of date ranges in a work-history section (e.g. 2019-2022).
    Rule-based and imperfect by design — see README tradeoffs.
    """
    lowered = text.lower()

    explicit = re.findall(r"(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience", lowered)
    if explicit:
        return max(float(x) for x in explicit)

    # Fallback: look for year ranges like 2019 - 2022 / 2019-Present
    ranges = re.findall(r"(20\d{2}|19\d{2})\s*[-–to]+\s*(20\d{2}|present|current)", lowered)
    total = 0.0
    for start, end in ranges:
        start_y = int(start)
        end_y = 2026 if end in ("present", "current") else int(end)
        if end_y >= start_y:
            total += (end_y - start_y)
    return round(total, 1)


EDUCATION_KEYWORDS = [
    ("phd", "PhD"),
    ("doctorate", "PhD"),
    ("m.tech", "Master's"),
    ("mtech", "Master's"),
    ("msc", "Master's"),
    ("m.sc", "Master's"),
    ("master", "Master's"),
    ("mba", "MBA"),
    ("b.tech", "Bachelor's"),
    ("btech", "Bachelor's"),
    ("bsc", "Bachelor's"),
    ("b.sc", "Bachelor's"),
    ("bachelor", "Bachelor's"),
    ("b.e.", "Bachelor's"),
]


def extract_education(text: str) -> str:
    lowered = text.lower()
    for keyword, label in EDUCATION_KEYWORDS:
        if keyword in lowered:
            return label
    return "Not specified"


def guess_candidate_name(text: str, fallback: str) -> str:
    """
    Heuristic: the first non-empty line of a resume is very commonly the
    candidate's name. Falls back to the filename if that line looks wrong
    (too long, contains @ or digits typical of contact lines).
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= 40 and "@" not in line and not re.search(r"\d", line):
            return line
        break
    return fallback


def parse_resume(file_path: str) -> dict:
    text = extract_text(file_path)
    name = guess_candidate_name(text, fallback=Path(file_path).stem)
    return {
        "file": Path(file_path).name,
        "name": name,
        "raw_text": text,
        "skills": extract_skills(text),
        "years_experience": extract_years_experience(text),
        "education": extract_education(text),
    }


def parse_all_resumes(folder: str) -> list:
    resumes = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if Path(fname).suffix.lower() in (".pdf", ".docx", ".txt", ".md"):
            resumes.append(parse_resume(fpath))
    return resumes
