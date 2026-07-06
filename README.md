# Resume Screening Agent

**My agent takes a job description + a folder of resumes (PDF/DOCX/TXT) and
produces a ranked, scored shortlist of candidates with clear reasoning for
each rank.**

Built for the Rooman 24-Hour AI Agent Challenge — Resume Screening Agent track.

---

## 1. What it does

```
Job Description + Resumes (PDF/DOCX/TXT)
              │
              ▼
   Parse resumes → extract skills, years of experience, education
              │
              ▼
   Score each resume against the JD:
     - Skill overlap (required vs preferred)
     - TF-IDF cosine similarity (NLP text similarity)
     - Experience match
              │
              ▼
   Rank candidates + generate a plain-English reasoning sentence
              │
              ▼
   ranked_candidates.json / ranked_candidates.csv
```

## 2. Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd resume-screening-agent

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Optional: enable LLM-phrased reasoning

The agent works fully **without** any API key (see [Design Choice #1](#4-design-choices--tradeoffs)
below). If you want Claude to phrase the per-candidate reasoning as a more
natural sentence instead of the built-in template:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
export $(cat .env | xargs)      # or use `python-dotenv` / your shell's method
```

## 3. Running the agent

Sample data is already included in `data/` (1 job description + 12 resumes
spanning PDF, DOCX, and TXT).

```bash
python src/agent.py --jd data/job_description.txt --resumes data/resumes --out output/ranked_candidates
```

**Output:**
```
[1/4] Reading job description from data/job_description.txt
      requires ~4.0 years experience; required skills: communication, data visualization, machine learning, python, sql; preferred skills: cloud, deep learning, etl, nlp, statistics
[2/4] Parsing resumes from data/resumes
      Parsed 12 resume(s)
[3/4] Scoring and ranking candidates
[4/4] Generating reasoning for each candidate

=== Ranked Shortlist ===
#1   72.8  Priya Sharma                   (priya_sharma.pdf)
#2   71.3  Ananya Iyer                    (ananya_iyer.txt)
#3   57.1  Rahul Verma                    (rahul_verma.docx)
#4   55.9  Karthik Reddy                  (karthik_reddy.pdf)
#5   55.3  Meera Pillai                   (meera_pillai.pdf)
#6   48.7  Suresh Kumar                   (suresh_kumar.txt)
#7   40.1  Lakshmi Rao                    (lakshmi_rao.docx)
#8   37.5  Vikram Singh                   (vikram_singh.docx)
#9   30.4  Sneha Nair                     (sneha_nair.txt)
#10  26.1  Rohan Das                      (rohan_das.txt)
#11  17.3  Divya Menon                    (divya_menon.txt)
#12  15.5  Arjun Kapoor                   (arjun_kapoor.txt)

Saved: output/ranked_candidates.json
Saved: output/ranked_candidates.csv
```

Full sample outputs are committed at `output/ranked_candidates.json` and
`output/ranked_candidates.csv` so you can inspect results without re-running
anything.

To run on your own data, point `--jd` and `--resumes` at your own files —
no code changes needed.

## 4. Approach / scoring method

Each resume gets a **single 0–100 score** built from three independent
signals, so the ranking is explainable rather than a single opaque LLM
judgment:

| Component | Weight | What it measures |
|---|---|---|
| Skill match | 45% | Overlap between the candidate's extracted skills and the JD's required/preferred skills (required weighted 4x higher than preferred) |
| TF-IDF cosine similarity | 35% | Classic NLP similarity between the full resume text and full JD text (scikit-learn `TfidfVectorizer` + cosine similarity) — this is what catches relevant context the keyword list misses |
| Experience match | 20% | Candidate's years of experience vs. the JD's required years, capped at 100% |

Skills, years of experience, and education are extracted with **regex +
keyword matching** against a small canonical skill taxonomy with synonyms
(e.g. "ML" / "scikit-learn" both map to `machine learning`). Word-boundary
matching is used throughout so short keywords don't false-positive inside
unrelated words (e.g. "excel" no longer matches inside "excellent").

Each candidate's JSON/CSV row also includes the three component scores and
the specific matched / missing skills, so a recruiter can see *why* a
candidate ranked where they did, not just the final number.

## 5. Design choices & tradeoffs

1. **Core scoring never depends on an LLM call.** Skill/experience
   extraction and the TF-IDF similarity are 100% deterministic and run
   offline. The LLM (Claude) is used only to *phrase* the reasoning
   sentence more naturally — if no `ANTHROPIC_API_KEY` is set, or the API
   call fails for any reason, the agent falls back to a template that uses
   the exact same structured data, and still runs correctly end-to-end.
   This was a deliberate reliability tradeoff: a resume screener that
   silently breaks when an API is down or rate-limited is worse than one
   that degrades gracefully to slightly less natural phrasing.

2. **Keyword taxonomy vs. embeddings.** A curated ~20-skill synonym map
   (`SKILL_SYNONYMS` in `parser.py`) was used instead of a large fuzzy
   taxonomy or embedding-based skill extraction. This is faster to build,
   fully explainable, and easy to extend, but it means skills outside the
   list (or phrased in an unusual way) won't be picked up. The TF-IDF
   similarity component partially compensates for this by scoring on the
   full text, not just the keyword list.

3. **Years-of-experience extraction is heuristic.** It looks for explicit
   phrases ("5+ years of experience") first, then falls back to summing
   date ranges found in the text (e.g. "2019–2023"). This works well for
   resumes with a clear format but can over/under-count for resumes with
   overlapping roles, career gaps, or unconventional date formats.

4. **Name extraction assumes the first non-empty line is the candidate's
   name**, which holds for the vast majority of resume templates but can
   fail on resumes that lead with a header/logo/address instead.

5. **PDF parsing uses `pdfplumber`**, which handles text-based PDFs well
   but will return empty/partial text for scanned image-only resumes (no
   OCR step is included in this build).

### What I'd improve with more time
- Add an OCR fallback (e.g. `pytesseract`) for scanned/image PDFs.
- Replace the keyword skill list with a small embedding-based skill matcher
  (e.g. sentence-transformers) so skills phrased unusually are still caught.
- Add a config file so the score weights (45/35/20) and skill taxonomy are
  editable without touching code.
- Add unit tests for the parsing edge cases (multi-column resumes, resumes
  without a clear "Skills" section, non-English resumes).
- Batch and cache LLM reasoning calls to reduce latency/cost on large runs.

## 6. Project structure

```
resume-screening-agent/
├── src/
│   ├── agent.py       # CLI entry point — wires the full pipeline
│   ├── parser.py       # Resume/JD text extraction + field extraction
│   ├── scorer.py       # NLP similarity + skill/experience scoring
│   └── llm.py          # Optional Claude-based reasoning + offline fallback
├── data/
│   ├── job_description.txt
│   └── resumes/         # 12 sample resumes (PDF/DOCX/TXT mix)
├── output/
│   ├── ranked_candidates.json
│   └── ranked_candidates.csv
├── requirements.txt
├── .env.example
└── README.md
```

## 7. Sample data note

The 12 sample resumes are synthetic (created for this challenge) and span
strong, moderate, and weak matches against the sample JD — including a
frontend developer and a fresh graduate — specifically to sanity-check that
the ranking correctly separates clearly-unqualified candidates from
strong ones.
"# resume-screening-agent" 
