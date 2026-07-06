"""
llm.py
------
Generates a short, human-readable "why this rank" explanation for each
scored candidate.

Design choice: the CORE scoring pipeline (parser.py + scorer.py) never
needs an LLM to produce a correct, ranked, working output — that is a
deliberate reliability choice for a 24-hour build (see README tradeoffs).

The LLM is used only for the *reasoning sentence*, an enhancement layer:
- If ANTHROPIC_API_KEY is set, we call Claude to turn the structured score
  breakdown into a natural, recruiter-friendly sentence.
- If no key is set, we fall back to a deterministic template that uses the
  exact same structured data. The agent still runs end-to-end and produces
  real, correct reasoning either way — it just reads more naturally with
  the LLM turned on.
"""

import os


def _template_reasoning(result: dict) -> str:
    matched = result["matched_skills"]
    missing = result["missing_required_skills"]
    comp = result["component_scores"]

    parts = []
    if matched:
        parts.append(f"matches {len(matched)} required/preferred skill(s) ({', '.join(matched[:5])}{'...' if len(matched) > 5 else ''})")
    else:
        parts.append("matches none of the JD's detected skills")

    if missing:
        parts.append(f"is missing {len(missing)} required skill(s) ({', '.join(missing[:4])}{'...' if len(missing) > 4 else ''})")

    parts.append(f"has {result['years_experience']} years of experience")
    parts.append(f"education: {result['education']}")

    return (
        f"Ranked #{result['rank']} with a score of {result['score']}/100. "
        f"Candidate {', '.join(parts)}. "
        f"(skill match {comp['skill_match']}%, text similarity {comp['tfidf_similarity']}%, "
        f"experience match {comp['experience_match']}%)"
    )


def _llm_reasoning(result: dict, jd_summary: str) -> str:
    """Calls Anthropic Claude to phrase the reasoning naturally. Raises on
    any failure so the caller can fall back to the template."""
    import anthropic  # imported lazily so the module still loads without the package

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    prompt = (
        "You are helping a recruiter understand a resume-screening result. "
        "Write ONE concise sentence (max 35 words) explaining why this "
        "candidate received this rank/score. Be specific and factual — use "
        "only the data given, do not invent skills or experience.\n\n"
        f"Job requirements summary: {jd_summary}\n\n"
        f"Candidate data: {result}\n\n"
        "Respond with only the sentence, no preamble."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    if not text:
        raise ValueError("Empty LLM response")
    return text


def generate_reasoning(result: dict, jd_summary: str) -> dict:
    """Attaches a 'reasoning' string to the result dict. Tries the LLM first
    (if an API key is configured), falls back to the template otherwise."""
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    reasoning = None
    mode = "template"

    if use_llm:
        try:
            reasoning = _llm_reasoning(result, jd_summary)
            mode = "llm"
        except Exception as e:  # noqa: BLE001 - any failure -> safe fallback
            reasoning = None
            mode = f"template (llm fallback: {type(e).__name__})"

    if reasoning is None:
        reasoning = _template_reasoning(result)

    result["reasoning"] = reasoning
    result["reasoning_mode"] = mode
    return result
