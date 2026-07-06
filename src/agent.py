"""
agent.py
--------
CLI entry point for the Resume Screening Agent.

Usage:
    python src/agent.py --jd data/job_description.txt --resumes data/resumes \
        --out output/ranked_candidates

This wires the full pipeline:
  Job description + resumes  --(parser.py)-->  structured fields
                              --(scorer.py)-->  NLP + skill + experience score
                              --(llm.py)    -->  natural-language reasoning
                              --> ranked JSON + CSV
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_all_resumes
from scorer import rank_resumes, parse_job_description
from llm import generate_reasoning


def jd_summary_string(jd_info: dict) -> str:
    return (
        f"requires ~{jd_info['required_years']} years experience; "
        f"required skills: {', '.join(jd_info['required_skills']) or 'none detected'}; "
        f"preferred skills: {', '.join(jd_info['preferred_skills']) or 'none detected'}"
    )


def write_json(results: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def write_csv(results: list, path: str):
    if not results:
        return
    fieldnames = [
        "rank", "name", "file", "score",
        "matched_skills", "missing_required_skills",
        "years_experience", "education", "reasoning", "reasoning_mode",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            row["matched_skills"] = "; ".join(row["matched_skills"])
            row["missing_required_skills"] = "; ".join(row["missing_required_skills"])
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Resume Screening Agent")
    ap.add_argument("--jd", required=True, help="Path to job description text file")
    ap.add_argument("--resumes", required=True, help="Folder containing resumes (PDF/DOCX/TXT)")
    ap.add_argument("--out", default="output/ranked_candidates", help="Output path prefix (no extension)")
    args = ap.parse_args()

    with open(args.jd, "r", encoding="utf-8", errors="ignore") as f:
        jd_text = f.read()

    print(f"[1/4] Reading job description from {args.jd}")
    jd_info = parse_job_description(jd_text)
    summary = jd_summary_string(jd_info)
    print(f"      {summary}")

    print(f"[2/4] Parsing resumes from {args.resumes}")
    resumes = parse_all_resumes(args.resumes)
    print(f"      Parsed {len(resumes)} resume(s)")
    if len(resumes) == 0:
        print("      No resumes found — check the folder path.")
        sys.exit(1)

    print("[3/4] Scoring and ranking candidates")
    ranked = rank_resumes(resumes, jd_text)

    print("[4/4] Generating reasoning for each candidate")
    for r in ranked:
        generate_reasoning(r, summary)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json_path = f"{args.out}.json"
    csv_path = f"{args.out}.csv"
    write_json(ranked, json_path)
    write_csv(ranked, csv_path)

    print("\n=== Ranked Shortlist ===")
    for r in ranked:
        print(f"#{r['rank']:<2} {r['score']:>5.1f}  {r['name']:<30} ({r['file']})")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
