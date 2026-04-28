#!/usr/bin/env python3
# main.py
# ─────────────────────────────────────────────────────────────────────────────
# AI Resume Screening System — Main Entry Point
# Groq API + LangSmith tracing
#
# Usage:
#   python main.py
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langsmith import traceable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()


# ── Validate required env vars ────────────────────────────────────────────────
def _missing_or_placeholder(key: str) -> bool:
    value = os.getenv(key, "").strip()
    return not value or value.startswith("your_")


def _check_env() -> None:
    required = {
        "GROQ_API_KEY": "https://console.groq.com/keys",
        "LANGCHAIN_API_KEY":        "https://smith.langchain.com → Settings → API Keys",
    }
    missing = [(k, url) for k, url in required.items() if _missing_or_placeholder(k)]
    if missing:
        print("\n❌  Missing environment variables:\n")
        for k, url in missing:
            print(f"     {k}")
            print(f"       → Create it at: {url}\n")
        print("  Copy .env.example → .env and fill in your keys, then re-run.\n")
        sys.exit(1)

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    if tracing != "true":
        print("⚠️   LANGCHAIN_TRACING_V2 is not 'true' — LangSmith tracing DISABLED.\n")

_check_env()

# ── Import project modules (after env validation) ────────────────────────────
from chains.screening_chain import ResumeScreeningPipeline, ScreeningResult
from resumes.sample_data import CANDIDATES, JOB_DESCRIPTION


# ── Pretty-print helper ───────────────────────────────────────────────────────
def _print_result(result: ScreeningResult, label: str) -> None:
    exp = result.explanation
    sb  = result.score_result.get("score_breakdown", {})

    print(f"\n{'═'*65}")
    print(f"  CANDIDATE : {result.candidate_name}  ({label})")
    print(f"  SCORE     : {result.score}/100   │   TIER: {result.tier}")
    print(f"{'─'*65}")
    print(f"  Score Breakdown:")
    print(f"    Required Skills : {sb.get('required_skills_score', '?')}/40")
    print(f"    Experience      : {sb.get('experience_score', '?')}/20")
    print(f"    Tools/Frameworks: {sb.get('tools_score', '?')}/20")
    print(f"    Education       : {sb.get('education_score', '?')}/10")
    print(f"    Bonus Skills    : {sb.get('bonus_score', '?')}/10")
    print(f"{'─'*65}")

    print(f"  ✅ Strengths:")
    for s in exp.get("strengths", []):
        print(f"     • {s}")

    print(f"\n  ❌ Gaps:")
    for g in exp.get("gaps", []):
        print(f"     • {g}")

    print(f"\n  📋 Summary:")
    print(f"     {exp.get('summary', 'N/A')}")
    print(f"\n  🎯 Recommendation: {result.recommendation}")
    print(f"{'═'*65}")


def _save_results(results: list[ScreeningResult], output_path: str) -> None:
    data = [
        {
            "candidate_name":        r.candidate_name,
            "score":                 r.score,
            "tier":                  r.tier,
            "hiring_recommendation": r.recommendation,
            "score_breakdown":       r.score_result.get("score_breakdown", {}),
            "matched_skills":        r.matching_analysis.get("matched_skills", []),
            "missing_required":      r.matching_analysis.get("missing_required_skills", []),
            "strengths":             r.explanation.get("strengths", []),
            "gaps":                  r.explanation.get("gaps", []),
            "summary":               r.explanation.get("summary", ""),
        }
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  💾 Results saved → {output_path}")


# ── LangSmith traceable wrapper ───────────────────────────────────────────────
@traceable(
    name="resume_screening_run",
    tags=["resume-screening", "groq", "assignment"],
)
def run_screening(
    pipeline: ResumeScreeningPipeline,
    resume_text: str,
    job_description: str,
    candidate_label: str,
) -> ScreeningResult:
    """
    LangSmith @traceable wrapper.
    Every call appears as a named, tagged trace in the LangSmith UI
    with all 5 pipeline steps visible as nested spans.
    """
    return pipeline.run(
        resume_text     = resume_text,
        job_description = job_description,
        candidate_label = candidate_label,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "█"*65)
    print("  AI RESUME SCREENING SYSTEM")
    print("  Powered by LangChain + Groq | Traced by LangSmith")
    print("█"*65)
    print(f"  Run started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model        : {os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')}")
    print(f"  LangSmith    : {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print(f"  Tracing      : {os.getenv('LANGCHAIN_TRACING_V2', 'false')}")
    print(f"  Candidates   : {len(CANDIDATES)}")

    # ── Instantiate pipeline ──────────────────────────────────────────────────
    pipeline = ResumeScreeningPipeline()

    all_results: list[ScreeningResult] = []

    # ── Run screening for each candidate ─────────────────────────────────────
    for label, resume_text in CANDIDATES.items():
        result = run_screening(
            pipeline        = pipeline,
            resume_text     = resume_text,
            job_description = JOB_DESCRIPTION,
            candidate_label = label,
        )
        all_results.append(result)
        _print_result(result, label)

    # ── Ranking summary ───────────────────────────────────────────────────────
    print("\n\n  🏆 CANDIDATE RANKING SUMMARY")
    print(f"  {'─'*55}")
    for rank, r in enumerate(sorted(all_results, key=lambda r: r.score, reverse=True), 1):
        bar = "█" * (r.score // 5) + "░" * (20 - r.score // 5)
        print(f"  #{rank}  {r.candidate_name:<30} {r.score:>3}/100  {bar}")
        print(f"       Tier: {r.tier:<20} Rec: {r.recommendation}")
    print(f"  {'─'*55}")

    # ── Save JSON output ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_results(all_results, f"screening_results_{ts}.json")

    print("\n  ✅ All done!")
    print(f"  🔗 View traces: https://smith.langchain.com/projects/{os.getenv('LANGCHAIN_PROJECT', 'default')}\n")


if __name__ == "__main__":
    main()
