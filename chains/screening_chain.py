# chains/screening_chain.py
# ─────────────────────────────────────────────────────────────────────────────
# Modular LangChain chains built with LCEL (LangChain Expression Language).
# Uses Groq through its OpenAI-compatible API.
#
# Pipeline:
#   Resume Text ──► extract_skills_chain
#                      │
#   Job Description ──► extract_job_chain
#                      │
#                   matching_chain
#                      │
#                   scoring_chain
#                      │
#                   explanation_chain
#                      │
#                   ScreeningResult (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from prompts.templates import (
    EXPLANATION_TEMPLATE,
    JOB_REQUIREMENTS_TEMPLATE,
    MATCHING_TEMPLATE,
    SCORING_TEMPLATE,
    SKILL_EXTRACTION_TEMPLATE,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# ─────────────────────────────────────────────────────────────────────────────


# ── Pydantic model for the final structured output ────────────────────────────
class ScreeningResult(BaseModel):
    """Structured result returned for each candidate after the full pipeline."""

    candidate_name: str
    resume_profile: dict
    job_requirements: dict
    matching_analysis: dict
    score_result: dict
    explanation: dict

    @property
    def score(self) -> int:
        return self.score_result.get("score", 0)

    @property
    def tier(self) -> str:
        return self.score_result.get("tier", "Unknown")

    @property
    def recommendation(self) -> str:
        return self.explanation.get("hiring_recommendation", "N/A")


# ── Utility: safely parse JSON from LLM response ─────────────────────────────
def _parse_json(text: str) -> dict:
    """
    Strip markdown fences, then extract and parse the first JSON object found.
    Falls back to an error dict so the pipeline never crashes silently.
    """
    # 1. Remove ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # 2. Try to find the first {...} block (handles models that add preamble text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Return a structured error so the pipeline can continue and be debugged
        return {"_parse_error": str(e), "_raw_response": text[:500]}


# ── LLM / Groq configuration ─────────────────────────────────────────────────
def build_llm():
    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
        temperature=0.01,
        max_tokens=1024,
    )



# ── Chain builders (LCEL pattern: prompt | llm | parser) ─────────────────────

def build_skill_extraction_chain(llm: ChatOpenAI):
    """Chain 1: Resume text → extracted candidate profile (JSON dict)."""
    return (
        SKILL_EXTRACTION_TEMPLATE
        | llm
        | StrOutputParser()
        | RunnableLambda(_parse_json)
    )


def build_job_requirements_chain(llm: ChatOpenAI):
    """Chain 2: Job description → structured job requirements (JSON dict)."""
    return (
        JOB_REQUIREMENTS_TEMPLATE
        | llm
        | StrOutputParser()
        | RunnableLambda(_parse_json)
    )


def build_matching_chain(llm: ChatOpenAI):
    """Chain 3: Candidate profile + job requirements → matching analysis (JSON dict)."""
    return (
        MATCHING_TEMPLATE
        | llm
        | StrOutputParser()
        | RunnableLambda(_parse_json)
    )


def build_scoring_chain(llm: ChatOpenAI):
    """Chain 4: Matching analysis → score + breakdown (JSON dict)."""
    return (
        SCORING_TEMPLATE
        | llm
        | StrOutputParser()
        | RunnableLambda(_parse_json)
    )


def build_explanation_chain(llm: ChatOpenAI):
    """Chain 5: Score + matching → explainable AI report (JSON dict)."""
    return (
        EXPLANATION_TEMPLATE
        | llm
        | StrOutputParser()
        | RunnableLambda(_parse_json)
    )


# ── Full orchestrated pipeline ────────────────────────────────────────────────
class ResumeScreeningPipeline:
    """
    Orchestrates the full 5-step screening pipeline for a single candidate.

    Usage:
        pipeline = ResumeScreeningPipeline()
        result   = pipeline.run(resume_text, job_description, candidate_label)
    """

    def __init__(self):
        print(f"  🔧 Loading Groq model: {GROQ_MODEL}")
        self.llm = build_llm()
        self.skill_chain   = build_skill_extraction_chain(self.llm)
        self.job_chain     = build_job_requirements_chain(self.llm)
        self.match_chain   = build_matching_chain(self.llm)
        self.score_chain   = build_scoring_chain(self.llm)
        self.explain_chain = build_explanation_chain(self.llm)
        print("  ✅ Model ready.\n")

    # ── private step helpers ──────────────────────────────────────────────────

    def _extract_skills(self, resume_text: str) -> dict:
        """Step 1 – Extract skills and profile from raw resume text."""
        return self.skill_chain.invoke({"resume_text": resume_text})

    def _extract_job_requirements(self, job_description: str) -> dict:
        """Step 2 – Extract structured requirements from job description."""
        return self.job_chain.invoke({"job_description": job_description})

    def _match(self, candidate_profile: dict, job_requirements: dict) -> dict:
        """Step 3 – Match candidate profile against job requirements."""
        return self.match_chain.invoke(
            {
                "candidate_profile": json.dumps(candidate_profile, indent=2),
                "job_requirements":  json.dumps(job_requirements, indent=2),
            }
        )

    def _score(self, matching_analysis: dict, job_requirements: dict) -> dict:
        """Step 4 – Assign score using the rubric in the prompt."""
        return self.score_chain.invoke(
            {
                "matching_analysis": json.dumps(matching_analysis, indent=2),
                "job_requirements":  json.dumps(job_requirements, indent=2),
            }
        )

    def _explain(
        self,
        candidate_name: str,
        score_result: dict,
        matching_analysis: dict,
    ) -> dict:
        """Step 5 – Generate explainable AI output for hiring manager."""
        return self.explain_chain.invoke(
            {
                "candidate_name":    candidate_name,
                "score_result":      json.dumps(score_result, indent=2),
                "matching_analysis": json.dumps(matching_analysis, indent=2),
            }
        )

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        resume_text: str,
        job_description: str,
        candidate_label: str = "Candidate",
    ) -> ScreeningResult:
        """
        Execute the full 5-step screening pipeline and return a ScreeningResult.

        Args:
            resume_text      : Raw resume as a plain-text string.
            job_description  : Raw job description as a plain-text string.
            candidate_label  : Human-readable label used in logs and LangSmith.

        Returns:
            ScreeningResult  : Pydantic model with all pipeline outputs.
        """
        print(f"\n{'─'*60}")
        print(f"  Processing: {candidate_label}")
        print(f"{'─'*60}")

        # Step 1 – Skill extraction
        print("  [1/5] Extracting skills from resume…")
        resume_profile = self._extract_skills(resume_text)
        candidate_name = resume_profile.get("candidate_name", candidate_label)

        # Step 2 – Job requirements extraction
        print("  [2/5] Extracting job requirements…")
        job_requirements = self._extract_job_requirements(job_description)

        # Step 3 – Matching
        print("  [3/5] Matching candidate against job requirements…")
        matching_analysis = self._match(resume_profile, job_requirements)

        # Step 4 – Scoring
        print("  [4/5] Scoring the candidate…")
        score_result = self._score(matching_analysis, job_requirements)

        # Step 5 – Explanation
        print("  [5/5] Generating explainable AI report…")
        explanation = self._explain(candidate_name, score_result, matching_analysis)

        print(f"  ✅ Done → Score: {score_result.get('score', '?')}/100 | "
              f"Tier: {score_result.get('tier', '?')}")

        return ScreeningResult(
            candidate_name    = candidate_name,
            resume_profile    = resume_profile,
            job_requirements  = job_requirements,
            matching_analysis = matching_analysis,
            score_result      = score_result,
            explanation       = explanation,
        )
