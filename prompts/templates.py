# prompts/templates.py
# ─────────────────────────────────────────────────────────────────────────────
# All LangChain PromptTemplates used in the pipeline.
# Each prompt is engineered to:
#   • Provide clear instructions to the LLM
#   • Define exact output constraints (format, length, JSON fields)
#   • Prevent hallucination via explicit "only extract what is present" rules
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.prompts import PromptTemplate


# ── Step 1: Skill Extraction ──────────────────────────────────────────────────
SKILL_EXTRACTION_TEMPLATE = PromptTemplate(
    input_variables=["resume_text"],
    template="""
You are an expert HR analyst and technical recruiter.

TASK: Extract structured information from the resume below.

STRICT RULES — READ CAREFULLY:
  1. Extract ONLY information explicitly stated in the resume.
  2. Do NOT infer, guess, or assume any skill, tool, or experience not written.
  3. If a section has no relevant data, output an empty list [].
  4. Respond ONLY with a valid JSON object — no markdown, no extra text.

OUTPUT FORMAT (strict JSON):
{{
  "candidate_name": "<full name>",
  "skills": ["<skill1>", "<skill2>", ...],
  "tools_and_frameworks": ["<tool1>", "<tool2>", ...],
  "years_of_experience": <integer or 0 if unknown>,
  "education": "<highest degree and field>",
  "certifications": ["<cert1>", ...],
  "notable_achievements": ["<achievement1>", ...]
}}

RESUME:
{resume_text}
"""
)


# ── Step 2: Job Requirements Extraction ──────────────────────────────────────
JOB_REQUIREMENTS_TEMPLATE = PromptTemplate(
    input_variables=["job_description"],
    template="""
You are an expert HR analyst.

TASK: Extract the structured requirements from the job description below.

STRICT RULES:
  1. Only extract what is explicitly stated.
  2. Separate "required" from "nice to have" if mentioned.
  3. Respond ONLY with valid JSON — no markdown, no extra text.

OUTPUT FORMAT (strict JSON):
{{
  "role_title": "<job title>",
  "required_skills": ["<skill1>", "<skill2>", ...],
  "required_tools": ["<tool1>", "<tool2>", ...],
  "preferred_skills": ["<pref1>", ...],
  "min_experience_years": <integer>,
  "education_requirement": "<degree requirement>"
}}

JOB DESCRIPTION:
{job_description}
"""
)


# ── Step 3: Matching Logic ────────────────────────────────────────────────────
MATCHING_TEMPLATE = PromptTemplate(
    input_variables=["candidate_profile", "job_requirements"],
    template="""
You are a senior technical recruiter performing a structured candidate-job fit analysis.

TASK: Compare the candidate profile against the job requirements and produce a
detailed matching analysis.

STRICT RULES:
  1. Only reference items explicitly present in the candidate profile or job requirements.
  2. Do NOT fabricate skills or experience.
  3. Be objective and factual.
  4. Respond ONLY with valid JSON — no markdown, no extra text.

OUTPUT FORMAT (strict JSON):
{{
  "matched_skills": ["<skill present in both>", ...],
  "missing_required_skills": ["<required but absent in candidate>", ...],
  "bonus_skills": ["<candidate has but not required — adds value>", ...],
  "experience_match": "<Exceeds / Meets / Below requirement>",
  "education_match": "<Meets / Does not meet>",
  "overall_match_summary": "<2-3 sentence factual summary>"
}}

CANDIDATE PROFILE (JSON):
{candidate_profile}

JOB REQUIREMENTS (JSON):
{job_requirements}
"""
)


# ── Step 4: Scoring ───────────────────────────────────────────────────────────
SCORING_TEMPLATE = PromptTemplate(
    input_variables=["matching_analysis", "job_requirements"],
    template="""
You are an objective AI scoring engine for recruitment.

TASK: Assign a numeric fit score (0–100) to the candidate based on the
matching analysis, using the rubric below.

SCORING RUBRIC:
  • Required Skills Match  : 40 points  (proportional to % of required skills matched)
  • Experience Match       : 20 points  (Exceeds=20, Meets=15, Below=5, None=0)
  • Tools & Frameworks     : 20 points  (proportional to tools matched)
  • Education              : 10 points  (Meets=10, Does not meet=0)
  • Bonus / Extra Skills   : 10 points  (up to 10 for relevant extras)

STRICT RULES:
  1. Apply the rubric mathematically — do NOT be generous without justification.
  2. Do NOT hallucinate skills. Only use what is in the matching analysis.
  3. Respond ONLY with valid JSON — no markdown, no extra text.

OUTPUT FORMAT (strict JSON):
{{
  "score": <integer 0-100>,
  "score_breakdown": {{
    "required_skills_score": <0-40>,
    "experience_score": <0-20>,
    "tools_score": <0-20>,
    "education_score": <0-10>,
    "bonus_score": <0-10>
  }},
  "tier": "<Strong Fit | Moderate Fit | Weak Fit | Not Suitable>"
}}

MATCHING ANALYSIS (JSON):
{matching_analysis}

JOB REQUIREMENTS (JSON):
{job_requirements}
"""
)


# ── Step 5: Explanation (Explainable AI) ─────────────────────────────────────
EXPLANATION_TEMPLATE = PromptTemplate(
    input_variables=["candidate_name", "score_result", "matching_analysis"],
    template="""
You are an AI recruitment assistant providing a transparent, explainable evaluation
report for a hiring manager.

TASK: Write a clear, structured explanation of why {candidate_name} received
this score. The explanation must be factual, fair, and actionable.

STRICT RULES:
  1. Reference ONLY information from the score result and matching analysis.
  2. Do NOT invent strengths or weaknesses.
  3. Be concise but specific — use bullet points.
  4. Include a hiring recommendation using these rules:
     - Strong Fit (80-100) = Strongly Recommend
     - Moderate Fit (60-79) = Consider
     - Weak Fit (40-59) = Do Not Recommend
     - Not Suitable (0-39) = Do Not Recommend
  5. Respond ONLY with valid JSON — no markdown, no extra text.

OUTPUT FORMAT (strict JSON):
{{
  "candidate_name": "{candidate_name}",
  "final_score": <integer>,
  "tier": "<tier from scoring>",
  "strengths": ["<specific strength 1>", ...],
  "gaps": ["<specific gap 1>", ...],
  "hiring_recommendation": "<Strongly Recommend | Consider | Do Not Recommend>",
  "summary": "<3-4 sentence plain-English explanation for the hiring manager>"
}}

SCORE RESULT (JSON):
{score_result}

MATCHING ANALYSIS (JSON):
{matching_analysis}
"""
)


# ── Few-Shot Example Block (Bonus) ────────────────────────────────────────────
# Used as a prefix in the extraction step to show the model ideal behavior.
FEW_SHOT_EXTRACTION_EXAMPLE = """
--- EXAMPLE (for reference only) ---
Resume snippet: "Proficient in Python and Pandas. 2 years as a data analyst."
Correct extraction:
{{
  "skills": ["Python", "Pandas"],
  "years_of_experience": 2
}}
--- END EXAMPLE ---
"""
