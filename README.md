# 🤖 AI Resume Screening System

> **Data Science Internship — GenAI Assignment**  
> Built with **LangChain · Groq · LangSmith**

---

## 📋 Overview

An end-to-end AI-powered resume screening pipeline that:
- Extracts skills, tools, and experience from resumes
- Matches candidates against a job description
- Assigns a transparent 0–100 fit score using a rubric
- Generates explainable AI output for hiring managers
- Traces every pipeline step via **LangSmith**

---

## 🏗️ Project Structure

```
ai_resume_screener/
│
├── main.py                    # Entry point — runs all 3 candidates
├── notebook.ipynb             # Jupyter notebook for interactive exploration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
│
├── prompts/
│   ├── __init__.py
│   └── templates.py           # All 5 LangChain PromptTemplates
│
├── chains/
│   ├── __init__.py
│   └── screening_chain.py     # LCEL chains + pipeline orchestrator
│
└── resumes/
    ├── __init__.py
    └── sample_data.py         # 3 sample resumes + 1 job description
```

---

## ⚙️ Pipeline Architecture

```
Resume Text
    │
    ▼
[Chain 1] Skill Extraction      ← PromptTemplate | LLM | StrOutputParser | JSON
    │
    ▼
[Chain 2] Job Requirements      ← PromptTemplate | LLM | StrOutputParser | JSON
    │
    ▼
[Chain 3] Matching Logic        ← Compare candidate ↔ job requirements
    │
    ▼
[Chain 4] Scoring (0–100)       ← Rubric-based scoring with breakdown
    │
    ▼
[Chain 5] Explanation           ← Explainable AI output for recruiter
    │
    ▼
LangSmith Trace                 ← All steps visible as nested spans
```

---

## 🚀 Setup & Run

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/ai-resume-screener.git
cd ai-resume-screener
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your keys:
#   GROQ_API_KEY=gsk_...
#   GROQ_MODEL=llama-3.1-8b-instant
#   LANGCHAIN_API_KEY=ls__...
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_PROJECT=ai-resume-screener
```

> **Get LangSmith API key:** https://smith.langchain.com → Settings → API Keys

### 3. Run

```bash
# Option A: Python script
python main.py

# Option B: Jupyter Notebook
jupyter notebook notebook.ipynb
```

---

## 📊 Scoring Rubric

| Category             | Max Points |
|----------------------|-----------|
| Required Skills Match | 40       |
| Experience Match      | 20       |
| Tools & Frameworks    | 20       |
| Education             | 10       |
| Bonus Skills          | 10       |
| **TOTAL**             | **100**  |

| Score Range | Tier            |
|-------------|-----------------|
| 80 – 100    | Strong Fit      |
| 60 – 79     | Moderate Fit    |
| 40 – 59     | Weak Fit        |
| 0 – 39      | Not Suitable    |

---

## 🔍 LangSmith Tracing

Once you run the pipeline with `LANGCHAIN_TRACING_V2=true`:

1. Go to https://smith.langchain.com
2. Open your project (`ai-resume-screener`)
3. You will see **4 runs** (3 candidates + 1 debug):
   - `Strong Candidate – Priya Sharma`
   - `Average Candidate – Rahul Verma`
   - `Weak Candidate – Amit Gupta`
   - `DEBUG – Malformed Resume`
4. Each run shows **5 nested spans** (one per pipeline step)
5. Click any span to inspect the exact prompt, LLM response, and latency

---

## 🧠 Prompt Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| `temperature=0.0` | Deterministic scoring — no randomness in rubric application |
| "Only extract what is present" rule | Prevents hallucination of skills |
| JSON-only output constraint | Enables reliable parsing downstream |
| Rubric in scoring prompt | Forces consistent, mathematical scoring |
| Separate extraction vs. matching | Modular — easy to swap or debug each step |

