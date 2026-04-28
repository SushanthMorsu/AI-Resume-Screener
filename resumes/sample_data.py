# resumes/sample_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Three sample resumes: strong / average / weak candidate.
# Each resume is a plain-text string that mimics a realistic candidate profile.
# ─────────────────────────────────────────────────────────────────────────────

STRONG_RESUME = """
Name: Priya Sharma
Email: priya.sharma@email.com | LinkedIn: linkedin.com/in/priyasharma

SUMMARY
Results-driven Data Scientist with 4 years of industry experience building
end-to-end ML pipelines. Passionate about NLP and large language model
applications. Proven track record of delivering models that reach production.

EXPERIENCE
Senior Data Scientist – TechCorp India (2022–Present)
  • Built a customer churn prediction model (XGBoost) that reduced churn by 18%.
  • Developed and deployed an NLP-based ticket classification system using
    BERT fine-tuning; achieved 94% accuracy in production.
  • Led a 3-person team to implement a real-time recommendation engine using
    PySpark and AWS SageMaker.
  • Conducted A/B testing and statistical analysis to validate model performance.

Data Analyst – DataWave Solutions (2020–2022)
  • Performed EDA and built dashboards in Tableau and Power BI for C-suite.
  • Wrote complex SQL queries to extract and transform data from PostgreSQL.
  • Automated monthly reporting pipelines using Python and Apache Airflow.

EDUCATION
M.Tech – Computer Science, IIT Hyderabad (2020) | GPA 9.1/10
B.Tech – Information Technology, JNTU (2018)

SKILLS
Languages    : Python, SQL, R
ML/DL        : Scikit-learn, TensorFlow, PyTorch, Transformers
LLM/GenAI    : LangChain, Groq API, Prompt Engineering, RAG, LangSmith
Data          : Pandas, NumPy, PySpark, Hadoop
Visualization : Matplotlib, Seaborn, Tableau, Power BI
Cloud/MLOps  : AWS SageMaker, Docker, MLflow, CI/CD, Git

CERTIFICATIONS
  • AWS Certified Machine Learning – Specialty
  • Deep Learning Specialization – Coursera (Andrew Ng)

PROJECTS
  • Open-source RAG chatbot (500+ GitHub stars) using LangChain + Pinecone.
  • Kaggle: Top 5% in NLP competition (BERT-based solution).
"""

AVERAGE_RESUME = """
Name: Rahul Verma
Email: rahul.verma@email.com

SUMMARY
Enthusiastic Data Science professional with 1.5 years of experience.
Comfortable working with Python and basic ML algorithms.
Looking to grow in an AI-focused environment.

EXPERIENCE
Junior Data Analyst – Analytics Hub (2023–Present)
  • Cleaned and preprocessed datasets using Pandas and Excel.
  • Built basic classification models (Logistic Regression, Decision Tree)
    using Scikit-learn for internal dashboards.
  • Created visualizations in Matplotlib and Power BI.
  • Assisted the senior team with SQL queries and data extraction.

Intern – CodeBase Startup (2022–2023)
  • Worked on web scraping using BeautifulSoup and Requests.
  • Explored sentiment analysis using TextBlob and VADER.

EDUCATION
B.Tech – Computer Science, VIT Vellore (2022) | GPA 7.8/10

SKILLS
Languages    : Python, SQL (basic)
ML Frameworks: Scikit-learn (intermediate), TensorFlow (beginner)
Data Tools   : Pandas, NumPy, Excel
Visualization: Matplotlib, Power BI (basic)
Other        : Git, Jupyter Notebook

PROJECTS
  • House price prediction using Linear Regression (college project).
  • Twitter sentiment dashboard — used VADER, deployed on Streamlit.
"""

WEAK_RESUME = """
Name: Amit Gupta
Email: amit2000@gmail.com

EDUCATION
B.Sc – Mathematics, Delhi University (2023) | 60%

SKILLS
• MS Office (Word, Excel, PowerPoint)
• Basic Python (learned from YouTube)
• HTML / CSS (self-taught)

EXPERIENCE
Freelance Tutor (2022–2023)
  • Taught maths and science to school students.
  • Prepared study material in MS Word.

Intern – Local NGO (2023, 1 month)
  • Entered data into spreadsheets.
  • Prepared reports in PowerPoint.

INTERESTS
  • Interested in data science and AI.
  • Completed 1 week Python bootcamp (certificate not available).

PROJECTS
  • None listed.
"""

JOB_DESCRIPTION = """
Role: Data Scientist

Company: InnovateTech Solutions

About the Role:
We are looking for a skilled Data Scientist to join our AI team.
You will build and deploy ML/AI solutions that solve real business problems
and integrate modern LLM/GenAI capabilities into our products.

Key Responsibilities:
  1. Design, develop, and deploy machine learning models (classification,
     regression, NLP) to production.
  2. Work with large language models and GenAI frameworks (LangChain, Groq).
  3. Build and maintain data pipelines using Python and SQL.
  4. Collaborate with product and engineering teams on MLOps and CI/CD.
  5. Conduct A/B testing, statistical analysis, and model evaluation.
  6. Visualize insights using dashboards (Tableau, Power BI, or similar).

Required Skills:
  - Python (advanced), SQL (intermediate+)
  - Machine Learning: Scikit-learn, XGBoost, or equivalent
  - Deep Learning: TensorFlow or PyTorch
  - NLP / LLM: Groq API, LangChain
  - Data: Pandas, NumPy, PySpark (preferred)
  - Cloud: AWS, GCP, or Azure (at least one)
  - MLOps: Docker, MLflow, or similar
  - Version control: Git

Nice to Have:
  - LangSmith / tracing experience
  - RAG systems
  - Kaggle or open-source contributions

Experience: 2+ years in a data science or related role
Education: B.Tech / M.Tech / M.S. in CS, Statistics, or related field
"""

DEBUG_RESUME = """
Name: Unknown Candidate

SUMMARY
asdfghjkl random text no clear formatting

SKILLS
??? ### Python maybe

EXPERIENCE
Worked somewhere maybe 2 years maybe not sure

EDUCATION
N/A

PROJECTS
none
"""

# Dictionary of all candidates for iteration
CANDIDATES = {
    "Strong Candidate – Priya Sharma": STRONG_RESUME,
    "Average Candidate – Rahul Verma": AVERAGE_RESUME,
    "Weak Candidate – Amit Gupta": WEAK_RESUME,
    "Debug Candidate – Unknown": DEBUG_RESUME,
}
    
