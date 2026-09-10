# interview-trainer-agent
# InterviewTwin — Your resume. Your job. Your interview.

An AI-powered, RAG-grounded personalized interview coach built for **IBM SkillsBuild AICTE Internship — Problem Statement 22 (Interview Trainer Agent)**.

Built with **IBM Granite** (via watsonx.ai on IBM Cloud Lite) and **IBM Bob** as the development environment.

## What it does

Unlike a generic interview question generator, InterviewTwin simulates the interview a specific candidate is likely to face for a specific job:

1. **Resume analysis** — upload a PDF/DOCX/TXT resume; IBM Granite extracts a structured candidate profile (skills, projects, education, certifications) without inventing information.
2. **Job description analysis** — paste a real JD; Granite extracts required/preferred skills, responsibilities, and experience expectations.
3. **Gap analysis** — compares the two, producing a job match score, strong matches, partial matches, and missing/weak skills.
4. **RAG-grounded retrieval** — a 1,400-question knowledge base (technical, behavioral, HR questions across 8 roles) is embedded using IBM's Granite embedding model. Questions are retrieved via semantic similarity, not random selection.
5. **Personalized, adaptive mock interview** — retrieved questions are tailored by Granite using the candidate's actual resume details and the job's specific requirements, so questions sound like a real interviewer for that role. Difficulty adapts based on answer quality.
6. **Structured evaluation** — every answer is scored (1–5), with specific feedback and a model answer, via Granite.
7. **Final report, retry-weakest-answer, and a personalized 7-day prep plan** — all generated dynamically from the candidate's actual performance and skill gaps.

## Architecture

```
Resume/JD → IBM Granite (structured extraction) → Gap Analysis
                                                        ↓
Question Bank (1,400 Qs) → Granite Embeddings → Semantic Retrieval
                                                        ↓
                              Retrieved Questions + Profile + Gap → IBM Granite → Personalized Interview
                                                        ↓
                                        User Answer → IBM Granite → Score + Feedback + Model Answer
```

**Retrieval → Context → IBM Granite → Personalized Response**, as required by the problem statement.

## Tech stack

- **IBM Granite** (`granite-4-h-small` for generation, `granite-embedding-278m-multilingual` for retrieval) via watsonx.ai
- **IBM Cloud Lite** (watsonx.ai project + Watson Machine Learning)
- **IBM Bob** — used as the AI-assisted development environment throughout the build
- **Streamlit** — UI
- **Python** (requests, pdfplumber, python-docx)

## Data source

Question bank built from two Kaggle datasets (HR/behavioral questions and software engineering technical questions), preprocessed and embedded — see `preprocess_data.py` and `embed_questions.py`.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GRANITE_API_KEY=your_ibm_cloud_api_key
GRANITE_PROJECT_ID=your_watsonx_project_id
GRANITE_API_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-4-h-small
```

Run:
```bash
streamlit run app.py
```

## Notes

Interviewer personality modes, an "interviewer trap detector," and persisted multi-session interview history are planned future enhancements beyond this submission's scope.

---
*Built as part of the IBM SkillsBuild AICTE Internship (Edunet Foundation), September 2026.*
