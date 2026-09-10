"""
InterviewTwin - Your resume. Your job. Your interview.
(Interview Trainer Agent - personalized RAG + IBM Granite mock interview)
"""

import streamlit as st
from src.granite_client import GraniteClient
from src.retrieval import Retriever
from src.document_parser import extract_text
from src.auth import sign_up, sign_in

st.set_page_config(page_title="InterviewTwin", page_icon="🎯", layout="centered")

st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.main .block-container { animation: fadeIn 0.6s ease-out; }
h1 {
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6em 1.5em;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
}
</style>
""", unsafe_allow_html=True)

# --- Sign up / Sign in gate ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

if not st.session_state.authenticated:
    st.title("InterviewTwin")
    st.write("Sign in or create an account to start your personalized interview prep.")

    tab_signin, tab_signup = st.tabs(["Sign In", "Sign Up"])

    with tab_signin:
        with st.form("signin_form"):
            si_email = st.text_input("Email", key="si_email")
            si_password = st.text_input("Password", type="password", key="si_password")
            si_submit = st.form_submit_button("Sign In")
        if si_submit:
            success, message = sign_in(si_email, si_password)
            if success:
                st.session_state.authenticated = True
                st.session_state.user_email = si_email.strip().lower()
                st.rerun()
            else:
                st.error(message)

    with tab_signup:
        with st.form("signup_form"):
            su_email = st.text_input("Email", key="su_email")
            su_password = st.text_input("Password (min 6 characters)", type="password", key="su_password")
            su_submit = st.form_submit_button("Create Account")
        if su_submit:
            success, message = sign_up(su_email, su_password)
            if success:
                st.success(message + " Please sign in now.")
            else:
                st.error(message)

    st.stop()


@st.cache_resource
def load_client():
    return GraniteClient()


@st.cache_resource
def load_retriever(_client):
    return Retriever(_client)


client = load_client()
retriever = load_retriever(client)

st.title("InterviewTwin")
st.caption("Your resume. Your job. Your interview.")

# --- Session state ---
defaults = {
    "stage": "setup",  # setup -> analyzed -> interviewing -> ended
    "profile": None,
    "jd": None,
    "gap": None,
    "history": [],
    "seen_questions": set(),
    "question_batch": None,
    "batch_index": 0,
    "retry_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

ROLE_OPTIONS = ["Software Engineer", "Data Scientist", "Product Manager",
                "DevOps Engineer", "QA Analyst", "UX Designer",
                "Marketing Associate", "HR Specialist"]

# =========================================================
# STAGE 1: SETUP - resume + JD + role/experience/interview type
# =========================================================
if st.session_state.stage == "setup":
    st.markdown("### 1. Candidate Profile")
    resume_file = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

    st.markdown("### 2. Job Description")
    jd_text = st.text_area("Paste the job description", height=180,
                            placeholder="Paste the full JD here...")

    st.markdown("### 3. Interview Setup")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Your name")
        job_role = st.selectbox("Target job role", ROLE_OPTIONS)
    with col2:
        experience_level = st.selectbox("Experience level",
                                        ["fresher", "1 year", "2 years", "3 years", "5+ years"])
        interview_type = st.selectbox("Interview type", ["Mixed", "Technical", "Behavioral", "HR"])

    num_questions = st.select_slider("Number of questions", options=[5, 10, 15], value=5)

    if st.button("Analyze & Continue"):
        if not resume_file:
            st.warning("Please upload your resume.")
        elif not jd_text.strip():
            st.warning("Please paste a job description.")
        elif not name.strip():
            st.warning("Please enter your name.")
        else:
            with st.spinner("Extracting resume text..."):
                try:
                    resume_text = extract_text(resume_file)
                except Exception as e:
                    st.error(f"Couldn't read resume file: {e}")
                    st.stop()

            st.write("DEBUG - extracted resume length:", len(resume_text))
            st.write("DEBUG - first 300 chars:", resume_text[:300])

            with st.spinner("Analyzing your resume with IBM Granite..."):
                profile = client.analyze_resume(resume_text)

            if "error" in profile:
                st.warning("Resume analysis had trouble parsing.")
                with st.expander("Resume analysis debug info"):
                    st.json(profile.get("debug_full_response", {}))

            with st.spinner("Analyzing the job description..."):
                jd = client.analyze_job_description(jd_text)

            if "error" in jd:
                st.warning("JD analysis had trouble parsing.")
                with st.expander("JD analysis debug info"):
                    st.json(jd.get("debug_full_response", {}))

            with st.spinner("Comparing your profile against the job requirements..."):
                gap = client.gap_analysis(profile, jd)

            st.session_state.profile = profile
            st.session_state.jd = jd
            st.session_state.gap = gap
            st.session_state.name = name
            st.session_state.job_role = job_role
            st.session_state.experience_level = experience_level
            st.session_state.interview_type = interview_type
            st.session_state.num_questions = num_questions
            st.session_state.stage = "analyzed"
            st.rerun()

# =========================================================
# STAGE 2: ANALYZED - show candidate analysis / gap
# =========================================================
if st.session_state.stage == "analyzed":
    profile = st.session_state.profile
    gap = st.session_state.gap

    st.markdown("### 📊 Candidate Analysis")

    if "error" in gap:
        st.warning("Gap analysis had trouble parsing — showing raw output below.")
        st.write(gap.get("raw", ""))
    else:
        match_score = gap.get("match_score", "N/A")
        st.metric("Job Match Score", f"{match_score}%" if isinstance(match_score, (int, float)) else match_score)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**✓ Strong Matches**")
            for s in gap.get("strong_matches", []):
                st.write(f"- {s}")
        with col2:
            st.write("**⚠ Partial Matches**")
            for s in gap.get("partial_matches", []):
                st.write(f"- {s}")
        with col3:
            st.write("**✗ Missing / Weak**")
            for s in gap.get("missing_skills", []):
                st.write(f"- {s}")

        st.info(gap.get("explanation", ""))

    with st.expander("View extracted candidate profile"):
        st.json(profile)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start Personalized Interview"):
            st.session_state.stage = "interviewing"
            st.rerun()
    with col2:
        if st.button("Start over"):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

# =========================================================
# STAGE 3: INTERVIEWING - batch-personalized, reordered adaptively
# =========================================================
if st.session_state.stage == "interviewing":
    profile = st.session_state.profile
    gap = st.session_state.gap
    target = st.session_state.num_questions

    # Build the personalized batch once, up front
    if "question_batch" not in st.session_state or st.session_state.question_batch is None:
        itype = st.session_state.interview_type
        question_types = None if itype == "Mixed" else [itype.lower()]

        with st.spinner("Retrieving relevant questions..."):
            base_questions = retriever.retrieve(
                job_role=st.session_state.job_role,
                experience_level=st.session_state.experience_level,
                question_types=question_types,
                top_k=target,
            )

        with st.spinner("Personalizing your interview with IBM Granite..."):
            personalized = client.generate_personalized_batch(
                questions=base_questions, profile=profile, gap=gap,
                jd=st.session_state.jd, job_role=st.session_state.job_role,
            )

        # Merge personalized text back with original metadata (ideal_answer etc.)
        batch = []
        for base, p in zip(base_questions, personalized):
            tailored_q = p.get("question", "").strip()
            # Sanity check: a real question is more than a couple words.
            # If personalization got corrupted/truncated, fall back to the
            # original retrieved question instead of showing garbage.
            if len(tailored_q.split()) < 5:
                tailored_q = base["question"]
                why = "Selected from the role-relevant question bank."
                testing = base["type"]
            else:
                why = p.get("why_this_question", "")
                testing = p.get("testing", "")

            batch.append({
                "question": tailored_q,
                "type": base["type"],
                "ideal_answer": base.get("ideal_answer", ""),
                "why": why,
                "testing": testing,
                "original_question": base["question"],
            })
        st.session_state.question_batch = batch
        st.session_state.batch_index = 0

    batch = st.session_state.question_batch
    idx = st.session_state.batch_index

    st.caption(f"Interviewing **{st.session_state.name}** for **{st.session_state.job_role}** "
               f"({st.session_state.experience_level}) — Question {min(idx + 1, len(batch))} of {len(batch)}")
    st.progress(min(idx / len(batch), 1.0) if batch else 0)

    for i, h in enumerate(st.session_state.history):
        with st.container(border=True):
            st.markdown(f"**Q{i+1} [{h['type'].upper()}]** {h['question']}")
            st.write(f"*Your answer:* {h['answer']}")
            emoji = "🟢" if h["score"] >= 4 else "🟡" if h["score"] == 3 else "🔴"
            st.write(f"{emoji} **Score: {h['score']}/5** — {h['feedback']}")
            if h.get("model_answer"):
                with st.expander("See a model answer"):
                    st.write(h["model_answer"])

    if idx >= len(batch):
        st.session_state.stage = "ended"
        st.rerun()

    q = batch[idx]
    st.markdown(f"### Question {idx + 1} [{q['type'].upper()}]")
    st.markdown(f"**{q['question']}**")

    with st.expander("Why am I being asked this?"):
        st.write(f"**Why this question:** {q['why']}")
        st.write(f"**What's being tested:** {q['testing']}")
        st.caption(f"📚 Retrieved from knowledge base: *\"{q['original_question']}\"* — personalized above using your resume and this job's requirements.")

    with st.form(f"answer_form_{idx}"):
        answer = st.text_area("Your answer", height=150)
        col1, col2 = st.columns(2)
        with col1:
            submit_answer = st.form_submit_button("Submit answer")
        with col2:
            end_now = st.form_submit_button("End interview now")

    if submit_answer:
        if not answer.strip():
            st.warning("Please write an answer.")
        else:
            with st.spinner("Evaluating with IBM Granite..."):
                result = client.evaluate_answer_structured(
                    question=q["question"],
                    user_answer=answer,
                    ideal_answer=q.get("ideal_answer", ""),
                    job_role=st.session_state.job_role,
                    experience_level=st.session_state.experience_level,
                )
            score = result.get("score", 3)
            st.session_state.history.append({
                "question": q["question"], "type": q["type"], "answer": answer,
                "score": score, "feedback": result.get("feedback", ""),
                "strength": result.get("strength", ""), "weakness": result.get("weakness", ""),
                "model_answer": result.get("model_answer", ""),
            })

            # Adaptive reorder (no extra API calls): if weak, bring a same-type
            # question from later in the batch forward to reinforce that area.
            if score <= 2:
                remaining = batch[idx + 1:]
                same_type_pos = next((j for j, r in enumerate(remaining) if r["type"] == q["type"]), None)
                if same_type_pos is not None:
                    swap_idx = idx + 1 + same_type_pos
                    batch[idx + 1], batch[swap_idx] = batch[swap_idx], batch[idx + 1]
                    st.session_state.question_batch = batch

            st.session_state.batch_index += 1
            st.rerun()

    if end_now:
        st.session_state.stage = "ended"
        st.rerun()

# =========================================================
# STAGE 4: ENDED - report, retry weakest, prep plan
# =========================================================
if st.session_state.stage == "ended":
    history = st.session_state.history
    st.success("Interview complete!")

    if not history:
        st.info("No questions were answered.")
    else:
        overall_pct = sum(h["score"] for h in history) / (len(history) * 5) * 100

        by_type = {}
        for h in history:
            by_type.setdefault(h["type"], []).append(h["score"])
        type_pct = {t: sum(s) / (len(s) * 5) * 100 for t, s in by_type.items()}

        st.markdown("### 📋 Interview Performance Report")
        st.metric("Overall Score", f"{overall_pct:.0f}%")

        cols = st.columns(len(type_pct)) if type_pct else []
        for col, (t, pct) in zip(cols, type_pct.items()):
            col.metric(t.capitalize(), f"{pct:.0f}%")

        strengths = [t for t, p in type_pct.items() if p >= 70]
        weaknesses = [t for t, p in type_pct.items() if p < 60]

        col1, col2 = st.columns(2)
        with col1:
            st.write("**✓ Strong Areas**")
            for s in strengths or ["Keep practicing to build strong areas!"]:
                st.write(f"- {s}")
        with col2:
            st.write("**⚠ Areas to Improve**")
            for s in weaknesses or ["No major weak areas detected!"]:
                st.write(f"- {s}")

        # Weakest answer + retry
        weakest = min(history, key=lambda h: h["score"])
        st.markdown("### 🔁 Retry Your Weakest Answer")
        with st.container(border=True):
            st.write(f"**Question:** {weakest['question']}")
            st.write(f"**Previous score:** {weakest['score']}/5")
            st.write(f"**Feedback:** {weakest['feedback']}")

            retry_answer = st.text_area("Try answering again", key="retry_answer_box", height=120)
            if st.button("Submit retry"):
                if retry_answer.strip():
                    with st.spinner("Evaluating your improved answer..."):
                        retry_result = client.evaluate_answer_structured(
                            question=weakest["question"],
                            user_answer=retry_answer,
                            ideal_answer="",
                            job_role=st.session_state.job_role,
                            experience_level=st.session_state.experience_level,
                        )
                    st.session_state.retry_result = retry_result

            if st.session_state.retry_result:
                r = st.session_state.retry_result
                st.write(f"**New score: {r.get('score', '?')}/5** — {r.get('feedback', '')}")
                if r.get("score", 0) > weakest["score"]:
                    st.success(f"Improvement! {weakest['score']}/5 → {r.get('score')}/5")

        # Prep plan
        st.markdown("### 🗓️ Your Personalized Preparation Plan")
        if st.button("Generate my 7-day prep plan"):
            with st.spinner("Building your personalized plan..."):
                plan = client.generate_prep_plan(
                    name=st.session_state.name,
                    gap=st.session_state.gap,
                    history=history,
                )
            with st.container(border=True):
                st.markdown(plan)

    if st.button("Start a completely new session"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()
