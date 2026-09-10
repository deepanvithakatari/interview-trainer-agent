"""
Granite client - wraps IBM Granite via watsonx.ai's chat API.
"""

import os
import re
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
GRANITE_API_URL = os.getenv("GRANITE_API_URL", "https://us-south.ml.cloud.ibm.com")
GRANITE_API_KEY = os.getenv("GRANITE_API_KEY", "")
GRANITE_PROJECT_ID = os.getenv("GRANITE_PROJECT_ID", "")
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-4-h-small")


class GraniteClient:
    """Client for interacting with IBM Granite via watsonx.ai's chat API."""

    def __init__(self, api_url: str = GRANITE_API_URL, api_key: str = GRANITE_API_KEY,
                 project_id: str = GRANITE_PROJECT_ID, model_id: str = GRANITE_MODEL_ID):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.project_id = project_id
        self.model_id = model_id
        self._access_token = None
        self._token_expiry = 0

    def _get_access_token(self) -> str:
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        response = requests.post(
            IAM_TOKEN_URL,
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self.api_key,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600)
        return self._access_token

    def _chat(self, messages: list, max_tokens: int = 600, temperature: float = 0.5) -> str:
        """Call watsonx.ai's chat endpoint with retry + backoff on 429 rate limits."""
        response = None
        payload = {
            "model_id": self.model_id,
            "project_id": self.project_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        endpoint = f"{self.api_url}/ml/v1/text/chat?version=2023-05-29"

        delay = 5  # seconds; doubles on each 429
        for attempt in range(5):
            # Proactive inter-call delay to stay under the rate limit ceiling
            if attempt > 0:
                time.sleep(delay)
                delay = min(delay * 2, 60)

            token = self._get_access_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            response = requests.post(endpoint, json=payload, headers=headers, timeout=90)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", delay))
                time.sleep(max(retry_after, delay))
                continue

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        # All attempts exhausted — raise the last 429
        if response is not None:
            response.raise_for_status()
        raise RuntimeError("No response received after retries")

    def _extract_json_object(self, text: str):
        """Extract a JSON object from text, correctly handling nested braces."""
        text = text.strip()
        text = re.sub(r"^```json\s*|^```\s*|\s*```$", "", text)

        # Try parsing the whole thing first
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Fall back to outermost brace span (handles preamble/trailing text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        return None

    def _chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> dict:
        """Call chat, expecting a JSON object back. Parses robustly, retries once if empty/invalid."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_text = ""
        for attempt in range(2):
            raw_text = self._chat(messages, max_tokens=max_tokens, temperature=0.3)
            parsed = self._extract_json_object(raw_text) if raw_text else None
            if parsed is not None:
                return parsed
            if attempt == 0:
                continue

        return {"error": "parse_failed", "raw": raw_text}

    # --- Core coaching methods ---

    def evaluate_answer(self, question: str, answer: str) -> str:
        """Free-text evaluation of a candidate's answer (legacy/simple use)."""
        messages = [
            {"role": "system", "content": "You are an expert technical interviewer giving constructive feedback."},
            {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\n\nProvide constructive feedback."},
        ]
        return self._chat(messages, max_tokens=400, temperature=0.7)

    def generate_prep_package(self, name: str, job_role: str, experience_level: str, questions: list) -> list:
        """Generate structured prep data: question, type, model_answer, tip for each question."""
        questions_block = "\n".join(
            f"{i+1}. [{q['type']}] {q['question']}" for i, q in enumerate(questions)
        )
        system_prompt = (
            f"You are an expert interview coach preparing {name}, a {experience_level} "
            f"candidate for a {job_role} role. Respond with ONLY a valid JSON array, no other text."
        )
        user_prompt = (
            f"Here are exactly {len(questions)} interview questions:\n{questions_block}\n\n"
            f"Generate exactly {len(questions)} JSON objects, one per question, each with keys: "
            f'"question", "type", "model_answer" (2-3 sentences), "tip" (1-2 sentences).'
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_text = ""
        for attempt in range(2):
            raw_text = self._chat(messages, max_tokens=2048, temperature=0.5)
            text = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw_text.strip())
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                    if isinstance(parsed, list) and len(parsed) >= len(questions):
                        return parsed[:len(questions)]
                except json.JSONDecodeError:
                    pass
            if attempt == 0:
                continue

        return [{"question": "Parsing error", "type": "error", "model_answer": raw_text, "tip": ""}]

    def evaluate_answer_structured(self, question: str, user_answer: str,
                                    ideal_answer: str, job_role: str, experience_level: str) -> dict:
        """Evaluate an answer, returning a structured score, feedback, and model answer."""
        system_prompt = (
            f"You are an expert interview coach evaluating a {experience_level} candidate "
            f"for a {job_role} role. Respond with ONLY a valid JSON object, no other text."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Candidate's answer: {user_answer}\n"
            f"Reference (ideal) answer, if available: {ideal_answer}\n\n"
            f'Respond with a JSON object with keys: "score" (integer 1-5), '
            f'"feedback" (2-3 sentences on the candidate\'s specific answer), '
            f'"strength" (short phrase), "weakness" (short phrase), '
            f'"model_answer" (a strong 2-3 sentence example answer to THIS exact question, '
            f'appropriate for a {experience_level} candidate).'
        )
        result = self._chat_json(system_prompt, user_prompt, max_tokens=500)
        if "error" in result:
            return {"score": 3, "feedback": result.get("raw", "Could not evaluate."),
                    "strength": "N/A", "weakness": "N/A", "model_answer": ""}
        return result

    def generate_session_summary(self, name: str, job_role: str, history: list) -> str:
        """Generate an overall encouraging session summary."""
        transcript = "\n".join(
            f"Q{i+1} [{h['type']}] {h['question']} — Score: {h['score']}/5, "
            f"Strength: {h['strength']}, Weakness: {h['weakness']}"
            for i, h in enumerate(history)
        )
        messages = [
            {"role": "system", "content": "You are an encouraging, honest interview coach."},
            {"role": "user", "content": (
                f"{name} just completed a mock interview for a {job_role} role. "
                f"Performance:\n{transcript}\n\n"
                f"Write a brief, encouraging summary (4-5 sentences): key strengths, "
                f"top 1-2 areas to improve, and a motivating closing note."
            )},
        ]
        return self._chat(messages, max_tokens=400, temperature=0.6)

    # --- InterviewTwin: resume / JD / gap / personalization ---

    def analyze_resume(self, resume_text: str) -> dict:
        system_prompt = (
            "Extract a structured candidate profile from resume text. Only include information "
            "actually present — never invent anything. Respond with ONLY a valid JSON object."
        )
        user_prompt = (
            f"Resume text:\n{resume_text[:4000]}\n\n"
            f'Respond with a JSON object with keys: "name", "education" (list), "skills" (list), '
            f'"soft_skills" (list), "projects" (list), "experience_summary" (string), '
            f'"certifications" (list), "tools" (list).'
        )
        return self._chat_json(system_prompt, user_prompt, max_tokens=1024)

    def analyze_job_description(self, jd_text: str) -> dict:
        system_prompt = "Extract structured requirements from a job description. Respond with ONLY a valid JSON object."
        user_prompt = (
            f"Job description:\n{jd_text[:4000]}\n\n"
            f'Respond with a JSON object with keys: "required_skills" (list), "preferred_skills" (list), '
            f'"responsibilities" (list), "soft_skills" (list), "experience_level" (string).'
        )
        return self._chat_json(system_prompt, user_prompt, max_tokens=800)

    def gap_analysis(self, profile: dict, jd: dict) -> dict:
        system_prompt = "Compare a candidate profile against job requirements. Respond with ONLY a valid JSON object."
        user_prompt = (
            f"Candidate profile: {json.dumps(profile)}\n\n"
            f"Job requirements: {json.dumps(jd)}\n\n"
            f'Respond with a JSON object with keys: "match_score" (integer 0-100), '
            f'"strong_matches" (list), "partial_matches" (list), "missing_skills" (list), '
            f'"explanation" (2-3 sentences).'
        )
        return self._chat_json(system_prompt, user_prompt, max_tokens=600)

    def generate_personalized_question(self, base_question: str, base_type: str,
                                        profile: dict, gap: dict, job_role: str) -> dict:
        system_prompt = (
            f"You are an AI interviewer for a {job_role} role, personalizing questions using "
            f"a candidate's resume and skill gaps. Respond with ONLY a valid JSON object."
        )
        user_prompt = (
            f"Base question from question bank: \"{base_question}\" (type: {base_type})\n\n"
            f"Candidate profile: {json.dumps(profile)}\n"
            f"Skill gap analysis: {json.dumps(gap)}\n\n"
            "Tailor this question to reference the candidate's actual resume details where relevant, "
            "OR if a missing/weak skill fits this question's theme, adjust it to probe that gap. "
            "Keep the core intent. If personalization doesn't naturally apply, keep it close to original.\n\n"
            'Respond with a JSON object with keys: "question", "why_this_question" (1 sentence), '
            '"testing" (short phrase).'
        )
        result = self._chat_json(system_prompt, user_prompt, max_tokens=400)
        if "error" in result or "question" not in result:
            return {"question": base_question, "why_this_question": "Selected from role-relevant question bank.",
                    "testing": base_type}
        return result


    def generate_personalized_batch(self, questions: list, profile: dict, gap: dict,
                                     jd: dict, job_role: str) -> list:
        """
        Personalize an entire batch of retrieved questions in a single call:
        grounds wording in the actual job description so it feels like a real
        interviewer for this specific role, while weaving in resume/gap context.
        """
        questions_block = "\n".join(
            f"{i+1}. [{q['type']}] {q['question']}" for i, q in enumerate(questions)
        )
        system_prompt = (
            f"You are simulating a real interviewer for this specific job. Use the actual job "
            f"description's responsibilities and required skills to phrase questions the way a "
            f"genuine interviewer for THIS role would — not generic textbook phrasing. "
            f"Respond with ONLY a valid JSON array, no other text."
        )
        user_prompt = (
            f"Job description details: {json.dumps(jd)}\n\n"
            f"Candidate profile: {json.dumps(profile)}\n"
            f"Skill gap analysis: {json.dumps(gap)}\n\n"
            f"Here are exactly {len(questions)} base questions from our question bank:\n{questions_block}\n\n"
            f"For EACH question: ground it in the job description's actual responsibilities/skills "
            f"where relevant, and reference the candidate's resume details (specific projects/skills) "
            f"where it fits naturally. If a missing/weak skill from the gap analysis fits the question's "
            f"theme, adjust it to probe that gap the way a real interviewer would follow up. Keep each "
            f"question's core intent — don't invent unrelated content.\n\n"
            f"Generate exactly {len(questions)} JSON objects, in the same order, each with keys: "
            f'"question" (tailored, realistic interviewer phrasing), "type", '
            f'"why_this_question" (1 sentence), "testing" (short phrase).'
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(2):
            raw_text = self._chat(messages, max_tokens=2500, temperature=0.4)
            text = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw_text.strip())
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                    if isinstance(parsed, list) and len(parsed) >= len(questions):
                        return parsed[:len(questions)]
                except json.JSONDecodeError:
                    pass
            if attempt == 0:
                continue

        # Fallback: return originals untouched if personalization fails
        return [{"question": q["question"], "type": q["type"],
                 "why_this_question": "Selected from role-relevant question bank.",
                 "testing": q["type"]} for q in questions]


    def generate_prep_plan(self, name: str, gap: dict, history: list) -> str:
        weak_types = {}
        for h in history:
            weak_types.setdefault(h["type"], []).append(h["score"])
        weak_summary = {t: round(sum(s) / len(s), 1) for t, s in weak_types.items()}

        messages = [
            {"role": "system", "content": "You are an interview coach creating a personalized study plan."},
            {"role": "user", "content": (
                f"Create a personalized 7-day interview prep plan for {name}.\n\n"
                f"Missing/weak skills: {gap.get('missing_skills', [])}\n"
                f"Average scores by question type (out of 5): {weak_summary}\n\n"
                f"Write a concise day-by-day plan (Day 1-7), each with one focused task based on "
                f"their actual weak areas above."
            )},
        ]
        return self._chat(messages, max_tokens=600, temperature=0.6)
