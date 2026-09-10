"""
Retrieval engine - finds the most relevant interview questions for a
given user profile using semantic similarity (cosine similarity) over
Granite embeddings.
"""

import json
import math

DATA_FILE = "data/questions_with_embeddings.json"


def _load_questions():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Retriever:
    """Retrieves relevant interview questions using semantic search."""

    def __init__(self, granite_client, data_file: str = DATA_FILE):
        self.client = granite_client
        self.questions = _load_questions() if data_file == DATA_FILE else self._load_custom(data_file)

    def _load_custom(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, job_role: str, experience_level: str = "any",
                 question_types: list = None, top_k: int = 5, exclude: set = None):
        """
        Retrieve top_k relevant questions per type, deduplicated, excluding
        any questions already seen (by exact question text).
        """
        if question_types is None:
            question_types = ["technical", "behavioral", "hr"]
        if exclude is None:
            exclude = set()

        query_text = f"{experience_level} {job_role} interview questions"
        query_embedding = self._embed_query(query_text)

        final_results = []
        seen_questions = set(exclude)

        base_k = top_k // len(question_types)
        remainder = top_k % len(question_types)

        for idx, q_type in enumerate(question_types):
            per_type_k = base_k + (1 if idx < remainder else 0)
            per_type_k = max(1, per_type_k)

            candidates = [q for q in self.questions if q.get("type") == q_type]

            scored = []
            for q in candidates:
                if q["question"] in seen_questions:
                    continue
                score = _cosine_similarity(query_embedding, q["embedding"])
                if q.get("role", "").lower() == job_role.lower():
                    score += 0.15
                scored.append((score, q))

            scored.sort(key=lambda x: x[0], reverse=True)

            added = 0
            for score, q in scored:
                if q["question"] in seen_questions:
                    continue
                seen_questions.add(q["question"])
                final_results.append(q)
                added += 1
                if added >= per_type_k:
                    break

        return [{k: v for k, v in q.items() if k != "embedding"} for q in final_results]

    def _embed_query(self, text: str):
        import requests
        token = self.client._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model_id": "ibm/granite-embedding-278m-multilingual",
            "inputs": [text],
            "project_id": self.client.project_id,
        }
        endpoint = f"{self.client.api_url}/ml/v1/text/embeddings?version=2023-05-29"
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["results"][0]["embedding"]

    def get_next_question(self, job_role: str, experience_level: str = "any",
                          exclude: set = None, preferred_type: str = None,
                          background_text: str = "") -> dict | None:
        """
        Retrieve a single next question for the adaptive interview.

        Args:
            job_role: Target job role.
            experience_level: Candidate experience level.
            exclude: Set of question strings already seen.
            preferred_type: If set, restrict to this question type.
            background_text: Extra context (skills/gaps) to enrich the query embedding.

        Returns:
            A single question dict (without embedding), or None if exhausted.
        """
        if exclude is None:
            exclude = set()

        query_text = f"{experience_level} {job_role} interview questions {background_text}".strip()
        query_embedding = self._embed_query(query_text)

        candidates = [
            q for q in self.questions
            if q["question"] not in exclude
            and (preferred_type is None or q.get("type") == preferred_type)
        ]

        if not candidates:
            return None

        scored = []
        for q in candidates:
            score = _cosine_similarity(query_embedding, q["embedding"])
            if q.get("role", "").lower() == job_role.lower():
                score += 0.15
            scored.append((score, q))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        return {k: v for k, v in best.items() if k != "embedding"}
