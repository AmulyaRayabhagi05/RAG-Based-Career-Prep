import os
import chromadb
from sentence_transformers import SentenceTransformer
from functools import lru_cache

model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "ingestion", "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
jobs_collection = chroma_client.get_or_create_collection(name="jobs")


def embed_text(text: str) -> list:
    return model.encode(text).tolist()


def query_jobs(query: str, n_results: int = 10, title_filter=None) -> list[dict]:
    embedding = embed_text(query)

    if isinstance(title_filter, list):
        where_clause = {"title": {"$in": title_filter}}
    elif title_filter:
        where_clause = {"title": {"$eq": title_filter}}
    else:
        where_clause = None

    results = jobs_collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where_clause
    )

    # Fallback: if title filter matched nothing, retry with semantic search only
    if not results["documents"][0] and where_clause is not None:
        results = jobs_collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

    formatted = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        formatted.append({
            "text": doc,
            "title": meta.get("title", ""),
            "source": meta.get("source", "")
        })

    return formatted


def retrieve_for_resume(student_profile: dict, target_title=None) -> dict:
    skills_str = ", ".join(student_profile.get("skills", []))
    coursework_str = ", ".join(student_profile.get("coursework", []))
    projects_str = ", ".join(student_profile.get("projects", []))
    major_str = str(student_profile.get("major", "")).strip()
    target_role_str = str(student_profile.get("target_role", "")).strip()
    target_title_str = ", ".join(target_title) if isinstance(target_title, list) else (target_title or "")
    query = " ".join(part for part in [skills_str, coursework_str, projects_str, major_str, target_role_str, target_title_str] if part).strip()

    job_chunks = query_jobs(query, n_results=10, title_filter=target_title)

    return {"job_chunks": job_chunks}

def retrieve_for_interview(target_job: str) -> dict:

    task_results = jobs_collection.query(
        query_texts=[f"tasks and responsibilities for {target_job}"],
        n_results=5
    )

    question_results = jobs_collection.query(
        query_texts=[f"interview questions for {target_job}"],
        n_results=5
    )

    if task_results["documents"]:
        tasks = task_results["documents"][0]

       
    else:
        tasks = []

    if question_results["documents"]:
        questions = question_results["documents"][0]

        
    else:
        questions = []

    return {
        "tasks": tasks,
        "questions": questions
    }

@lru_cache(maxsize=100)
def retrieve_for_interview_cached(target_job: str) -> dict:
    return retrieve_for_interview(target_job)

