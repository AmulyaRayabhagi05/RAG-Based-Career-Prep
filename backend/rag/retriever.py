"""
[RAG-5 RETRIEVE] Online retrieval over the `jobs` Chroma collection.

Built offline by rag/ingestion/ingest_mongo.py (see rag/chunking.py for chunk shape).
Consumers:
  * modules/resume_builder.py   -> retrieve_for_resume()           (resume bullets)
  * modules/interview_router.py -> retrieve_for_interview_cached() (question grounding)
  * evals/run_evals.py          -> retrieval-quality eval (LLM-as-a-judge precision@k)
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# Must be the same model used at ingestion time ([RAG-3 EMBED]).
# Loaded once at import so each request only pays for encode(), not model load.
model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "ingestion", "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# get_or_create keeps the cosine space configured at ingestion; if ingestion has not
# been run yet this creates an empty collection and queries simply return nothing.
jobs_collection = chroma_client.get_or_create_collection(name="jobs")


def embed_text(text: str) -> list:
    return model.encode(text).tolist()


def query_jobs(query: str, n_results: int = 10, title_filter=None) -> list[dict]:
    """
    Hybrid retrieval: exact metadata filter on `title` + dense vector similarity.

    1. Pre-filter: restrict the candidate set to chunks whose title matches the
       target role(s). This is cheap and very precise when the user names an
       O*NET-style title ("Data Scientists").
    2. Rank: cosine similarity between the query embedding and chunk embeddings.
    3. Fallback: if the filter matched nothing (user typed "SWE" instead of
       "Software Developers"), retry with pure semantic search so we still return
       the closest occupations instead of an empty context.
    """
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
    """
    Query construction for the resume builder: concatenate the student's skills,
    coursework, projects, major and target role into a single "profile query".
    Embedding the whole profile retrieves occupations whose skills/tasks overlap
    with what the student has actually done, which is what the bullets are built from.
    """
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
    """
    Two intent-specific queries against the same index: one phrased to pull task
    statements (Technical questions), one for behavioural question material (HR).
    Note: `query_texts` lets Chroma embed the query with its default embedding
    function, which is also all-MiniLM-L6-v2, so the vector space matches ingestion.
    """

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

# The interview UI asks for a new question every turn for the same role; caching
# the retrieval per job title removes two vector queries from every turn after the first.
@lru_cache(maxsize=100)
def retrieve_for_interview_cached(target_job: str) -> dict:
    return retrieve_for_interview(target_job)
