"""
[RAG-5 RETRIEVE] Online retrieval over the `jobs` Chroma collection.

Built offline by rag/ingestion/ingest_mongo.py (see rag/chunking.py for chunk shape).
Consumers:
  * modules/resume_builder.py   -> retrieve_for_resume()           (resume bullets)
  * modules/interview_router.py -> retrieve_for_interview_cached() (question grounding)
  * evals/run_evals.py          -> retrieval-quality eval (LLM-as-a-judge precision@k)

Two-stage retrieval (retrieve -> rerank)
  Stage 1  bi-encoder (MiniLM): query and chunks are embedded SEPARATELY, so chunk vectors
           can be precomputed and searched in milliseconds. Fast, but the query never "sees"
           the chunk, so fine-grained relevance is approximate. -> fetch a wide candidate set.
  Stage 2  cross-encoder [RAG-6 RERANK]: reads (query, chunk) TOGETHER and outputs one
           relevance score. Much more accurate, but too slow to run over the whole index,
           so it only rescores the ~20 stage-1 candidates and keeps the best n_results.
  Toggle with RERANK=0 to measure the difference: python -m evals.run_evals --suite retrieval
"""

import os
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from functools import lru_cache
from typing import Optional

# Must be the same model used at ingestion time ([RAG-3 EMBED]).
# Loaded once at import so each request only pays for encode(), not model load.
model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "ingestion", "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# get_or_create keeps the cosine space configured at ingestion; if ingestion has not
# been run yet this creates an empty collection and queries simply return nothing.
jobs_collection = chroma_client.get_or_create_collection(name="jobs")

# [RAG-6 RERANK] ms-marco-MiniLM-L-6-v2: a 22M-param cross-encoder trained on MS MARCO
# query/passage relevance. Small enough for CPU; ~20 pairs score in well under a second.
RERANK_ENABLED = os.getenv("RERANK", "1") != "0"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = 20  # stage-1 recall budget: wide enough to contain the right chunk


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    # Loaded lazily so importing the retriever (e.g. in tests) doesn't pay for it.
    return CrossEncoder(RERANK_MODEL)


# Query normalisation: expand common role abbreviations before embedding/reranking.
# Found via eval: the cross-encoder scored every chunk ~-11 for "SWE" (it has never seen
# the abbreviation) but ranked the right occupation first once expanded to "software engineer".
ROLE_ALIASES = {
    "swe": "software engineer",
    "sde": "software development engineer",
    "mle": "machine learning engineer",
    "ml engineer": "machine learning engineer",
    "ds": "data scientist",
    "pm": "product manager",
    "qa": "quality assurance engineer",
    "devops": "devops engineer",
    "rn": "registered nurse",
}


def expand_role(role: str) -> str:
    return ROLE_ALIASES.get(role.strip().lower(), role.strip())


def embed_text(text: str) -> list:
    return model.encode(text).tolist()


def rerank_chunks(query: str, chunks: list[dict], top_k: int, model=None) -> list[dict]:
    """
    [RAG-6 RERANK] Rescore stage-1 candidates with a cross-encoder and keep the top_k.
    `model` lets tests inject a fake scorer.
    """
    if not chunks:
        return []
    model = model or get_reranker()
    scores = model.predict([(query, c["text"]) for c in chunks])
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


def query_jobs(query: str, n_results: int = 10, title_filter=None,
               rerank: Optional[bool] = None, rerank_query: Optional[str] = None) -> list[dict]:
    """
    Hybrid retrieval: exact metadata filter on `title` + dense vector similarity.

    1. Pre-filter: restrict the candidate set to chunks whose title matches the
       target role(s). This is cheap and very precise when the user names an
       O*NET-style title ("Data Scientists").
    2. Rank: cosine similarity between the query embedding and chunk embeddings.
    3. Fallback: if the filter matched nothing (user typed "SWE" instead of
       "Software Developers"), retry with pure semantic search so we still return
       the closest occupations instead of an empty context.
    4. Rerank (optional, on by default): over-fetch RERANK_CANDIDATES, rescore with the
       cross-encoder against `rerank_query` (a natural-language version of the query),
       return the best n_results.
    """
    rerank = RERANK_ENABLED if rerank is None else rerank
    fetch_k = max(n_results, RERANK_CANDIDATES) if rerank else n_results
    embedding = embed_text(query)

    if isinstance(title_filter, list):
        where_clause = {"title": {"$in": title_filter}}
    elif title_filter:
        where_clause = {"title": {"$eq": title_filter}}
    else:
        where_clause = None

    results = jobs_collection.query(
        query_embeddings=[embedding],
        n_results=fetch_k,
        where=where_clause
    )

    # Fallback: if title filter matched nothing, retry with semantic search only
    if not results["documents"][0] and where_clause is not None:
        results = jobs_collection.query(
            query_embeddings=[embedding],
            n_results=fetch_k
        )

    formatted = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        formatted.append({
            "text": doc,
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "distance": round(dist, 4),  # cosine distance from stage 1 (lower = closer)
        })

    if rerank:
        return rerank_chunks(rerank_query or query, formatted, top_k=n_results)
    return formatted[:n_results]


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
    titles = target_title if isinstance(target_title, list) else ([target_title] if target_title else [])
    # Expanded titles are used for the *query text* only; the metadata filter still uses the
    # raw title and falls back to semantic search when it doesn't match exactly.
    target_title_str = ", ".join(expand_role(t) for t in titles)
    target_role_str = expand_role(target_role_str) if target_role_str else ""
    query = " ".join(part for part in [skills_str, coursework_str, projects_str, major_str, target_role_str, target_title_str] if part).strip()

    # The cross-encoder was trained on natural-language queries, so give it a sentence
    # that leads with the role instead of the raw keyword soup used for embedding.
    role = target_title_str or target_role_str or major_str
    rerank_query = f"{role} job requiring {skills_str or coursework_str}".strip()

    job_chunks = query_jobs(query, n_results=10, title_filter=target_title, rerank_query=rerank_query)

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
