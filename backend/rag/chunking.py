"""
[RAG-2 CHUNK] Chunking strategy for the Elevatr knowledge base.

Strategy: record-level ("one entity = one chunk") chunking
-----------------------------------------------------------
Our sources are *structured* records, not long free-text documents:

  * O*NET occupations   (MongoDB: onet_occupations)     -> 1 chunk per occupation
  * AI career recs      (MongoDB: ai-based_career_recs) -> 1 chunk per candidate row

Each record is flattened into a small labelled text block ("Job Title: ...\nSkills: ...").
We deliberately do NOT use fixed-size / sliding-window splitting because:

  1. Every downstream consumer wants a whole occupation back (title + skills + tasks),
     not a fragment that mentions "SQL" but has lost which job it belongs to.
  2. The "Job Title:" header line is repeated at the top of every chunk, so the
     embedding is always anchored to the role it describes.
  3. The title is also stored as *metadata*, which lets the retriever apply an exact
     metadata filter (title == target role) before doing vector similarity.
     See rag/retriever.py::query_jobs.

Known trade-off (good interview talking point)
----------------------------------------------
The embedding model (all-MiniLM-L6-v2) only reads the first 256 word-pieces of the
input. Long O*NET task lists get silently truncated *for the embedding*, although the
full text is still stored and returned to the generator. Field order is therefore
chosen most-important-first: Title -> Description -> Skills -> Knowledge -> Tasks.
If recall on task-level questions becomes a problem, the next step is to split each
occupation into an "overview" chunk and N "tasks" chunks that share the same
`title` / `mongo_id` metadata (parent-document retrieval).
"""

import ast
from typing import Any, Dict

# all-MiniLM-L6-v2 `max_seq_length`; anything beyond this is ignored when embedding.
EMBEDDING_TOKEN_LIMIT = 256


def parse_list_field(value: Any) -> str:
    """
    O*NET stores skills/knowledge as stringified lists of dicts like:
    "[{'name': 'Programming', 'importance': 4.0}, ...]"
    Extract just the names as a clean comma-separated string.
    """
    if not value:
        return ""
    if isinstance(value, list):
        # Already a real list of dicts
        return ", ".join(item.get("name", "") for item in value if isinstance(item, dict))
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return ", ".join(item.get("name", "") for item in parsed if isinstance(item, dict))
        except Exception:
            pass
        return value
    return str(value)


def parse_task_statements(value: Any) -> str:
    """Parse task_statements which is a stringified list of strings."""
    if not value:
        return ""
    if isinstance(value, list):
        return " ".join(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return " ".join(str(t) for t in parsed)
        except Exception:
            pass
        return value
    return str(value)


def build_career_rec_chunk(doc: Dict[str, Any]) -> tuple[str, Dict[str, str]]:
    """
    One `ai-based_career_recs` row -> (chunk_text, metadata).
    Source fields: CandidateID, Name, Age, Education, Skills, Interests, Recommended_Career.
    PII fields (Name, Age, CandidateID) are intentionally dropped; they add no retrieval signal.
    """
    title = doc.get("Recommended_Career", "")
    text = (
        f"Job Title: {title}\n"
        f"Skills: {parse_list_field(doc.get('Skills', ''))}\n"
        f"Interests: {parse_list_field(doc.get('Interests', ''))}\n"
        f"Education: {doc.get('Education', '')}"
    )
    metadata = {
        "title": title,                        # used for exact-match `where` filtering
        "source": "career_recs",               # lets us attribute / debug retrieved chunks
        "mongo_id": str(doc.get("_id", "")),   # pointer back to the source-of-truth record
    }
    return text, metadata


def build_onet_chunk(doc: Dict[str, Any]) -> tuple[str, Dict[str, str]]:
    """
    One `onet_occupations` document -> (chunk_text, metadata).
    Source fields: soc_code, occupation_title, occupation_description,
                   job_zone, skills, knowledge, abilities, task_statements.
    Fields are ordered most-important-first because of EMBEDDING_TOKEN_LIMIT.
    """
    title = doc.get("occupation_title", "")
    text = (
        f"Job Title: {title}\n"
        f"Description: {doc.get('occupation_description', '')}\n"
        f"Skills: {parse_list_field(doc.get('skills', ''))}\n"
        f"Knowledge: {parse_list_field(doc.get('knowledge', ''))}\n"
        f"Tasks: {parse_task_statements(doc.get('task_statements', ''))}"
    )
    metadata = {
        "title": title,
        "source": "onet",
        "mongo_id": str(doc.get("_id", "")),
    }
    return text, metadata
