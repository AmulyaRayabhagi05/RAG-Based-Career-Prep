import os
import uuid
import ast
from dotenv import load_dotenv
from pymongo import MongoClient
import chromadb
from sentence_transformers import SentenceTransformer

# --- Setup Environment Variables ---
load_dotenv()

mongo_url = os.getenv("MONGO_URI")
client = MongoClient(mongo_url)
db = client["elevatr_db"]

career_recs_collection = db["ai-based_career_recs"]
onet_collection = db["onet_occupations"]

# --- Setup Embedding Model ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Setup Chroma ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)

try:
    chroma_client.delete_collection(name="jobs")
    print("  Cleared existing 'jobs' collection")
except Exception:
    pass

jobs_chroma = chroma_client.get_or_create_collection(
    name="jobs",
    metadata={"hnsw:space": "cosine"}
)


def embed_text(text: str) -> list:
    return embed_model.encode(text).tolist()


def parse_list_field(value) -> str:
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


def parse_task_statements(value) -> str:
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


def ingest_all():
    ids, embeddings, documents, metadatas = [], [], [], []

    # ── 1. ai-based_career_recs ──────────────────────────────────────────
    # Fields: CandidateID, Name, Age, Education, Skills, Interests, Recommended_Career
    career_docs = list(career_recs_collection.find())
    print(f" Found {len(career_docs)} docs in ai-based_career_recs")

    for doc in career_docs:
        title    = doc.get("Recommended_Career", "")
        skills    = parse_list_field(doc.get("Skills", ""))
        interests = parse_list_field(doc.get("Interests", ""))
        education = doc.get("Education", "")

        text = (
            f"Job Title: {title}\n"
            f"Skills: {skills}\n"
            f"Interests: {interests}\n"
            f"Education: {education}"
        )

        ids.append(str(uuid.uuid4()))
        embeddings.append(embed_text(text))
        documents.append(text)
        metadatas.append({
            "title": title,
            "source": "career_recs",
            "mongo_id": str(doc.get("_id", ""))
        })

    # ── 2. onet_occupations ──────────────────────────────────────────────
    # Fields: soc_code, occupation_title, occupation_description,
    #         job_zone, skills, knowledge, abilities, task_statements
    onet_docs = list(onet_collection.find())
    print(f" Found {len(onet_docs)} docs in onet_occupations")

    for doc in onet_docs:
        title       = doc.get("occupation_title", "")
        description = doc.get("occupation_description", "")
        skills      = parse_list_field(doc.get("skills", ""))
        knowledge   = parse_list_field(doc.get("knowledge", ""))
        tasks       = parse_task_statements(doc.get("task_statements", ""))

        text = (
            f"Job Title: {title}\n"
            f"Description: {description}\n"
            f"Skills: {skills}\n"
            f"Knowledge: {knowledge}\n"
            f"Tasks: {tasks}"
        )

        ids.append(str(uuid.uuid4()))
        embeddings.append(embed_text(text))
        documents.append(text)
        metadatas.append({
            "title": title,
            "source": "onet",
            "mongo_id": str(doc.get("_id", ""))
        })

    # ── 3. Batch insert ──────────────────────────────────────────────────
    if not ids:
        print("  No documents found. Check your MongoDB connection.")
        return

    BATCH_SIZE = 2000
    for i in range(0, len(ids), BATCH_SIZE):
        jobs_chroma.add(
            ids=ids[i:i+BATCH_SIZE],
            embeddings=embeddings[i:i+BATCH_SIZE],
            documents=documents[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE]
        )
        print(f"  ↳ Batch {i // BATCH_SIZE + 1}: inserted {min(i+BATCH_SIZE, len(ids))} / {len(ids)}")

    print(f"\n Done! {len(ids)} total docs in ChromaDB 'jobs' at {db_path}")
    print(f"   • career_recs : {len(career_docs)}")
    print(f"   • onet        : {len(onet_docs)}")


def debug_sample():
    print("\n── career_recs sample ──")
    for doc in career_recs_collection.find().limit(2):
        print({k: str(v)[:80] for k, v in doc.items() if k != "_id"})
    print("\n── onet_occupations sample ──")
    for doc in onet_collection.find().limit(2):
        print({k: str(v)[:80] for k, v in doc.items() if k != "_id"})


if __name__ == "__main__":
    # debug_sample()
    ingest_all()


