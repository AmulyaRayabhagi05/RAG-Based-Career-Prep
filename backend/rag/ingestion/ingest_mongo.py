"""
Offline ingestion pipeline: MongoDB -> chunks -> embeddings -> ChromaDB.

    [RAG-1 LOAD]   read raw records from MongoDB Atlas
    [RAG-2 CHUNK]  flatten each record into one labelled text chunk   (rag/chunking.py)
    [RAG-3 EMBED]  encode each chunk with all-MiniLM-L6-v2 (384-dim)
    [RAG-4 INDEX]  batch-write vectors + text + metadata into Chroma (HNSW, cosine)

Run from backend/:   python rag/ingestion/ingest_mongo.py
This is a full rebuild: the `jobs` collection is dropped and re-created every run,
which keeps the index idempotent (no duplicate chunks across re-runs).
"""

import os
import sys
import uuid

from dotenv import load_dotenv
from pymongo import MongoClient
import chromadb
from sentence_transformers import SentenceTransformer

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rag.chunking import build_career_rec_chunk, build_onet_chunk

# --- Setup Environment Variables ---
load_dotenv()

mongo_url = os.getenv("MONGO_URI")
client = MongoClient(mongo_url)
db = client["elevatr_db"]

career_recs_collection = db["ai-based_career_recs"]
onet_collection = db["onet_occupations"]

# --- [RAG-3 EMBED] Embedding model ---
# all-MiniLM-L6-v2: 384-dim sentence embeddings, ~80MB, runs on CPU.
# Chosen for zero API cost and low latency; the SAME model must be used at query
# time (rag/retriever.py), otherwise query and document vectors live in different spaces.
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- [RAG-4 INDEX] Chroma persistent store ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)


def embed_text(text: str) -> list:
    return embed_model.encode(text).tolist()


def reset_jobs_collection():
    """Drop and re-create the `jobs` collection (full rebuild, keeps ingestion idempotent)."""
    try:
        chroma_client.delete_collection(name="jobs")
        print("  Cleared existing 'jobs' collection")
    except Exception:
        pass

    # hnsw:space=cosine -> similarity is direction-only, which is what sentence
    # embeddings are trained for. (Chroma's default is L2.)
    return chroma_client.get_or_create_collection(
        name="jobs",
        metadata={"hnsw:space": "cosine"}
    )


def ingest_all():
    jobs_chroma = reset_jobs_collection()
    ids, embeddings, documents, metadatas = [], [], [], []

    def add_chunk(text: str, metadata: dict):
        ids.append(str(uuid.uuid4()))
        embeddings.append(embed_text(text))   # [RAG-3 EMBED]
        documents.append(text)                # raw text is stored so the generator can read it
        metadatas.append(metadata)            # title/source/mongo_id for filtering + attribution

    # ── 1. ai-based_career_recs ──────────────────────────────────────────
    # [RAG-1 LOAD] -> [RAG-2 CHUNK]: one row = one chunk
    career_docs = list(career_recs_collection.find())
    print(f" Found {len(career_docs)} docs in ai-based_career_recs")
    for doc in career_docs:
        add_chunk(*build_career_rec_chunk(doc))

    # ── 2. onet_occupations ──────────────────────────────────────────────
    # [RAG-1 LOAD] -> [RAG-2 CHUNK]: one occupation = one chunk
    onet_docs = list(onet_collection.find())
    print(f" Found {len(onet_docs)} docs in onet_occupations")
    for doc in onet_docs:
        add_chunk(*build_onet_chunk(doc))

    # ── 3. [RAG-4 INDEX] Batch insert ────────────────────────────────────
    if not ids:
        print("  No documents found. Check your MongoDB connection.")
        return

    # Chroma caps a single add() call (~5k records); 2000 keeps each request small.
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
