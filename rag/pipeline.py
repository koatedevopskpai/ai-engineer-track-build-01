# rag/pipeline.py
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import os

DSN = os.getenv("PGVECTOR_DSN", "postgresql://ai:ai@localhost:5433/rag")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_table(cur):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY, content TEXT, embedding vector(384))""")


def ingest(text_path):
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    build_table(cur)
    with open(text_path, encoding="utf-8") as f:
        for para in f.read().split("\n\n"):
            if not para.strip():
                continue
            emb = MODEL.encode(para).tolist()
            cur.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s,%s)", (para, emb)
            )
    conn.commit()
    cur.close()
    conn.close()


def retrieve(query, k=3):
    conn = psycopg2.connect(DSN)
    register_vector(conn)
    cur = conn.cursor()
    emb = MODEL.encode(query).tolist()
    cur.execute(
        "SELECT content FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
        (emb, k),
    )
    docs = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return docs
