from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import psycopg2
from psycopg2.extras import RealDictCursor
import os


hf_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


DB_CONFIG = {
    "dbname": "gen_ai_db",
    "user": "students",
    "password": "", 
    "host": "localhost",
    "port": "5432"
}

app = APIRouter()

class AnswerRequest(BaseModel):
    question: str
    temperature: float = 0.5
    lang: str = "en"

class AnswerResponse(BaseModel):
    answer: str
    source: str | None
    focus_area: str | None
    similarity_score: float

class SourceDocument(BaseModel):
    source: str | None
    focus_area: str | None
    similarity_score: float
    similarity_type: str
    content: str | None

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

def embedding_to_str(embedding: list[float]) -> str:
    """Convertit une liste de floats en littéral de vecteur compatible PGVector."""
    return "[" + ", ".join(map(str, embedding)) + "]"

@app.post("/answer_from_table", response_model=AnswerResponse)
async def get_answer(request: AnswerRequest):
    query_embedding = hf_model.encode(request.question).tolist()
    query_embedding_str = embedding_to_str(query_embedding)
    
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, answer, source, focus_area, content,
                   embedding <=> %s::vector(768) AS similarity
            FROM qa_table
            ORDER BY similarity
            LIMIT 1
        """, (query_embedding_str,))
        
        result = cur.fetchone()
        if result:
            return AnswerResponse(
                answer=result["answer"],
                source=result["source"],
                focus_area=result["focus_area"],
                similarity_score=float(result["similarity"])
            )
        
        raise HTTPException(404, "No matching answer found")
        
    except Exception as e:
        raise HTTPException(500, f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

@app.get("/get_sources", response_model=list[SourceDocument])
async def get_sources(question: str, temperature: float = 0.5, lang: str = "en"):
    query_embedding = hf_model.encode(question).tolist()
    query_embedding_str = embedding_to_str(query_embedding)
    
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT source, focus_area, content,
                   embedding <=> %s::vector(768) AS similarity
            FROM qa_table
            ORDER BY similarity
            LIMIT 3
        """, (query_embedding_str,))
        
        return [{
            "source": row["source"],
            "focus_area": row["focus_area"],
            "similarity_score": float(row["similarity"]),
            "similarity_type": "cosine",
            "content": row["content"]
        } for row in cur.fetchall()]
        
    except Exception as e:
        raise HTTPException(500, f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()