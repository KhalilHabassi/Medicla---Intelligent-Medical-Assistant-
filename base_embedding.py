import pandas as pd
import torch
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import psycopg2.extras
import numpy as np
from tqdm import tqdm

DB_PASS = ""
DB_HOST = "localhost"
INSTANCE = "gen-ai-instance"
DATABASE = "gen_ai_db"
DB_USER = "students"
DB_NAME = "gen_ai_db"
DB_PORT = "5432"

db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)

csv_path = "med_query.csv"
df = pd.read_csv(csv_path, sep=";", encoding="utf-8", on_bad_lines="warn", engine="python")
df.columns = df.columns.str.strip()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de: {device}")

model = SentenceTransformer("all-mpnet-base-v2")
model.to(device)

contents = []
for idx, row in df.iterrows():
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    content = f"Question: {question}\nAnswer: {answer}"
    contents.append(content)

BATCH_SIZE = 16
total_batches = (len(contents) + BATCH_SIZE - 1) // BATCH_SIZE

all_embeddings = []
print("Génération des embeddings en cours...")
for i in tqdm(range(0, len(contents), BATCH_SIZE)):
    batch = contents[i:i+BATCH_SIZE]
    batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
    all_embeddings.append(batch_embeddings)

all_embeddings = np.vstack(all_embeddings)

records = []
for idx, row in df.iterrows():
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    source = str(row.get("source", "")).strip()[:254]
    focus_area = str(row.get("focus_area", "")).strip()[:254]
    content = f"Question: {question}\nAnswer: {answer}"
    embedding = all_embeddings[idx].tolist()
    
    records.append({
        "question": question,
        "answer": answer,
        "source": source,
        "focus_area": focus_area,
        "content": content,
        "embedding": embedding
    })

INSERTION_BATCH_SIZE = 50
print("Insertion en base de données...")

with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE qa_table RESTART IDENTITY"))
    
    raw_conn = engine.raw_connection()
    cursor = raw_conn.cursor()
    
    for i in tqdm(range(0, len(records), INSERTION_BATCH_SIZE)):
        batch_records = records[i:i+INSERTION_BATCH_SIZE]
        
        values = []
        for rec in batch_records:
            values.append((
                rec["question"],
                rec["answer"],
                rec["source"],
                rec["focus_area"],
                rec["content"],
                rec["embedding"]
            ))
        
        psycopg2.extras.execute_batch(
            cursor,
            """
            INSERT INTO qa_table (question, answer, source, focus_area, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            values
        )
        
        raw_conn.commit()
    
    cursor.close()
    raw_conn.close()

print(f"Ingestion complète : {len(records)} documents insérés dans kqa_table.")