import pandas as pd
import requests
import random
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

df = pd.read_csv("med_query.csv", sep=";", encoding="utf-8", on_bad_lines="warn", engine="python")

sample_df = df.sample(10)

model = SentenceTransformer("all-mpnet-base-v2")

API_URL = "http://localhost:8181/answer_from_table"

results = []

for _, row in sample_df.iterrows():
    question = row['question']
    expected_answer = row['answer']
    
    start_time = time.time()
    response = requests.post(API_URL, json={"question": question, "temperature": 0.5, "lang": "en"})
    response_time = time.time() - start_time
    
    if response.status_code == 200:
        actual_answer = response.json().get("answer", "")
        
        expected_embedding = model.encode(expected_answer)
        actual_embedding = model.encode(actual_answer)
        similarity = cosine_similarity([expected_embedding], [actual_embedding])[0][0]
        
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "response_time": response_time,
            "similarity_score": similarity
        })
    else:
        print(f"Error for question: {question}")

avg_response_time = sum(r["response_time"] for r in results) / len(results)
avg_similarity = sum(r["similarity_score"] for r in results) / len(results)

print(f"=== ÉVALUATION DU CHATBOT MEDICLA ===")
print(f"Nombre d'exemples testés: {len(results)}")
print(f"Temps de réponse moyen: {avg_response_time:.2f} secondes")
print(f"Score de similarité moyen: {avg_similarity:.2f}")

for i, r in enumerate(results):
    print(f"\nExemple {i+1}:")
    print(f"Question: {r['question']}")
    print(f"Similarité: {r['similarity_score']:.2f}")
    print(f"Temps: {r['response_time']:.2f}s")