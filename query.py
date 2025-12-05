import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# -----------------------------
# Database Connection
# -----------------------------
conn = psycopg2.connect(
    host="localhost",
    database="startup_assistant",  # make sure this matches your DB
    user="postgres",
    password="300234"  # <-- replace with your real password
)
cur = conn.cursor()

# -----------------------------
# Embedding Model
# -----------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# Retriever
# -----------------------------
def retrieve_relevant_context(query, top_k=5):
    query_vec = embedder.encode(query).tolist()

    cur.execute("SELECT doc_id, section, content, embedding FROM legal_docs")
    rows = cur.fetchall()

    scored = []
    for row in rows:
        sim = cosine(query_vec, row[3])
        scored.append((sim, row))

    top_results = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]

    context = "\n\n".join(
        [f"[{r[1][0]} - {r[1][1]}]\n{r[1][2][:500]}..." for r in top_results]
    )
    return context

# -----------------------------
# LLM Query (Ollama Mistral)
# -----------------------------
def ask_legal_assistant(query):
    context = retrieve_relevant_context(query)

    prompt = f"""
You are a Startup Legal Assistant.
Answer the user’s question based ONLY on the following legal documents:

{context}

User Question: {query}

Give a clear, concise, legal answer with references to the context.
"""

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "You are a helpful legal assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("🔎 Startup Legal Assistant (RAG + Mistral)")
    while True:
        query = input("\nAsk a legal question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = ask_legal_assistant(query)
        print("\n🤖 Answer:\n", answer)
