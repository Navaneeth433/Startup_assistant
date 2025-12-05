import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Connect to Postgres
conn = psycopg2.connect(
    dbname="startup_assistant",
    user="postgres",
    password="300234",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

def search_and_rerank(query, top_k=3, model_name="mistral"):  # Use smaller model by default
    # Embed query
    query_embedding = model.encode([query])[0].tolist()

    # Fetch docs
    cur.execute("SELECT id, section, content, embedding FROM legal_docs;")
    rows = cur.fetchall()

    docs = []
    for row in rows:
        doc_id, section, content, embedding = row
        if embedding is None:
            continue
        # Cosine similarity
        sim = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        docs.append((sim, section, content))

    # Sort and keep top-k
    top_docs = sorted(docs, key=lambda x: x[0], reverse=True)[:top_k]
    print("\n🔹 Top 3 relevant sections:")
    for i, (sim, sec, txt) in enumerate(top_docs, start=1):
        print(f"\n{i}. Section: {sec}\nSimilarity: {sim:.4f}\nContent: {txt[:200]}...")
    # Format context for Ollama
    context = "\n\n".join([f"Section: {sec}\nContent: {txt}" for _, sec, txt in top_docs])

    # Send to Ollama
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a legal assistant. Use the context to answer queries."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"❌ Error contacting Ollama: {e}"

if __name__ == "__main__":
    q = input("Enter your legal query: ")
    answer = search_and_rerank(q, top_k=3, model_name="mistral")  # Use smaller model
    print("\n📌 Answer:\n", answer)
