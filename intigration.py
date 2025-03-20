from fastapi import FastAPI
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi.testclient import TestClient

app = FastAPI()

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Chroma
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
categories = ["beginner", "intermediate", "expert"]
collections = {cat: chroma_client.get_or_create_collection(name=cat) for cat in categories}

# Verify data in ChromaDB
print("Verifying data in ChromaDB...")
for cat in categories:
    count = collections[cat].count()
    print(f"Category {cat} has {count} documents")

# Query ChromaDB
def query_chroma(query, category, top_k=3):
    query_embedding = model.encode(query)
    results = collections[category].query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results.get('documents', [])

@app.get("/search")
async def search(query: str, category: str):
    if category not in collections:
        return {"error": "Invalid category"}
    
    try:
        results = query_chroma(query, category)
        if not results:
            return {"results": []}
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}

# Test and run FastAPI
if __name__ == "__main__":
    import uvicorn
    from fastapi.testclient import TestClient

    client = TestClient(app)

    def test_search():
        response = client.get("/search?query=artificial intelligence&category=beginner")
        print("Test response:", response.json())

    test_search()

    uvicorn.run(app, host="0.0.0.0", port=8000)