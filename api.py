from fastapi import FastAPI
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import os
# Add this before the FastAPI app definition
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

# Initialize paths and models
db_path = os.path.expanduser("~/Documents/chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up collections
categories = ["beginner", "intermediate", "expert"]
collections = {cat: chroma_client.get_or_create_collection(name=cat) for cat in categories}
# Add this after creating the collections
print("Verifying data in ChromaDB...")
for cat in categories:
    count = collections[cat].count()
    print(f"Category {cat} has {count} documents")

# Query function
def query_chroma(query, category, top_k=3):
    if category not in collections:
        return {"error": "Invalid category"}
    
    query_embedding = model.encode(query)
    results = collections[category].query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"]

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

# Run FastAPI (for local testing)
if __name__ == "__main__":
    import uvicorn
    from fastapi.testclient import TestClient

    client = TestClient(app)

    def test_search():
        response = client.get("/search?query=artificial intelligence&category=beginner")
        print("Test response:", response.json())

    test_search()

    uvicorn.run(app, host="0.0.0.0", port=8000)