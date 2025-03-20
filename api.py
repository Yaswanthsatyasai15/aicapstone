import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer  # Added import
import chromadb
from embeddeddata import initialize_data

# Modern lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize data and models
    initialize_data()
    global chroma_client, model, collections
    db_path = os.path.expanduser("~/Documents/chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    categories = ["beginner", "intermediate", "expert"]
    collections = {cat: chroma_client.get_or_create_collection(name=cat) for cat in categories}
    yield

app = FastAPI(lifespan=lifespan)

<<<<<<< HEAD
@app.get("/")
def health_check():
    return {"status": "OK"}
=======
# Initialize paths and models
db_path = os.path.expanduser("~/Documents/chroma_project")
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
>>>>>>> d16323bb09c8206ded40c22650dc8a36cf373a06

# Query function
def query_chroma(query, category, top_k=3):
    if category not in collections:
        return {"error": "Invalid category"}
    
    query_embedding = model.encode(query)
    results = collections[category].query(
        query_embeddings=[query_embedding.tolist()],  # Convert numpy array to list
        n_results=top_k
    )
    return results["documents"]

@app.get("/search")
async def search(query: str, category: str):
    try:
        results = query_chroma(query, category)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
<<<<<<< HEAD
    uvicorn.run(app, host="0.0.0.0", port=8000)
=======
    from fastapi.testclient import TestClient

    client = TestClient(app)

    def test_search():
        response = client.get("/search?query=artificial intelligence&category=beginner")
        print("Test response:", response.json())

    test_search()

    uvicorn.run(app, host="0.0.0.0", port=8000)
>>>>>>> d16323bb09c8206ded40c22650dc8a36cf373a06
