import os
os.environ["PYTORCH_ENABLE_MPS"] = "0"  # Disable MPS backend

import chromadb
from sentence_transformers import SentenceTransformer

print("Initializing Chroma client...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create collections for categories
categories = ["beginner", "intermediate", "expert"]
collections = {cat: chroma_client.get_or_create_collection(name=cat) for cat in categories}

# Function to embed and store data
def store_data_in_chroma(file_path, category):
    print(f"Processing file: {file_path} for category: {category}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.readlines()

    print("Encoding content...")
    embeddings = model.encode(content)

    # Add to Chroma
    for i, text in enumerate(content):
        print(f"Adding document {i} to {category} collection")
        collections[category].add(
            embeddings=[embeddings[i]],
            documents=[text],
            ids=[f"{category}_{i}"]
        )

    print(f"✅ Data stored in {category} collection.")

store_data_in_chroma("wiki_AI_clean.txt", "beginner")
store_data_in_chroma("wiki_MLs_clean.txt", "intermediate")