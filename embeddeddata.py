import os
from sentence_transformers import SentenceTransformer
import chromadb

os.environ["PYTORCH_ENABLE_MPS"] = "0"  # Disable MPS backend

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_FILE = os.path.join(BASE_DIR, "wiki_AI_clean.txt")
ML_FILE = os.path.join(BASE_DIR, "wiki_MLs_clean.txt")

# Initialize Chroma
chroma_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
categories = ["beginner", "intermediate", "expert"]
collections = {cat: chroma_client.get_or_create_collection(name=cat) for cat in categories}

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 80MB vs 120MB

def store_data_in_chroma(file_path: str, category: str) -> None:
    """Store document chunks in ChromaDB with embeddings."""
    try:
        print(f"🔍 Processing {os.path.basename(file_path)}...")
        
        # Read and chunk data
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().split("\n\n")  # Split by paragraphs
        content = [para.strip() for para in content if para.strip()]
        
        if not content:
            print(f"⚠️ No content found in {os.path.basename(file_path)}!")
            return

        # Batch embed and store
        print(f"📥 Loaded {len(content)} documents. Generating embeddings...")
        embeddings = model.encode(content)
        
        # Convert numpy arrays to lists for Chroma
        embeddings_list = embeddings.tolist()
        
        # Add to Chroma in batches (avoids memory issues)
        batch_size = 100  # Adjust based on your system's memory
        for i in range(0, len(content), batch_size):
            batch_ids = [f"{category}_{i + j}" for j in range(len(content[i:i + batch_size]))]
            collections[category].add(
                embeddings=embeddings_list[i:i + batch_size],
                documents=content[i:i + batch_size],
                ids=batch_ids
            )
            print(f"➕ Added batch {i // batch_size + 1} to {category} collection")
        
        print(f"✅ Successfully stored {len(content)} documents in '{category}' collection\n")
        
    except FileNotFoundError:
        print(f"❌ File {os.path.basename(file_path)} not found! Check file paths.")
    except Exception as e:
        print(f"❌ Unexpected error in {category}: {str(e)}")

def initialize_data() -> None:
    """Main function to load all data into ChromaDB."""
    print("🚀 Starting data initialization...")
    store_data_in_chroma(AI_FILE, "beginner")
    store_data_in_chroma(ML_FILE, "intermediate")
    print("🎉 All data loaded successfully!")

if __name__ == "__main__":
    initialize_data()
