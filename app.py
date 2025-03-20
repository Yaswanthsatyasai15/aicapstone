import chromadb # type: ignore

# Create a ChromaDB client (persistent storage)
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
collection = client.get_or_create_collection(name="my_collection")

# Check if embeddings exist before adding
existing_ids = collection.get()["ids"]

if "1" not in existing_ids and "2" not in existing_ids:
    # Add some data (vectors and metadata)
    collection.add(
        ids=["1", "2"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        metadatas=[{"info": "first"}, {"info": "second"}]
    )
else:
    print("Embeddings already exist in the database.")

# Retrieve results
results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3]],
    n_results=1,
    include=["embeddings", "metadatas"]
)

print(results)