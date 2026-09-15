import chromadb

# Connect directly to the persisted local Chroma storage
client = chromadb.PersistentClient(path="./chroma_db")

# Inspect all collections created in the DB
collections = client.list_collections()
print(f"Total collections found: {len(collections)}\n")

for col in collections:
    print(f"--- Collection Name: {col.name} ---")
    data = col.get()  # Fetches all stored entries
    
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    
    print(f"Total items stored: {len(ids)}\n")
    for idx, (doc_id, doc_text) in enumerate(zip(ids, documents), start=1):
        print(f"[{idx}] ID: {doc_id}")
        print(f"     Content: {doc_text}\n")