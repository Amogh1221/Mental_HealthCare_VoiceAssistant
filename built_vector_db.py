import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# CONFIG

RAG_FILE = "RAG_data.txt"
PERSIST_DIR = "MH_db"
COLLECTION_NAME = "Docs"
DELIMITER = "=" * 80
BATCH_SIZE = 3000
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LOAD + CHUNK

def load_rag_documents(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    entries = [
        entry.strip()
        for entry in raw_text.split(DELIMITER)
        if entry.strip()
    ]

    documents = [Document(page_content=e) for e in entries]
    return documents

# BUILD VECTOR DB

def build_vector_db(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"Inserted {i + len(batch)} / {len(documents)}")

    print("Vector DB build complete.")
    return vector_store

# MAIN

if __name__ == "__main__":
    print("Loading RAG data...")
    documents = load_rag_documents(RAG_FILE)
    print(f"Total documents: {len(documents)}")

    print("Building vector database...")
    build_vector_db(documents)

    print(f"Vector DB saved to: {PERSIST_DIR}")
