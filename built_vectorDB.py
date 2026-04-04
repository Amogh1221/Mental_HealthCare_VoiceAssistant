"""
build_pinecone_db.py - Restored original suite of chunking modes for Pinecone Cloud.
Handles the Compumacy/Psych_data dataset with user_message, assistant_message structure.
"""

import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Configuration from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mhcva-index")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATASET_NAME = "Compumacy/Psych_data"
BATCH_SIZE = 100

def download_dataset():
    print("📥 Downloading dataset from HuggingFace...")
    try:
        ds = load_dataset(DATASET_NAME)
        print("Dataset downloaded successfully.")
        return ds
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def create_documents_from_dataset(dataset, mode="assistant_only"):
    """
    Restored original direct document creation logic with metadata preservation.
    
    Modes:
    - "assistant_only": Only save assistant messages (cleanest answers)
    - "qa_pairs": Save Question + Answer pairs together
    - "both_separate": Save questions and answers as separate documentation
    """
    print(f" Creating documents from dataset (mode: {mode})...")
    
    data = dataset['train']
    documents = []
    
    for idx, item in enumerate(tqdm(data, desc="Processing documents")):
        user_msg = item.get('user_message', '')
        assistant_msg = item.get('assistant_message', '')
        metadata = item.get('metadata', {})
        
        doc_metadata = {
            "chunk_id": idx,
            "source_pdf": metadata.get('source_pdf', 'unknown'),
            "page_number": metadata.get('page_number', -1),
            "confidence_score": metadata.get('confidence_score', 0.0),
        }
        
        if mode == "assistant_only":
            if assistant_msg and assistant_msg.strip():
                documents.append(
                    Document(page_content=assistant_msg.strip(), metadata=doc_metadata)
                )
                
        elif mode == "qa_pairs":
            if user_msg and assistant_msg:
                combined = f"Question: {user_msg.strip()}\n\nAnswer: {assistant_msg.strip()}"
                documents.append(
                    Document(page_content=combined, metadata=doc_metadata)
                )
                
        elif mode == "both_separate":
            if user_msg and user_msg.strip():
                q_meta = doc_metadata.copy()
                q_meta["type"] = "question"
                documents.append(
                    Document(page_content=f"Question: {user_msg.strip()}", metadata=q_meta)
                )
            
            if assistant_msg and assistant_msg.strip():
                a_meta = doc_metadata.copy()
                a_meta["type"] = "answer"
                documents.append(
                    Document(page_content=assistant_msg.strip(), metadata=a_meta)
                )
    
    print(f" Created {len(documents)} document objects.")
    return documents

def build_pinecone_db(documents):
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY missing from .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Get dimensions from model initialization
    print(f" Initializing embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    test_dim = len(embeddings.embed_query("test"))
    print(f" Model dimension identified: {test_dim}")

    # Create index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
        print(f" Creating new serverless index on Pinecone: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=test_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    print(f" Uploading batches to Pinecone...")
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    print(" Pinecone DB build complete!")
    return vector_store

def main():
    print("-" * 50)
    print(" PINE-PSYCH KNOWLEDGE BUILDER")
    print("-" * 50)
    
    print(" Choose indexing mode:")
    print("  1. Assistant messages only (default)")
    print("  2. Q&A pairs (combined)")
    print("  3. Both separate (individual docs)")
    
    choice = input("\nEnter choice (1/2/3) [1]: ").strip() or "1"
    mode_map = {"1": "assistant_only", "2": "qa_pairs", "3": "both_separate"}
    mode = mode_map.get(choice, "assistant_only")

    try:
        dataset = download_dataset()
        documents = create_documents_from_dataset(dataset, mode=mode)
        build_pinecone_db(documents)
        print("\n✅ Success! Your knowledge base is now live on Pinecone.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    main()