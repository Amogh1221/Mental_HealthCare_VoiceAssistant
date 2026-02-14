"""
build_vectordb.py - Complete pipeline to build psychiatric knowledge vector database
Handles the Compumacy/Psych_data dataset with user_message, assistant_message structure
"""

import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import json

DATASET_NAME = "Compumacy/Psych_data"
RAG_FILE = "RAG_data.txt"
PERSIST_DIR = "MH_db"
COLLECTION_NAME = "Docs"
DELIMITER = "=" * 80
BATCH_SIZE = 1000  
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Choose what to index:
# "assistant_only" - Only assistant messages (recommended for most use cases)
# "qa_pairs" - Both question and answer together
# "both_separate" - Questions and answers as separate documents
INDEX_MODE = "assistant_only"

def download_dataset():
    """Download the psychiatric dataset from HuggingFace"""
    print("📥 Downloading dataset from HuggingFace...")
    try:
        ds = load_dataset(DATASET_NAME)
        print(f" Dataset downloaded successfully")
        print(f" Dataset structure: {ds}")
        return ds
    except Exception as e:
        print(f" Error downloading dataset: {e}")
        raise

def save_dataset_to_txt(dataset, output_file: str, mode: str = "assistant_only"):
    """
    Save dataset to a text file with delimiters
    
    Args:
        dataset: HuggingFace dataset
        output_file: Path to output file
        mode: How to save the data
            - "assistant_only": Only save assistant messages
            - "qa_pairs": Save Q&A pairs together
            - "both_separate": Save questions and answers separately
    """
    print(f" Saving dataset to {output_file} (mode: {mode})...")
    
    data = dataset['train']
    doc_count = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc="Processing documents"):
            user_msg = item.get('user_message', '')
            assistant_msg = item.get('assistant_message', '')
            metadata = item.get('metadata', {})
            
            if mode == "assistant_only":
                if assistant_msg and assistant_msg.strip():
                    f.write(assistant_msg.strip())
                    f.write("\n\n" + DELIMITER + "\n\n")
                    doc_count += 1
                    
            elif mode == "qa_pairs":
                if user_msg and assistant_msg:
                    combined = f"Q: {user_msg.strip()}\n\nA: {assistant_msg.strip()}"
                    f.write(combined)
                    f.write("\n\n" + DELIMITER + "\n\n")
                    doc_count += 1
                    
            elif mode == "both_separate":
                if user_msg and user_msg.strip():
                    f.write(f"[Question] {user_msg.strip()}")
                    f.write("\n\n" + DELIMITER + "\n\n")
                    doc_count += 1
                
                if assistant_msg and assistant_msg.strip():
                    f.write(assistant_msg.strip())
                    f.write("\n\n" + DELIMITER + "\n\n")
                    doc_count += 1
    
    print(f" Saved {doc_count} documents to {output_file}")
    return doc_count


def load_documents_from_txt(file_path: str):
    """Load and chunk documents from text file"""
    print(f" Loading documents from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" {file_path} not found")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    entries = [
        entry.strip()
        for entry in raw_text.split(DELIMITER)
        if entry.strip()
    ]

    documents = [
        Document(
            page_content=entry,
            metadata={"source": file_path, "chunk_id": idx}
        )
        for idx, entry in enumerate(entries)
    ]
    
    print(f" Loaded {len(documents)} document chunks")
    return documents


def create_documents_directly(dataset, mode: str = "assistant_only"):
    """
    Create Document objects directly from dataset without intermediate file
    This is more efficient and preserves metadata
    """
    print(f" Creating documents directly from dataset (mode: {mode})...")
    
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
            "medical_context": metadata.get('medical_context', ''),
        }
        
        if mode == "assistant_only":
            if assistant_msg and assistant_msg.strip():
                documents.append(
                    Document(
                        page_content=assistant_msg.strip(),
                        metadata=doc_metadata
                    )
                )
                
        elif mode == "qa_pairs":
            if user_msg and assistant_msg:
                combined = f"Question: {user_msg.strip()}\n\nAnswer: {assistant_msg.strip()}"
                doc_metadata["has_question"] = True
                documents.append(
                    Document(
                        page_content=combined,
                        metadata=doc_metadata
                    )
                )
                
        elif mode == "both_separate":
            if user_msg and user_msg.strip():
                q_metadata = doc_metadata.copy()
                q_metadata["type"] = "question"
                documents.append(
                    Document(
                        page_content=f"Question: {user_msg.strip()}",
                        metadata=q_metadata
                    )
                )
            
            if assistant_msg and assistant_msg.strip():
                a_metadata = doc_metadata.copy()
                a_metadata["type"] = "answer"
                documents.append(
                    Document(
                        page_content=assistant_msg.strip(),
                        metadata=a_metadata
                    )
                )
    
    print(f" Created {len(documents)} document objects")
    return documents


def build_vector_db(documents):
    """Build and persist the vector database"""
    print(" Building vector database...")
    
    print(f" Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(PERSIST_DIR):
        print(f"  Vector DB already exists at {PERSIST_DIR}")
        response = input("Do you want to overwrite it? (y/n): ").lower()
        if response != 'y':
            print(" Aborted. Exiting...")
            return None
        
        import shutil
        shutil.rmtree(PERSIST_DIR)
        print("  Removed existing vector DB")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    print(f" Adding documents in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Inserting batches"):
        batch = documents[i:i + BATCH_SIZE]
        vector_store.add_documents(batch)

    print(" Vector DB build complete!")
    return vector_store


def verify_vector_db(persist_dir: str):
    """Verify that the vector DB was created successfully"""
    print(" Verifying vector database...")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        test_queries = [
            "depression symptoms",
            "anxiety treatment",
            "hallucinations"
        ]
        
        print(f" Vector DB verified successfully!")
        print(f"\n Testing sample queries:")
        print("-" * 80)
        
        for query in test_queries:
            results = vector_store.similarity_search(query, k=2)
            print(f"\n🔍 Query: '{query}'")
            print(f"   Found {len(results)} results")
            if results:
                print(f"   Preview: {results[0].page_content[:200]}...")
        
        print("-" * 80)
        
        return True
    except Exception as e:
        print(f" Verification failed: {e}")
        return False


def main():
    """Main execution pipeline"""
    print("="*80)
    print(" PSYCHIATRIC KNOWLEDGE BASE - VECTOR DB BUILDER")
    print("="*80)
    print()
    
    print(" Choose indexing mode:")
    print("  1. Assistant messages only (recommended - clean answers)")
    print("  2. Q&A pairs (question + answer together)")
    print("  3. Both separate (questions and answers as separate docs)")
    
    mode_choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    mode_map = {
        "1": "assistant_only",
        "2": "qa_pairs",
        "3": "both_separate"
    }
    
    mode = mode_map.get(mode_choice, "assistant_only")
    print(f" Selected mode: {mode}\n")
    
    try:
        dataset = download_dataset()
        print()
        
        print("📚 Creating documents from dataset...")
        use_direct = input("Use direct document creation (faster, preserves metadata)? (y/n) [default: y]: ").lower() or 'y'
        print()
        
        if use_direct == 'y':
            documents = create_documents_directly(dataset, mode=mode)
        else:
            doc_count = save_dataset_to_txt(dataset, RAG_FILE, mode=mode)
            print()
            documents = load_documents_from_txt(RAG_FILE)
        
        print()
        
        vector_store = build_vector_db(documents)
        print()
        
        if vector_store:
            verify_vector_db(PERSIST_DIR)
            print()
        
        print("="*80)
        print(" VECTOR DATABASE BUILD COMPLETE!")
        print("="*80)
        print(f" Location: {os.path.abspath(PERSIST_DIR)}")
        print(f" Total documents indexed: {len(documents)}")
        print(f" Collection name: {COLLECTION_NAME}")
        print(f" Embedding model: {EMBED_MODEL}")
        print(f" Indexing mode: {mode}")
        print()
        print("💡 You can now use this vector DB with your RAG engine!")
        print("="*80)
        
        if os.path.exists(RAG_FILE) and use_direct != 'y':
            cleanup = input("\n  Do you want to delete the intermediate RAG_data.txt file? (y/n): ").lower()
            if cleanup == 'y':
                os.remove(RAG_FILE)
                print(f"✅ Deleted {RAG_FILE}")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        print("Build process failed. Please check the error message above.")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()