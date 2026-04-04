import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Cloud-hosted Hugging Face Embeddings for Pinecone
        self.embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # Load the cloud vector store directly
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )

    def retrieve(self, query: str, k: int = 3):
        results = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])

rag_engine = RAGEngine()