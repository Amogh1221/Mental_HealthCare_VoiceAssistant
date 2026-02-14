from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class RAGEngine:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        self.vector_store = Chroma(
            collection_name="Docs",
            persist_directory="MH_db",
            embedding_function=self.embeddings
        )

    def retrieve(self, query: str, k: int = 5) -> str:
        """Retrieve relevant documents from vector store"""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)


rag_engine = RAGEngine()