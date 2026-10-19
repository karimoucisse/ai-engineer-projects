from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
embedding = OllamaEmbeddings(model = "nomic-embed-text")

def create_db():
	vector_store = Chroma(
		collection_name="hybrid_rag",
		embedding_function= embedding,
		persist_directory="./chroma_langchain_db"
	)
	return vector_store
