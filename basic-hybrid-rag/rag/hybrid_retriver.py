from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from rag.vector_store import create_db
from rag.ingestion import split_file



def hybrid_retriever():
	documents = split_file()
	vector_store = create_db()
	vector_store.add_documents(documents)
	bm25_retriver = BM25Retriever.from_documents(
		documents,
		k=3
	)
	vector_retriver = vector_store.as_retriever(
		search_type="similarity", search_kwargs={'k': 3}
	)
	ensemble_retriever = EnsembleRetriever(
		retrievers = [bm25_retriver, vector_retriver],
		weights= [0.25, 0.25]
	)
	return ensemble_retriever


