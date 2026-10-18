from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_kb():
	""" Creating a vector db for knowledg base """
	# File loading
	with open("nelson_mandela.txt") as f:
		texts = f.read()

	# Chunking
	text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
	chunks = text_splitter.create_documents([texts])

	# Embedding
	embedding_model = OllamaEmbeddings(model = "nomic-embed-text")

	# Vector DB
	vector_store = Chroma(
		collection_name="rag_collection",
		embedding_function=embedding_model,
		persist_directory="./chroma_langchain_db"
    )
	if vector_store._collection.count() == 0:
		vector_store.add_documents(chunks)

	return vector_store

def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)

def basic_rag():
	vector_store = create_kb()
	retriver = vector_store.as_retriever(
		search_type="similarity", search_kwargs={'k': 2}
	)
	llm = ChatOllama(
		model = "llama3.2:3b",
		temperature = 0,
    )
	prompt = PromptTemplate.from_template(
		"""
            Answer the question based only on the following context:
			Context: {context}
			Question: {question}
			Answer:
			Make sure to answer in a concise manner, and if you don't know the answer, just say "i don't know"
        """
    )

	# langchain lcel
	rag_chain = (
		{"context": retriver | format_docs, "question": RunnablePassthrough()}
		| prompt
		| llm
		| StrOutputParser()
	)

	questions = [
		"When was nelson mandela elected president ?",
		"Who is Kevin de bruyne ?",
		"Where and when was nelson mandela born ?"
	]

	for question in questions:
		print(f"Question: {question}")
		answer = rag_chain.invoke(question)
		print(f"Answer: {answer.strip()}\n\n")

basic_rag()
