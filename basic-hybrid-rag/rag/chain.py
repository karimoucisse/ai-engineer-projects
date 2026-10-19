from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from rag.hybrid_retriver import hybrid_retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)

def rag_chain():
	llm = ChatOllama(model="llama3.2:3b")
	prompt = ChatPromptTemplate.from_template(
		"""
			Using this context, answer the question located between the delimiters <<< and >>>
			Context: {context}

			Question: <<<{question}>>>

			Answer: Answer this question simply and concisely. If you don't know the answer, just reply: "I do not know the answer to this question."
		"""
	)
	retriever = hybrid_retriever()
	
	# langchain lcel
	chain = (
			{"context": retriever | format_docs, "question": RunnablePassthrough()}
			| prompt
			| llm
			| StrOutputParser()
	)
	return chain

