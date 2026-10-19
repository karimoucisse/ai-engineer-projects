from rag.chain import rag_chain


def run_rag():
	questions = [
				"When was nelson mandela elected president ?",
				"Who is Kevin de bruyne ?",
				"Where and when was nelson mandela born ?"
	]
	chain = rag_chain()

	for question in questions:
		print(f"Question: {question}")
		answer = chain.invoke(question)
		print(f"Answer: {answer.strip()}\n\n")

if __name__ == "__main__":
	run_rag()
