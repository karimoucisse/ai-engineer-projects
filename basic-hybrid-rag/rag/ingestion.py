from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(file_path: str):
	with open(file_path) as f:
		texts = f.read()
	return texts

def split_file():
	texts = load_file("./nelson_mandela.txt")
	splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
	chunks = splitter.create_documents([texts])
	return chunks
