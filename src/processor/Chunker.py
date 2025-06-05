from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk(self, docs):
        return self.splitter.split_documents(docs)
