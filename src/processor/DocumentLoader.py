from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

class DocumentLoader:
    def load(self, file_path: str):
        if file_path.endswith(".pdf"):
            return PyPDFLoader(file_path).load()
        elif file_path.endswith(".txt"):
            return TextLoader(file_path).load()
        elif file_path.endswith(".csv"):
            return CSVLoader(file_path).load()
        else:
            raise ValueError("Unsupported file type.")
