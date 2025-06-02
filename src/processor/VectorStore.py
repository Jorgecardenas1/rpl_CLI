from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def create_index(self, docs):
        return FAISS.from_documents(docs, self.embeddings)

    def save(self, vectorstore, path="faiss_lab_index"):
        vectorstore.save_local(path)

    def load(self, path="faiss_lab_index", allow_dangerous_deserialization=False):
        return FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )