from langchain_community.vectorstores import FAISS
import faiss
import numpy as np

class VectorStoreManager:
    def __init__(self, embedding_model, normalize=False, index_type="FlatL2"):
        self.embedding_model = embedding_model
        self.normalize = normalize
        self.index_type = index_type

    def normalize_vectors(self, vectors):
        return [v / np.linalg.norm(v) for v in vectors]

    def create_index(self, documents):
        if not documents:
            raise ValueError("❌ Cannot index an empty document list.")

        texts = [doc.page_content for doc in documents]
        vectors = self.embedding_model.embed_documents(texts)

        if not vectors:
            raise ValueError("❌ Embedding failed — got empty vector list.")

        if self.normalize:
            vectors = self.normalize_vectors(vectors)

        dim = len(vectors[0])
        np_vectors = np.array(vectors).astype("float32")

        # 🔧 Choose index type
        if self.index_type == "FlatIP":
            index = faiss.IndexFlatIP(dim)
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 32)
        elif self.index_type == "FlatL2":
            index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")

        index.add(np_vectors)

        # Wrap it in LangChain FAISS
        return FAISS(embedding_function=self.embedding_model, index=index, documents=documents)

    def save(self, vectorstore, path):
        vectorstore.save_local(path)

    def load(self, path, allow_dangerous_deserialization=True):
        return FAISS.load_local(path, self.embedding_model, allow_dangerous_deserialization=allow_dangerous_deserialization)

    def as_retriever(self, vectorstore, k=5):
        return vectorstore.as_retriever(search_kwargs={"k": k})
