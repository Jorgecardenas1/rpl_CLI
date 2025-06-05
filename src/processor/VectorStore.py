import uuid
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np

class VectorStoreManager:
    def __init__(self, embedding_model, normalize=True, index_type="FlatIP"):
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
            print("🧪 Vectors normalized for cosine similarity.")

        dim = len(vectors[0])
        np_vectors = np.array(vectors).astype("float32")

        # 🔧 Choose index type
        if self.index_type == "FlatIP":
            index = faiss.IndexFlatIP(dim)
        elif self.index_type == "FlatL2":
            index = faiss.IndexFlatL2(dim)
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")

        # Add vectors
        index.add(np_vectors)

        # 🔑 Create document ID mapping
        ids = [str(uuid.uuid4()) for _ in documents]
        docstore = InMemoryDocstore(dict(zip(ids, documents)))
        index_to_docstore_id = {i: doc_id for i, doc_id in enumerate(ids)}

        # ✅ Final FAISS object
        faiss_store = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        print(f"✅ FAISS index created: {self.index_type}, dim={dim}, docs={len(documents)}")

        return faiss_store

    def save(self, vectorstore, path):
        """Saves the FAISS vectorstore to disk at the given path."""
        vectorstore.save_local(path)

    def load(self, path, allow_dangerous_deserialization=True):
        return FAISS.load_local(
            path,
            self.embedding_model,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
