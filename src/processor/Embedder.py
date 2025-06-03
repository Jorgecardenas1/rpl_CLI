from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

class Embedder:
    def __init__(self, openai_api_key):
        self.model = OpenAIEmbeddings(model="text-embedding-3-small")  # uses your OPENAI_API_KEY

    def embed(self, docs):
        return self.model.embed_documents(docs)
