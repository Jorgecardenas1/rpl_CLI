from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import os


class Embedder:
    def __init__(self, openai_api_key):
        #self.model = OpenAIEmbeddings(model="text-embedding-3-large")  # uses your OPENAI_API_KEY
        self.model = OpenAIEmbeddings(
            model="text-embedding-3-large" # or hardcoded (not recommended)
        )
    def embed(self, docs):
        return self.model.embed_documents(docs)
