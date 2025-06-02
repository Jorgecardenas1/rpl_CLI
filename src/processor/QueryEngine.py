from langchain_openai import ChatOpenAI  # ✅ modern, stable
from langchain.chains import RetrievalQA
import os

groq_base_url = "https://api.groq.com/openai/v1"

class QueryEngine:
    def __init__(self, vectorstore):
        self.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=os.getenv("GROQ_API_KEY"),
                       openai_api_base=groq_base_url,
                       model_name="llama3-70b-8192"),
            retriever=vectorstore.as_retriever()
        )



    def ask(self, query: str):
        return self.qa.invoke({"query": query})  # ✅ newer `invoke()` replaces `run()`
