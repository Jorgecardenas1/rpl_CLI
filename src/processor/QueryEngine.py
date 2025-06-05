from langchain_openai import ChatOpenAI  # ✅ modern, stable
from langchain.chains import RetrievalQA
import os

groq_base_url = "https://api.groq.com/openai/v1"


class QueryEngine:
    def __init__(self, vectorstore, top_k=10):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        self.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                openai_api_key=os.getenv("GROQ_API_KEY"),
                openai_api_base="https://api.groq.com/openai/v1",
                model_name="llama3-70b-8192"
            ),
            retriever=self.retriever
        )

    def ask(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        print(f"🔍 Retrieved {len(docs)} documents")
        for doc in docs[:3]:
            print("🧠", doc.page_content[:200])
        return self.qa.invoke({"query": query})
