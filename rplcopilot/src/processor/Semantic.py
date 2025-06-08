from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import os
class SemanticEngine:
    def __init__(self, vectorstore, top_k=3):
        """
        :param vectorstore: FAISS or hybrid retriever object (optional for related search)
        """
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k}) if vectorstore else None
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_base="https://api.groq.com/openai/v1",
            model_name="llama3-70b-8192"
        )


    def detect_type(self, text: str) -> str:
        """
        Classify the document type using LLM.
        """
        prompt = f"""
        You are a lab assistant. Classify this document as one of the following types:
        - research paper
        - technical paper
        - scientific article
        - journal article
        - conference paper
        - experiment log
        - dataset
        - protocol
        - monograph
        - review article
        - thesis
        - technical report
        - patent
        - conference paper
        - book chapter
        - book 
        - case study
        - white paper
        - survey
        - meta-analysis
        - systematic review
        - clinical trial report
        - data analysis report
        - project report
        - grant proposal
        - policy brief
        - presentation
        - poster
        - infographic
        - news article
        - blog post
        - opinion piece
        - editorial
        - commentary
        - interview transcript
        - press release
        - podcast transcript
        - video transcript
        - lecture notes
        - workshop summary
        - seminar notes
        - conference proceedings
        - technical manual
        - user guide
        - troubleshooting guide
        - FAQ document
        - instructional material
        - training material
        - course syllabus
        - curriculum
        - lesson plan
        - educational resource
        - case report
        - clinical guideline
        - treatment protocol
        - diagnostic criteria
        - epidemiological study
        - cohort study
        - case-control study
        - randomized controlled trial
        - observational study
        - meta-analysis
        - systematic review
        - literature review
        - scoping review
        - rapid review
        - narrative review
        - integrative review
        - critical review
        - theoretical paper
        - conceptual paper
        - methodological paper
        - empirical paper
        - qualitative study
        - quantitative study
        - mixed methods study
        - data collection report
        - data processing report
        - data visualization report
        - data mining report
        - data analysis report
        - data interpretation report
        - data management report
        - data sharing report
        - data governance report
        - data security report
        - data privacy report
        - data ethics report
        - data quality report
        - data standards report
        - data interoperability report
        - data integration report
        - data architecture report
        - data modeling report
        - data warehousing report
        - data analytics report
        - data science report
        - data engineering report
        - data visualization report
        - data storytelling report
        - data journalism report
        - data literacy report
        - data education report
        - data training report
        - data certification report
        - data competency report
        - data skills report    
        - data career report
        - data job market report
        - data industry report
        - data trends report
        - data challenges report
        - data opportunities report
        - data future report
        - data impact report
        - data value report
        - data benefits report
        - data risks report   
        - literature summary
        - literature overview
        - literature synthesis
        - literature analysis
        - literature mapping
        - literature review protocol
        - literature review methodology
        - literature review framework
        - literature review guidelines
        - literature review checklist
        - other

        Document content:
        {text[:1500]}
        Type:"""
        response = self.llm.invoke(prompt)
        return response.content.strip()


    def enrich_metadata(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract a short summary and a list of keywords.
        """
        prompt = f"""
    Summarize this scientific content in 1–2 sentences and extract 5 to 7 keywords.

    Text:
    {text[:1500]}

    Respond with:
    Summary: ...
    Keywords: ...
    """
        response = self.llm.invoke(prompt).content  # ✅ Use `.content`

        summary = ""
        keywords = []

        for line in response.split("\n"):
            if line.lower().startswith("summary:"):
                summary = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("keywords:"):
                keywords = [k.strip() for k in line.split(":", 1)[-1].split(",")]

        return summary, keywords


    def find_related_experiments(self, chunks: List[Document], k: int = 3) -> List[Document]:
        if not self.retriever:
            return []  # Nothing to compare yet

        candidates = []
        for chunk in chunks[:3]:
            related = self.retriever.get_relevant_documents(chunk.page_content)
            candidates.extend(related)
        return candidates
