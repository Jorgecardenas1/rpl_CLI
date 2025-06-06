import os
import sys
import json
import glob
import shutil
import pickle
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import typer
from langchain_community.vectorstores.faiss import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Typer CLI instance
app = typer.Typer()

# Add src/ to import path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Local modules
from processor import DocumentLoader, Chunker, Embedder, VectorStore, QueryEngine
from processor.project_vector import ProjectVectorManager

# === Constants ===
PROJECTS_DIR = ".rpl/projects"
CONFIG_PATH = ".rpl/config.json"


# === Helper: Load API Keys from Hosted Server ===
def get_keys_from_backend(project_id="demo-lab-1"):
    url = f"https://rpl-render.onrender.com/keys/{project_id}"
    res = requests.get(url)
    if res.status_code != 200:
        raise RuntimeError("❌ Failed to fetch API keys from backend.")
    keys = res.json()
    os.environ["OPENAI_API_KEY"] = keys["openai_key"]
    os.environ["GROQ_API_KEY"] = keys["groq_key"]
    print("🔐 API keys set from backend")


# === Helper: Get Current Project Context ===
class ProjectContext:
    @staticmethod
    def current():
        if not os.path.exists(CONFIG_PATH):
            raise typer.Exit("❌ No project initialized. Run `rpl init <name>`.")
        with open(CONFIG_PATH) as f:
            return json.load(f)["current_project"]

    @staticmethod
    def set(project_name):
        os.makedirs(".rpl", exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump({"current_project": project_name}, f, indent=2)


# === Command: Init a New Project ===
@app.command()
def init(project_name: str):
    path = os.path.join(PROJECTS_DIR, project_name)
    os.makedirs(os.path.join(path, "uploads"), exist_ok=True)
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump({"project": project_name, "files": []}, f, indent=2)
    ProjectContext.set(project_name)
    print(f"✅ Initialized project '{project_name}'.")


# === Command: Log a New Experiment (placeholder logic) ===
@app.command()
def log(message: str):
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    log_path = os.path.join(path, "log.txt")
    with open(log_path, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()}: {message}\n")
    print("📝 Logged:", message)


# === Command: Upload & Embed Documents ===
@app.command()
def upload(folder_path: str):
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    uploads_dir = os.path.join(path, "uploads")
    meta_path = os.path.join(path, "metadata.json")

    files = [Path(f).name for f in glob.glob(folder_path + "/*")]
    vectorstore = None

    # Try to load existing index
    try:
        vectorstore = store_mgr.load(os.path.join(path, "faiss_index"), allow_dangerous_deserialization=True)
        print("🔁 Loaded existing vectorstore.")
    except:
        print("🧠 Creating new vectorstore.")

    for file in files:
        full_path = os.path.join(folder_path, file)
        if not os.path.exists(full_path):
            print(f"⚠️ Skipping missing file: {full_path}")
            continue

        print(f"📥 Uploading `{file}` to `{project}`...")

        # Load and chunk
        docs = doc_loader.load(full_path)
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            chunk.metadata["source"] = file

        # Add to vectorstore
        if vectorstore:
            vectorstore.add_documents(chunks)
        else:
            vectorstore = store_mgr.create_index(chunks)

        # Copy to uploads/
        os.makedirs(uploads_dir, exist_ok=True)
        shutil.copy(full_path, os.path.join(uploads_dir, file))

        # Update metadata
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        metadata["files"].append({
            "file_name": file,
            "uploaded_at": datetime.utcnow().isoformat()
        })
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("✅ File processed and saved.")

    store_mgr.save(vectorstore, os.path.join(path, "faiss_index"))
    print("💾 Index saved.")


# === Command: Query FAISS Only ===
@app.command()
def query(question: str):
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    vectorstore = store_mgr.load(os.path.join(path, "faiss_index"), allow_dangerous_deserialization=True)
    engine = QueryEngine.QueryEngine(vectorstore)
    answer = engine.ask(question)
    print("🤖 Answer:", answer)


# === Command: Hybrid FAISS + BM25 Retrieval ===
@app.command()
def hybrid(query: str, k: int = 5):
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    uploads_dir = os.path.join(path, "uploads")
    meta_path = os.path.join(path, "metadata.json")

    # Reload and chunk all uploaded docs
    all_docs = []
    with open(meta_path, "r") as f:
        meta = json.load(f)

    for entry in meta["files"]:
        file_path = os.path.join(uploads_dir, entry["file_name"])
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping missing file: {file_path}")
            continue
        docs = doc_loader.load(file_path)
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            chunk.metadata["source"] = entry["file_name"]
        all_docs.extend(chunks)

    if not all_docs:
        raise typer.Exit("❌ No documents loaded for hybrid search.")

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    vectorstore = store_mgr.load(os.path.join(path, "faiss_index"), allow_dangerous_deserialization=True)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    hybrid = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    results = hybrid.get_relevant_documents(query)
    print(f"\n🔍 Results for: '{query}'\n")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip().replace("\n", " ")[:300]
        print(f"[{i}] 📄 {source}")
        print(f"    {content}\n")


# === Command: Push (Preview sync — future API upload) ===
@app.command()
def push():
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    meta_path = os.path.join(path, "metadata.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(f"📤 Preparing to push project: {project}")
    for file in meta["files"]:
        file_path = os.path.join(path, "uploads", file["file_name"])
        size = os.path.getsize(file_path) / 1024
        print(f" - {file['file_name']}: {size:.1f} KB")
    print("✅ Push preview complete.")


# === Startup Setup ===
# Inject keys and initialize components
get_keys_from_backend()
doc_loader = DocumentLoader.DocumentLoader()
chunker = Chunker.TextChunker(chunk_size=500, chunk_overlap=50)
embedder = Embedder.Embedder(os.environ["OPENAI_API_KEY"])
store_mgr = VectorStore.VectorStoreManager(embedder.model)
