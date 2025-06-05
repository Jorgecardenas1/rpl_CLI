#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import typer
import glob
from langchain_community.vectorstores.faiss import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import shutil




# Typer CLI app
app = typer.Typer()

# Local modules
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from processor import DocumentLoader, Chunker, Embedder, VectorStore, QueryEngine
from processor.project_vector import ProjectVectorManager

# Load API keys
load_dotenv()

# Initialize components
groq_api_key = os.getenv("GROQ_API_KEY")  # Or paste directly (not recommended)
openAI = os.getenv("OPENAI_API_KEY")  # Or paste directly (not recommended)
openai_base_url = "https://api.openai.com/v1/chat/completions"


# Setup
doc_loader = DocumentLoader.DocumentLoader()
chunker = Chunker.TextChunker(chunk_size=1000, chunk_overlap=200)
embedder = Embedder.Embedder(openAI)

store_mgr = VectorStore.VectorStoreManager(
    embedding_model=embedder.model,
    normalize=True,         # cosine similarity
    index_type="FlatIP"     # default for semantic search
)


BASE_DIR = ".rpl"
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
print(CONFIG_PATH)

# -----------------------------
# 📁 Project Context Management
# -----------------------------
def copy_file_to_uploads(src_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(src_path))
    shutil.copy(src_path, dest_path)
    return dest_path

def clean_metadata(metadata_path, uploads_dir):
    """Remove metadata entries pointing to missing files."""
    if not os.path.exists(metadata_path):
        return

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    original_count = len(meta.get("files", []))
    cleaned_files = [f for f in meta["files"] if os.path.exists(os.path.join(uploads_dir, f["file_name"]))]
    cleaned_count = len(cleaned_files)

    if cleaned_count < original_count:
        print(f"🧹 Cleaned {original_count - cleaned_count} broken metadata entries")

    meta["files"] = cleaned_files
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
        
def load_or_cache_chunks(project_path: str, uploads_dir: str, meta_path: str):
    cache_path = os.path.join(project_path, "doc_chunks.pkl")

    # 🔁 Load from cache if it exists
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print("🔁 Loading cached document chunks...")
            return pickle.load(f)

    # 📥 Otherwise, load and chunk
    doc_loader = DocumentLoader.DocumentLoader()
    chunker = Chunker.TextChunker(chunk_size=500, chunk_overlap=50)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    all_docs = []

    for entry in metadata.get("files", []):
        file_path = os.path.join(uploads_dir, entry["file_name"])

        if not os.path.exists(file_path):
            print(f"⚠️ Skipping missing file: {file_path}")
            continue

        docs = doc_loader.load(file_path)
        if not docs:
            print(f"⚠️ No documents loaded from: {file_path}")
            continue

        chunks = chunker.chunk(docs)
        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(file_path)

        all_docs.extend(chunks)

    if not all_docs:
        raise ValueError("❌ No valid documents found.")

    # 💾 Cache it
    with open(cache_path, "wb") as f:
        pickle.dump(all_docs, f)

    print(f"✅ Chunked and cached {len(all_docs)} chunks.")
    return all_docs


def normalize_embeddings(vectors):
    return [v / np.linalg.norm(v) for v in vectors]


class ProjectContext:
    @staticmethod
    def ensure_initialized():
        if not os.path.exists(CONFIG_PATH):
            typer.echo("❌ No RPL project initialized. Run `rpl init <project>`.")
            raise typer.Exit(1)

    @staticmethod
    def current():
        ProjectContext.ensure_initialized()
        with open(CONFIG_PATH) as f:
            return json.load(f)["current_project"]

    @staticmethod
    def set_current(project):
        os.makedirs(BASE_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump({"current_project": project}, f)

    @staticmethod
    def show_current():
        ProjectContext.ensure_initialized()
        project = ProjectContext.current()
        typer.echo(f"🔍 Current project: {project}")


# -----------------------------
# 🧪 Commands
# -----------------------------

@app.command()
def init(project: str, description: str = typer.Option("", help="Project description")):
    path = os.path.join(PROJECTS_DIR, project)
    os.makedirs(path, exist_ok=True)
    metadata = {
        "project": project,
        "description": description,
        "created_at": datetime.utcnow().isoformat(),
        "logs": [],
        "files": []
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    ProjectContext.set_current(project)
    typer.echo(f"✅ Initialized RPL project \"{project}\" in .rpl/")


@app.command()
def log(
    title: str = typer.Option(..., help="Experiment title"),
    notes: str = typer.Option(..., help="Experiment notes"),
    tags: str = typer.Option("", help="Comma-separated tags")):

    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "title": title,
        "notes": notes,
        "tags": tags.split(",") if tags else []
    }

    logs_path = os.path.join(path, "logs")
    os.makedirs(logs_path, exist_ok=True)
    log_file = os.path.join(logs_path, f"{datetime.utcnow().isoformat()}.json")
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2)

    meta_path = os.path.join(path, "metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    metadata["logs"].append(log_entry)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    typer.echo(f"📝 Logged experiment under project \"{project}\".")


@app.command()
def upload(file_path: str):
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)

    manager = ProjectVectorManager(
        project_path=path,
        store_mgr=store_mgr,
        doc_loader=doc_loader,
        chunker=chunker
    )
    manager.upload_folder(file_path)

@app.command()
def query(question: str):
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project, "faiss_index")
    typer.echo(f"🔍 Querying `{project}`...")

    vs = store_mgr.load(path, allow_dangerous_deserialization=True)
    qe = QueryEngine.QueryEngine(vs)
    answer = qe.ask(question)
    typer.echo("🤖 Answer: " + answer["result"])

@app.command()
def list():
    """
    List all available RPL projects in local folders.
    """
    project_paths = []
    for root, dirs, files in os.walk("."):
        if ".rpl" in dirs:
            meta_path = os.path.join(root, ".rpl", "config.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    config = json.load(f)
                project_paths.append((root, config.get("current_project")))

    if not project_paths:
        typer.echo("⚠️  No RPL projects found in current folder tree.")
    else:
        typer.echo("📁 Available projects:")
        for path, proj in project_paths:
            typer.echo(f" - {proj} (at {path})")

@app.command()
def switch(project: str):
    """
    Switch the current active project.
    """
    project_path = os.path.join(PROJECTS_DIR, project)
    if not os.path.exists(project_path):
        typer.echo(f"❌ Project '{project}' not found.")
        raise typer.Exit(1)

    ProjectContext.set_current(project)
    typer.echo(f"🔄 Switched to project: {project}")


@app.command()
def push():
    """
    Prepare files for upload (shows summary — no actual sync yet).
    """
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)

    typer.echo(f"📡 Preparing to sync project: {project}")
    summary = {
        "metadata.json": os.path.getsize(os.path.join(path, "metadata.json")),
        "logs": 0,
        "uploads": 0,
        "total_size_bytes": 0
    }

    # Logs
    logs_dir = os.path.join(path, "logs")
    if os.path.exists(logs_dir):
        for f in os.listdir(logs_dir):
            full = os.path.join(logs_dir, f)
            if os.path.isfile(full):
                summary["logs"] += os.path.getsize(full)

    # Uploads
    uploads_dir = os.path.join(path, "uploads")
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            full = os.path.join(uploads_dir, f)
            if os.path.isfile(full):
                summary["uploads"] += os.path.getsize(full)

    summary["total_size_bytes"] = (
        summary["metadata.json"] + summary["logs"] + summary["uploads"]
    )

    typer.echo("📦 Payload Summary:")
    typer.echo(f" - metadata.json: {summary['metadata.json']} B")
    typer.echo(f" - logs: {summary['logs']} B")
    typer.echo(f" - uploads: {summary['uploads']} B")
    typer.echo(f" - total: {summary['total_size_bytes'] / 1024:.2f} KB")

    typer.echo("🚧 Upload not yet implemented. API integration pending.")

@app.command()
def hybrid(query: str, k: int = 5):
    """
    Hybrid semantic + keyword search (FAISS + BM25).
    """
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    vectorstore = store_mgr.load(os.path.join(path, "faiss_index"), allow_dangerous_deserialization=True)

    uploads_dir = os.path.join(path, "uploads")
    meta_path = os.path.join(path, "metadata.json")

    # 🔁 Load or regenerate cached chunks
    all_docs = load_or_cache_chunks(path, uploads_dir, meta_path)

    # ✅ Build hybrid retriever
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    hybrid = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    results = hybrid.get_relevant_documents(query)

    print(f"\n🔍 Hybrid results for query: '{query}'\n")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip().replace("\n", " ")[:300]
        print(f"[{i}] 📄 {source}")
        print(f"    {content}\n")



@app.command()
def current():
    ProjectContext.show_current()

@app.command()
def debug(query: str, k: int = 5):
    """
    Inspect top-k retrieved chunks before answering with LLM.
    """
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)

    print(f"\n🔍 [bold cyan]Project:[/bold cyan] {project}")
    print(f"[bold cyan]Query:[/bold cyan] {query}")

    # Load vectorstore
    vectorstore = store_mgr.load(
        os.path.join(path, "faiss_index"),
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    print(f"\n[bold green]Top {k} Retrieved Chunks:[/bold green]\n")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content[:300].strip().replace("\n", " ")
        print(f"[{i}] [yellow]{source}[/yellow]")
        print(f"    [dim]{content}[/dim]\n")

    print("✅ Use [bold]rpl query[/bold] to synthesize an answer from these.")
# -----------------------------
# 🚀 Entrypoint
# -----------------------------
if __name__ == "__main__":
    app()

