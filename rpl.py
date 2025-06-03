#!/usr/bin/env python3

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import typer
import glob
from langchain_community.vectorstores.faiss import FAISS

# Typer CLI app
app = typer.Typer()

# Local modules
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from processor import DocumentLoader, Chunker, Embedder, VectorStore, QueryEngine

# Load API keys
load_dotenv(dotenv_path=Path(".env"))

# Initialize components
groq_api_key = os.getenv("GROQ_API_KEY")  # Or paste directly (not recommended)
openAI = os.getenv("OPENAI_API_KEY")  # Or paste directly (not recommended)

openai_base_url = "https://api.openai.com/v1/chat/completions"


# Setup
doc_loader = DocumentLoader.DocumentLoader()
chunker = Chunker.TextChunker(chunk_size=500, chunk_overlap=50)
embedder = Embedder.Embedder(openAI)
store_mgr = VectorStore.VectorStoreManager(embedder.model)

BASE_DIR = ".rpl"
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


# -----------------------------
# 📁 Project Context Management
# -----------------------------

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
    tags: str = typer.Option("", help="Comma-separated tags")
):
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
    # Load the existing FAISS index (if any)
    # Add new chunks
    # Save the updated vectorstore back


    files = [Path(file).name for file in glob.glob(file_path+"/*")]
    
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    index_path = os.path.join(path, "faiss_index")

    try:
        vectorstore = store_mgr.load(index_path, allow_dangerous_deserialization=True)
    except Exception:
        vectorstore = None

    for file in files:
        print(file)
        typer.echo(f"📥 Uploading `{file}` into `{project}`...")

        docs = doc_loader.load(file_path+"/"+file)
        chunks = chunker.chunk(docs)

        if vectorstore:
            vectorstore.add_documents(chunks)  # Accumulate
        else:
            vectorstore = store_mgr.create_index(chunks)
        
        store_mgr.save(vectorstore, os.path.join(path, "faiss_index"))

        meta_path = os.path.join(path, "metadata.json")
        
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        metadata["files"].append({
            "file_name": os.path.basename(file),
            "uploaded_at": datetime.utcnow().isoformat()
    })
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        typer.echo("✅ File embedded and linked.")





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
def current():
    ProjectContext.show_current()


# -----------------------------
# 🚀 Entrypoint
# -----------------------------
if __name__ == "__main__":
    app()

