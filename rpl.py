#!/usr/bin/env python3

import os
import sys
import json
import numpy as np

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
from processor.project_vector import ProjectVectorManager

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
store_mgr = VectorStore.VectorStoreManager(
    embedding_model=embedder.model,
    normalize=True  # Only one supported for now
)


BASE_DIR = ".rpl"
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


# -----------------------------
# 📁 Project Context Management
# -----------------------------


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
def current():
    ProjectContext.show_current()


# -----------------------------
# 🚀 Entrypoint
# -----------------------------
if __name__ == "__main__":
    app()

