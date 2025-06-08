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
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from datetime import datetime, timedelta

# Typer CLI instance
app = typer.Typer()

# Add src/ to import path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Local modules
from processor import DocumentLoader, Chunker, Embedder, VectorStore, QueryEngine
from processor import Semantic,Export
from processor import Formatters


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
def log(
    title: str = typer.Option(..., help="Title of the experiment"),
    notes: str = typer.Option("", help="Detailed notes or description"),
    tags: str = typer.Option("", help="Comma-separated tags")
):
    """Log an experiment entry under the current project with semantic enrichment."""
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    logs_dir = os.path.join(path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    typer.secho(f"\n📝 Logging new experiment in project: {project}", fg="cyan", bold=True)

    # Load FAISS if available
    vectorstore = None
    try:
        vectorstore = store_mgr.load(os.path.join(path, "faiss_index"), allow_dangerous_deserialization=True)
        typer.secho("🔁 Vector index loaded successfully.", fg="green")
    except:
        typer.secho("⚠️ No vector index found. Proceeding without similarity linking.", fg="yellow")

    # Init LLM
    llm = ChatOpenAI(
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        model_name="llama3-70b-8192"
    )
    semantic_engine = Semantic.SemanticEngine(vectorstore)
    semantic_engine.llm = llm

    summary, enriched_tags, related = "", [], []

    try:
        if notes.strip():
            typer.secho("🧠 Enriching log with semantic info...", fg="blue")
            summary, enriched_tags = semantic_engine.enrich_metadata(notes)
            related = semantic_engine.find_related_experiments([
                Document(page_content=notes, metadata={"source": "log entry"})
            ])
            typer.secho("✅ Semantic enrichment complete!", fg="green")
    except Exception as e:
        typer.secho(f"⚠️ Semantic enrichment failed: {e}", fg="red")

    # Combine tags
    if tags:
        user_tags = [t.strip() for t in tags.split(",")]
        combined_tags = list(set(user_tags + enriched_tags))
    else:
        combined_tags = enriched_tags
    # Save log
    entry = {
        "title": title,
        "notes": notes,
        "summary": summary,
        "tags": combined_tags,
        "related": [r.metadata.get("source", "") for r in related],
        "timestamp": datetime.utcnow().isoformat()
    }

    log_file = os.path.join(logs_dir, f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_file, "w") as f:
        json.dump(entry, f, indent=2)

    # Append to logbook
    logbook_path = os.path.join(logs_dir, "logbook.json")
    if os.path.exists(logbook_path):
        with open(logbook_path, "r") as f:
            all_logs = json.load(f)
    else:
        all_logs = []
    all_logs.append(entry)
    with open(logbook_path, "w") as f:
        json.dump(all_logs, f, indent=2)

    # Report
    typer.secho(f"\n✅ Experiment '{title}' logged.", fg="green", bold=True)
    if summary:
        typer.secho(f"\n🧠 Summary:\n{summary}", fg="cyan")
    if combined_tags:
        typer.secho(f"\n🏷️ Tags: {', '.join(combined_tags)}", fg="magenta")
    if related:
        typer.secho(f"\n🔗 Related Files:", fg="yellow")
        for r in related:
            typer.secho(f"   - {r.metadata.get('source', 'unknown')}", fg="white")

# === Command: List Projects ===
@app.command()
def ls():
    """List all initialized projects."""
    projects = os.listdir(PROJECTS_DIR)
    if not projects:
        print("❌ No projects found.")
    for project in projects:
        print(f"📁 {project}")

@app.command()
def switch(project: str):
    """Switch the active project context."""
    ProjectContext.set(project)
    print(f"🔄 Switched context to project: {project}")


@app.command()
def current():
    """Switch the active project context."""
   
    print(f"🔄 Current context to project: { ProjectContext.current()}")
# === Command: Upload & Embed Documents ===

@app.command()
def upload(folder_path: str):
    """Upload and embed all documents from a folder into the current project's knowledge base."""
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    uploads_dir = os.path.join(path, "uploads")
    meta_path = os.path.join(path, "metadata.json")

    os.makedirs(uploads_dir, exist_ok=True)

    files = [Path(f).name for f in glob.glob(os.path.join(folder_path, "*"))]
    if not files:
        typer.secho("❌ No files found in the provided folder.", fg="red")
        raise typer.Exit()

    # Load or create FAISS index
    try:
        vectorstore = store_mgr.load(os.path.join(path, "faiss_index"), allow_dangerous_deserialization=True)
        typer.secho("🔁 Loaded existing vectorstore.", fg="green")
    except:
        vectorstore = None
        typer.secho("🧠 No vectorstore found. A new one will be created.", fg="yellow")

    # Init semantic engine
    llm = ChatOpenAI(
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        model_name="llama3-70b-8192"
    )
    semantic_engine = Semantic.SemanticEngine(vectorstore)
    semantic_engine.llm = llm

    for file in files:
        full_path = os.path.join(folder_path, file)
        if not os.path.exists(full_path):
            typer.secho(f"⚠️ Skipping missing file: {file}", fg="yellow")
            continue

        typer.secho(f"\n📥 Uploading `{file}` to `{project}`...", fg="cyan", bold=True)

        try:
            docs = doc_loader.load(full_path)
            chunks = chunker.chunk(docs)
            for chunk in chunks:
                chunk.metadata["source"] = file

            if vectorstore:
                vectorstore.add_documents(chunks)
            else:
                vectorstore = store_mgr.create_index(chunks)

            shutil.copy(full_path, os.path.join(uploads_dir, file))

            # 🧠 Semantic Enrichment
            summary, keywords, doc_type, related = "", [], "unknown", []
            try:
                text = " ".join([doc.page_content for doc in chunks[:3]])
                doc_type = semantic_engine.detect_type(text)
                summary, keywords = semantic_engine.enrich_metadata(text)
                related = semantic_engine.find_related_experiments(chunks)
            except Exception as e:
                typer.secho(f"⚠️ Enrichment failed: {e}", fg="red")

            # 📝 Update metadata
            with open(meta_path, "r") as f:
                metadata = json.load(f)

            metadata["files"].append({
                "file_name": file,
                "uploaded_at": datetime.utcnow().isoformat(),
                "summary": summary,
                "keywords": keywords,
                "type": doc_type,
                "related": [r.metadata.get("source", "unknown") for r in related]
            })

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            typer.secho("✅ File indexed and metadata enriched.", fg="green")

        except Exception as e:
            typer.secho(f"❌ Failed to process `{file}`: {e}", fg="red")

    # Save vectorstore
    if vectorstore:
        store_mgr.save(vectorstore, os.path.join(path, "faiss_index"))
        typer.secho("\n💾 Vectorstore saved.\n", fg="green", bold=True)
    else:
        typer.secho("⚠️ No vectorstore created.", fg="red")


# === Command: Query FAISS Only ===
@app.command()
def query(
    question: str = typer.Argument(..., help="Your question about the project"),
    top_k: int = typer.Option(5, help="How many chunks to retrieve from the index")
):
    """Ask a natural language question against the current project's knowledge base."""
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)

    typer.secho(f"\n🔎 Current project: {project}", fg="cyan", bold=True)
    typer.secho(f"🧠 Question: {question}\n", fg="magenta")

    # Load the FAISS index
    try:
        vectorstore = store_mgr.load(os.path.join(path, "faiss_index"), allow_dangerous_deserialization=True)
        typer.secho("✅ Vector index loaded successfully.\n", fg="green")
    except Exception:
        typer.secho("❌ No FAISS index found. Use `rpl upload` to add documents first.", fg="red", bold=True)
        raise typer.Exit()

    # 🔌 Use the QueryEngine handler
    try:
        engine = QueryEngine.QueryEngine(vectorstore, top_k=top_k)
        response = engine.ask(question)

        answer = response.strip() if isinstance(response, str) else response.get("result", "").strip()
        typer.secho("\n🤖 Answer:\n", fg="green", bold=True)
        typer.echo(answer)

    except Exception as e:
        typer.secho(f"\n❌ Query failed: {e}", fg="red", bold=True)



# === Command: Hybrid FAISS + BM25 Retrieval ===
@app.command()
def hybrid(query: str, k: int = 5, export: str = typer.Option(None, help="Export format: json, bib, tex")):
    """Run a hybrid search (BM25 + vector) across uploaded documents."""
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    uploads_dir = os.path.join(path, "uploads")
    meta_path = os.path.join(path, "metadata.json")

    typer.secho(f"\n🔎 Hybrid search in project: {project}", fg="cyan", bold=True)
    typer.secho(f"🔍 Query: {query}", fg="magenta")

    # Load and chunk all uploaded docs
    all_docs = []
    with open(meta_path, "r") as f:
        meta = json.load(f)

    for entry in meta.get("files", []):
        file_path = os.path.join(uploads_dir, entry["file_name"])
        if not os.path.exists(file_path):
            typer.secho(f"⚠️  Skipping missing file: {file_path}", fg="yellow")
            continue
        docs = doc_loader.load(file_path)
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            chunk.metadata["source"] = entry["file_name"]
        all_docs.extend(chunks)

    if not all_docs:
        typer.secho("❌ No documents found to run hybrid search.", fg="red")
        raise typer.Exit()

    # Build BM25 + FAISS retrievers
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    try:
        vectorstore = store_mgr.load(os.path.join(path, "faiss_index"), allow_dangerous_deserialization=True)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception:
        typer.secho("❌ Could not load FAISS index. Please run `rpl upload` first.", fg="red")
        raise typer.Exit()

    hybrid = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    results = hybrid.get_relevant_documents(query)
    # Format results into exportable dictionaries
    results_data = [{
        "source": doc.metadata.get("source", "unknown"),
        "content": doc.page_content.strip()
    } for doc in results]

    typer.secho(f"\n📄 Top {k} hybrid-matched chunks:\n", fg="green", bold=True)

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:300]
        typer.secho(f"{i}. 📁 Source: {source}", fg="cyan")
        typer.echo(f"   📄 {preview}\n")
    
    if export:
        formatter = Export.ExportFormatter(results_data)
        if export == "json":
            path = formatter.save_json()
        elif export == "bib":
            path = formatter.save_bib()
        elif export == "tex":
            path = formatter.save_tex()
        else:
            typer.secho("❌ Invalid export format. Use: json, bib, tex", fg="red")
            raise typer.Exit()
        typer.secho(f"💾 Exported to {path}", fg="green")

    typer.secho("✅ Hybrid search complete.\n", fg="green", bold=True)


@app.command()
def digest(
    last: str = typer.Option("7d", help="Time window to include (e.g. 3d, 1w, 1m)"),
    format: str = typer.Option("md", help="Export format: md | tex | pdf"),
):
    """Generate a digest summary of recent uploads."""
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    meta_path = os.path.join(path, "metadata.json")
    uploads_dir = os.path.join(path, "uploads")

    # --- Parse time window
    unit = last[-1]
    count = int(last[:-1])
    now = datetime.utcnow()
    delta = {"d": timedelta(days=1), "w": timedelta(weeks=1), "m": timedelta(days=30)}.get(unit, timedelta(days=7))
    cutoff = now - (count * delta)

    # --- Load metadata
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    recent_files = [
        f for f in metadata["files"]
        if datetime.fromisoformat(f["uploaded_at"]) > cutoff
    ]

    if not recent_files:
        print("📭 No recent files found in the last", last)
        raise typer.Exit()

    # --- Semantic engine
    vectorstore = store_mgr.load(os.path.join(path, "faiss_index"), allow_dangerous_deserialization=True)
    semantic_engine = Semantic.SemanticEngine(vectorstore)

    # --- Collect digests
    digest_data = []

    for f in recent_files:
        full_path = os.path.join(uploads_dir, f["file_name"])
        docs = doc_loader.load(full_path)
        text = "\n".join([doc.page_content for doc in docs])
        summary, keywords = semantic_engine.enrich_metadata(text)

        digest_data.append({
            "file": f["file_name"],
            "uploaded": f["uploaded_at"],
            "summary": summary,
            "keywords": keywords
        })

    # --- Format output
    formatter = Formatters.DigestFormatter()
    report = formatter.to_format(digest_data, format=format)

    outfile = os.path.join(path, f"digest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{format}")
    with open(outfile, "w") as f:
        f.write(report)

    print(f"📄 Digest saved to: {outfile}")



@app.command()
def trace(
    concept: str = typer.Argument(..., help="Concept or keyword to trace"),
    k: int = 10,
    format: str = typer.Option("md", help="Export format: md | json | bib"),
):
    """
    Trace a concept across your project: finds where it's mentioned and in what context.
    """
    project = ProjectContext.current()
    path = os.path.join(PROJECTS_DIR, project)
    uploads_dir = os.path.join(path, "uploads")
    meta_path = os.path.join(path, "metadata.json")

    # --- Load metadata and files
    if not os.path.exists(meta_path):
        print("❌ No metadata found for this project.")
        raise typer.Exit()

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    all_chunks = []
    trace_results = []

    for entry in metadata.get("files", []):
        full_path = os.path.join(uploads_dir, entry["file_name"])
        if not os.path.exists(full_path):
            print(f"⚠️ Skipping missing file: {full_path}")
            continue

        docs = doc_loader.load(full_path)
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            chunk.metadata.update({
                "source": entry["file_name"],
                "uploaded_at": entry["uploaded_at"]
            })
        all_chunks.extend(chunks)

    # --- Retrieve matching chunks with BM25
    bm25 = BM25Retriever.from_documents(all_chunks)
    bm25.k = k
    matched_docs = bm25.get_relevant_documents(concept)

    if not matched_docs:
        print("📭 No matches found for:", concept)
        raise typer.Exit()

    # --- Prepare results
    for doc in matched_docs:
        trace_results.append({
            "file": doc.metadata.get("source", "unknown"),
            "uploaded": doc.metadata.get("uploaded_at", "unknown"),
            "excerpt": doc.page_content.strip()[:300]
        })

    # --- Export or display
    formatter = Formatters.DigestFormatter()
    output = formatter.to_format(trace_results, format=format, mode="trace")

    outfile = os.path.join(path, f"trace_{concept.replace(' ', '_')}.{format}")
    with open(outfile, "w") as f:
        f.write(output)

    print(f"🔍 Found {len(trace_results)} matches for: '{concept}'")
    print(f"📄 Saved trace to: {outfile}")


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

