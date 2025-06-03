import os
import json
from datetime import datetime
from pathlib import Path
from glob import glob

class ProjectVectorManager:
    def __init__(self, project_path, store_mgr, doc_loader, chunker):
        self.project_path = project_path
        self.index_path = os.path.join(project_path, "faiss_index")
        self.meta_path = os.path.join(project_path, "metadata.json")
        self.store_mgr = store_mgr
        self.doc_loader = doc_loader
        self.chunker = chunker
        self.vectorstore = None

    def load_or_init_index(self):
        try:
            self.vectorstore = self.store_mgr.load(self.index_path, allow_dangerous_deserialization=True)
            print("🔁 Existing FAISS index loaded.")
        except Exception:
            self.vectorstore = None
            print("🆕 No existing index found. A new one will be created.")

    def upload_folder(self, folder_path: str):
        files = [Path(file).name for file in glob(f"{folder_path}/*")]
        self.load_or_init_index()

        for file in files:
            full_path = os.path.join(folder_path, file)
            print(f"📥 Processing: {file}")
            self._process_file(full_path)

        self._save_index()
        print("✅ All files processed and saved.")

    def upload_file(self, file_path: str):
        self.load_or_init_index()
        self._process_file(file_path)
        self._save_index()
        print(f"✅ File `{file_path}` processed and saved.")

    def _process_file(self, file_path: str):
        print(f"🔍 Loading file: {file_path}")
        docs = self.doc_loader.load(file_path)
        print(f"📄 Loaded {len(docs)} document(s)")

        if not docs:
            raise ValueError(f"❌ No documents could be loaded from: {file_path}")

        chunks = self.chunker.chunk(docs)
        print(f"✂️ Chunked into {len(chunks)} chunks")

        if not chunks:
            raise ValueError(f"❌ No chunks were generated from: {file_path}")

        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = self.store_mgr.create_index(chunks)

        self._update_metadata(os.path.basename(file_path))


    def _update_metadata(self, file_name):
        with open(self.meta_path, "r") as f:
            metadata = json.load(f)
        metadata["files"].append({
            "file_name": file_name,
            "uploaded_at": datetime.utcnow().isoformat()
        })
        with open(self.meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_index(self):
        self.store_mgr.save(self.vectorstore, self.index_path)
