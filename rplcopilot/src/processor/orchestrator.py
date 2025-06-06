# orchestrator.py
from .handlers.create_project import CreateProjectHandler
from .knowledge.store import KnowledgeStore
from .handlers.log_experiment import LogExperimentHandler
from .handlers.upload_file import UploadFileHandler
from pathlib import Path


class Orchestrator:
    def __init__(self, store):
        self.store = store
        self.last_active_project = None

        # ✅ Register available intent handlers
        self.handlers = {
            "create_project": CreateProjectHandler(self.store),
            "log_experiment": LogExperimentHandler(self.store),
            "upload_file": UploadFileHandler(self.store),
            # Add others here
        }


    def process_intent(self, intent_data):
        intent = intent_data.get("intent")
        data = intent_data.get("data", {})

        # Normalize fields
        if "project_name" in data and "project" not in data:
            data["project"] = data["project_name"]

        # ✅ Auto-update last active project
        if data.get("project"):
            self.last_active_project = data["project"]
        else:
            data["project"] = self.last_active_project

        # ✅ Route to the proper handler
        handler = self.handlers.get(intent)
        if handler:
            return handler.handle(data)
        else:
            return {"status": "error", "message": f"Unknown or unsupported intent: '{intent}'"}
