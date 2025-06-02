from .base import BaseHandler
from datetime import datetime

class LogExperimentHandler(BaseHandler):
    def handle(self, data):

        project = data.get("project") or self.last_active_project
        if not project:
            return {"status": "error", "message": "No project specified or active."}



        description = data.get("description", "")   
        version = data.get("version", "v1")
        result = data.get("results", "")
        name = data.get("experiment_name", "")
        timestamp = data.get("timestamp")
        results_file=data.get("file_name", None),

        return self.store.log_experiment(
            project=project,
            description=description,
            results=result,
            name=name,
            results_file=results_file,
            version=version,
            timestamp=timestamp
        )

