from .base import BaseHandler
from datetime import datetime
import os
import shutil
class UploadFileHandler(BaseHandler):

    

    def handle(self, data):
        project = data.get("project")
        experiment_name = data.get("experiment")
        file_name = data.get("file_name")

        # Check required fields
        
        if not project or not file_name:
            return {
                "status": "error",
                "message": "Missing project or file name."
            }

        # Fallback to link to the latest experiment if no name provided
        if not experiment_name:
            latest_exp = self.store.get_latest_experiment(project)
            if latest_exp:
                experiment_name = latest_exp.get("name")
            else:
                return {
                    "status": "error",
                    "message": f"No experiment name provided and no existing experiments in project '{project}'. Please log the experiment first."
                }

        # Link the file to the experiment
        return self.store.link_file_to_experiment(project, experiment_name, file_name)

