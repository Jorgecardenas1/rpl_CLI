from datetime import datetime
import os
import json

class KnowledgeStore:
    def __init__(self, filepath="ripple_store.json"):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            self._save({"projects": {}, "experiments": []})
        self.data = self._load()

        # Ensure keys exist
        self.data.setdefault("projects", {})
        self.data.setdefault("experiments", [])
        self._save(self.data)

    def _load(self):
        with open(self.filepath, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def create_project(self, name, description):
        if name in self.data["projects"]:
            return {"status": "exists", "message": "Project already exists."}

        self.data["projects"][name] = {
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "experiments": []
        }
        self._save(self.data)
        return {"status": "success", "message": f"Project '{name}' created."}
    
    def project_exists(self, name):
        """Check if a project already exists by name."""
        return name in self.data.get("projects", {})



    def log_experiment(self, project, description="",results="",name="", results_file=None, version="v1", timestamp=None):
        if project not in self.data["projects"]:
            return {"status": "error", "message": f"Project '{project}' not found."}

        entry = {
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "project": project,
            "name": name,
            "description": description,
            "version": version,
            "results": results
        }

        if results_file:
            entry["results_file"] = results_file

        self.data["projects"][project]["experiments"].append(entry)
        self.data["experiments"].append(entry)
        self._save(self.data)

        return {"status": "success", "message": f"Experiment logged to '{project}'."}


    def link_file_to_project(self, project, file_name):
        if project not in self.data["projects"]:
            return {"status": "error", "message": f"Project '{project}' not found."}

        # Add the file name to the latest experiment or to project metadata
        self.data["projects"][project].setdefault("files", []).append({
            "file_name": file_name,
            "timestamp": datetime.utcnow().isoformat()
        })
        self._save(self.data)

        return {"status": "success", "message": f"File '{file_name}' linked to '{project}'."}

    def link_file_to_experiment(self, project, experiment_name, file_name):
        if project not in self.data["projects"]:
            return {"status": "error", "message": f"Project '{project}' not found."}

        for exp in self.data["projects"][project]["experiments"]:
            if exp.get("name") == experiment_name:
                exp.setdefault("files", []).append({
                    "file_name": file_name,
                    "timestamp": datetime.utcnow().isoformat()
                })
                self._save(self.data)
                return {"status": "success", "message": f"File '{file_name}' linked to experiment '{experiment_name}' in project '{project}'."}

        return {"status": "error", "message": f"Experiment '{experiment_name}' not found in project '{project}'."}

    def get_latest_experiment(self, project):
        if project not in self.data["projects"]:
            return None

        experiments = self.data["projects"][project].get("experiments", [])
        if experiments:
            return experiments[-1]  # latest experiment
        return None

    def delete_project(self, project_name):
        if project_name in self.data["projects"]:
            del self.data["projects"][project_name]

            # Optionally: remove experiments from global list
            self.data["experiments"] = [
                exp for exp in self.data["experiments"]
                if exp.get("project") != project_name
            ]

            self._save(self.data)
            return {"status": "success", "message": f"Project '{project_name}' deleted."}
        else:
            return {"status": "error", "message": f"Project '{project_name}' not found."}
