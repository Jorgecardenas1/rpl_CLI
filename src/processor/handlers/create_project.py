from .base import BaseHandler

class CreateProjectHandler(BaseHandler):
    def handle(self, data):
        # name = data.get("project_name")
        # description = data.get("description", "")
        # self.store.create_project(name, description)

        project_name = data.get("name")
        description = data.get("description", "")

        if not project_name:
            return {"status": "error", "message": "Project name is required."}

        result = self.store.create_project(project_name, description)

        if result.get("status") == "success":
            self.last_active_project = project_name
        
        return result
        
        #return {"status": "success", "message": f"Project '{name}' created."}
