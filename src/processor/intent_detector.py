import requests
import os
import json
import re
from pathlib import Path


from dotenv import load_dotenv


dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Or paste directly (not recommended)
MODEL = "llama3-8b-8192"

class IntentDetector:
    def detect(self, user_input):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": """
    You are an assistant that detects user intent and extracts structured data.
    Intents: create_project, log_experiment, upload_file, other.
    Return JSON like: {"intent": "...", "data": {...}}.
                 The "data" field should contain:
                - For create_project: {"name": "...", "description": "..."}
                - For log_experiment: {"project": "...", "experiment_name", "description": "...", "results": "...", "version": "..."}
                - For upload_file: {"project": "...", "file_name": "...", "experiment_name": "..."}
                - For other: {"message": "..."} if no intent is detected.
    """     },
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.2
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        # ✅ Step 1: Get parsed JSON from requests
        result = response.json()

        # ✅ Step 2: Check if the response is valid

        content_str = result["choices"][0]["message"]["content"]

        # Clean markdown artifacts like **...** or ```json ... ```
        content_str = content_str.strip()

        # Remove bold markdown if present
        if content_str.startswith("**") and content_str.endswith("**"):
            content_str = content_str[2:-2].strip()

        # Optional: clean triple backticks if needed
        content_str = self.extract_json_from_response(content_str)
        

        if not content_str:
            return {
                "intent": "error",
                "data": {"error": "No content returned from model."}
            }

      
        print("[IntentDetector] Raw model response content:", content_str)

        try:
            parsed = json.loads(content_str)
        except json.JSONDecodeError as e:
            print("[IntentDetector] Model content not valid JSON:", content_str)
            print("[IntentDetector] Error:", e)
            return {"intent": "error", "data": {"error": "Model response not valid JSON."}}

        return parsed

    def clean_json_response(self,content_str):
        """
        Cleans model response string of markdown-style formatting like ```json ... ```
        or bold wrappers like **{...}**.
        """
        # Remove leading/trailing whitespace
        content_str = content_str.strip()

        # Remove triple backticks and optional "json"
        content_str = re.sub(r"^```(?:json)?\s*", "", content_str)
        content_str = re.sub(r"\s*```$", "", content_str)

        # Remove bold markdown wrapping like **{...}**
        if content_str.startswith("**") and content_str.endswith("**"):
            content_str = content_str[2:-2].strip()

        return content_str
    
    
    def extract_json_from_response(self,content_str):
        """
        Extracts the first JSON object from a possibly messy model response.
        Handles cases with leading text, backticks, etc.
        """
        content_str = content_str.strip()

        # Try to extract JSON from a ``` block first
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_str, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to extract first JSON-looking object manually if not in backticks
        match = re.search(r"(\{.*\})", content_str, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Nothing found
        return None