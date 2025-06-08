from datetime import datetime
import os
import json
from typing import List, Dict

class ExportFormatter:
    def __init__(self, results: List[Dict], output_dir: str = "exports"):
        self.results = results
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_json(self, filename: str = None):
        filename = filename or f"results_{self._timestamp()}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        return path

    def save_bib(self, filename: str = None):
        filename = filename or f"results_{self._timestamp()}.bib"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            for i, entry in enumerate(self.results):
                f.write(self.to_bibtex(entry, i) + "\n\n")
        return path

    def save_tex(self, filename: str = None):
        filename = filename or f"results_{self._timestamp()}.tex"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write("\\section*{Hybrid Search Results}\n\\begin{itemize}\n")
            for entry in self.results:
                title = entry.get("source", "unknown")
                content = entry.get("content", "").replace("\n", " ")[:200]
                f.write(f"  \\item \\textbf{{File:}} {title}\n\n")
                f.write(f"  \\texttt{{{content}}}\n\n")
            f.write("\\end{itemize}\n")
        return path

    def to_bibtex(self, entry: Dict, index: int = 0) -> str:
        title = entry.get("title", entry.get("source", f"doc{index}"))
        author = entry.get("author", "Unknown")
        year = entry.get("year", "2024")
        key = f"{title.replace(' ', '_').lower()}_{year}"
        return f"""@article{{{key},
  title={{ {title} }},
  author={{ {author} }},
  year={{ {year} }},
  note={{ Exported from Ripple Copilot }}
}}"""

    def _timestamp(self):
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
