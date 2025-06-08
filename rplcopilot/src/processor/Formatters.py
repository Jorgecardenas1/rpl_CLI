from typing import List
import textwrap

class DigestFormatter:
    def to_format(self, data: List[dict], format: str = "md",mode="trace") -> str:
        """
        Formats the digest data into Markdown, LaTeX, or plain text.
        """
        if format == "md":
            return self._to_markdown(data)
        elif format == "tex":
            return self._to_latex(data)
        elif format == "txt":
            return self._to_text(data)
        else:
            raise ValueError("Unsupported format: choose md | tex | txt")

    def _to_markdown(self, data, mode="digest"):
        lines = []

        if mode == "trace":
            lines.append("# 🧠 Ripple Trace Report\n")
            for i, entry in enumerate(data, 1):
                lines.append(f"## [{i}] 📄 {entry.get('file', 'unknown')}")
                lines.append(f"🕒 Uploaded: `{entry.get('uploaded', 'unknown')}`")
                lines.append(f"**Excerpt:**\n> {entry.get('excerpt', '[No content]')}\n")
                lines.append("---")
        else:
            lines.append("# 📊 Ripple Digest\n")
            for entry in data:
                lines.append(f"## 📄 {entry.get('file', 'unknown')}")
                lines.append(f"🕒 Uploaded: `{entry.get('uploaded', 'unknown')}`")
                lines.append(f"**Summary:**\n{entry.get('summary', '[No summary]')}")
                lines.append(f"**Keywords:** `{', '.join(entry.get('keywords', []))}`")
                lines.append("---")

        return "\n".join(lines)



    def _to_text(self, data):
        lines = ["RIPPLE DIGEST REPORT\n"]
        for entry in data:
            lines.append(f"File: {entry['file']}")
            lines.append(f"Uploaded: {entry['uploaded']}")
            lines.append("Summary:\n" + textwrap.fill(entry['summary'], width=80))
            lines.append("Keywords: " + ", ".join(entry['keywords']))
            lines.append("=" * 40)
        return "\n".join(lines)

    def _to_latex(self, data):
        lines = [r"\section*{Ripple Copilot Digest}"]
        for entry in data:
            lines.append(rf"\subsection*{{{entry['file']}}}")
            lines.append(rf"\textbf{{Uploaded:}} {entry['uploaded']}\\\\")
            lines.append(r"\textbf{Summary:}\\")
            lines.append(textwrap.fill(entry['summary'], width=100).replace("\n", " ") + r"\\")
            lines.append(r"\textbf{Keywords:} " + ", ".join(entry['keywords']) + r"\\")
            lines.append(r"\hrule\vspace{1em}")
        return "\n".join(lines)
