pip install -e .
rpl init
rpl log "Experiment X"


# 🚀 Ripple Copilot CLI (`rpl`)

> AI-powered command-line lab assistant for researchers.  
> Log experiments, upload files, and build a searchable knowledge base using natural language.

---

## 📦 Installation

```bash
git clone https://github.com/your-org/rpl-cli.git
cd rpl-cli
chmod +x rpl.py
ln -s $(pwd)/rpl.py /usr/local/bin/rpl   # optional: system-wide install
```

Now you can use `rpl` globally from the terminal:

```bash
rpl init ...
```

---

## 🧪 Commands

### 🔹 `rpl init <project>`

Initialize a new RPL project in the current directory. Creates a `.rpl/` folder.

```bash
rpl init quantum --description "Quantum film experiments"
```

---

### 🔹 `rpl log`

Log an experiment under the current project.

```bash
rpl log --title "Day 1" --notes "Tested sapphire substrate." --tags "GHz,permittivity"
```

---

### 🔹 `rpl upload <folder>`

Upload and index **all files** in a folder into your knowledge base.

```bash
rpl upload data/experiments
rpl upload "/Users/jorgehernancardenas/Downloads/PaperReview" 

```

### 🔹 `rpl upload <file>`

Upload and embed a **single file**.

```bash
rpl upload results_day1.pdf
```

---

### 🔹 `rpl query "<question>"`

Ask a natural language question against the project’s indexed documents.

```bash
rpl query "What did we learn about sapphire at 10 GHz?"
```

---

### 🔹 `rpl list`

List all available `.rpl` projects found in subdirectories.

```bash
rpl list
```

---

### 🔹 `rpl switch <project>`

Switch the active project context.

```bash
rpl switch metasurfaces
```

---

### 🔹 `rpl current`

Show the currently active project.

```bash
rpl current
```

---

### 🔹 `rpl push`

Simulates syncing with a remote API. Calculates and reports the size of files to be synced.

```bash
rpl push
```

---

## 🗃 Project Structure

Each initialized project will follow this structure:

```
your-lab-project/
└── .rpl/
    ├── config.json               # Points to current project
    └── projects/
        └── <project-name>/
            ├── metadata.json
            ├── logs/
            ├── uploads/
            └── faiss_index/
```

---

## 📬 Coming Soon

- `rpl pull` to fetch a project from the remote backend
- `rpl auth` for login and team access
- Web UI to view, search, and visualize experiments
- Multi-user project collaboration

---

## 👩‍🔬 Built for Lab Researchers

Ripple Copilot is designed for:
- Materials science labs
- Biotech and nanotech researchers
- Anyone running complex experiments with hard-to-track results

---

## 🧠 Credits

Crafted with ❤️ by the Ripple Copilot team  
Feel free to fork, extend, or contribute!
