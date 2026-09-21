# manual-rag

A fully-local RAG (retrieval-augmented generation) pipeline for scanning PDF manuals — e.g. motorcycle or equipment service manuals — and answering questions against them. No API keys, nothing leaves your machine.

## Stack

- **Parsing:** [Docling](https://github.com/DS4SD/docling) (IBM) — strong table/layout extraction to Markdown
- **Chunking:** Docling's `HybridChunker` — structure-aware, won't split spec tables mid-row
- **Vectors:** [Chroma](https://www.trychroma.com/) — embedded, persists to `./chroma_db`
- **Embeddings:** [Ollama](https://ollama.com) (`nomic-embed-text`)
- **LLM:** Ollama (`llama3.1`)

## Why this exists

Service manual PDFs are full of tables (torque specs, part numbers, clearances) that naive text-splitting RAG pipelines mangle. This project uses Docling's layout-aware chunking specifically so spec tables survive intact, and keeps the entire stack local — useful for manuals you don't want touching a third-party API.

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:

   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.1
   ```

   (Ollama runs a local server on `http://localhost:11434` automatically.)

2. Install Python deps (Python 3.10+):

   ```bash
   pip install docling chromadb ollama
   ```

   The first Docling run downloads its layout/table models (a few hundred MB). A GPU speeds this up but isn't required.

## Usage

Ingest a folder of PDFs into the local vector store:

```bash
python app.py ingest ./manuals
```

Ask a question:

```bash
python app.py ask "What is the front axle torque spec?"
```

Interactive Q&A loop:

```bash
python app.py chat
```

## Notes

This is a starting point meant to be customized, not a black box — the pipeline steps (parse → chunk → embed → retrieve → generate) are each small, readable functions in `app.py`. Versions of Docling/Chroma move fast; check their docs if an import path changes.
