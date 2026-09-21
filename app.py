"""
manual_rag.py — a fully-local RAG pipeline for scanning manuals (e.g. motorcycle
service manuals) and answering questions against them.

STACK (all local, no API keys, nothing leaves your machine):
  - Parsing:    Docling  (IBM, strong table/layout extraction -> Markdown)
  - Chunking:   Docling HybridChunker (structure-aware; won't split spec tables)
  - Vectors:    Chroma   (embedded, persists to ./chroma_db)
  - Embeddings: Ollama    (default: nomic-embed-text)
  - LLM:        Ollama    (default: llama3.1)

------------------------------------------------------------------------------
SETUP
  1. Install Ollama (https://ollama.com) and pull the models:
        ollama pull nomic-embed-text
        ollama pull llama3.1
     (Ollama runs a local server on http://localhost:11434 automatically.)

  2. Install Python deps (Python 3.10+):
        pip install docling chromadb ollama

     NOTE: the first Docling run downloads its layout/table models (a few
     hundred MB). A GPU speeds it up but is not required.

USAGE
  Ingest a folder of PDFs into the local vector store:
        python manual_rag.py ingest ./manuals

  Ask a question:
        python manual_rag.py ask "What is the front axle torque spec?"

  Interactive Q&A loop:
        python manual_rag.py chat
------------------------------------------------------------------------------
Versions move fast; if an import path changes, check the Docling and Chroma
docs. This is a starting point meant to be customized, not a black box.
"""

import sys
import glob
import os

import chromadb
import ollama
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# ---- Config -----------------------------------------------------------------
EMBED_MODEL = "nomic-embed-text"     # ollama pull nomic-embed-text
LLM_MODEL = "llama3.1"               # ollama pull llama3.1
DB_PATH = "./chroma_db"
COLLECTION = "manuals"
TOP_K = 5                            # how many chunks to retrieve per question
# -----------------------------------------------------------------------------


def get_collection():
    """Open (or create) the persistent local Chroma collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def embed(texts):
    """Embed a list of strings locally via Ollama."""
    vectors = []
    for t in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        vectors.append(resp["embedding"])
    return vectors


def ingest(folder):
    """Parse every PDF in `folder` with Docling, chunk it, embed, and store."""
    pdfs = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        return

    converter = DocumentConverter()
    chunker = HybridChunker()          # structure-aware: respects tables/sections
    collection = get_collection()

    for pdf in pdfs:
        name = os.path.basename(pdf)
        print(f"Parsing {name} ...")
        doc = converter.convert(pdf).document

        docs, metadatas, ids = [], [], []
        for i, chunk in enumerate(chunker.chunk(doc)):
            # contextualize() prepends section headers/captions to the chunk text,
            # which noticeably improves retrieval on manuals.
            text = chunker.contextualize(chunk=chunk)
            page = None
            try:
                page = chunk.meta.doc_items[0].prov[0].page_no
            except Exception:
                pass
            docs.append(text)
            metadatas.append({"source": name, "page": page})
            ids.append(f"{name}-{i}")

        if not docs:
            continue

        print(f"  -> {len(docs)} chunks, embedding + storing ...")
        # Embed and add in batches to keep memory sane on big manuals.
        BATCH = 64
        for start in range(0, len(docs), BATCH):
            sl = slice(start, start + BATCH)
            collection.add(
                documents=docs[sl],
                embeddings=embed(docs[sl]),
                metadatas=metadatas[sl],
                ids=ids[sl],
            )

    print(f"Done. Collection now holds {collection.count()} chunks.")


def retrieve(question, k=TOP_K):
    """Return the top-k most relevant chunks for a question."""
    collection = get_collection()
    q_vec = embed([question])[0]
    res = collection.query(query_embeddings=[q_vec], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return list(zip(docs, metas))


def ask(question, k=TOP_K):
    """Retrieve context and generate a grounded answer with a local LLM."""
    hits = retrieve(question, k)
    if not hits:
        print("Nothing in the index yet — run `ingest` first.")
        return

    context_blocks = []
    for text, meta in hits:
        tag = meta.get("source", "?")
        if meta.get("page") is not None:
            tag += f" p.{meta['page']}"
        context_blocks.append(f"[{tag}]\n{text}")
    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "You are a technical assistant answering questions from equipment "
        "manuals. Use ONLY the context below. If the answer is not in the "
        "context, say you couldn't find it. Cite the [source p.X] tags you used.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    print("\n" + resp["message"]["content"].strip() + "\n")
    print("Sources: " + ", ".join(
        f"{m.get('source','?')}" + (f" p.{m['page']}" if m.get("page") else "")
        for _, m in hits
    ))


def chat():
    """Simple interactive loop."""
    print("Ask questions about your manuals. Ctrl-C or 'quit' to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            ask(q)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "ingest" and len(sys.argv) == 3:
        ingest(sys.argv[2])
    elif cmd == "ask" and len(sys.argv) >= 3:
        ask(" ".join(sys.argv[2:]))
    elif cmd == "chat":
        chat()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()