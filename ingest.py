"""Build the Chroma vector index from the FIA technical regulations PDF.

Run once locally (or at Docker build time):  python ingest.py
"""

import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# 1. Configuration
# -----------------------------

CANDIDATE_PDFS = [
    os.getenv("PDF_PATH", ""),
    "data/raw/fia/technical_regulations.pdf",
    "technical_regulations.pdf",
]
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

PDF_PATH = next((p for p in CANDIDATE_PDFS if p and Path(p).exists()), None)
if PDF_PATH is None:
    raise SystemExit("Could not find technical_regulations.pdf (set PDF_PATH)")

# -----------------------------
# 2. Load PDF
# -----------------------------

print(f"Loading PDF from {PDF_PATH}...")
documents = PyPDFLoader(PDF_PATH).load()
print(f"Loaded {len(documents)} pages")

# -----------------------------
# 3. Split documents
# -----------------------------

print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# -----------------------------
# 4. Create embedding model
# -----------------------------

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# -----------------------------
# 5. Store in Chroma (fresh, so re-running doesn't duplicate chunks)
# -----------------------------

if Path(CHROMA_PATH).exists():
    shutil.rmtree(CHROMA_PATH)

print("Creating vector database...")
Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)

print("Ingestion completed successfully!")
