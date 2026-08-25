from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------------
# 1. Configuration
# -----------------------------

PDF_PATH = "data/raw/fia/technical_regulations.pdf"
CHROMA_PATH = "data/chroma"


# -----------------------------
# 2. Load PDF
# -----------------------------

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Loaded {len(documents)} pages")


# -----------------------------
# 3. Split documents
# -----------------------------

print("Splitting documents...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# -----------------------------
# 4. Create embedding model
# -----------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 5. Store in Chroma
# -----------------------------

print("Creating vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)


print("Ingestion completed successfully!")