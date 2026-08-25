from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Test documents
texts = [
    "Formula 1 cars must comply with the FIA technical regulations.",
    "The front wing must comply with the specified dimensional requirements.",
    "The power unit is subject to specific FIA technical regulations."
]


# Create vector database
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    persist_directory="data/chroma"
)


print("Vector database created successfully!")