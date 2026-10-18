from langchain_community.vectorstores import Chroma
from rag import get_embeddings


# Embedding model
embeddings = get_embeddings()


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