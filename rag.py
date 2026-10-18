"""Shared RAG pipeline for RaceMind AI (used by the web app and the CLI)."""

import os
from functools import lru_cache

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = int(os.getenv("TOP_K", "4"))

PROMPT_TEMPLATE = """You are RaceMind, an AI Formula 1 technical assistant.

Answer the user's question using ONLY the provided FIA
technical regulation context.

If the answer cannot be found in the provided context,
say that the information was not found in the retrieved
FIA regulations.

Do not invent technical regulations.

USER QUESTION:
{question}

FIA CONTEXT:
{context}
"""


class MiniLMOnnxEmbeddings(Embeddings):
    """all-MiniLM-L6-v2 (384-dim) run through ONNX Runtime via ChromaDB.

    Same model as sentence-transformers/all-MiniLM-L6-v2, but without torch,
    so the whole app fits in a 512MB container.
    """

    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [list(map(float, v)) for v in self._fn(list(texts))]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings():
    return MiniLMOnnxEmbeddings()


@lru_cache(maxsize=1)
def get_retriever():
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


@lru_cache(maxsize=1)
def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")
    return ChatGroq(model=GROQ_MODEL, temperature=0)


def retrieve(question: str):
    return get_retriever().invoke(question)


def format_sources(documents):
    sources = []
    for doc in documents:
        page = doc.metadata.get("page_label", doc.metadata.get("page"))
        sources.append(
            {
                "page": str(page) if page is not None else None,
                "source": os.path.basename(str(doc.metadata.get("source", ""))),
                "snippet": doc.page_content[:300].strip(),
            }
        )
    return sources


def ask(question: str) -> dict:
    documents = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    response = get_llm().invoke(prompt)
    return {"answer": response.content, "sources": format_sources(documents)}
