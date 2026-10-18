from rag import get_embeddings
from langchain_community.vectorstores import Chroma


CHROMA_PATH = "data/chroma"


# Load embedding model
embeddings = get_embeddings()


# Load Chroma database
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)


# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# Questions to test
questions = [
    "What are the general principles of the technical regulations?",
    "What are the requirements for the front wing?",
    "What are the regulations concerning the power unit?"
]


# Test each question
for question in questions:

    print("\n")
    print("=" * 100)
    print("QUESTION:")
    print(question)
    print("=" * 100)

    results = retriever.invoke(question)

    print(f"\nRetrieved {len(results)} chunks\n")

    for i, document in enumerate(results, start=1):

        print("-" * 100)
        print(f"RESULT {i}")
        print("-" * 100)

        print(document.page_content[:700])

        print("\nPAGE:", document.metadata.get("page_label"))
        print("SOURCE:", document.metadata.get("source"))