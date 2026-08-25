from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CHROMA_PATH = "data/chroma"


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Load vector database
# --------------------------------------------------

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)


# --------------------------------------------------
# 4. Create retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# --------------------------------------------------
# 5. Load Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# 6. Ask user a question
# --------------------------------------------------

question = input("\nAsk RaceMind: ")


# --------------------------------------------------
# 7. Retrieve relevant documents
# --------------------------------------------------

documents = retriever.invoke(question)


# --------------------------------------------------
# 8. Build context
# --------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)


# --------------------------------------------------
# 9. Create prompt
# --------------------------------------------------

prompt = f"""
You are RaceMind, an AI Formula 1 technical assistant.

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


# --------------------------------------------------
# 10. Generate answer
# --------------------------------------------------

response = llm.invoke(prompt)


# --------------------------------------------------
# 11. Display answer
# --------------------------------------------------

print("\n" + "=" * 80)
print("RACEMIND ANSWER")
print("=" * 80)

print(response.content)


# --------------------------------------------------
# 12. Display sources
# --------------------------------------------------

print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)

for i, document in enumerate(documents, start=1):

    print(
        f"{i}. "
        f"Page {document.metadata.get('page_label', document.metadata.get('page'))} "
        f"- {document.metadata.get('source')}"
    )