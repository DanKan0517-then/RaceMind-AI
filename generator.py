from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


response = llm.invoke(
    "Explain Formula 1 technical regulations in simple terms."
)


print(response.content)