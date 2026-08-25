from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


text = "Formula 1 technical regulations"

vector = embeddings.embed_query(text)

print("Embedding dimensions:", len(vector))
print("First 10 values:", vector[:10])