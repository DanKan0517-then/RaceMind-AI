from rag import get_embeddings


embeddings = get_embeddings()


text = "Formula 1 technical regulations"

vector = embeddings.embed_query(text)

print("Embedding dimensions:", len(vector))
print("First 10 values:", vector[:10])