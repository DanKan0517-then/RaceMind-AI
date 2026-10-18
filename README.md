# RaceMind-AI
AI-powered Formula 1 technical assistant using Retrieval-Augmented Generation (RAG), LangChain, vector search, and LLMs.

RaceMind AI is a RAG-based AI assistant that can answer questions about Formula 1 technical regulations by retrieving relevant information from FIA documents and using an LLM to generate grounded responses.

In this project, I implemented a basic RAG pipeline using:

FIA Formula 1 Technical Regulations
LangChain
Sentence Transformers
ChromaDB
Groq LLM

#project Structure
<img width="316" height="645" alt="image" src="https://github.com/user-attachments/assets/877a4a30-4fb3-4572-ae1d-a118334dd053" />


#This is how Our RAG system looks
<img width="431" height="587" alt="image" src="https://github.com/user-attachments/assets/d82e7524-4ca5-49b6-860d-578232d18769" />

#How it works
1. Load the PDF

The FIA technical regulations are loaded using LangChain's PDF loader.

2. Split the document
The document is divided into smaller chunks using a recursive text splitter.

277 pages
   ↓
897 chunks
3. Create embeddings

Each chunk is converted into a vector using:

sentence-transformers/all-MiniLM-L6-v2

The embedding size used in this project is 384 dimensions.

4. Store embeddings
The embeddings are stored in ChromaDB.

This allows the system to search for documents based on semantic similarity.

5. Retrieve relevant information

When the user asks a question, the question is converted into an embedding and the most relevant chunks are retrieved from ChromaDB.
6. Generate the answer

The retrieved FIA content is passed to the Groq LLM along with the user's question.

The model then generates an answer using the retrieved context.



#OUTPUT
<img width="1034" height="743" alt="image" src="https://github.com/user-attachments/assets/8d95996b-8a5e-4a0b-8c18-272e4c6b4532" />

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python ingest.py            # builds data/chroma from the PDF (one time)
python main.py              # web app at http://localhost:7860
python rag_pipeline.py      # or ask from the terminal
```

API: `POST /api/ask` with `{"question": "..."}` returns the answer plus page-level sources. `GET /health` reports whether the index and API key are present.

## Deployment

Runs on Render's free tier as a Docker web service (config in `render.yaml`).

- Embeddings use the ONNX build of `all-MiniLM-L6-v2` that ships with ChromaDB. It's the same model, so the vectors are the same, but there's no torch dependency, which keeps the app inside Render's 512MB memory limit.
- The Docker build downloads the model and builds the Chroma index from the PDF into the image, so the service starts ready to answer.
- `GROQ_API_KEY` is set as an environment variable in the Render dashboard.
- Every push to `main` redeploys automatically.
- The free tier sleeps after about 15 minutes idle, so the first request after that takes around 30–60 seconds.
