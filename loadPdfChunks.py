import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Load environment variables from .env file
load_dotenv("api.env")

# Get API key and environment
api_key = os.environ.get("PINECONE_API_KEY")
environment = os.environ.get("PINECONE_ENVIRONMENT")

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

print(f"Pinecone initialized: {pc is not None}")
print(f"Environment: {environment}")
print(pc)

# --------- Step 1: Load your PDF document ---------
pdf_path = "D:/GenAITraining/Files/Exercise_11.pdf"  # Update path as needed
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# --------- Step 2: Split Document in Chunks ---------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print(f"✅ Loaded {len(chunks)} chunks from PDF")

# Step 5: View Chunks results
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)

# --------- Step 6: Load Ollama embedding model ---------
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# --------- Step 7: Convert chunks into vectors ---------
texts = [chunk.page_content for chunk in chunks]
embeddings = embedding_model.embed_documents(texts)

print(f"✅ Converted {len(texts)} chunks into vectors.")
print(f"🔢 Vector size: {len(embeddings[0])} dimensions")

# Print the embedding vector values
print("Embedding vector values:")
for i, value in enumerate(embeddings):
    print(f"{i}: {value}")

# Connect to the index
index = pc.Index("my-test-index-a1")

# Prepare data to insert
to_upsert = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    vector_id = f"pdf-chunk-{i}"
    metadata = {"text": chunk.page_content}
    to_upsert.append((vector_id, embedding, metadata))

# Insert into Pinecone
index.upsert(vectors=to_upsert)

print("Inserted PDF chunks into Pinecone.")
