import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings



# Load environment variables from .env file
load_dotenv("api.env")

# Get API key and environment
api_key = os.environ.get("PINECONE_API_KEY")
environment = os.environ.get("PINECONE_ENVIRONMENT")

# Initialize Pinecone
pc = Pinecone(api_key=api_key, environment=environment)


# Print for verification
print(f"Pinecone initialized: {pc is not None}")
print(f"Environment: {environment}")
print(pc)
#print(os.environ.get("PINECONE_API_KEY"))
#print(pc.list_indexes())


# --------- Step 1: Load your document ---------
# For a text file
loader = TextLoader("D:/Softwares/KnowledgeBase/AWS Certification/GenAI/Sessions/CodePiece/GenAITraining/Files/poem1.txt", encoding="utf-8")
# For PDF: loader = PyPDFLoader("example.pdf")
documents = loader.load()

# --------- Step 2: Split Document in Chunks ---------
text_splitter = RecursiveCharacterTextSplitter(
   chunk_overlap=50,
     chunk_size=500
    
)
chunks = text_splitter.split_documents(documents)

print(f"✅ Loaded {len(chunks)} chunks")

# Step 5: View Chunks results
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)

# --------- Step 6: Load Ollama embedding model ---------
embedding_model = OllamaEmbeddings(model="nomic-embed-text")  # or other embedding-capable model

# --------- Step 7: Convert chunks into vectors ---------
texts = [chunk.page_content for chunk in chunks]
embeddings = embedding_model.embed_documents(texts)

# --------- Step 5: Print embeddings info ---------
print(f"✅ Converted {len(texts)} chunks into vectors.")
print(f"🔢 Vector size: {len(embeddings[0])} dimensions")

# Print the embedding vector values
print("Embedding vector values:")
for i, value in enumerate(embeddings):
    print(f"{i}: {value}")

# Connect to the index
index = pc.Index("knowledge-repo")

# Prepare data to insert
to_upsert = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    vector_id = f"chunk-{i}"
    metadata = {"text": chunk.page_content} 
    to_upsert.append((vector_id, embedding, metadata))

# Insert into Pinecone
index.upsert(vectors=to_upsert)

print("Inserted chunks into Pinecone.")

