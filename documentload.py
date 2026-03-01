import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec

# Load environment variables from .env file
load_dotenv("api.env")

# Get API key and environment
api_key = os.environ.get("PINECONE_API_KEY")
environment = os.environ.get("PINECONE_ENVIRONMENT")

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Print for verification
print(f"Pinecone initialized: {pc is not None}")
print(f"Environment: {environment}")
print(pc)
print(os.environ.get("PINECONE_API_KEY"))
print(pc.list_indexes())


# Define index parameters
index_name = "my-test-index-a1"
dimension = 768  # This should match your embedding model's output
metric = "cosine"  # Other options: 'dotproduct', 'euclidean'

# Check if the index already exists
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension=dimension,
        metric=metric,
    )
    print(f"✅ Index '{index_name}' created.")
else:
    print(f"⚠️ Index '{index_name}' already exists.")

# Optionally, describe the index
index_description = pc.describe_index(index_name)
print("📄 Index Description:", index_description)
print(pc.list_indexes())