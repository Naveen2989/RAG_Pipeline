import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain import PromptTemplate
from langchain.chains import LLMChain




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
index = pc.Index("knowledge-repo")

#vector_store=pc.from_documents(chunks,embeddings,index_name="my-test-index-a1")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")  # or other embedding-capable model
query_text="What matter if we go clear to the west"

# Convert query string to vector
query_vector = embedding_model.embed_query(query_text)



# Query Pinecone for similar vectors
results = index.query(
    vector=query_vector,
    top_k=5,                # Number of most similar results to return
    include_metadata=True   # Include stored metadata like original text
)

# # Print the results
print("Top matching results:")
for match in results['matches']:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']}")
    print(f"Text: {match['metadata'].get('text', '[No text metadata]')}")
    print("-" * 50)


    
# Extract texts from metadata
retrieved_chunks = [match['metadata'].get("text", "") for match in results['matches']]
context = "\n".join(retrieved_chunks)

llm=OllamaLLM(model="llama3.1")

# Define prompt template
prompt = PromptTemplate.from_template(
    """
    You are an assistant answering questions based on the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

# Create the chain
#chain = LLMChain(llm=llm, prompt=prompt)
chain = prompt | llm 
# Run the chain with retrieved context

response = chain.invoke({
    "context": context,
    "question": query_text
})

print("💬 LLM Response:\n", response)

