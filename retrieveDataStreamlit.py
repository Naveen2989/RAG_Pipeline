import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate


# Load environment variables from .env file
load_dotenv("api.env")

# Get API key and environment
api_key = os.environ.get("PINECONE_API_KEY")
environment = os.environ.get("PINECONE_ENVIRONMENT")

# Initialize Pinecone
pc = Pinecone(api_key=api_key)
index = pc.Index("financekr")

# Embedding & LLM setup
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
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
chain = prompt | llm

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔍 AI-Powered Search with Pinecone + Ollama")
st.markdown("Enter your query below and get context-aware answers powered by **Pinecone retrieval** and **LLM reasoning**.")

query_text = st.text_input("Enter your question:")

if st.button("Search") and query_text.strip():
    # Convert query string to vector
    query_vector = embedding_model.embed_query(query_text)

    # Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    # Display retrieval results
    st.subheader("📚 Retrieved Chunks")
    retrieved_chunks = []
    for match in results["matches"]:
        text = match["metadata"].get("text", "[No text metadata]")
        retrieved_chunks.append(text)
        st.write(f"**Score:** {match['score']:.4f}")
        st.write(text)
        st.markdown("---")

    # Prepare context
    context = "\n".join(retrieved_chunks)

    # Run the LLM chain
    response = chain.invoke({
        "context": context,
        "question": query_text
    })

    st.subheader("💡 LLM Response")
    st.write(response)
