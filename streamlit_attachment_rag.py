import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="HR Policy RAG", page_icon="📚")
st.title("📚 HR Policy Assistant (Upload + RAG)")

# -------------------------------
# Load Environment
# -------------------------------
load_dotenv("api.env")
api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
index = pc.Index("knowledge-repo")

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3.1")

# -------------------------------
# File Upload Section
# -------------------------------
st.sidebar.header("📎 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file is not None:
    with st.spinner("Processing document..."):

        # Save file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load document
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Convert chunks into vectors and upload to Pinecone
        for chunk in chunks:
            vector = embedding_model.embed_query(chunk.page_content)

            index.upsert(
                vectors=[(
                    str(uuid.uuid4()),
                    vector,
                    {"text": chunk.page_content}
                )]
            )

        st.success("✅ Document uploaded and indexed successfully!")

# -------------------------------
# Query Section
# -------------------------------
st.header("🔎 Ask a Question")

query_text = st.text_input("Enter your question:")

if query_text:
    with st.spinner("Searching vector store..."):

        query_vector = embedding_model.embed_query(query_text)

        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )

        retrieved_chunks = [
            match['metadata'].get("text", "")
            for match in results['matches']
        ]

        context = "\n".join(retrieved_chunks)

    with st.expander("📄 Retrieved Context"):
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.markdown(f"**Chunk {i}**")
            st.write(chunk)
            st.write("---")

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

    with st.spinner("Generating answer..."):
        response = chain.invoke({
            "context": context,
            "question": query_text
        })

    st.subheader("💬 Answer")
    st.write(response)