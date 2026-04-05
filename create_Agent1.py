from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate

from pinecone import Pinecone
import os
import requests
from dotenv import load_dotenv

load_dotenv("api.env")

# -------------------------------
# Weather Tool
# -------------------------------
def get_weather(location: str) -> str:
    """Get current weather for a given city."""
    
    api_key = os.getenv("WEATHER_API_KEY")   
    if not api_key:
        return "Weather API key is missing."

    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return f"Failed: {response.text}"

    data = response.json()
    condition = data["current"]["condition"]["text"]
    temp = data["current"]["temp_c"]

    return f"{location}: {condition}, {temp}°C"


# -------------------------------
# Stock Tool
# -------------------------------
def get_stock_price(symbol: str) -> str:
    """Fetch latest stock price for a given symbol (e.g. AAPL, TSLA)."""
    
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Stock API key is missing."

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)

    data = r.json()

    try:
        price = data["Global Quote"]["05. price"]
        return f"{symbol} current price is ${price}"
    except:
        return f"Could not fetch stock price for {symbol}"


# -------------------------------
# RAG Tool
# -------------------------------
def rag_tool(question: str) -> str:
    """Answer questions using Pinecone vector database and Ollama embeddings."""

    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    index = pc.Index("knowledge-repo")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Embed query
    query_vector = embedding_model.embed_query(question)

    # Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    # Build context
    retrieved_chunks = [
        match['metadata'].get("text", "") for match in results['matches']
    ]
    context = "\n".join(retrieved_chunks)

    # LLM
    llm = OllamaLLM(model="llama3.1")

    prompt = PromptTemplate.from_template("""
    You are an assistant answering questions based on the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response


# -------------------------------
# ✅ Correct Model Initialization
# -------------------------------
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# -------------------------------
# Agent
# -------------------------------
agent = create_agent(
    model=llm,   # ✅ pass object (IMPORTANT)
    tools=[get_weather, get_stock_price, rag_tool],
    system_prompt="You are a helpful assistant"
)

# -------------------------------
# Run
# -------------------------------
query = input("Ask something: ")

response = agent.invoke({
    "messages": [{"role": "user", "content": query}]
})

print(response["messages"][-1].content)
