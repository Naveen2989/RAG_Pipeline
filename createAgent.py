import os
import requests
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv("api.env")

# Initialize Ollama LLM
llm = OllamaLLM(
    model="llama3.1:latest",
    temperature=0
)

def rag_tool (location: str):
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

    # Create the chain  (LCEL >>  LangChain Expression Language)
    #chain = LLMChain(llm=llm, prompt=prompt)
    chain = prompt | llm 
    # Run the chain with retrieved context

    response = chain.invoke({
        "context": context,
        "question": query_text
    })

    print("💬 LLM Response:\n", response)

# -------------------------------
# Weather Tool
# -------------------------------
def get_weather(location: str) -> str:
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Weather API key is missing."

    url = (
        "http://api.weatherapi.com/v1/current.json"
        f"?key={api_key}&q={location}"
    )

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return f"Failed to fetch weather: {response.text}"

    data = response.json()
    condition = data["current"]["condition"]["text"]
    temp = data["current"]["temp_c"]

    return f"The weather in {location} is {condition} at {temp}°C."

def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

#
weather_tool = Tool(
    name="GetWeather",
    func=get_weather,
    description="Get the current weather for a given city or location."
)

stock_tool = Tool(
    name="GetStockPrice",
    func=get_stock_price,
    description="Get the current stock price for a given stock symbol."
)

rag_tool = Tool(
    name="getHotelInfo",
    func=rag_tool,
    description="Get information about hotels in a given location."
)

# -------------------------------
# Create Agent
# -------------------------------
agent = initialize_agent(
    tools=[weather_tool, stock_tool, rag_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# -------------------------------
# Run Loop
# -------------------------------
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter city name (or 'exit'): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = agent.invoke({"input": user_input})
            print(f"\nAgent: {result['output']}")
        except Exception as e:
            print("Error:", e)
