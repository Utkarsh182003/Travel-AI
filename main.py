# main.py
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq
import uvicorn
import datetime
from knowledge_base import initialize_rag_system, get_relevant_documents
from neo4j_utils import get_neo4j_driver, clear_and_load_sample_data, query_knowledge_graph # NEW import

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Workflow Assistant Backend")

# Get API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Initialize RAG system globally
rag_retriever = None

@app.on_event("startup")
async def startup_event():
    """
    Event handler for FastAPI application startup.
    Initializes the RAG system and Neo4j connection, loads sample data.
    """
    global rag_retriever
    print("Initializing RAG system on startup...")
    rag_retriever = initialize_rag_system()
    print("RAG system initialized.")

    print("Initializing Neo4j connection and loading sample data...")
    # Attempt to get driver and load data. This will print errors if Neo4j is not running.
    neo4j_driver = get_neo4j_driver()
    if neo4j_driver:
        clear_and_load_sample_data()
    else:
        print("Neo4j driver could not be initialized. Knowledge Graph features will be unavailable.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Event handler for FastAPI application shutdown.
    Closes the Neo4j driver connection.
    """
    from neo4j_utils import close_neo4j_driver
    close_neo4j_driver()


# --- Pydantic Models for Tool Inputs ---
class BookFlightParams(BaseModel):
    origin: str = Field(description="The departure airport or city.")
    destination: str = Field(description="The arrival airport or city.")
    departure_date: str = Field(description="The date of departure in YYYY-MM-DD format.")
    num_passengers: int = Field(default=1, description="Number of passengers.")

class BookHotelParams(BaseModel):
    city: str = Field(description="The city where the hotel is to be booked.")
    check_in_date: str = Field(description="The check-in date in YYYY-MM-DD format.")
    check_out_date: str = Field(description="The check-out date in YYYY-MM-DD format.")
    num_guests: int = Field(default=1, description="Number of guests.")

class BlockCalendarParams(BaseModel):
    event_name: str = Field(description="Name of the event to be added to the calendar.")
    start_time: str = Field(description="Start time of the event in YYYY-MM-DD HH:MM format.")
    end_time: str = Field(description="End time of the event in YYYY-MM-DD HH:MM format.")

class QueryKnowledgeGraphParams(BaseModel):
    # This tool is designed to let the LLM generate a Cypher query based on the user's need.
    # In a more robust system, you might have pre-defined KG query types for the LLM to choose from.
    cypher_query: str = Field(description="A Cypher query to execute against the Neo4j knowledge graph to retrieve structured information. Examples: 'MATCH (c:City)-[:LOCATED_IN]->(a:Airport) RETURN c.name, a.name', 'MATCH (h:Hotel {city: 'London'}) RETURN h.name, h.stars'.")


# --- Simulated External Tools (Python functions) ---
def simulate_book_flight(origin: str, destination: str, departure_date: str, num_passengers: int = 1):
    """
    Simulates booking a flight.
    """
    print(f"Simulating flight booking: {origin} to {destination} on {departure_date} for {num_passengers} passengers.")
    if not (origin and destination and departure_date):
        return {"status": "error", "message": "Missing required flight details."}
    if num_passengers <= 0:
        return {"status": "error", "message": "Number of passengers must be at least 1."}
    return {"status": "success", "message": f"Flight from {origin} to {destination} on {departure_date} for {num_passengers} passengers booked successfully (simulated). Confirmation ID: FL12345"}

def simulate_book_hotel(city: str, check_in_date: str, check_out_date: str, num_guests: int = 1):
    """
    Simulates booking a hotel.
    """
    print(f"Simulating hotel booking: {city} from {check_in_date} to {check_out_date} for {num_guests} guests.")
    if not (city and check_in_date and check_out_date):
        return {"status": "error", "message": "Missing required hotel details."}
    if num_guests <= 0:
        return {"status": "error", "message": "Number of guests must be at least 1."}
    return {"status": "success", "message": f"Hotel in {city} from {check_in_date} to {check_out_date} for {num_guests} guests booked successfully (simulated). Confirmation ID: HTL9876"}

def simulate_block_calendar(event_name: str, start_time: str, end_time: str):
    """
    Simulates blocking a time slot in the calendar.
    """
    print(f"Simulating calendar block: '{event_name}' from {start_time} to {end_time}.")
    try:
        datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M")
    except ValueError:
        return {"status": "error", "message": "Invalid date/time format. Use YYYY-MM-DD HH:MM."}

    if not event_name:
        return {"status": "error", "message": "Event name cannot be empty."}
    return {"status": "success", "message": f"Calendar blocked for '{event_name}' from {start_time} to {end_time} (simulated)."}


# --- Tools definition for Groq LLM ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "simulate_book_flight",
            "description": "Book a flight for a user.",
            "parameters": BookFlightParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_book_hotel",
            "description": "Book a hotel for a user.",
            "parameters": BookHotelParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_block_calendar",
            "description": "Block a time slot in the user's calendar for a specific event.",
            "parameters": BlockCalendarParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "Query the Neo4j knowledge graph for structured information about flights, hotels, cities, airports, airlines, or user preferences. Provide a direct Cypher query as argument.",
            "parameters": QueryKnowledgeGraphParams.model_json_schema()
        }
    }
]

# --- FastAPI Chat Endpoint with Tool Calling, RAG, and KG Logic ---
class ChatMessageInput(BaseModel):
    message: str
    history: list[dict] = []

@app.post("/chat")
async def chat_with_llm(chat_input: ChatMessageInput):
    """
    Handles incoming chat messages.
    1. Retrieves relevant documents using RAG.
    2. Sends user message + RAG context to LLM with tool definitions (including KG query tool).
    3. If LLM suggests a tool call (booking or KG query), executes the tool.
    4. Sends tool output back to LLM for a final user-friendly response.
    """
    user_message = chat_input.message
    chat_history = chat_input.history

    print(f"Received message from frontend: {user_message}")
    print(f"Chat history: {chat_history}")

    # --- RAG Step ---
    retrieved_docs = get_relevant_documents(user_message, rag_retriever)
    rag_context = ""
    if retrieved_docs:
        rag_context = "\n\nRelevant Information from Knowledge Base:\n"
        for i, doc in enumerate(retrieved_docs):
            rag_context += f"--- Document {i+1} ---\n{doc.page_content}\n\n"
        print(f"RAG Context added:\n{rag_context}")

    # Construct the message for the LLM, including RAG context in the user's prompt
    full_user_message_for_llm = user_message + rag_context

    messages_for_llm = chat_history + [{"role": "user", "content": full_user_message_for_llm}]

    try:
        # Step 1: Send user message (with RAG context) to LLM with all tool definitions
        chat_completion = client.chat.completions.create(
            messages=messages_for_llm,
            model="llama3-8b-8192",
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )

        response_message = chat_completion.choices[0].message
        tool_calls = response_message.tool_calls

        # Step 2: Check if the LLM decided to call a tool
        if tool_calls:
            print(f"LLM proposed tool calls: {tool_calls}")
            messages_for_llm.append(response_message) # Append the tool call request

            tool_outputs = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"Executing tool: {function_name} with args: {function_args}")

                # Execute the simulated tool function or the KG query function
                if function_name == "simulate_book_flight":
                    tool_output = simulate_book_flight(**function_args)
                elif function_name == "simulate_book_hotel":
                    tool_output = simulate_book_hotel(**function_args)
                elif function_name == "simulate_block_calendar":
                    tool_output = simulate_block_calendar(**function_args)
                elif function_name == "query_knowledge_graph": # NEW TOOL EXECUTION
                    cypher_query = function_args.get("cypher_query")
                    if cypher_query:
                        tool_output = query_knowledge_graph(cypher_query)
                    else:
                        tool_output = {"status": "error", "message": "Missing 'cypher_query' for knowledge graph tool."}
                else:
                    tool_output = {"status": "error", "message": f"Unknown tool: {function_name}"}

                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(tool_output)
                })
                print(f"Tool '{function_name}' output: {tool_output}")

            # Step 3: Send tool output back to LLM to get a user-friendly response
            for output in tool_outputs:
                messages_for_llm.append(
                    {
                        "tool_call_id": output["tool_call_id"],
                        "role": "tool",
                        "content": output["output"],
                    }
                )

            final_chat_completion = client.chat.completions.create(
                messages=messages_for_llm,
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )
            llm_response_content = final_chat_completion.choices[0].message.content
            return {"response": llm_response_content, "role": "assistant"}

        else:
            # If no tool call, just return the LLM's direct response
            llm_response_content = response_message.content
            print(f"LLM direct response: {llm_response_content}")
            return {"response": llm_response_content, "role": "assistant"}

    except Exception as e:
        print(f"Error during chat or tool execution: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# To run this file directly for development, uncomment the following:
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
