import os
import json
import re # regex for robust parsing
import logging
import asyncio
from typing import Dict, Any, List, Optional
import datetime # For calendar and timestamp handling

# FastAPI and Pydantic for API structure
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Environment variable loading
from dotenv import load_dotenv

# Groq client for LLM interaction
from groq import Groq
# Import specific types for tool calls from Groq library
from groq.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

# For running the FastAPI app directly
import uvicorn

# For web scraping/HTTP requests if needed (from original snippet, though not explicitly used in current tools)
import httpx
from bs4 import BeautifulSoup

# Import components from your project files
# RAG system for knowledge retrieval
from knowledge_base import initialize_rag_system, get_relevant_documents, add_documents_to_rag

# Neo4j utilities for Knowledge Graph interaction
from neo4j_utils import get_neo4j_driver, clear_and_load_sample_data, query_knowledge_graph, add_flight_route_to_kg, add_hotel_to_kg

# External API simulations (flight, hotel)
from external_apis import simulate_flight_search, simulate_google_hotels

# Load environment variables from .env file
load_dotenv()

# Configure logging for better visibility in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="AI Workflow Assistant Backend")

# Global variables for Groq client and a simple session state mock
# In a real FastAPI app, session state for chat history would typically be managed
# using a database (e.g., Redis, PostgreSQL) or a more robust session management library.
# This mock is for demonstration purposes to mimic Streamlit's session_state for the LLM context.
groq_client: Groq = None

class SessionStateMock:
    def __init__(self):
        self._state = {
            "groq_model": "llama3-8b-8192", # Default Groq model
            "messages": [], # To store conversation history for the LLM
            "rag_retriever": None, # RAG retriever instance
            "neo4j_driver": None, # Neo4j driver instance
            "last_flight_search_results": None, # To store results for follow-up actions like booking
            "last_hotel_search_results": None, # To store results for follow-up actions like booking
            "booking_id_counter": 1000 # Simple counter for mock booking IDs
        }

    def __getitem__(self, key):
        # Allow direct access like st["key"]
        return self._state[key]

    def __setitem__(self, key, value):
        # Allow direct assignment like st["key"] = value
        self._state[key] = value

    def get(self, key, default=None):
        # Allow .get() method like st.get("key", default)
        return self._state.get(key, default)

    def update(self, new_values: Dict[str, Any]):
        # Allow .update() method like st.update({"key": value})
        self._state.update(new_values)

st = SessionStateMock() # Instantiate the mock session state

# Initialize Groq client, RAG system, and Neo4j driver on startup
@app.on_event("startup")
async def startup_event():
    global groq_client
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logging.error("GROQ_API_KEY not found in environment variables.")
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
    groq_client = Groq(api_key=groq_api_key)
    logging.info("Groq client initialized.")

    # Initialize RAG system
    logging.info("Initializing RAG system...")
    st["rag_retriever"] = initialize_rag_system()
    if st["rag_retriever"]:
        logging.info("RAG system initialized successfully.")
        # Add some initial documents to RAG if needed (for demonstration)
        add_documents_to_rag("Our airline policy states that international flights require a passport valid for at least six months beyond your intended stay. Baggage allowance is 23kg for economy class.", source="airline_policy")
        add_documents_to_rag("Hotel booking cancellations are free up to 24 hours before check-in. After that, a one-night fee applies.", source="hotel_policy")
    else:
        logging.warning("RAG system could not be initialized. Knowledge retrieval functions may not work.")

    # Initialize Neo4j driver and load sample data
    logging.info("Initializing Neo4j driver and loading sample data...")
    neo4j_driver_instance = get_neo4j_driver()
    if neo4j_driver_instance:
        st["neo4j_driver"] = neo4j_driver_instance
        clear_and_load_sample_data() 
        logging.info("Neo4j driver initialized and sample data loaded.")
    else:
        logging.warning("Neo4j driver could not be initialized. Knowledge Graph functions may not work.")

@app.on_event("shutdown")
async def shutdown_event():
    if st.get("neo4j_driver"):
        st.get("neo4j_driver").close()
        logging.info("Neo4j driver connection closed.")

# Pydantic model for the chat request body
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = [] # To receive chat history from frontend

# Define tools for Groq
tools = [
    {
        "type": "function",
        "function": {
            "name": "simulate_flight_search",
            "description": "Simulate searching for flights between two airports on a specific date. Returns a list of available flights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_iata": {
                        "type": "string",
                        "description": "The IATA code of the origin airport (e.g., 'LHR' for London Heathrow)."
                    },
                    "destination_iata": {
                        "type": "string",
                        "description": "The IATA code of the destination airport (e.g., 'CDG' for Paris Charles de Gaulle)."
                    },
                    "flight_date": {
                        "type": "string",
                        "format": "date",
                        "description": "The date of the flight in YYYY-MM-DD format (e.g., '2025-07-20')."
                    }
                },
                "required": ["origin_iata", "destination_iata", "flight_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_google_hotels",
            "description": "Simulate searching for hotels in a specific city for a given check-in and check-out date, number of guests, and optional amenities. Returns a list of available hotels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city for which to search hotels (e.g., 'Paris')."
                    },
                    "check_in_date": {
                        "type": "string",
                        "format": "date",
                        "description": "The check-in date in YYYY-MM-DD format (e.g., '2025-08-10')."
                    },
                    "check_out_date": {
                        "type": "string",
                        "format": "date",
                        "description": "The check-out date in YYYY-MM-DD format (e.g., '2025-08-15')."
                    },
                    "num_guests": {
                        "type": "integer",
                        "description": "The number of guests."
                    },
                    "amenities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of desired amenities (e.g., ['pool', 'gym'])."
                    }
                },
                "required": ["city", "check_in_date", "check_out_date", "num_guests"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "Queries the Neo4j Knowledge Graph for information based on a Cypher query. Use this for general knowledge queries about airports, cities, or specific entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Cypher query to execute. Example: 'MATCH (a:Airport) RETURN a.name AS airportName, a.iata AS iataCode LIMIT 5'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_relevant_documents",
            "description": "Retrieves relevant documents from the RAG system based on a user query. Use this for policy questions or specific information retrieval from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for relevant documents in the RAG system."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_documents_to_rag",
            "description": "Adds new text content to the RAG system's knowledge base. Useful for dynamically updating information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_content": {
                        "type": "string",
                        "description": "The text content to add to the RAG system."
                    },
                    "source": {
                        "type": "string",
                        "description": "The source of the document (e.g., 'user_input', 'new_policy')."
                    }
                },
                "required": ["text_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_flight_route_to_kg",
            "description": "Adds a new flight route to the Neo4j Knowledge Graph. Use this to update flight information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_iata": {"type": "string"},
                    "destination_iata": {"type": "string"},
                    "flight_number": {"type": "string"},
                    "airline": {"type": "string"},
                    "departure_time": {"type": "string", "format": "time"},
                    "arrival_time": {"type": "string", "format": "time"},
                    "price": {"type": "number"},
                    "flight_date": {"type": "string", "format": "date"}
                },
                "required": ["origin_iata", "destination_iata", "flight_number", "airline", "departure_time", "arrival_time", "price", "flight_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_hotel_to_kg",
            "description": "Adds a new hotel to the Neo4j Knowledge Graph. Use this to update hotel information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "city": {"type": "string"},
                    "stars": {"type": "integer"},
                    "price_per_night": {"type": "number"},
                    "address": {"type": "string"},
                    "amenities": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "city", "stars", "price_per_night", "address"]
            }
        }
    }
]

# In-memory storage for simulated bookings
simulated_flight_bookings = {}
simulated_hotel_bookings = {}


def simulate_book_flight(flight_number: str, origin_iata: str, destination_iata: str, flight_date: str, price: float, passenger_name: str, passenger_email: str):
    booking_id = f"FKB{len(simulated_flight_bookings) + 1:04d}"
    booking_details = {
        "flight_number": flight_number,
        "origin_iata": origin_iata,
        "destination_iata": destination_iata,
        "flight_date": flight_date,
        "price": price,
        "passenger_name": passenger_name,
        "passenger_email": passenger_email,
        "status": "confirmed",
        "booking_time": datetime.datetime.now().isoformat()
    }
    simulated_flight_bookings[booking_id] = booking_details
    
    # --- Send Confirmation Email for Flight ---
    # Removed email sending logic
    # email_subject = f"Flight Booking Confirmation - {booking_id}"
    # email_body = (
    #     f"Dear {passenger_name},\n\n"
    #     f"Your flight booking has been confirmed!\n\n"
    #     f"Booking ID: {booking_id}\n"
    #     f"Flight: {flight_number} ({origin_iata} to {destination_iata})\n"
    #     f"Date: {flight_date}\n"
    #     f"Price: ${price:.2f}\n\n"
    #     f"Thank you for booking with us!\n\n"
    #     f"Please note: This is a simulated booking for demonstration purposes."
    # )
    # send_email(passenger_email, email_subject, email_body)

    return {"status": "success", "message": f"Flight {flight_number} from {origin_iata} to {destination_iata} on {flight_date} booked successfully for {passenger_name}. Your booking ID is {booking_id}. (Email simulation removed).", "booking_id": booking_id}

def simulate_book_hotel(hotel_id: str, check_in_date: str, check_out_date: str, num_guests: int, guest_name: str, guest_email: str):
    booking_id = f"HKB{len(simulated_hotel_bookings) + 1:04d}"
    booking_details = {
        "hotel_id": hotel_id,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "num_guests": num_guests,
        "guest_name": guest_name,
        "guest_email": guest_email,
        "status": "confirmed",
        "booking_time": datetime.datetime.now().isoformat()
    }
    simulated_hotel_bookings[booking_id] = booking_details

    # --- Send Confirmation Email for Hotel ---
    # Removed email sending logic
    # email_subject = f"Hotel Booking Confirmation - {booking_id}"
    # email_body = (
    #     f"Dear {guest_name},\n\n"
    #     f"Your hotel booking has been confirmed!\n\n"
    #     f"Booking ID: {booking_id}\n"
    #     f"Hotel: {hotel_id}\n"
    #     f"Check-in Date: {check_in_date}\n"
    #     f"Check-out Date: {check_out_date}\n"
    #     f"Number of Guests: {num_guests}\n\n"
    #     f"We look forward to your stay!\n\n"
    #     f"Please note: This is a simulated booking for demonstration purposes."
    # )
    # send_email(guest_email, email_subject, email_body)

    return {"status": "success", "message": f"Hotel {hotel_id} from {check_in_date} to {check_out_date} booked successfully for {guest_name}. Your booking ID is {booking_id}. (Email simulation removed).", "booking_id": booking_id}


def get_flight_booking_status(booking_id: str):
    booking = simulated_flight_bookings.get(booking_id)
    if booking:
        return {"status": "success", "booking_details": booking, "message": f"Status for flight booking {booking_id}: {booking['status']}."}
    else:
        return {"status": "error", "message": f"Flight booking {booking_id} not found."}

def get_hotel_booking_status(booking_id: str):
    booking = simulated_hotel_bookings.get(booking_id)
    if booking:
        return {"status": "success", "booking_details": booking, "message": f"Status for hotel booking {booking_id}: {booking['status']}."}
    else:
        return {"status": "error", "message": f"Hotel booking {booking_id} not found."}

async def add_url_to_knowledge_base(url: str, source_name: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status() # Raise an exception for HTTP errors
            
            # Use BeautifulSoup to extract text content
            soup = BeautifulSoup(response.text, 'html.parser')
            page_content = soup.get_text(separator=' ', strip=True)
            
            if page_content:
                # Add to RAG system
                success = add_documents_to_rag(page_content, source=source_name)
                if success:
                    return {"status": "success", "message": f"Content from {url} (source: {source_name}) added to knowledge base."}
                else:
                    return {"status": "error", "message": f"Failed to add content from {url} to knowledge base."}
            else:
                return {"status": "error", "message": f"No readable text content found at {url}."}
    except httpx.RequestError as e:
        return {"status": "error", "message": f"Network or HTTP error fetching URL {url}: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred while processing URL {url}: {e}"}


# Define additional tools for booking and status checking
# These are internal and called by the LLM
tools.extend([
    {
        "type": "function",
        "function": {
            "name": "simulate_book_flight",
            "description": "Simulates booking a flight. Requires full flight details and a user email. Returns a booking confirmation or error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {"type": "string"},
                    "origin_iata": {"type": "string"},
                    "destination_iata": {"type": "string"},
                    "flight_date": {"type": "string", "format": "date"},
                    "price": {"type": "number"},
                    "passenger_name": {"type": "string"},
                    "passenger_email": {"type": "string", "format": "email"},
                },
                "required": ["flight_number", "origin_iata", "destination_iata", "flight_date", "price", "passenger_name", "passenger_email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_book_hotel",
            "description": "Simulates booking a hotel. Requires hotel ID, dates, and user email. Returns a booking confirmation or error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotel_id": {"type": "string"},
                    "check_in_date": {"type": "string", "format": "date"},
                    "check_out_date": {"type": "string", "format": "date"},
                    "num_guests": {"type": "integer"},
                    "guest_name": {"type": "string"},
                    "guest_email": {"type": "string", "format": "email"},
                },
                "required": ["hotel_id", "check_in_date", "check_out_date", "num_guests", "guest_name", "guest_email"],
            },
        },
    },
    # Removed simulate_block_calendar tool
    {
        "type": "function",
        "function": {
            "name": "get_flight_booking_status",
            "description": "Retrieves the status of a previously simulated flight booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string", "description": "The booking ID of the flight."},
                },
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hotel_booking_status",
            "description": "Retrieves the status of a previously simulated hotel booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string", "description": "The booking ID of the hotel."},
                },
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_url_to_knowledge_base",
            "description": "Fetches content from a given URL and adds it to the RAG knowledge base for future information retrieval. Useful for dynamically updating the RAG with external webpage content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to fetch content from.",
                    },
                    "source_name": {
                        "type": "string",
                        "description": "A brief name for the source of this content (e.g., 'Airline Policy Page').",
                    },
                },
                "required": ["url", "source_name"],
            },
        },
    },
])


async def execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a single tool call based on the function name and arguments.
    This function acts as a dispatcher to your external API simulations and KG/RAG functions.
    """
    # Tool call structure from Groq is a dict with 'function' key, which itself is a dict
    # with 'name' and 'arguments' (JSON string).
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    logging.info(f"Executing tool: {function_name} with args: {arguments}")

    try:
        # Handle functions from external_apis.py
        if function_name == "simulate_flight_search":
            result = simulate_flight_search(**arguments)
            if result.get("status") == "success" and result.get("flights"):
                st["last_flight_search_results"] = result["flights"]
            return result
        elif function_name == "simulate_google_hotels":
            result = simulate_google_hotels(**arguments)
            if result.get("status") == "success" and result.get("hotels"):
                st["last_hotel_search_results"] = result["hotels"]
            return result
        # Removed block_calendar, send_email, get_current_weather calls
        # elif function_name == "block_calendar":
        #     result = block_calendar(**arguments)
        #     return result
        # elif function_name == "send_email":
        #     result = send_email(**arguments)
        #     return result
        # elif function_name == "get_current_weather":
        #     result = get_current_weather(**arguments)
        #     return result

        # Handle functions from knowledge_base.py (RAG)
        elif function_name == "get_relevant_documents":
            retriever = st.get("rag_retriever")
            if not retriever:
                return {"status": "error", "detail": "RAG system not initialized."}
            docs = get_relevant_documents(query=arguments["query"], retriever=retriever)
            if docs:
                return {"status": "success", "documents": [d.page_content for d in docs]}
            else:
                return {"status": "success", "documents": [], "message": "No relevant documents found."}
        elif function_name == "add_documents_to_rag":
            success = add_documents_to_rag(text_content=arguments["text_content"], source=arguments.get("source", "user_added"))
            return {"status": "success" if success else "error", "message": "Document added." if success else "Failed to add document."}

        # Handle functions from neo4j_utils.py (Knowledge Graph)
        elif function_name == "query_knowledge_graph":
            driver = st.get("neo4j_driver")
            if not driver:
                return {"status": "error", "detail": "Neo4j driver not initialized."}
            result = query_knowledge_graph(driver, arguments["query"])
            return result 
        elif function_name == "add_flight_route_to_kg":
            driver = st.get("neo4j_driver")
            if not driver:
                return {"status": "error", "detail": "Neo4j driver not initialized."}
            result = add_flight_route_to_kg(driver, **arguments)
            return result
        elif function_name == "add_hotel_to_kg":
            driver = st.get("neo4j_driver")
            if not driver:
                return {"status": "error", "detail": "Neo4j driver not initialized."}
            result = add_hotel_to_kg(driver, **arguments)
            return result

        # --- Booking Functions ---
        elif function_name == "simulate_book_flight":
            result = simulate_book_flight(**arguments)
            return result

        elif function_name == "simulate_book_hotel":
            result = simulate_book_hotel(**arguments)
            return result

        # Removed simulate_block_calendar call
        # elif function_name == "simulate_block_calendar": # This refers to the function defined in main.py
        #     result = simulate_block_calendar(**arguments)
        #     return result

        elif function_name == "get_flight_booking_status": # Specific booking status functions
            result = get_flight_booking_status(**arguments)
            return result
        elif function_name == "get_hotel_booking_status":
            result = get_hotel_booking_status(**arguments)
            return result
        
        elif function_name == "add_url_to_knowledge_base":
            result = await add_url_to_knowledge_base(**arguments) # AWAIT the async call
            return result

        else:
            raise ValueError(f"Unknown tool: {function_name}")

    except Exception as e:
        logging.error(f"Error executing tool '{function_name}': {e}")
        return {"status": "error", "detail": str(e)}


@app.post("/chat")
async def chat_with_llm(chat_input: ChatRequest):
    global groq_client
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized. Check API key.")

    user_message = chat_input.message
    chat_history = chat_input.history # Received history from frontend

    # Append user message to history for LLM context
    st["messages"].append({"role": "user", "content": user_message})

    # --- RAG Step ---
    retrieved_docs = get_relevant_documents(user_message, st.get("rag_retriever"), k=1) 
    rag_context = ""
    if retrieved_docs:
        rag_context = "\n\nRelevant Information from Knowledge Base:\n"
        for i, doc in enumerate(retrieved_docs):
            # Truncate content for display to avoid overwhelming LLM context
            truncated_content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            rag_context += f"--- Document {i+1} ---\n{truncated_content}\n\n"

    full_user_message_for_llm = user_message + rag_context

    system_prompt = (
        "You are an AI Workflow Assistant. "
        "Use tools for travel (flights, hotels) and calendar. "
        "**Strict Rules:**\n"
        "1.  **Tool Use:** Prioritize tools for clear requests. Manage multi-step tasks (search then book).\n"
        "2.  **Parameters:** Use ONLY defined tool parameters. DO NOT invent parameters.\n"
        "3.  **Disambiguation:** If tool parameters are missing/unclear, ASK specific, polite questions. Specify formats (YYYY-MM-DD, HH:MM, IATA codes).\n"
        "4.  **Constraints:** After search, check results against user constraints (e.g., price, amenities). Report if no results meet ALL criteria, and offer alternatives.\n"
        "5.  **Flow:** Search before booking. Use search results for booking parameters. Book directly if all details provided. Assume current year 2025; DO NOT book past dates.\n"
        "6.  **Tool Output:** Use tool output for responses. ALWAYS generate a natural language response to the user. NEVER output tool call markup (such as /tool-use>, <tool-use>, or JSON tool calls) or any raw JSON, even in tool outputs. ONLY output user-facing text. If you need to call a tool, do so using the function call mechanism, not by outputting tool call markup or echoing the tool call directly in your final response.\n"
        "7.  **Calendar/Status:** (Removed calendar functionality) Use `get_flight_booking_status`, `get_hotel_booking_status`.\n" # Updated prompt
        "8.  **KG Query:** Use `query_knowledge_graph` for specific structured data (Cypher queries only).\n"
        "9.  **RAG:** Use 'Relevant Information' for general knowledge/policies.\n"
        "10. **Dynamic Knowledge (URL):** Use `add_url_to_knowledge_base` (ask for URL & source name).\n"
        "11. **Dynamic Knowledge (KG):** Use `add_flight_route_to_kg` or `add_hotel_to_kg` for new structured flight/hotel data. Confirm details.\n"
        "12. **Conversation:** Maintain flow, remember previous turns.\n"
        "13. For any booking (flight or hotel), ALWAYS include the user's email address in the tool call parameters if provided in the user's message. If not provided, ask the user for their email before booking."
    )

    # Use the history received from the frontend for the LLM context
    # This ensures the LLM has the full conversation history
    messages_for_llm = [{"role": "system", "content": system_prompt}] + chat_history
    messages_for_llm.append({"role": "user", "content": full_user_message_for_llm})

    try:
        # First call to LLM to decide on tool use or direct response
        chat_completion = groq_client.chat.completions.create( # Use groq_client
            messages=messages_for_llm,
            model=st["groq_model"], 
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            stop=None,
            stream=False,
        )

        response_message = chat_completion.choices[0].message
        
        # --- STRICTLY HANDLE TOOL CALLS: Prioritize structured tool_calls, then parse content ---
        tool_calls = response_message.tool_calls
        
        # If no structured tool_calls, but content might contain raw markup, parse it NOW.
        # This block is for robustness against LLMs that might output tool calls as raw text.
        if not tool_calls and response_message.content and "<tool-use>" in response_message.content:
            logging.warning("Detected raw tool-use markup in content. Attempting to parse.")
            try:
                # Use regex to find the JSON string within <tool-use> tags
                match = re.search(r'<tool-use>\s*({.*})\s*</tool-use>', response_message.content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    parsed_tool_data = json.loads(json_str)
                    
                    if "tool_calls" in parsed_tool_data and isinstance(parsed_tool_data["tool_calls"], list):
                        tool_calls_from_content = []
                        for tc_data in parsed_tool_data["tool_calls"]:
                            if tc_data.get("type") == "function" and "function" in tc_data and "name" in tc_data["function"]:
                                # Create ChatCompletionMessageToolCall objects from parsed data
                                tool_calls_from_content.append(
                                    ChatCompletionMessageToolCall(
                                        id=tc_data.get("id", f"call_from_content_{datetime.datetime.now().timestamp()}"),
                                        type=tc_data["type"],
                                        function=Function(
                                            name=tc_data["function"]["name"],
                                            arguments=json.dumps(tc_data["function"].get("parameters", {})) 
                                        )
                                    )
                                )
                        if tool_calls_from_content:
                            tool_calls = tool_calls_from_content 
                            logging.info(f"Successfully parsed tool calls from content: {tool_calls}")
                        else:
                            logging.warning("Parsed JSON but found no valid tool_calls within.")
                            tool_calls = None 
                    else:
                        logging.warning("Parsed JSON but 'tool_calls' key is missing or not a list.")
                        tool_calls = None
                else:
                    logging.warning("Tool-use markup found, but no parsable JSON within.")
                    tool_calls = None 
            except json.JSONDecodeError as e:
                logging.error(f"JSONDecodeError parsing tool-use content: {e}")
                tool_calls = None
            except Exception as e:
                logging.error(f"General error parsing tool-use content: {e}")
                tool_calls = None
        # --- End of strict tool-use content parsing ---
        
        if tool_calls: # If tool_calls are now present (either structured or parsed)
            # Append a structured assistant message for the history
            # This teaches the LLM how it *should* respond with tool calls.
            structured_assistant_message = {
                "role": "assistant",
                "content": None, 
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type, 
                        "function": {
                            "name": tc.function.name, 
                            "arguments": tc.function.arguments 
                        }
                    } for tc in tool_calls
                ]
            }
            st["messages"].append(structured_assistant_message) 
            messages_for_llm.append(structured_assistant_message) 
            
            tool_outputs_for_llm = []
            display_message = ""

            for tool_call in tool_calls:
                function_name = tool_call.function.name 
                # Ensure arguments are parsed from JSON string to Python dict
                function_args = json.loads(tool_call.function.arguments) 

                logging.info(f"Executing tool: {function_name} with args: {function_args}")

                tool_output_data = {"status": "error", "message": f"Unknown tool: {function_name}"}
                output_str = ""

                # --- Tool Execution Logic ---
                # All calls to execute_tool_call must now be awaited
                tool_output_data = await execute_tool_call({"function": {"name": function_name, "arguments": json.dumps(function_args)}})
                
                # Format tool output for LLM and for display
                if function_name == "simulate_flight_search":
                    flights = tool_output_data.get("flights", [])
                    if tool_output_data.get("status") == "success" and flights:
                        summary_lines = [f"Found {len(flights)} flights from {function_args.get('origin_iata')} to {function_args.get('destination_iata')} on {function_args.get('flight_date')}:"]
                        for i, flight in enumerate(flights[:3]):
                            summary_lines.append(f"  - {flight.get('flight_number', 'N/A')} ({flight.get('airline', 'N/A')}): {flight.get('departure_time', 'N/A')} for ${flight.get('price_estimate', 0.0):.2f}")
                        if len(flights) > 3:
                            summary_lines.append(f"  ...and {len(flights) - 3} more flights. Ask for details if needed.")
                        output_str = "\n".join(summary_lines)
                        display_message += output_str + "\n"
                    else:
                        output_str = tool_output_data.get("message", "No flights found for the given criteria.")
                        display_message += output_str + "\n"
                        
                elif function_name == "simulate_google_hotels":
                    hotels = tool_output_data.get("hotels", [])
                    if tool_output_data.get("status") == "success" and hotels:
                        summary_lines = [f"Found {len(hotels)} hotels in {function_args.get('city')}:"]
                        for i, hotel in enumerate(hotels[:3]):
                            summary_lines.append(f"  - {hotel.get('name', 'N/A')} ({hotel.get('stars', 'N/A')} stars) for ${hotel.get('price_per_night', 0.0):.2f} per night.")
                        if len(hotels) > 3:
                            summary_lines.append(f"  ...and {len(hotels) - 3} more hotels. Ask for details if needed.")
                        output_str = "\n".join(summary_lines)
                        display_message += output_str + "\n"
                    else:
                        output_str = tool_output_data.get("message", "No hotels found for the given criteria.")
                        display_message += output_str + "\n"

                elif function_name == "simulate_book_flight":
                    output_str = tool_output_data.get("message", "Flight booking status unknown.")
                    display_message += output_str + "\n"
                elif function_name == "simulate_book_hotel":
                    output_str = tool_output_data.get("message", "Hotel booking status unknown.")
                    display_message += output_str + "\n"
                # Removed simulate_block_calendar, send_email, get_current_weather handling
                # elif function_name == "simulate_block_calendar":
                #     output_str = tool_output_data.get("message", "Calendar blocking status unknown.")
                #     display_message += output_str + "\n"
                # elif function_name == "send_email":
                #     output_str = tool_output_data.get("message", "Email sending status unknown.")
                #     display_message += output_str + "\n"
                # elif function_name == "get_current_weather":
                #     output_str = tool_output_data.get("message", "Weather retrieval status unknown.")
                #     display_message += output_str + "\n"
                elif function_name == "query_knowledge_graph":
                    if tool_output_data.get("status") == "success" and tool_output_data.get("data"):
                        output_str = "Knowledge Graph Query Result:\n" + json.dumps(tool_output_data["data"], indent=2)
                        display_message += output_str + "\n"
                    else:
                        output_str = tool_output_data.get("message", "Knowledge Graph query failed or returned no data.")
                        display_message += output_str + "\n"
                elif function_name == "get_relevant_documents":
                    if tool_output_data.get("status") == "success" and tool_output_data.get("documents"):
                        output_str = "Relevant Documents Found:\n" + "\n".join([doc[:150] + "..." for doc in tool_output_data["documents"]])
                        display_message += output_str + "\n"
                    else:
                        output_str = tool_output_data.get("message", "No relevant documents found.")
                        display_message += output_str + "\n"
                elif function_name == "add_documents_to_rag":
                    output_str = tool_output_data.get("message", "RAG update status unknown.")
                    display_message += output_str + "\n"
                elif function_name == "add_url_to_knowledge_base":
                    output_str = tool_output_data.get("message", "URL knowledge base addition status unknown.")
                    display_message += output_str + "\n"
                elif function_name == "add_flight_route_to_kg":
                    output_str = tool_output_data.get("message", "Flight route addition status unknown.")
                    display_message += output_str + "\n"
                elif function_name == "add_hotel_to_kg":
                    output_str = tool_output_data.get("message", "Hotel addition status unknown.")
                    display_message += output_str + "\n"
                elif function_name == "get_flight_booking_status" or function_name == "get_hotel_booking_status":
                    output_str = tool_output_data.get("message", "Booking status unknown.")
                    display_message += output_str + "\n"
                else:
                    output_str = f"Tool '{function_name}' executed. Output: {json.dumps(tool_output_data)}"
                    display_message += output_str + "\n"

                if not isinstance(output_str, str):
                    output_str = str(output_str) 

                if len(output_str) > 2000: 
                    output_str = output_str[:2000] + "... (truncated)"

                tool_outputs_for_llm.append({
                    "tool_call_id": tool_call.id, 
                    "role": "tool",
                    "content": output_str
                })
                logging.info(f"Tool '{function_name}' output (for LLM): {output_str}") 


            messages_for_llm.extend(tool_outputs_for_llm)

            final_chat_completion = groq_client.chat.completions.create(
                messages=messages_for_llm, 
                model=st["groq_model"],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )
            llm_response_content = final_chat_completion.choices[0].message.content
            
            if isinstance(llm_response_content, str) and (
                "<tool-use>" in llm_response_content or 
                "tool_calls" in llm_response_content and '{' in llm_response_content
            ):
                logging.warning(f"CRITICAL WARNING: LLM still outputted tool markup in final response despite strict parsing: {llm_response_content}")
                if display_message:
                    llm_response_content = "Operation completed. " + display_message.strip()
                else:
                    llm_response_content = "I have processed your request. However, I encountered an unexpected response format."
            elif not llm_response_content and display_message:
                 logging.warning("LLM's final response was empty. Using manually constructed display message.")
                 llm_response_content = display_message.strip()

            final_display_content = ""
            if display_message:
                final_display_content += display_message.strip()
                if llm_response_content and llm_response_content.strip() != display_message.strip():
                    final_display_content += "\n\n"
            
            final_display_content += llm_response_content.strip()

            if not final_display_content:
                 final_display_content = "Operation completed. No specific message generated for display."

            st["messages"].append({"role": "assistant", "content": final_display_content})
            logging.info(f"Final response content for display: {final_display_content}")
            return {"response": final_display_content, "role": "assistant"}

        else:
            llm_response_content = response_message.content
            if isinstance(llm_response_content, str) and (
                "<tool-use>" in llm_response_content or 
                "tool_calls" in llm_response_content and '{' in llm_response_content
            ):
                logging.warning(f"Direct LLM response contained tool markup: {llm_response_content}")
                return {"response": "I'm sorry, I encountered an unexpected response format. Please try rephrasing your request.", "role": "assistant"}
            logging.info(f"LLM direct response: {llm_response_content}")
            
            st["messages"].append({"role": "assistant", "content": llm_response_content})
            return {"response": llm_response_content, "role": "assistant"}

    except Exception as e:
        logging.error(f"Error during chat or tool execution: {e}")
        if "context_length_exceeded" in str(e):
            raise HTTPException(status_code=500, detail="Error: Conversation context too long. Please try a shorter query or start a new conversation.")
        elif "authentication" in str(e).lower() or "api_key" in str(e).lower():
            raise HTTPException(status_code=500, detail="Authentication error with Groq API. Please check your GROQ_API_KEY.")
        else:
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")




