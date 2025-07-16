# main.py
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq
import uvicorn
import datetime
import httpx # For fetching URL content
from bs4 import BeautifulSoup # For parsing HTML content
import smtplib # For sending emails (if needed)
from email.mime.text import MIMEText # For creating email messages

from knowledge_base import initialize_rag_system, get_relevant_documents, add_documents_to_rag 
from neo4j_utils import get_neo4j_driver, clear_and_load_sample_data, query_knowledge_graph, add_flight_route_to_kg, add_hotel_to_kg
from external_apis import simulate_flight_search, simulate_search_hotels 

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

# In-memory storage for mock bookings (for demonstration purposes)
mock_flight_bookings = {} # Stores {booking_id: {details}}
mock_hotel_bookings = {} # Stores {booking_id: {details}}
next_flight_booking_id = 1
next_hotel_booking_id = 1

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

# --- function for sending email to the user ---
def send_booking_email(to_email: str, subject: str, body: str):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_server or not smtp_user or not smtp_password:
        print("SMTP credentials missing.")
        return

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = to_email
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], msg.as_string())
        print(f"Booking email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")


# --- Pydantic Models for Tool Inputs ---
class BookFlightParams(BaseModel):
    origin: str = Field()
    destination: str = Field()
    departure_date: str = Field()
    num_passengers: int = Field(default=1)
    flight_number: str | None = Field(default=None)
    email: str =Field()

class BookHotelParams(BaseModel):
    city: str = Field()
    check_in_date: str = Field()
    check_out_date: str = Field()
    num_guests: int = Field(default=1)
    hotel_id: str | None = Field(default=None)
    hotel_name: str | None = Field(default=None)
    price_per_night: float | None = Field(default=None)
    email: str= Field()

class BlockCalendarParams(BaseModel):
    event_name: str = Field()
    start_time: str = Field()
    end_time: str = Field()

class QueryKnowledgeGraphParams(BaseModel):
    cypher_query: str = Field()

class SimulateFlightSearchParams(BaseModel):
    origin_iata: str = Field()
    destination_iata: str = Field()
    flight_date: str = Field()

class SearchHotelsParams(BaseModel):
    city: str = Field()
    check_in_date: str = Field()
    check_out_date: str = Field()
    num_guests: int = Field(default=1)
    amenities: list[str] = Field(default_factory=list)

class GetFlightBookingStatusParams(BaseModel):
    booking_id: str = Field()

class GetHotelBookingStatusParams(BaseModel):
    booking_id: str = Field()

class AddUrlToKnowledgeBaseParams(BaseModel):
    url: str = Field()
    source_name: str = Field()

class AddFlightRouteToKgParams(BaseModel):
    origin_iata: str = Field()
    destination_iata: str = Field()
    airline_name: str = Field()
    flight_number: str | None = Field(default=None)
    duration_hours: float | None = Field(default=None)

class AddHotelToKgParams(BaseModel):
    hotel_name: str = Field()
    city: str = Field()
    stars: int = Field()
    has_pool: bool = Field(default=False)
    has_wifi: bool = Field(default=False)


# --- Simulated External Tools (Python functions) ---
def simulate_book_flight(origin: str, destination: str, departure_date: str, num_passengers: int = 1, flight_number: str | None = None, email: str | None = None):
    """
    Simulates booking a flight and stores it in mock_flight_bookings.
    Includes validation for future dates.
    """
    try:
        dep_date_obj = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()
        if dep_date_obj < datetime.date.today():
            return {"status": "error", "message": f"Cannot book a flight for a past date: {departure_date}. Please provide a future date."}
    except ValueError:
        return {"status": "error", "message": "Invalid departure date format. Please use YYYY-MM-DD."}

    global next_flight_booking_id
    booking_id = f"FKB{next_flight_booking_id:04d}"
    next_flight_booking_id += 1
    booking_details = {
        "origin": origin, "destination": destination, "departure_date": departure_date,
        "num_passengers": num_passengers, "flight_number": flight_number if flight_number else "Not Specified",
        "status": "CONFIRMED", "booking_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    mock_flight_bookings[booking_id] = booking_details
    print(f"Simulated flight booking stored: {booking_id} -> {booking_details}")
    if email:
        subject = f"Flight Booking Confirmation: {origin} to {destination}"
        body = f"Your flight from {origin} to {destination} on {departure_date} for {num_passengers} passengers has been booked successfully.\n\nBooking ID: {booking_id}\nFlight Number: {flight_number if flight_number else 'N/A'}\n\nThank you for choosing our service!"
        send_booking_email(email, subject, body)
    return {"status": "success", "message": f"Flight from {origin} to {destination} on {departure_date} for {num_passengers} passengers (Flight No: {flight_number if flight_number else 'N/A'}) booked successfully (simulated). Your booking ID is {booking_id}.", "booking_id": booking_id, "details": booking_details}

def simulate_book_hotel(city: str, check_in_date: str, check_out_date: str, num_guests: int = 1, hotel_id: str | None = None, hotel_name: str | None = None, price_per_night: float | None = None, email: str | None = None):
    """
    Simulates booking a hotel and stores it in mock_hotel_bookings.
    Includes validation for future dates.
    """
    try:
        check_in_date_obj = datetime.datetime.strptime(check_in_date, "%Y-%m-%d").date()
        check_out_date_obj = datetime.datetime.strptime(check_out_date, "%Y-%m-%d").date()
        if check_in_date_obj < datetime.date.today():
            return {"status": "error", "message": f"Cannot book a hotel for a past check-in date: {check_in_date}. Please provide a future date."}
        if check_out_date_obj <= check_in_date_obj:
            return {"status": "error", "message": "Check-out date must be after check-in date."}
    except ValueError:
        return {"status": "error", "message": "Invalid date format for check-in/out. Please use YYYY-MM-DD."}

    global next_hotel_booking_id
    booking_id = f"HKB{next_hotel_booking_id:04d}"
    next_hotel_booking_id += 1
    booking_details = {
        "city": city, "check_in_date": check_in_date, "check_out_date": check_out_date,
        "num_guests": num_guests, "hotel_id": hotel_id if hotel_id else "Not Specified",
        "hotel_name": hotel_name if hotel_name else "Not Specified", "price_per_night": price_per_night,
        "status": "CONFIRMED", "booking_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    mock_hotel_bookings[booking_id] = booking_details
    print(f"Simulated hotel booking stored: {booking_id} -> {booking_details}")
    if email:
        subject = f"Hotel Booking Confirmation in {city}"
        body = f"Your hotel in {city} from {check_in_date} to {check_out_date} for {num_guests} guests has been booked successfully.\n\nBooking ID: {booking_id}\nHotel Name: {hotel_name if hotel_name else 'N/A'}\nPrice per Night: {price_per_night if price_per_night else 'N/A'}\n\nThank you for choosing our service!"
        send_booking_email(email, subject, body)
    return {"status": "success", "message": f"Hotel '{hotel_name if hotel_name else 'N/A'}' in {city} from {check_in_date} to {check_out_date} for {num_guests} guests booked successfully (simulated). Your booking ID is {booking_id}.", "booking_id": booking_id, "details": booking_details}

def simulate_block_calendar(event_name: str, start_time: str, end_time: str):
    print(f"Simulating calendar block: '{event_name}' from {start_time} to {end_time}.")
    try:
        start_time_obj = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        end_time_obj = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M")
        if start_time_obj < datetime.datetime.now():
            return {"status": "error", "message": f"Cannot block calendar for a past start time: {start_time}. Please provide a future date/time."}
        if end_time_obj <= start_time_obj:
            return {"status": "error", "message": "End time must be after start time."}
    except ValueError:
        return {"status": "error", "message": "Invalid date/time format. Please use YYYY-MM-DD HH:MM."}

    if not event_name:
        return {"status": "error", "message": "Event name cannot be empty."}
    return {"status": "success", "message": f"Calendar blocked for '{event_name}' from {start_time} to {end_time} (simulated)."}

def get_flight_booking_status(booking_id: str):
    booking = mock_flight_bookings.get(booking_id)
    if booking:
        return {"status": "success", "booking": booking}
    else:
        return {"status": "error", "message": f"Flight booking with ID '{booking_id}' not found."}

def get_hotel_booking_status(booking_id: str):
    booking = mock_hotel_bookings.get(booking_id)
    if booking:
        return {"status": "success", "booking": booking}
    else:
        return {"status": "error", "message": f"Hotel booking with ID '{booking_id}' not found."}

async def add_url_to_knowledge_base(url: str, source_name: str):
    """
    Fetches content from a given URL, extracts text, and adds it to the RAG knowledge base.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text(separator='\n', strip=True)
            
            formatted_content = f"--- Content from {source_name} ({url}) ---\n\n{text_content}"

            success = add_documents_to_rag(formatted_content, source=url)
            if success:
                return {"status": "success", "message": f"Content from {url} (Source: {source_name}) successfully added to knowledge base."}
            else:
                return {"status": "error", "message": "Failed to add content to knowledge base after fetching."}
    except httpx.RequestError as exc:
        return {"status": "error", "message": f"Network error fetching URL {url}: {exc}"}
    except httpx.HTTPStatusError as exc:
        return {"status": "error", "message": f"HTTP error fetching URL {url}: {exc.response.status_code} - {exc.response.text}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred while processing URL {url}: {e}"}


# --- Tools definition for Groq LLM ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "simulate_book_flight",
            "description": "Book flight.",
            "parameters": BookFlightParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_book_hotel",
            "description": "Book hotel.",
            "parameters": BookHotelParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_block_calendar",
            "description": "Block calendar time.",
            "parameters": BlockCalendarParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "Query Neo4j KG.",
            "parameters": QueryKnowledgeGraphParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_flight_search",
            "description": "Search flights.",
            "parameters": SimulateFlightSearchParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_search_hotels",
            "description": "Search hotels.",
            "parameters": SearchHotelsParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_booking_status",
            "description": "Get flight status.",
            "parameters": GetFlightBookingStatusParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_hotel_booking_status",
            "description": "Get hotel status.",
            "parameters": GetHotelBookingStatusParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_url_to_knowledge_base",
            "description": "Add URL to RAG.",
            "parameters": AddUrlToKnowledgeBaseParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_flight_route_to_kg",
            "description": "Add flight route to KG.",
            "parameters": AddFlightRouteToKgParams.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_hotel_to_kg",
            "description": "Add hotel to KG.",
            "parameters": AddHotelToKgParams.model_json_schema()
        }
    }
]

# --- FastAPI Chat Endpoint with Tool Calling, RAG, and KG Logic ---
class ChatMessageInput(BaseModel):
    message: str
    history: list[dict] = []

@app.post("/chat")
async def chat_with_llm(chat_input: ChatMessageInput):
    user_message = chat_input.message
    chat_history = chat_input.history

    print(f"Received message from frontend: {user_message}")
    print(f"Chat history: {chat_history}")

    # --- RAG Step ---
    retrieved_docs = get_relevant_documents(user_message, rag_retriever, k=1) 
    rag_context = ""
    if retrieved_docs:
        rag_context = "\n\nRelevant Information from Knowledge Base:\n"
        for i, doc in enumerate(retrieved_docs):
            # Further reduced truncation length to 100 characters
            truncated_content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            rag_context += f"--- Document {i+1} ---\n{truncated_content}\n\n"
        print(f"RAG Context added (truncated):\n{rag_context}")

    # Construct the message for the LLM, including RAG context in the user's prompt
    full_user_message_for_llm = user_message + rag_context

    # The system prompt is crucial for guiding the LLM's behavior.
    # Drastically condensed system prompt
    system_prompt = (
        "You are an AI Workflow Assistant. "
        "Use tools for travel (flights, hotels) and calendar. "
        "**Strict Rules:**\n"
        "1.  **Tool Use:** Prioritize tools for clear requests. Manage multi-step tasks (search then book).\n"
        "2.  **Parameters:** Use ONLY defined tool parameters. DO NOT invent parameters.\n"
        "3.  **Disambiguation:** If tool parameters are missing/unclear, ASK specific, polite questions. Specify formats (YYYY-MM-DD, HH:MM, IATA codes).\n"
        "4.  **Constraints:** After search, check results against user constraints (e.g., price, amenities). Report if no results meet ALL criteria, and offer alternatives.\n"
        "5.  **Flow:** Search before booking. Use search results for booking parameters. Book directly if all details provided. Assume current year 2025; DO NOT book past dates.\n"
        "6.  **Tool Output:** Use tool output for responses. ALWAYS generate a natural language response for the user. NEVER output tool call markup (such as /tool-use>, <tool-use>, or JSON tool calls) or any raw JSON. ONLY output user-facing text. If you need to call a tool, do so using the function call mechanism, not by outputting tool call markup.\n"
        "7.  **Calendar/Status:** Use `simulate_block_calendar`, `get_flight_booking_status`, `get_hotel_booking_status`.\n"
        "8.  **KG Query:** Use `query_knowledge_graph` for specific structured data (Cypher queries only).\n"
        "9.  **RAG:** Use 'Relevant Information' for general knowledge/policies.\n"
        "10. **Dynamic Knowledge (URL):** Use `add_url_to_knowledge_base` (ask for URL & source name).\n"
        "11. **Dynamic Knowledge (KG):** Use `add_flight_route_to_kg` or `add_hotel_to_kg` for new structured flight/hotel data. Confirm details.\n"
        "12. **Conversation:** Maintain flow, remember previous turns.\n"
        "13. For any booking (flight or hotel), ALWAYS include the user's email address in the tool call parameters if provided in the user's message. If not provided, ask the user for their email before booking."
    )

    # Reduced history turns to 2 (1 user/assistant pair)
    MAX_HISTORY_TURNS = 2
    truncated_chat_history = chat_history[-MAX_HISTORY_TURNS:]

    messages_for_llm = [{"role": "system", "content": system_prompt}] + truncated_chat_history
    messages_for_llm.append({"role": "user", "content": full_user_message_for_llm})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages_for_llm,
            model="llama3-8b-8192",
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=2048, # Increased max_tokens
            top_p=1,
            stop=None,
            stream=False,
        )

        response_message = chat_completion.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            print(f"LLM proposed tool calls: {tool_calls}")
            messages_for_llm.append(response_message)

            tool_outputs_for_llm = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"Executing tool: {function_name} with args: {function_args}")

                tool_output_data = {"status": "error", "message": f"Unknown tool: {function_name}"}

                if function_name == "simulate_book_flight":
                    tool_output_data = simulate_book_flight(**function_args)
                elif function_name == "simulate_book_hotel":
                    tool_output_data = simulate_book_hotel(**function_args)
                elif function_name == "simulate_block_calendar":
                    tool_output_data = simulate_block_calendar(**function_args)
                elif function_name == "query_knowledge_graph":
                    cypher_query = function_args.get("cypher_query")
                    if cypher_query:
                        tool_output_data = query_knowledge_graph(cypher_query)
                    else:
                        tool_output_data = {"status": "error", "message": "Missing 'cypher_query' for knowledge graph tool."}
                elif function_name == "simulate_flight_search":
                    tool_output_data = simulate_flight_search(**function_args)
                elif function_name == "simulate_search_hotels":
                    tool_output_data = simulate_search_hotels(**function_args)
                elif function_name == "get_flight_booking_status":
                    tool_output_data = get_flight_booking_status(**function_args)
                elif function_name == "get_hotel_booking_status":
                    tool_output_data = get_hotel_booking_status(**function_args)
                elif function_name == "add_url_to_knowledge_base":
                    tool_output_data = await add_url_to_knowledge_base(**function_args)
                elif function_name == "add_flight_route_to_kg":
                    tool_output_data = add_flight_route_to_kg(**function_args)
                elif function_name == "add_hotel_to_kg":
                    tool_output_data = add_hotel_to_kg(**function_args)

                # Truncate tool output content if it's too long
                output_str = json.dumps(tool_output_data)
                if len(output_str) > 500: # Limit tool output content to 500 characters
                    output_str = output_str[:500] + "... (truncated)"

                tool_outputs_for_llm.append({
                    "tool_call_id": tool_call.id,
                    "output": output_str
                })
                print(f"Tool '{function_name}' output: {tool_output_data}")

            for output_item in tool_outputs_for_llm:
                messages_for_llm.append(
                    {
                        "tool_call_id": output_item["tool_call_id"],
                        "role": "tool",
                        "content": output_item["output"],
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
            llm_response_content = response_message.content
            print(f"LLM direct response: {llm_response_content}")
            return {"response": llm_response_content, "role": "assistant"}

    except Exception as e:
        print(f"Error during chat or tool execution: {e}")
        if "context_length_exceeded" in str(e):
            raise HTTPException(status_code=500, detail="Error: Conversation context too long. Please try a shorter query or start a new conversation.")
        else:
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


# To run this file use:
# uvicorn main:app --reload
