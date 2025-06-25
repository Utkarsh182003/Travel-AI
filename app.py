# app.py
import streamlit as st
import requests

# FastAPI backend URL (adjust if running on a different port/host)
FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Workflow Assistant", layout="centered")

st.title("✈️🏨🗓️ AI Workflow Assistant")
st.markdown("""
Welcome to your AI Workflow Assistant!
I can help you book flights, hotels, and manage your calendar.
Try asking:
- "Book a flight from London to Paris on 2025-07-20 for 2 people."
- "Find a hotel in New York for 2025-08-10 to 2025-08-15 for 3 guests."
- "Block my calendar for a meeting from 2025-07-01 10:00 to 2025-07-01 11:30 named 'Project Sync'."
""")

# Initialize chat history in Streamlit session state
# Each message will be a dictionary with 'role' and 'content'
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages to send to backend (including full history for context)
    # Only send 'role' and 'content' for history to match Groq API message format
    messages_to_send = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]

    # Send user message and history to FastAPI backend
    with st.spinner("Processing your request..."):
        try:
            response = requests.post(
                f"{FASTAPI_URL}/chat",
                json={"message": prompt, "history": messages_to_send[:-1]} # Exclude the current prompt from history sent
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()
            llm_response_content = response_data.get("response", "No response from AI.")
            llm_response_role = response_data.get("role", "assistant") # Get the role from backend response

        except requests.exceptions.ConnectionError:
            llm_response_content = "Could not connect to the backend. Please ensure FastAPI is running."
            llm_response_role = "assistant"
        except requests.exceptions.RequestException as e:
            llm_response_content = f"An error occurred: {e}"
            llm_response_role = "assistant"

    # Display assistant response in chat history
    st.session_state.messages.append({"role": llm_response_role, "content": llm_response_content})
    with st.chat_message(llm_response_role):
        st.markdown(llm_response_content)

# Save the chat history to session state
st.session_state.messages = st.session_state.messages[-20:]  # Keep only the last 20 messages to avoid memory issues
# This helps manage memory usage in Streamlit and keeps the chat responsive