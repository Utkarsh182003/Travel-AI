# # app.py
# import streamlit as st
# import requests
# import json 

# # FastAPI backend URL (adjust if running on a different port/host)
# FASTAPI_URL = "http://localhost:8000"

# st.set_page_config(page_title="AI Workflow Assistant", layout="centered")

# st.title("✈️🏨🗓️ AI Workflow Assistant")
# st.markdown("""
# Welcome to your AI Workflow Assistant!
# I can help you with travel planning, information retrieval, and calendar management.

# **Try asking:**
# - "Find flights from London (LHR) to Paris (CDG) for 2025-07-20."
# - "Show me hotels in New York from 2025-08-10 to 2025-08-15 for 2 guests with a pool."
# - "Now, book that flight." (after a flight search)
# - "Book the hotel 'The Grand London Hotel' in London for 2025-09-01 to 2025-09-05." (after a hotel search)
# - "What is the policy for international flights?" (RAG)
# - "List airports in Paris." (Knowledge Graph)
# - "Block my calendar for 'Team Sync' from 2025-07-01 10:00 to 2025-07-01 11:00."
# - "What is the status of my flight booking FKB0001?"
# - "What is the status of my hotel booking HKB0001?"
# - "Learn about the history of the Eiffel Tower from this page: `https://en.wikipedia.org/wiki/Eiffel_Tower` and call it 'Eiffel Tower History'." (Dynamic RAG)
# - "When was the Eiffel Tower completed?" (Querying dynamic RAG)
# - "Add a new flight route from Mumbai (BOM) to Dubai (DXB) with Emirates airline, flight number EK500, and a duration of 3.5 hours." (Dynamic KG Update)
# - "Store a new hotel called 'Grand Oasis Resort' in Dubai, 5 stars, it has a pool and wifi." (Dynamic KG Update)
# """)

# # Initialize chat history in Streamlit session state
# # Each message will be a dictionary with 'role' and 'content'
# # Tool outputs from the backend will be stored with role 'tool_output' for display purposes
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         # Only display content for 'user' and 'assistant' roles directly.
#         # 'tool_output' is for the LLM's internal context and should not be shown raw to the user.
#         if message["role"] in ["user", "assistant"]:
#             st.markdown(message["content"])
#         # If you want to debug and see tool outputs in the UI, you can uncomment this:
#         # elif message["role"] == "tool_output":
#         #     st.markdown("**Tool Output Received (for LLM context)**")
#         #     with st.expander("View Raw Tool Output"):
#         #         st.json(json.loads(message["content"]))


# # React to user input
# if prompt := st.chat_input("How can I help you today?"):
#     # Display user message in chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Prepare messages to send to backend (including full history for context)
#     # The LLM expects 'user', 'assistant', 'tool' roles.
#     # Convert 'tool_output' from Streamlit's session_state back to 'tool' for the backend/LLM.
#     messages_to_send_to_backend = []
#     for msg in st.session_state.messages:
#         if msg["role"] == "tool_output":
#             messages_to_send_to_backend.append({"role": "tool", "content": msg["content"]})
#         else:
#             messages_to_send_to_backend.append({"role": msg["role"], "content": msg["content"]})

#     # Send the history *excluding* the current user prompt (as it's sent in 'message' param)
#     history_for_llm = messages_to_send_to_backend[:-1]

#     # Send user message and history to FastAPI backend
#     with st.spinner("Processing your request..."):
#         try:
#             response = requests.post(
#                 f"{FASTAPI_URL}/chat",
#                 json={"message": prompt, "history": history_for_llm},
#                 timeout=30  # Set a timeout for the request
#             )
#             response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            
#             # Debugging line to see the raw response from the backend
#             print(f"Response from backend: {response.text}")  # Debugging line

#             # Parse the response JSON
#             response_data = response.json()
#             llm_response_content = response_data.get("response", "No response from AI.")
#             llm_response_role = response_data.get("role", "assistant")

#         except requests.exceptions.ConnectionError:
#             llm_response_content = "Could not connect to the backend. Please ensure FastAPI is running."
#             llm_response_role = "assistant"
#         except requests.exceptions.RequestException as e:
#             # Catch HTTP 500 error specifically for context length exceeded
#             if response is not None and response.status_code == 500 and "context_length_exceeded" in response.text:
#                 llm_response_content = "The conversation has become too long for the AI to process. Please try a shorter query or refresh the page to start a new conversation."
#             else:
#                 llm_response_content = f"An error occurred: {e}"
#             llm_response_role = "assistant"

#     # Display assistant response in chat history
#     st.session_state.messages.append({"role": llm_response_role, "content": llm_response_content})
#     with st.chat_message(llm_response_role):
#         st.markdown(llm_response_content)


# app.py
import streamlit as st
import requests
import json 

# FastAPI backend URL (adjust if running on a different port/host)
FASTAPI_URL = "http://localhost:8000"

# Configure the Streamlit page
st.set_page_config(page_title="AI Workflow Assistant", layout="centered")

st.title("✈️🏨🗓️ AI Workflow Assistant")
st.markdown("""
Welcome to your AI Workflow Assistant!
I can help you with travel planning, information retrieval, and calendar management.

**Try asking:**
- "Find flights from London (LHR) to Paris (CDG) for 2025-07-20."
- "Show me hotels in New York from 2025-08-10 to 2025-08-15 for 2 guests with a pool."
- "Now, book that flight." (after a flight search)
- "Book the hotel 'The Grand London Hotel' in London for 2025-09-01 to 2025-09-05." (after a hotel search)
- "What is the policy for international flights?" (RAG)
- "List airports in Paris." (Knowledge Graph)
- "Block my calendar for 'Team Sync' from 2025-07-01 10:00 to 2025-07-01 11:00."
- "What is the status of my flight booking FKB0001?"
- "What is the status of my hotel booking HKB0001?"
- "Learn about the history of the Eiffel Tower from this page: `https://en.wikipedia.org/wiki/Eiffel_Tower` and call it 'Eiffel Tower History'." (Dynamic RAG)
- "When was the Eiffel Tower completed?" (Querying dynamic RAG)
- "Add a new flight route from Mumbai (BOM) to Dubai (DXB) with Emirates airline, flight number EK500, and a duration of 3.5 hours." (Dynamic KG Update)
- "Store a new hotel called 'Grand Oasis Resort' in Dubai, 5 stars, it has a pool and wifi." (Dynamic KG Update)
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Only display content for 'user' and 'assistant' roles directly.
        # 'tool' role messages are for the LLM's internal context and are handled by the backend.
        if message["role"] in ["user", "assistant"]:
            st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages to send to backend for LLM context
    # The backend expects 'user', 'assistant', and 'tool' roles for history.
    # We send the full history *excluding* the current user prompt (as it's sent in 'message' param).
    messages_for_backend_history = []
    for msg in st.session_state.messages[:-1]: # Exclude the current user prompt
        # Ensure only roles the LLM understands are sent as history
        if msg["role"] in ["user", "assistant", "tool"]:
            messages_for_backend_history.append({"role": msg["role"], "content": msg["content"]})
        # If the backend sends structured tool calls, they might be stored differently.
        # For simplicity, we assume the backend summarizes tool outputs into 'assistant' role messages.

    # Process user input
    with st.spinner("Processing your request..."):
        try:
            # Send request to backend
            response = requests.post(
                f"{FASTAPI_URL}/chat",
                json={
                    "message": prompt,
                    "history": messages_for_backend_history # Send the prepared history
                },
                timeout=60 # Increased timeout for potentially longer LLM/tool operations
            )
            
            # Debug logging
            print(f"Request sent to backend with prompt: {prompt}")
            print(f"Response status code: {response.status_code}")
            print(f"Raw response: {response.text}")
            
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # If status code is 200, parse the JSON response
            response_data = response.json()
            print("Full response_data:", json.dumps(response_data, indent=2))

            # The backend is designed to send the final, human-readable response
            # in the 'response' field.
            assistant_response = response_data.get("response", "No response from AI.")
            llm_response_role = response_data.get("role", "assistant")

            # Add assistant's response to chat history and display it
            st.session_state.messages.append({
                "role": llm_response_role,
                "content": assistant_response
            })
            with st.chat_message(llm_response_role):
                st.markdown(assistant_response)

        except requests.exceptions.ConnectionError:
            error_msg = "⚠️ Could not connect to the backend. Please ensure the server is running."
            st.error(error_msg)
            print(error_msg)
            
        except requests.exceptions.Timeout:
            error_msg = "⚠️ Request timed out. Please try again."
            st.error(error_msg)
            print(error_msg)
            
        except requests.exceptions.RequestException as e:
            # Catch HTTP 500 error specifically for context length exceeded
            if response is not None and response.status_code == 500 and "context_length_exceeded" in response.text:
                error_msg = "The conversation has become too long for the AI to process. Please try a shorter query or refresh the page to start a new conversation."
            else:
                error_msg = f"⚠️ An error occurred: {str(e)}"
            st.error(error_msg)
            print(f"Request exception: {str(e)}")
            
        except json.JSONDecodeError as e:
            error_msg = f"⚠️ Error parsing response from backend: {str(e)}. Raw response: {response.text}"
            st.error(error_msg)
            print(error_msg)

        except Exception as e:
            error_msg = f"⚠️ Unexpected error: {str(e)}"
            st.error(error_msg)
            print(f"Unexpected error: {str(e)}")

# Add a clear chat button in the sidebar
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

