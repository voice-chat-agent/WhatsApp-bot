# # endpoints/chat.py
# import uuid
# import logging
# from fastapi import APIRouter, Request
# from models.chat import ChatRequest, ChatResponse
# from services.agent import get_agent_executor
# from langchain.schema import HumanMessage, AIMessage

# router = APIRouter()

# # Global in-memory session history store
# session_histories = {}

# def convert_history_to_messages(history):
#     messages = []
#     for item in history:
#         role = "user" if item.get("role") == "user" else "assistant"
#         messages.append(HumanMessage(content=item["content"]) if role=="user" else AIMessage(content=item["content"]))
#     return messages

# @router.get("/new_session")
# async def new_session():
#     session_id = str(uuid.uuid4())
#     session_histories[session_id] = []  # initialize empty history
#     logging.info("New session created with session_id: %s", session_id)
#     return {"session_id": session_id}

# @router.post("/chat/{session_id}", response_model=ChatResponse)
# async def chat_endpoint(session_id: str, chat_request: ChatRequest, request: Request):
#     # If no session history exists, initialize it
#     if session_id not in session_histories:
#         session_histories[session_id] = []
    
#     # Append the new user message to the session history
#     session_histories[session_id].append({"role": "user", "content": chat_request.input})
    
#     # Retrieve the conversation history as message objects
#     message_objects = convert_history_to_messages(session_histories[session_id])
    
#     # Reuse the global agent executor (attached to the request state if using middleware) or create one here
#     # For this example, we'll instantiate one (ensure your get_agent_executor() is optimized for reuse)
#     agent_executor = get_agent_executor()
#     agent_executor.memory.chat_memory.messages = message_objects
    
#     # Run the agent to get the assistant's response
#     response = agent_executor.run(chat_request.input)
    
#     # Append the assistant's response to the session history
#     session_histories[session_id].append({"role": "assistant", "content": response})
    
#     return ChatResponse(response=response, chat_history=session_histories[session_id])
import uuid
import logging
from fastapi import APIRouter, Request
from models.chat import ChatRequest, ChatResponse
from services.agent import get_agent_executor
from langchain.schema import HumanMessage, AIMessage

router = APIRouter()

# Global dictionary to hold session histories.
session_histories = {}

@router.get("/new_session")
async def new_session():
    session_id = str(uuid.uuid4())
    session_histories[session_id] = []  # Initialize empty history.
    logging.info("New session created with session_id: %s", session_id)
    return {"session_id": session_id}

@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat_endpoint(request: Request, session_id: str, chat_request: ChatRequest):
    # Ensure there's a session history for this session ID
    if session_id not in session_histories:
        session_histories[session_id] = []
    history = session_histories[session_id]
    
    # Append the new user input to history.
    history.append({"role": "user", "content": chat_request.input})
    
    # Convert stored history into LangChain message objects.
    message_objects = []
    for item in history:
        if item["role"] == "user":
            message_objects.append(HumanMessage(content=item["content"]))
        else:
            message_objects.append(AIMessage(content=item["content"]))
    
    # Retrieve (or use a global) agent executor.
    agent_executor = get_agent_executor()  # Make sure this function returns a ready-to-use executor.
    agent_executor.memory.chat_memory.messages = message_objects
    
    # Get the agent response.
    response_text = agent_executor.run(chat_request.input)
    
    # Append the assistant's response to history.
    history.append({"role": "assistant", "content": response_text})
    session_histories[session_id] = history

    return ChatResponse(response=response_text, chat_history=history)
