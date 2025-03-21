# # endpoints/chat.py
# import uuid
# import logging
# from fastapi import APIRouter
# from models.chat import ChatRequest, ChatResponse
# from services.agent import get_agent_executor
# from langchain.schema import HumanMessage, AIMessage

# router = APIRouter()

# # A simple in-memory session store.
# session_histories = {}

# @router.get("/new_session")
# async def new_session():
#     session_id = str(uuid.uuid4())
#     session_histories[session_id] = []
#     logging.info("New session created with session_id: %s", session_id)
#     return {"session_id": session_id}

# @router.post("/chat/{session_id}", response_model=ChatResponse)
# async def chat_endpoint(session_id: str, chat_request: ChatRequest):
#     if session_id not in session_histories:
#         session_histories[session_id] = chat_request.chat_history

#     full_history = session_histories[session_id] + chat_request.chat_history

#     def convert_dict_to_message(item):
#         role = item.get("role")
#         content = item.get("content")
#         if role == "user":
#             return HumanMessage(content=content)
#         elif role == "assistant":
#             return AIMessage(content=content)
#         else:
#             return HumanMessage(content=content)
    
#     message_objects = [convert_dict_to_message(item) for item in full_history if isinstance(item, dict)]
#     agent_executor = get_agent_executor()
#     agent_executor.memory.chat_memory.messages = message_objects

#     response = agent_executor.run(chat_request.input)

#     updated_history = []
#     for msg in agent_executor.memory.chat_memory.messages:
#         role = "assistant" if isinstance(msg, AIMessage) else "user"
#         updated_history.append({"role": role, "content": msg.content})
    
#     session_histories[session_id] = updated_history
#     return ChatResponse(response=response, chat_history=updated_history)



# endpoints/chat.py
import uuid
import logging
from fastapi import APIRouter, Request
from models.chat import ChatRequest, ChatResponse
from services.agent import get_agent_executor
from langchain.schema import HumanMessage, AIMessage

router = APIRouter()

# Global in-memory session history store
session_histories = {}

def convert_history_to_messages(history):
    messages = []
    for item in history:
        role = "user" if item.get("role") == "user" else "assistant"
        messages.append(HumanMessage(content=item["content"]) if role=="user" else AIMessage(content=item["content"]))
    return messages

@router.get("/new_session")
async def new_session():
    session_id = str(uuid.uuid4())
    session_histories[session_id] = []  # initialize empty history
    logging.info("New session created with session_id: %s", session_id)
    return {"session_id": session_id}

@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat_endpoint(session_id: str, chat_request: ChatRequest, request: Request):
    # If no session history exists, initialize it
    if session_id not in session_histories:
        session_histories[session_id] = []
    
    # Append the new user message to the session history
    session_histories[session_id].append({"role": "user", "content": chat_request.input})
    
    # Retrieve the conversation history as message objects
    message_objects = convert_history_to_messages(session_histories[session_id])
    
    # Reuse the global agent executor (attached to the request state if using middleware) or create one here
    # For this example, we'll instantiate one (ensure your get_agent_executor() is optimized for reuse)
    agent_executor = get_agent_executor()
    agent_executor.memory.chat_memory.messages = message_objects
    
    # Run the agent to get the assistant's response
    response = agent_executor.run(chat_request.input)
    
    # Append the assistant's response to the session history
    session_histories[session_id].append({"role": "assistant", "content": response})
    
    return ChatResponse(response=response, chat_history=session_histories[session_id])
