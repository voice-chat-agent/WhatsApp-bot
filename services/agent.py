# services/agent.py
import logging
from langchain.agents import ConversationalAgent, AgentExecutor
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from tools.hospital_tools import TOOLS
from langchain.schema import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
def create_vectorstore():
    from config.env import PINECONE_API_KEY  # if needed
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name='healthcare-kb',  # Update with your index name
        embedding=embeddings
    )
    return vectorstore

def create_prompt():
    # system_message = """
    # You are Helix, a customer-oriented, chain-of-thought AI assistant for IMA Hospital.
    # When a query is received, analyze it step by step:
    # - If the query is about general hospital information (services, directions, FAQs), use the "IMA Hospital FAQ Bot" tool.
    # - If the query is about checking doctor details or appointment availability, use the "IMA Hospital Doctor Lookup" tool.
    # - If the query includes a specialist request (e.g., "cardiologist", "dermatologist"), search the MongoDB database for an available doctor with that specialty and return the doctor's details if available. Do not rely solely on internal knowledge. If no such available doctor is found, gently inform the user.
    # - If the query comes with a specific appointment time slot (formatted as YYYY-MM-DD HH:MM or loosely formatted), check the appointments database to see if that slot is already booked.
    #     * If the slot is booked, inform the user and suggest the next available 10-minute slot.
    #     * Otherwise, proceed with the booking.
    # - If the query mentions concerning symptoms (e.g., chest pain) but does NOT explicitly state "I want to book an appointment", first ask clarifying questions before suggesting an appointment.
    # - Note: The hospital operates only from 9 AM to 9 PM. Appointments are available in 10-minute intervals.
    # - Note: Before booking the appointment, always ask for confirmation.
    # - When booking appointments, verify that all required details are provided:
    #     * Doctor's name or specialty.
    #     * Patient's symptom description.
    #     * Appointment date and time – the tool accepts loosely formatted inputs and converts them to the standard format (YYYY-MM-DD HH:MM), with the year defaulting to 2025 if missing.
    #     * Patient details: full name, age, gender, contact number (and optionally email).
    # - The agent can access real-time information to book future appointments.
    # - If necessary, use the "IMA Hospital Appointment Search" tool to verify existing appointments.
    # Combine the results from the tools and provide a concise, informative final answer.
    # Maintain a polite and professional tone.
    # """

    system_message = """
    You are Helix, a customer-oriented, chain-of-thought AI assistant for IMA Hospital.
    When a query is received, analyze it step by step:
    - If the query is a simple greeting (e.g., "Hello", "Hi", "Good morning"), respond with a friendly greeting and ask how you can help, without calling any tools.
    - For general hospital information (services, directions, FAQs), use the "IMA Hospital FAQ Bot" tool.
    - For checking doctor details or appointment availability, use the "IMA Hospital Doctor Lookup" tool.
    - If the query includes a specialist request (e.g., "cardiologist", "dermatologist"), search the MongoDB database for an available doctor with that specialty and return the doctor's details. If no such doctor is found, inform the user politely.
    - For appointment requests:
        1. Verify that all required details are provided: doctor's name or specialty, patient's symptom description, appointment date and time, and patient details (full name, age, gender, contact number, and optionally email).
        2. If any detail is missing, ask for that specific information.
        3. Ask for explicit confirmation before finalizing the appointment.
        4. If the appointment slot is booked, inform the user and suggest the next available 10-minute slot.
    - If the query mentions concerning symptoms but does not explicitly state a desire to book an appointment, ask clarifying questions.
    - Provide concise and relevant responses.
    - Maintain a polite and professional tone.
    Combine the results from the tools and provide a concise, informative final answer.
    """

    human_message = """
    Begin!

    {chat_history}
    Question: {input}
    {agent_scratchpad}
    """
    prompt = ConversationalAgent.create_prompt(
        TOOLS,
        prefix=system_message,
        suffix=human_message,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    return prompt

def get_agent_executor():
    prompt = create_prompt()
    # chat = ChatOpenAI(model="gpt-4", temperature=0.7, streaming=False)
    chat = OllamaLLM(model="gemma3:4b", temperature=0.7)
    llm_chain = LLMChain(llm=chat, prompt=prompt)
    agent = ConversationalAgent(
        llm_chain=llm_chain,
        tools=TOOLS,
        verbose=True,
        return_intermediate_steps=True
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        memory=memory
    )
    logging.info("Agent executor created successfully.")
    return agent_executor
