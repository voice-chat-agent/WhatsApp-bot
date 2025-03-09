# services/faq_tool.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from config.env import PINECONE_API_KEY  # if needed

def create_faq_tool():
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name='healthcare-kb',  # Update with your index name
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()
    chat = ChatOpenAI(model="gpt-4", temperature=0.7, streaming=False)
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)
    faq_tool = Tool(
        name="IMA Hospital FAQ Bot",
        func=qa.run,
        description="Use this tool for general hospital information such as services, directions, or FAQs."
    )
    return faq_tool
