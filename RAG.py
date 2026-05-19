import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# We wrap the entire ingestion and graph creation in a function
def create_agent_for_pdf(pdf_path: str):
    
    # 1. Initialize Models
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # 2. Load and Split the PDF
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)

    # 3. Create an Ephemeral Vector Store (No persist_directory)
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        collection_name="dynamic_paper"
    )
    
    retriever = vectorstore.as_retriever(search_type= "similarity", search_kwargs={"k": 5})

    # 4. Define the Tool (Dynamically bound to the new retriever)
    @tool
    def retriever_tool(query: str) -> str:
        """Searches and returns information from the uploaded research paper."""
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the document."
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}")
        return "\n\n".join(results)

    tools = [retriever_tool]
    llm_with_tools = llm.bind_tools(tools)

    # 5. Define Graph State
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    system_prompt = """
        You are an advanced AI research and document analysis assistant.

        Your primary responsibility is to answer questions ONLY using the information retrieved from the loaded document knowledge base.

        You have access to a retrieval tool that searches the document and returns relevant passages. Always use the retriever tool whenever:
        - the user asks factual questions about the document
        - clarification or evidence is needed
        - you are uncertain about an answer
        - the question references events, themes, characters, equations, concepts, or arguments from the document

        Guidelines:
        1. Base your answers strictly on the retrieved document context.
        2. Do NOT invent information that is not supported by the retrieved text.
        3. If the retrieved information is insufficient, clearly say:
        "I could not find enough information in the document to answer confidently."
        4. If multiple retrieved passages conflict, explain the conflict instead of guessing.
        5. You may make multiple retrieval calls if necessary.
        6. For follow-up questions, use previous conversation context when relevant.
        7. Keep answers clear, structured, and academically accurate.
        8. When possible, explain reasoning step-by-step.
        9. Always mention the source page numbers if available.
        10. If the user asks something unrelated to the document, politely state that your knowledge is limited to the loaded document.

        When answering, use this format:

        Answer:
        <your detailed answer>

        Sources Used:
        - Page X
        - Page Y

        Confidence:
        - High / Medium / Low

        Behavior Rules:
        - Never fabricate citations.
        - Never pretend to know information not found in retrieval results.
        - Prefer precise answers over broad vague summaries.
        - Quote small relevant excerpts when useful.
        - If the user asks for summaries, provide concise but complete summaries.
        - If the user asks analytical questions, combine evidence from multiple retrieved sections.

        Your goal is to behave like a trustworthy research assistant, not a generic chatbot.
        """

    # 6. Define Nodes
    def call_llm(state: AgentState) -> AgentState:
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        message = llm_with_tools.invoke(messages)
        return {'messages': [message]}

    def should_continue(state: AgentState):
        result = state['messages'][-1]
        if hasattr(result, 'tool_calls') and len(result.tool_calls) > 0:
            return "retriever_agent"
        return END

    # 7. Build Graph
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    
    # Using the clean ToolNode!
    retriever_node = ToolNode(tools=tools)
    graph.add_node("retriever_agent", retriever_node)

    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    # ---- ADDING MEMORY CHECKPOINTER HERE ----
    memory = MemorySaver()

    # Compile the graph with the checkpointer
    return graph.compile(checkpointer=memory)