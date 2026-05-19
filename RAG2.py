import os
import re
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

def create_multi_document_agent(pdf_data: list[dict]):
    """
    Accepts a list of dictionaries containing the temp path and original file name.
    Example: [{"path": "/tmp/file1.pdf", "name": "Attention_Paper.pdf"}, ...]
    """
    
    # 1. Initialize Models
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    tools = []
    
    # 2. Process each PDF into its own isolated vector store and tool
    for doc_info in pdf_data:
        pdf_path = doc_info["path"]
        doc_name = doc_info["name"]
        
        # Load and Split
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pages_split = text_splitter.split_documents(pages)

        # Create isolated collection. Clean the name to ensure it's a valid Chroma collection name.
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', doc_name)
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            collection_name=f"collection_{safe_name}"
        )
        
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # 3. Dynamic Tool Creation Wrapper
        # We use a closure wrapper to prevent Python's late-binding loop issue
        def create_search_func(r, n):
            def search_doc(query: str) -> str:
                docs = r.invoke(query)
                if not docs:
                    return f"I found no relevant information in {n}."
                
                results = []
                for doc in docs:
                    page_num = doc.metadata.get('page', 'Unknown Page')
                    results.append(f"--- Excerpt from {n} (Page {page_num}) ---\n{doc.page_content}")
                return "\n\n".join(results)
            return search_doc

        search_func = create_search_func(retriever, doc_name)

        # Create the LangChain tool
        tool = StructuredTool.from_function(
            func=search_func,
            name=f"search_{safe_name}",
            description=(
                f"Search the document named '{doc_name}'. "
                "Use this tool to find arguments, data, and theories specific to this paper. "
                "Input should be a highly specific search query."
            )
        )
        tools.append(tool)

    # 4. Bind all dynamic tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # 5. Define Graph State
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    system_prompt = f"""
        You are an advanced AI research assistant analyzing multiple research papers simultaneously.
        You have been provided with {len(tools)} distinct tools, each searching a specific document.

        Your primary responsibility is to answer questions using ONLY the information retrieved from these documents.
        You are exceptional at cross-referencing. If the user asks to compare theories, use MULTIPLE tools to query the different documents and synthesize the results.

        Guidelines:
        1. Base your answers strictly on the retrieved document context.
        2. Explicitly cite which document and page number your information comes from.
        3. If the retrieved information conflicts across papers, or if papers use different strategies, highlight the contrast clearly.
        4. For follow-up questions, use previous conversation context.

        When answering, use this format:
        Answer:
        <your detailed, comparative answer>

        Sources Used:
        - [Document 1 Name] - Page X
        - [Document 2 Name] - Page Y
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
    
    retriever_node = ToolNode(tools=tools)
    graph.add_node("retriever_agent", retriever_node)

    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)