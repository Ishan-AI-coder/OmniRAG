from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)# I want to minimize hallucination - temperature = 0 makes the model output more deterministic 

# Our Embedding Model - has to also be compatible with the LLM
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001" 
)


pdf_path = "sample_research_paper.pdf"


# Safety measure for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

persist_directory = "./chroma_db"
collection_name = "research_papers"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Sample_research_paper document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Sample_research_paper document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


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


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        # print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            # print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    # print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT ===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        final_message = result['messages'][-1]
        
        # Check if the content is a list (which contains text, signatures, extras, etc.)
        if isinstance(final_message.content, list):
            # Look for the dictionary inside the list that actually contains the text
            clean_text = ""
            for block in final_message.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    clean_text = block.get("text", "")
                    break
            print(clean_text)
        else:
            # If it's already a standard string, just print it normally
            print(final_message.content)


running_agent()