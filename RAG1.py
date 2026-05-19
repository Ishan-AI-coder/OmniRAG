import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import fitz  # PyMuPDF
import io
import base64
from PIL import Image
from langchain_core.documents import Document

load_dotenv()

def extract_multimodal_documents(pdf_path: str, llm):
    """Parses a PDF using PyMuPDF. Extracts text and generates visual summaries for images."""
    doc = fitz.open(pdf_path)
    processed_docs = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 1. Extract Page Text
        text = page.get_text()
        if text.strip():
            processed_docs.append(Document(
                page_content=text,
                metadata={"source": f"Page {page_num + 1}", "type": "text"}
            ))
            
        # 2. Extract Images / Figures from the Page
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.width < 100 or pil_img.height < 100:
                continue
                
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            
            caption_prompt = (
                "You are an expert academic illustrator. Describe this image, chart, "
                "or table from a research paper in meticulous detail. Include specific data points, "
                "axis labels, trends, formulas, or text visible within the image."
            )
            
            try:
                caption_response = llm.invoke([
                    SystemMessage(content=caption_prompt),
                    HumanMessage(content=[
                        {"type": "text", "text": "Please summarize this figure:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ])
                ])
                image_description = caption_response.content
                
                processed_docs.append(Document(
                    page_content=f"[Visual Figure Summary]: {image_description}",
                    metadata={
                        "source": f"Page {page_num + 1}", 
                        "type": "image",
                        "image_base64": encoded_image
                    }
                ))
            except Exception as e:
                print(f"Skipping image {img_idx} on page {page_num} due to captioning error: {e}")
                
    return processed_docs

def create_agent_for_pdf(pdf_path: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    raw_documents = extract_multimodal_documents(pdf_path, llm)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    
    final_chunks = []
    for doc in raw_documents:
        if doc.metadata["type"] == "text":
            final_chunks.extend(text_splitter.split_documents([doc]))
        else:
            final_chunks.append(doc)

    vectorstore = Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        collection_name="dynamic_paper"
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    @tool
    def retriever_tool(query: str) -> str:
        """Searches and returns textual information or figures from the uploaded research paper."""
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the document."
        
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown Page")
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "image":
                results.append(f"Snippet {i+1} [IMAGE FOUND ON {source}]:\n{doc.page_content}")
            else:
                results.append(f"Snippet {i+1} [TEXT FROM {source}]:\n{doc.page_content}")
                
        return "\n\n".join(results)

    tools = [retriever_tool]
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    system_prompt = """
        You are an advanced AI research assistant capable of analyzing both text and visual diagrams.
        Your primary responsibility is to answer questions using the information retrieved from the document knowledge base.
        
        Guidelines:
        1. Base your answers strictly on the retrieved document context or visual summaries provided.
        2. Always mention the source page numbers.
        3. Formulate equations and data trends accurately based on descriptions.
        
        When answering, use this format:
        Answer:
        <your detailed answer>

        Sources Used:
        - Page X

        Confidence:
        - High / Medium / Low
    """

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

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    
    retriever_node = ToolNode(tools=tools)
    graph.add_node("retriever_agent", retriever_node)

    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)