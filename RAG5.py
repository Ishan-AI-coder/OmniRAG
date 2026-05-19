import os
import re
import concurrent.futures
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
import fitz  # PyMuPDF
import io
import base64
from PIL import Image

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# --- NEW HYBRID SEARCH IMPORTS ---
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

load_dotenv()

def extract_multimodal_documents_fast(pdf_path: str, doc_name: str, llm, progress_callback=None):
    """Parses text sequentially, then uses multithreading to caption images in parallel."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    processed_docs = []
    images_to_process = []
    
    # --- PASS 1: Instant Text Extraction & Image Gathering ---
    for page_num in range(total_pages):
        page = doc[page_num]
        
        text = page.get_text()
        if text.strip():
            processed_docs.append(Document(
                page_content=text,
                metadata={"source": f"{doc_name} - Page {page_num + 1}", "type": "text"}
            ))
            
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.width > 200 and pil_img.height > 200:
                images_to_process.append({
                    "bytes": image_bytes,
                    "page_num": page_num + 1,
                    "img_idx": img_idx
                })

        if progress_callback:
            progress = (page_num / total_pages) * 0.20
            progress_callback(progress, f"[{doc_name}] Extracting text: Page {page_num + 1}...")

    # --- PASS 2: Parallel Image Captioning ---
    total_images = len(images_to_process)
    
    if total_images > 0:
        def caption_single_image(img_data):
            encoded_image = base64.b64encode(img_data["bytes"]).decode("utf-8")
            caption_prompt = (
                "You are an expert academic illustrator. Describe this image, chart, "
                "or table from a research paper in meticulous detail."
            )
            try:
                caption_response = llm.invoke([
                    SystemMessage(content=caption_prompt),
                    HumanMessage(content=[
                        {"type": "text", "text": "Please summarize this figure:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ])
                ])
                return Document(
                    page_content=f"[Visual Figure Summary]: {caption_response.content}",
                    metadata={
                        "source": f"{doc_name} - Page {img_data['page_num']}", 
                        "type": "image"
                    }
                )
            except Exception as e:
                print(f"Error captioning image: {e}")
                return None

        completed_images = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_image = {executor.submit(caption_single_image, img): img for img in images_to_process}
            
            for future in concurrent.futures.as_completed(future_to_image):
                result = future.result()
                if result:
                    processed_docs.append(result)
                
                completed_images += 1
                if progress_callback:
                    progress = 0.20 + ((completed_images / total_images) * 0.80)
                    progress_callback(progress, f"[{doc_name}] Analyzing chart {completed_images} of {total_images}...")

    return processed_docs


def create_multi_document_agent(pdf_data: list[dict], progress_callback=None):
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 1. Initialize Open-Source Hybrid Embedding Models
    # First run will download these models (~2.5GB total) to your local machine
    dense_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5") 
    sparse_embeddings = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")
    
    tools = []
    
    for doc_idx, doc_info in enumerate(pdf_data):
        pdf_path = doc_info["path"]
        doc_name = doc_info["name"]
        
        # 2. Multimodal Fast Ingestion
        raw_documents = extract_multimodal_documents_fast(pdf_path, doc_name, llm, progress_callback)
        
        if progress_callback:
            progress_callback(1.0, f"[{doc_name}] Building Qdrant Hybrid Index...")

        # 3. Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        final_chunks = []
        for doc in raw_documents:
            if doc.metadata["type"] == "text":
                final_chunks.extend(text_splitter.split_documents([doc]))
            else:
                final_chunks.append(doc)

        # 4. Native Hybrid Vector Store (Qdrant)
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', doc_name)
        
        # This replaces Chroma entirely. It calculates both dense and sparse vectors natively.
        qdrant_store = QdrantVectorStore.from_documents(
            documents=final_chunks,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            location=":memory:", # Runs completely in RAM for extreme speed
            collection_name=f"hybrid_{safe_name}_{doc_idx}",
            retrieval_mode=RetrievalMode.HYBRID
        )
        
        # Qdrant handles the algorithm fusion automatically!
        hybrid_retriever = qdrant_store.as_retriever(search_kwargs={"k": 5})

        # 5. Create Tool Wrapper
        def create_search_func(r, n):
            def search_doc(query: str) -> str:
                docs = r.invoke(query)
                if not docs:
                    return f"I found no relevant information in {n}."
                results = []
                for doc in docs:
                    source = doc.metadata.get('source', 'Unknown Page')
                    results.append(f"--- Excerpt from {source} ---\n{doc.page_content}")
                return "\n\n".join(results)
            return search_doc

        search_func = create_search_func(hybrid_retriever, doc_name)

        tool = StructuredTool.from_function(
            func=search_func,
            name=f"search_{safe_name}",
            description=f"Search '{doc_name}' for exact keywords, arguments, data, and theories. Input a specific query."
        )
        tools.append(tool)

    # 6. Build LangGraph Agent
    llm_with_tools = llm.bind_tools(tools)
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    system_prompt = f"""
        You are an advanced AI research assistant analyzing multiple research papers visually and textually.
        You have {len(tools)} hybrid search tools available.
        Base answers strictly on retrieved context. Explicitly cite the document and page number.
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