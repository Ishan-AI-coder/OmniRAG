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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.documents import Document

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
        
        # Extract Text
        text = page.get_text()
        if text.strip():
            processed_docs.append(Document(
                page_content=text,
                metadata={"source": f"{doc_name} - Page {page_num + 1}", "type": "text"}
            ))
            
        # Gather Images (Don't process them yet!)
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Filter out tiny logos/lines to save API calls
            if pil_img.width > 200 and pil_img.height > 200:
                images_to_process.append({
                    "bytes": image_bytes,
                    "page_num": page_num + 1,
                    "img_idx": img_idx
                })

        # Update UI during fast text pass (occupies first 20% of progress bar)
        if progress_callback:
            progress = (page_num / total_pages) * 0.20
            progress_callback(progress, f"[{doc_name}] Extracting text: Page {page_num + 1}...")

    # --- PASS 2: Parallel Image Captioning ---
    total_images = len(images_to_process)
    
    if total_images > 0:
        def caption_single_image(img_data):
            """Isolated function for the thread pool to execute."""
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

        # Execute API calls in parallel (max 5 at a time to prevent rate limits)
        completed_images = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # We use submit() instead of map() so we can update the progress bar as each thread finishes
            future_to_image = {executor.submit(caption_single_image, img): img for img in images_to_process}
            
            for future in concurrent.futures.as_completed(future_to_image):
                result = future.result()
                if result:
                    processed_docs.append(result)
                
                completed_images += 1
                if progress_callback:
                    # Maps the remaining 80% of the progress bar
                    progress = 0.20 + ((completed_images / total_images) * 0.80)
                    progress_callback(progress, f"[{doc_name}] Analyzing chart {completed_images} of {total_images}...")

    return processed_docs


def create_multi_document_agent(pdf_data: list[dict], progress_callback=None):
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    tools = []
    
    for doc_idx, doc_info in enumerate(pdf_data):
        pdf_path = doc_info["path"]
        doc_name = doc_info["name"]
        
        # 1. Multimodal Fast Ingestion
        raw_documents = extract_multimodal_documents_fast(pdf_path, doc_name, llm, progress_callback)
        
        # Update UI: Moving to embedding phase
        if progress_callback:
            progress_callback(1.0, f"[{doc_name}] Embedding vectors into ChromaDB...")

        # 2. Split text chunks but keep image summaries whole
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        final_chunks = []
        for doc in raw_documents:
            if doc.metadata["type"] == "text":
                final_chunks.extend(text_splitter.split_documents([doc]))
            else:
                final_chunks.append(doc)

        # 3. Create Hybrid Retrievers
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', doc_name)
        vectorstore = Chroma.from_documents(
            documents=final_chunks,
            embedding=embeddings,
            collection_name=f"collection_{safe_name}_{doc_idx}" 
        )
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        bm25_retriever = BM25Retriever.from_documents(final_chunks)
        bm25_retriever.k = 3

        # --- C. ENSEMBLE RETRIEVER (Hybrid Search)(written by reciprocal rank fusion method) ---
        def weighted_rrf(results: dict, k=60) -> list[Document]:
            # Define your weights here to match the old EnsembleRetriever
            weights = {"bm25": 0.4, "chroma": 0.6}
            fused_scores = {}
            
            # We iterate through the dictionary so we know which retriever we are scoring
            for retriever_name, docs in results.items():
                weight = weights[retriever_name]
                
                for rank, doc in enumerate(docs):
                    doc_str = doc.page_content
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = {"doc": doc, "score": 0.0}
                    
                    # Multiply the standard RRF score by the specific retriever's weight!
                    fused_scores[doc_str]["score"] += weight * (1.0 / (rank + k))
                    
            reranked_results = [
                item["doc"] for item in sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
            ]
            return reranked_results[:5]

        # 2. Run both retrievers simultaneously
        parallel_retrieval = RunnableParallel(
            bm25=bm25_retriever,
            chroma=chroma_retriever
        )
        
        # 3. Pass the resulting dictionary directly to our new weighted function
        hybrid_retriever = parallel_retrieval | RunnableLambda(weighted_rrf)

        
        # 4. Create Tool Wrapper
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

    # 5. Build Graph
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