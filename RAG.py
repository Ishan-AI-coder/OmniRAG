import os
import re
import time
import random
import io
import base64
import concurrent.futures
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Langchain & AI Imports
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Hybrid & Reranking Imports
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_cohere import CohereRerank

load_dotenv(override=True)

# Prevent FastEmbed caching errors on Linux/WSL
os.environ["FASTEMBED_CACHE_PATH"] = "./model_cache"

def execute_python_plot_tool(python_code: str) -> str:
    """
    Use this tool when the user asks for a chart, graph, or plot.
    Input MUST be a valid, complete Python script using matplotlib.pyplot (as plt) or numpy (as np).
    The code must define the data arrays directly based on the context.
    DO NOT include plt.show(). DO NOT wrap in markdown backticks.
    """
    try:
        
        clean_code = python_code.replace("```python", "").replace("```", "").strip()
        plt.clf()
        plt.close('all')

        local_vars = {"plt": plt, "np": np}
        safe_globals = {"__builtins__": __builtins__}

        exec(clean_code, safe_globals, local_vars)

        buf = io.BytesIO()
        
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close('all')
        
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        print("[Python Tool] Success! Plot generated in memory.")
        
        return f"Successfully generated plot. [PLOT_BASE64:{img_base64}] Tell the user the diagram is displayed."

    except Exception as e:
        print(f"❌ [Python Tool Error]: {str(e)}")
        return f"Error executing plot code: {str(e)}. Please correct your Python code and try again."

python_plotter = StructuredTool.from_function(
    func=execute_python_plot_tool,
    name="generate_python_plot",
    description="Executes Python code to generate a matplotlib graph. Input must be raw, executable Python code containing data arrays extracted from the documents."
)


# --- MULTIMODAL EXTRACTION ---
def extract_multimodal_documents_fast(pdf_path: str, doc_name: str, llm, progress_callback=None):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    processed_docs = []
    images_to_process = []


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
                    "bytes": image_bytes, "page_num": page_num + 1, "img_idx": img_idx
                })

        if progress_callback:
            progress = (page_num / total_pages) * 0.20
            progress_callback(progress, f"[{doc_name}] Extracting text: Page {page_num + 1}...")


    total_images = len(images_to_process)
    if total_images > 0:
        def caption_single_image(img_data):
            encoded_image = base64.b64encode(img_data["bytes"]).decode("utf-8")
            caption_prompt = "You are an expert academic illustrator. Describe this image or chart from a research paper in meticulous detail."
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = llm.invoke([
                        SystemMessage(content=caption_prompt),
                        HumanMessage(content=[
                            {"type": "text", "text": "Please summarize this figure:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ])
                    ])
                    return Document(
                        page_content=f"[Visual Figure Summary]: {response.content}",
                        metadata={"source": f"{doc_name} - Page {img_data['page_num']}", "type": "image"}
                    )
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        sleep_time = (2 ** attempt) + random.uniform(1, 3)
                        time.sleep(sleep_time)
                    else:
                        return None
            return None

        completed_images = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
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


# --- LANGGRAPH AGENT SETUP ---
def create_multi_document_agent(pdf_data: list[dict], progress_callback=None):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    dense_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    sparse_embeddings = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")
    cohere_reranker = CohereRerank(model="rerank-english-v3.0", top_n=6)
    
    tools = [python_plotter]
    
    for doc_idx, doc_info in enumerate(pdf_data):
        pdf_path, doc_name = doc_info["path"], doc_info["name"]
        
        raw_documents = extract_multimodal_documents_fast(pdf_path, doc_name, llm, progress_callback)
        
        if progress_callback:
            progress_callback(1.0, f"[{doc_name}] Building Hybrid Index & Reranker...")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        final_chunks = []
        for doc in raw_documents:
            if doc.metadata["type"] == "text":
                final_chunks.extend(text_splitter.split_documents([doc]))
            else:
                final_chunks.append(doc)

        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', doc_name)
        qdrant_store = QdrantVectorStore.from_documents(
            documents=final_chunks,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            path="./qdrant_db",
            collection_name=f"hybrid_{safe_name}_{doc_idx}",
            retrieval_mode=RetrievalMode.HYBRID
        )
        
        base_hybrid_retriever = qdrant_store.as_retriever(search_kwargs={"k": 15})

        def create_search_func(retriever, reranker, n):
            def search_doc(query: str) -> str:
                raw_docs = retriever.invoke(query) 
                if not raw_docs:
                    return f"I found no relevant information in {n}."
                
                best_docs = reranker.compress_documents(documents=raw_docs, query=query)
                if not best_docs:
                    return f"No highly relevant information found in {n} after reranking."
                
                return "\n\n".join([f"--- Excerpt from {d.metadata.get('source', 'Unknown')} ---\n{d.page_content}" for d in best_docs])
            return search_doc

        tools.append(StructuredTool.from_function(
            func=create_search_func(base_hybrid_retriever, cohere_reranker, doc_name),
            name=f"search_{safe_name}",
            description=f"Search '{doc_name}' for keywords and data. Do NOT use this to generate diagrams; only use it to retrieve existing information."
        ))

    llm_with_tools = llm.bind_tools(tools)
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    system_prompt = f"""
        You are an advanced AI research assistant analyzing multiple research papers visually and textually.
        You have {len(tools)} highly advanced search and execution tools available.
        
        When a user asks for a chart, graph, or visualization:
        1. Search the documents to find the relevant numerical data.
        2. Use the 'generate_python_plot' tool.
        3. Pass fully written, executable Python code to the tool. 
        4. Hardcode the extracted data into arrays (e.g., x = [1, 2, 3]). 
        5. Use 'plt' (matplotlib.pyplot) to plot and style it professionally. DO NOT use plt.show().

        Base all answers and data strictly on retrieved context. Explicitly cite the document and page number.
    """

    def call_llm(state: AgentState) -> AgentState:
        return {'messages': [llm_with_tools.invoke([SystemMessage(content=system_prompt)] + list(state['messages']))]}

    def should_continue(state: AgentState):
        if hasattr(state['messages'][-1], 'tool_calls') and len(state['messages'][-1].tool_calls) > 0:
            return "retriever_agent"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", ToolNode(tools=tools))
    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    return graph.compile(checkpointer=MemorySaver())