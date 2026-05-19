import streamlit as st
import tempfile
import os
import uuid
from langchain_core.messages import HumanMessage
from RAG4 import create_multi_document_agent

st.set_page_config(page_title="Multi-Paper Explorer", page_icon="📚", layout="wide")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- Sidebar: Upload & Config ---
with st.sidebar:
    st.title("⚙️ Setup Documents")
    
    uploaded_files = st.file_uploader("Upload Research Papers (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            
            # Create UI placeholders for the live progress bar
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            # The callback function we pass to the backend!
            def update_ui(progress_val: float, message: str):
                # Ensure value stays safely between 0.0 and 1.0
                safe_val = max(0.0, min(1.0, progress_val)) 
                progress_bar.progress(safe_val)
                status_text.text(message)

            try:
                pdf_data = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_data.append({
                            "path": tmp_file.name, 
                            "name": uploaded_file.name
                        })
                
                # Pass the files AND the callback to our fast backend
                st.session_state.active_agent = create_multi_document_agent(
                    pdf_data, 
                    progress_callback=update_ui
                )
                
                for doc in pdf_data:
                    os.remove(doc["path"])
                
                st.session_state.chat_history = []
                
                # Clear the progress bar and show success
                status_text.empty()
                progress_bar.empty()
                st.success(f"🚀 Successfully processed {len(uploaded_files)} documents! Agent is ready.")
                
            except Exception as e:
                st.error(f"Error processing documents: {e}")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4()) 
        st.rerun()

# --- Main UI Area ---
st.title("📚 Fast Multi-Paper Explorer")

if st.session_state.active_agent is None:
    st.info("👈 Please upload and process 2 or more PDF documents in the sidebar to get started.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question to compare theories..."):
        
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Querying documents..."):
                
                inputs = {"messages": [HumanMessage(content=user_query)]}
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                result = st.session_state.active_agent.invoke(inputs, config=config)
                final_message = result['messages'][-1]
                
                clean_text = ""
                if isinstance(final_message.content, list):
                    for block in final_message.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            clean_text = block.get("text", "")
                            break
                else:
                    clean_text = str(final_message.content)

                if clean_text:
                    st.markdown(clean_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": clean_text})
                else:
                    st.error("Empty response block returned.")