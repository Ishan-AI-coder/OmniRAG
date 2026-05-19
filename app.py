import streamlit as st
import tempfile
import os
from langchain_core.messages import HumanMessage
from RAG import create_agent_for_pdf
import uuid

st.set_page_config(page_title="Research Paper Explorer", page_icon="📚", layout="wide")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# We add a new state to hold the compiled graph agent
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
# Create a unique thread ID for LangGraph memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- Sidebar: Upload & Config ---
with st.sidebar:
    st.title("⚙️ Setup Document")
    
    uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Reading PDF and building AI graph..."):
                try:
                    # 1. Save the Streamlit upload to a temporary physical file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # 2. Pass the file path to our backend function
                    st.session_state.active_agent = create_agent_for_pdf(temp_path)
                    
                    # 3. Clean up the temp file from the system
                    os.remove(temp_path)
                    
                    # 4. Clear old chat history for the new document
                    st.session_state.chat_history = []
                    st.success("Document processed! Agent is ready.")
                    
                except Exception as e:
                    st.error(f"Error processing document: {e}")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4()) 
        st.rerun()

# --- Main UI Area ---
st.title("📚 Research Paper Explorer")

# Gatekeeper: Don't show chat until an agent exists
if st.session_state.active_agent is None:
    st.info("👈 Please upload and process a PDF document in the sidebar to get started.")
else:
    # Render Existing Chat Messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Query Input Handling
    if user_query := st.chat_input("Ask a question about the document..."):
        
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document graph nodes..."):
                
                inputs = {"messages": [HumanMessage(content=user_query)]}

                # Define the config with the current thread_id
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                # Pass the config to invoke
                result = st.session_state.active_agent.invoke(inputs, config=config)
                final_message = result['messages'][-1]
                
                # Safe Parsing
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