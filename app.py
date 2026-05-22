import streamlit as st
import tempfile
import os
import uuid
import base64
import re
from langchain_core.messages import HumanMessage

from RAG import create_multi_document_agent

st.set_page_config(page_title="Multi-Paper Explorer", page_icon="📚", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.title("⚙️ Setup Documents")
    uploaded_files = st.file_uploader("Upload Research Papers (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_ui(progress_val: float, message: str):
                progress_bar.progress(max(0.0, min(1.0, progress_val)))
                status_text.text(message)

            try:
                pdf_data = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_data.append({"path": tmp_file.name, "name": uploaded_file.name})
                
                st.session_state.active_agent = create_multi_document_agent(pdf_data, progress_callback=update_ui)
                
                for doc in pdf_data:
                    os.remove(doc["path"])
                
                st.session_state.chat_history = []
                status_text.empty()
                progress_bar.empty()
                st.success(f"🚀 Architecture Deployed! Successfully processed {len(uploaded_files)} documents.")
                
            except Exception as e:
                st.error(f"Error processing documents: {e}")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4()) 
        st.rerun()

# --- Main Chat UI ---
st.title("📚 Multi-Paper Explorer")

if st.session_state.active_agent is None:
    st.info("👈 Please upload and process PDF documents in the sidebar to get started.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message.get("images"):
                cols = st.columns(len(message["images"]))
                for idx, img_bytes in enumerate(message["images"]):
                    with cols[idx]:
                        st.image(img_bytes, width=300)
            
            if message.get("generated_b64"):
                decoded_img_bytes = base64.b64decode(message["generated_b64"])
                st.image(decoded_img_bytes, caption="AI Generated Matplotlib Plot", use_container_width=True)


    with st.expander("📎 Attach images to your next question"):
        chat_images = st.file_uploader("Upload images (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if user_query := st.chat_input("Ask a question, request a plot, or analyze data..."):
        
        lc_content = [{"type": "text", "text": user_query}]
        saved_image_bytes = []
        
        if chat_images:
            for img_file in chat_images:
                img_bytes = img_file.getvalue()
                saved_image_bytes.append(img_bytes)
                b64_encoded = base64.b64encode(img_bytes).decode("utf-8")
                mime_type = img_file.type
                
                lc_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_encoded}"}
                })

        with st.chat_message("user"):
            st.markdown(user_query)
            if saved_image_bytes:
                cols = st.columns(len(saved_image_bytes))
                for idx, img_bytes in enumerate(saved_image_bytes):
                    with cols[idx]:
                        st.image(img_bytes, width=300)
                        
        history_entry = {"role": "user", "content": user_query}
        if saved_image_bytes:
            history_entry["images"] = saved_image_bytes
        st.session_state.chat_history.append(history_entry)


        with st.chat_message("assistant"):
            with st.spinner("Analyzing data and generating response..."):
                inputs = {"messages": [HumanMessage(content=lc_content)]}
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                result = st.session_state.active_agent.invoke(inputs, config=config)
                final_message = result['messages'][-1]
                
                generated_b64 = None
                for msg in reversed(result['messages']):
                    if msg.type == 'human':
                        break
                    if msg.type == 'tool' and isinstance(msg.content, str):
                        match = re.search(r"\[PLOT_BASE64:(.*?)\]", msg.content)
                        if match:
                            generated_b64 = match.group(1)
                            break
                
                clean_text = ""
                if isinstance(final_message.content, list):
                    for block in final_message.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            clean_text = block.get("text", "")
                            break
                else:
                    clean_text = str(final_message.content)

                if clean_text:
                    clean_text = re.sub(r"\[PLOT_BASE64:.*?\]", "", clean_text).strip()
                    
                    st.markdown(clean_text)
                    assistant_history_entry = {"role": "assistant", "content": clean_text}
                    
                    if generated_b64:
                        decoded_img_bytes = base64.b64decode(generated_b64)
                        st.image(decoded_img_bytes, caption="AI Generated Matplotlib Plot", use_container_width=True)
                        assistant_history_entry["generated_b64"] = generated_b64
                    
                    st.session_state.chat_history.append(assistant_history_entry)
                else:
                    st.error("Empty response block returned.")