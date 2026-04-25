import streamlit as st
import requests
import os

# Backend address - update this for cloud deployments
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Debugging Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Debugging Assistant")
st.markdown("Upload your codebase as a ZIP file, and ask questions like **'Find bugs in this code'** or **'Explain how user authentication works.'**")

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: File Upload & Management ---
with st.sidebar:
    st.header("📂 Codebase Details")
    uploaded_file = st.file_uploader("Upload Codebase (ZIP)", type=["zip"])
    
    if st.button("Process & Embed Database", type="primary"):
        if uploaded_file is not None:
            with st.spinner("📦 Extracting, chunking, and embedding files... This might take a minute."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/zip")}
                try:
                    res = requests.post(f"{backend_url}/upload-code", files=files)
                    if res.status_code == 200:
                        st.success(f"✅ Success! Generated {res.json().get('chunks_created')} chunks.")
                    else:
                        st.error(f"Server Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Failed: Ensure backend is running. ({e})")
        else:
            st.warning("Please select a ZIP file to upload.")

    st.divider()
    
    if st.button("View Indexed Files"):
        try:
            res = requests.get(f"{backend_url}/list-files")
            if res.status_code == 200:
                files = res.json().get("files", [])
                st.write(f"**Found {len(files)} parsed files:**")
                with st.expander("Show all files"):
                    for f in files:
                        st.text(f)
            else:
                st.error("Could not fetch file list.")
        except:
            st.error("Connection Failed. Backend might be offline.")

# --- Main Interface ---

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "contexts" in message and message["contexts"]:
            with st.expander("See Exact Code References"):
                for doc in message["contexts"]:
                    st.markdown(f"**File:** `{doc['source']}`")
                    st.code(doc['content'])

# User Query Input
if query := st.chat_input("E.g., What are the main issues in main.py?"):
    
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Get and display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Brainstorming & Searching codebase..."):
            try:
                payload = {"query": query}
                res = requests.post(f"{backend_url}/ask", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No response.")
                    sources = data.get("sources", [])
                    raw_contexts = data.get("context", [])
                    
                    st.markdown(answer)
                    
                    # Format context for UI
                    display_contexts = []
                    if raw_contexts:
                        with st.expander("See Exact Code References"):
                            for doc in raw_contexts:
                                source_val = doc.get('source_file')
                                content_val = doc.get('content')
                                st.markdown(f"**File:** `{source_val}`")
                                st.code(content_val)
                                display_contexts.append({"source": source_val, "content": content_val})
                    
                    # 3. Add to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "contexts": display_contexts
                    })
                    
                else:
                    st.error(f"Error from server: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
