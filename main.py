import streamlit as st
import os
from rag_qa import load_and_split_docs, create_vector_store, get_qa_chain
from dotenv import load_dotenv

load_dotenv()

os.environ["STREAMLIT_WATCH_FILES"] = "false"

st.set_page_config(page_title="📄 Document Q&A with Claude", layout="centered")
st.title("📄 Chat with Your Document")

# --- Sidebar: Upload Document ---
st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# --- Session State ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Handle File Upload ---
if uploaded_file is not None:
    with st.spinner("Processing document..."):
        # Save PDF to disk
        file_path = os.path.join("docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load, split, and create vector store
        chunks = load_and_split_docs(file_path)
        vectorstore = create_vector_store(chunks)
        st.session_state.vectorstore = vectorstore

        # Create QA chain
        chain = get_qa_chain(vectorstore)
        st.session_state.chain = chain

    st.success("✅ Document processed. You can now ask questions!")

# --- Chat Interface ---
if st.session_state.chain:
    user_input = st.chat_input("Ask a question about your document...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"query": user_input})
            print(f"response : {response}")
            answer = response.get("result", "⚠️ No answer found.")
            st.session_state.chat_history.append(("bot", answer))

# --- Display Chat History ---
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)
