import streamlit as st
import os
from rag_qa import load_and_split_docs, create_vector_store, get_qa_chain , detect_call_me_intent , extract_callback_entities
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

        # Detect "call me" intent
        if detect_call_me_intent(user_input):
            st.session_state.collect_callback_info = True
            st.session_state.chat_history.append(("bot", "Sure! Let me collect a few details so we can call you."))

            # NEW: extract entities from the user message
            extracted = extract_callback_entities(user_input)
            st.session_state.pre_fill = extracted
        elif st.session_state.chain:
            with st.spinner("Thinking..."):
                chat_pairs = st.session_state.chat_history[:-1]
                # Prepare formatted history for prompt
                formatted_history =  "\n".join(
                f"{'User' if r == 'user' else 'Assistant'}: {m}" for r, m in chat_pairs
                )
                response = st.session_state.chain.invoke({
                    "query": user_input,
                    "chat_history": formatted_history
                })
                answer = response.get("result", "Sorry, I couldn't find an answer.")
                st.session_state.chat_history.append(("bot", answer))

# --- Display Chat History ---
for i, (role, message) in enumerate(st.session_state.chat_history):
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)

            # Only show form *after* the assistant says the callback line
            if (
                st.session_state.get("collect_callback_info")
                and message.startswith("Sure! Let me collect")
            ):
                extracted = st.session_state.get("pre_fill", {})

                with st.form("callback_form"):
                    name = st.text_input("Your Name", value=extracted.get("name", ""))
                    phone = st.text_input("Phone Number", value=extracted.get("phone", ""))
                    time_str = extracted.get("time", "")
                    time = st.time_input("Preferred Time for Call")  # Optional: parse time_str
                    msg = st.text_area("Optional Message")

                    submitted = st.form_submit_button("Submit Request")
                    if submitted:
                        st.session_state.callback_details = {
                            "name": name,
                            "phone": phone,
                            "time": str(time),
                            "message": msg
                        }
                        st.success("📞 Thank you! We'll reach out at your preferred time.")
                        st.session_state.chat_history.append(
                            ("bot", f"Thanks {name}, we’ve noted your callback request for {time}.")
                        )
                        st.session_state.collect_callback_info = False

