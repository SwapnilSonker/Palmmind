
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_anthropic import ChatAnthropic
from transformers import pipeline
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    grouped_entities=True,
    device=-1  # Ensures it runs on CPU
)

def extract_callback_entities(text):
    entities = ner_pipeline(text)
    result = {"name": "", "phone": "", "time": ""}

    for ent in entities:
        label = ent["entity_group"]
        value = ent["word"]

        if "PER" in label and not result["name"]:
            result["name"] = value
        elif "MISC" in label and ("am" in value.lower() or "pm" in value.lower()) and not result["time"]:
            result["time"] = value
        elif value.isdigit() and len(value) >= 10 and not result["phone"]:
            result["phone"] = value

    return result

def detect_call_me_intent(text: str) -> bool:
    keywords = ["call me", "someone call", "please call", "phone call", "callback", "give me a call"]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def load_and_split_docs(file_path):
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks

def create_vector_store(chunks, persist_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

def load_vector_store(persist_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(persist_path, embeddings)

def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

    # Custom prompt supporting chat history
    prompt_template = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""
You are a helpful assistant. Use the following context and previous conversation to answer the user's question.

Context:
{context}

Chat History:
{chat_history}

User Question:
{question}

Answer in a friendly and informative tone:
"""
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=prompt_template,
        return_source_documents=True
    )
    return qa_chain


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Document QA with Claude")
    parser.add_argument("--file", type=str, required=True, help="Path to PDF or text file")
    args = parser.parse_args()

    print("📄 Loading and splitting document...")
    chunks = load_and_split_docs(args.file)

    print("🔎 Creating vector store...")
    vectorstore = create_vector_store(chunks)

    print("🧠 Setting up QA chain with Claude...")
    qa = get_qa_chain(vectorstore)

    print("\n✅ Ready. Type your question (or type 'exit' to quit):")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa({"query": query})
        print(f"\nClaude: {response['result']}\n")
