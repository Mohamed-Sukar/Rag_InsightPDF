import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

st.set_page_config(page_title="PDF Analysis System", page_icon="📄", layout="wide")

st.title("📄 PDF Analysis System with Gemini")
st.markdown("Upload PDFs and ask questions based on their content.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200)
    
    if st.button("Process PDFs"):
        if not uploaded_files:
            st.error("Please upload at least one PDF file.")
        elif not os.environ.get("GEMINI_API_KEY"):
            st.error("GEMINI_API_KEY not found in environment variables.")
        else:
            with st.spinner("Processing PDFs..."):
                documents = []
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        for doc in docs:
                            # Keep the original file name in metadata
                            doc.metadata["source"] = file.name
                        documents.extend(docs)
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                
                if not documents:
                    st.error("No text could be extracted.")
                else:
                    try:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap)
                        )
                        chunks = text_splitter.split_documents(documents)
                        
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/embedding-001",
                            google_api_key=os.environ.get("GEMINI_API_KEY")
                        )
                        
                        vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            collection_name=f"rag_{uuid.uuid4().hex[:8]}"
                        )
                        
                        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                        st.success(f"Processing complete. Loaded {len(uploaded_files)} files, {len(chunks)} chunks.")
                    except Exception as e:
                        st.error(f"Error during processing: {e}")

def format_docs_with_sources(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        source = os.path.basename(source)
        page = doc.metadata.get("page", "Unknown")
        page_label = f"Page {int(page) + 1}" if str(page).isdigit() else f"Page {page}"
        formatted.append(f"[Source: {source} | {page_label}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def get_unique_sources(docs):
    seen = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "Unknown")
        page_label = f"Page {int(page) + 1}" if str(page).isdigit() else f"Page {page}"
        label = f"{source} ({page_label})"
        if label not in seen:
            seen.append(label)
    return seen

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_text := st.chat_input("Ask a question about your documents..."):
    if st.session_state.retriever is None:
        st.warning("Please upload and process PDF documents first.")
    elif not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not found in environment variables.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro-latest",
                        temperature=0,
                        google_api_key=os.environ.get("GEMINI_API_KEY")
                    )
                    
                    prompt_template = ChatPromptTemplate.from_template("""
You are a professional assistant analyzing document text.

Answer the user's question using ONLY the provided context below.
If the answer cannot be found in the context, output exactly: "I don't know." Do not invent information.

CRITICAL INSTRUCTIONS FOR TONE AND LANGUAGE:
1. You MUST answer the user in the EXACT SAME LANGUAGE they used to ask the question.
2. You MUST mirror the user's EXACT TONE, MANNER, and STYLE. For example, if the user speaks formally, respond formally. If they use slang, casual, or colloquial language, you must respond in the same casual slang or dialect. Adapt your persona completely to match how the user speaks to you.

Context:
{context}

Question:
{question}
""")
                    
                    docs = st.session_state.retriever.invoke(prompt_text)
                    context = format_docs_with_sources(docs)
                    
                    messages = prompt_template.format_messages(context=context, question=prompt_text)
                    response = llm.invoke(messages)
                    answer = response.content
                    
                    sources = get_unique_sources(docs)
                    if answer.strip().lower() not in ["i don't know.", "i don't know", '"i don\'t know."'] and sources:
                        sources_text = "\n\n**Sources:**\n- " + "\n- ".join(sources)
                        answer += sources_text
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error during response generation: {e}")
