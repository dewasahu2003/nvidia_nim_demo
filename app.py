import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import re

# Load API key
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# Initialize LLM
llm = ChatNVIDIA(model="deepseek-ai/deepseek-r1")

def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Loading and embedding documents..."):
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./pdfs")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Document Embedding Completed")

st.title("NVIDIA NIM DEMO")

prompt_template = """
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
<question>
Question: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def parse_thought_process(response):
    # Extract content within <think> tags
    thought_process = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
    # Remove <think> tags and their content from the response
    final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return thought_process, final_answer

prompt1 = st.text_input("Enter Your Question from Documents")
if st.button("Start Doc Embedding"):
    vector_embedding()
    

if prompt1:
    with st.spinner("Retrieving relevant documents..."):
        doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, doc_chain)
        response = retriever_chain.invoke({"input": prompt1})
    
    # Parse the model's response
    thought_process, final_answer = parse_thought_process(response['answer'])
    
    if thought_process:
        with st.expander("Model's Thought Process"):
            for step in thought_process:
                st.write(step)
                st.write("---------------------")

    st.success("Response Generated!")
    st.subheader("AI Response:")
    st.write(final_answer)
    

    
    with st.expander("DOC Similarity Search (Relevant Context)"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------")
