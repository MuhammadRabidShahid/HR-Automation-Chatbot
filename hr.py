# Activate the required Conda environment
# conda activate C:\Users\rabid\Desktop\internship\hr\rag

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Streamlit title
st.title("HR Automation Chatbot")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """If the question is general, such as:
    - "Hi"
    - "Hello"
    - "How are you?"
    - "What’s up?"
    - "Can you help me?"
    - "What’s your name?"
    - "Tell me a joke."
    - "What time is it?"
    - "Where are you located?"
    - "How do I contact support?"
    - "What is your purpose?"
    - "Can you tell me about yourself?"
    - "What can you do?"
    - "How can I get started?"
    - "What are your hours of operation?"
    - "What is your favorite color?"
    - "Do you have any recommendations?"
    - "Can you provide some information?"
    - "How do I use this service?"
    - "What are the latest updates?"
    - "Can you explain something to me?"
    - "What’s new?"
    - "Do you have any news?"
    - "What’s the best way to reach you?"

    Provide a friendly and relevant response for these questions.

    If the question is specific to the provided context, answer it based on the context only. Please provide the most accurate response based on the question. If an answer cannot be found in the context, provide the closest related answer.

    <context>
    {context}
    <context>
    Questions: {input}"""
)



# Function to initialize vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Custom CSS for chatbot styling
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display old messages
for message in st.session_state["messages"]:
    if message["role"] == "user":
      st.markdown(
    f"""
    <div style='font-size:20px;'>
        <b>🧔:</b> {message['content']}
    </div>
    """,
    unsafe_allow_html=True
)

    else:
        st.markdown(f"**🤖 HR Bot:** {message['content']}")

# Chatbot interface
prompt1 = st.chat_input("🤖  How may I help you?...")

# Process the input and generate a response
if prompt1:
    vector_embedding()  # Ensure vectors are initialized
    
    st.session_state["messages"].append({"role": "user", "content": prompt1})
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()  # Start timing the response generation
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start 

    # Display the user input
    st.markdown(
    f"""
    <div style='font-size:20px;'>
        <b>🧔:</b> {prompt1}
    </div>
    """,
    unsafe_allow_html=True
)


    # Display the assistant's response
    st.markdown(f"**🤖 HR Bot:** {response['answer']}")
    
    st.session_state["messages"].append({"role": "assistant", "content": response['answer']})
    
    st.write("----------------------------------")
