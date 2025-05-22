import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pickle
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your Hugging Face API key
HUGGING_FACE_API_KEY = os.getenv("your api key")
if not HUGGING_FACE_API_KEY:
    st.error("Hugging Face API key not found. Please check your .env file.")
    st.stop()

API_URL = "https://api-inference.huggingface.co/models/decapoda-research/llama-7b-hf"
headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

# Debugging: Print API key and headers
st.write(f"Using API Key: {HUGGING_FACE_API_KEY}")
st.write(f"Headers: {headers}")

# Function to query Hugging Face API
def query_hugging_face(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''ABOUT           
    This app is an LLM-powered app built using:
    Streamlit
    LangChain
    Hugging Face LLaMA''')

def main():
    st.header("CHAT WITH PDF")

    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        with st.spinner("Processing PDF..."):
            pdf_reader = PdfReader(pdf)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            chunks = text_splitter.split_text(text=text)

            store_name = pdf.name[:-4]

            # Check if vector store exists
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                # Use alternative embeddings if sentence-transformers is not available
                try:
                    from langchain.embeddings import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                except ImportError:
                    from langchain.embeddings import HuggingFaceInstructEmbeddings
                    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
                
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

        # Accept user query
        query = st.text_input("Ask questions about your PDF file")
        if query:
            with st.spinner("Searching for answers..."):
                docs = VectorStore.similarity_search(query=query,  k=3)

                # Use Hugging Face API to generate a response
                response = query_hugging_face(query)
                if response:
                    st.write(response)

if __name__ == '__main__':
    main()